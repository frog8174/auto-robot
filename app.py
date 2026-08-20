#!/usr/bin/env python3
"""
Flask LAN remote controller for a 2‑wheel robot with MJPEG video streaming.
- Tested on Raspberry Pi OS with Python 3.11+, Flask 3.x, gpiozero, opencv-python.
- Default pins are BCM numbering. Adjust to your wiring!
- Open your browser to: http://<RPI_LAN_IP>:8000

Features:
- Live video stream from USB webcam.
- Direction buttons: Forward / Back / Left / Right / Stop
- Relay toggle button (active-low relays supported via RELAY_ACTIVE_HIGH)
- Keyboard control: ↑ ↓ ← → for movement; Space = Stop; R = relay toggle
- Simple REST endpoints: POST /api/move {cmd} and POST /api/relay {state|toggle}

Safety:
- Motors stop and camera is released on server shutdown or exception.
- Commands are edge-triggered; movement persists until Stop or another command.
"""

from flask import Flask, request, jsonify, Response
from gpiozero import Motor, OutputDevice
from signal import signal, SIGINT, SIGTERM
import atexit
import os
import cv2
import threading
import time
import platform
import datetime
import numpy as np

# Use a mock pin factory on non-Pi systems
is_pi = platform.system() == "Linux"
if not is_pi:
    from gpiozero.pins.mock import MockFactory, MockPWMPin
    from gpiozero import Device
    Device.pin_factory = MockFactory(pin_class=MockPWMPin)
else:
    from adafruit_servokit import ServoKit

AUTO_STEER_ENABLED = False
# ===================== CONFIG =====================
# --- GPIO ---
LEFT_IN1  = int(os.getenv("LEFT_IN1", 17))
LEFT_IN2  = int(os.getenv("LEFT_IN2", 27))
RIGHT_IN1 = int(os.getenv("RIGHT_IN1", 22))
RIGHT_IN2 = int(os.getenv("RIGHT_IN2", 23))
PWM_SPEED_DEFAULT = float(os.getenv("PWM_SPEED", 0.8))
RELAY_PIN = int(os.getenv("RELAY_PIN", 24))
RELAY_ACTIVE_HIGH = (os.getenv("RELAY_ACTIVE_HIGH", "0") == "1")

# --- Camera ---
CAM_DEVICE_INDEX = int(os.getenv("CAM_DEVICE_INDEX", 0))
CAM_FPS = int(os.getenv("CAM_FPS", 30))
CAM_WIDTH = int(os.getenv("CAM_WIDTH", 1280))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", 720))

# --- YOLO & PID ---
YOLO_PATH = "yolo/yolo_paper.onnx" # Ensure this file exists in the directory
YOLO_CONF_THRESHOLD = 0.5
YOLO_NMS_THRESHOLD = 0.4
YOLO_INPUT_SIZE = (320, 320)

PID_KP = 0.002
PID_KI = 0.000
PID_KD = 0.0005
pid_state = {"error": 0, "integral": 0, "prev_error": 0}

yolo_net = None
try:
    if os.path.exists(YOLO_PATH):
        yolo_net = cv2.dnn.readNetFromONNX(YOLO_PATH)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"YOLO model loaded from {YOLO_PATH}")
    else:
        print(f"WARNING: YOLO model not found at {YOLO_PATH}. Object detection disabled.")
except Exception as e:
    print(f"ERROR loading YOLO model: {e}")

# --- Pan/Tilt Servo ---
SERVO_CHANNELS = 16
PAN_CHANNEL = 0
TILT_CHANNEL = 1
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 180
HOME_PAN, HOME_TILT = 106, 42
SERVO_SPEED = 2 # Degrees per update cycle

# ===================== APP / DEVICES =====================
app = Flask(__name__)

# --- Robot Components ---
left_motor  = Motor(forward=LEFT_IN1, backward=LEFT_IN2, pwm=True)
right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2, pwm=True)
relay = OutputDevice(RELAY_PIN, active_high=RELAY_ACTIVE_HIGH, initial_value=False)

# --- Servo Components ---
if is_pi:
    try:
        kit = ServoKit(channels=SERVO_CHANNELS)
        pan_servo = kit.servo[PAN_CHANNEL]
        tilt_servo = kit.servo[TILT_CHANNEL]
        pan_servo.angle = HOME_PAN
        tilt_servo.angle = HOME_TILT
    except Exception as e:
        print(f"Could not initialize ServoKit: {e}")
        kit = None # Disable servos if init fails
else:
    kit = None # Mock environment

state = {
    "motion": "stop",
    "speed": PWM_SPEED_DEFAULT,
    "relay": False,
    "pan": HOME_PAN,
    "tilt": HOME_TILT,
}

# --- Camera Components ---
camera = cv2.VideoCapture(CAM_DEVICE_INDEX)
if not camera.isOpened():
    raise RuntimeError("Could not start camera.")

camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, CAM_FPS)
outputFrame = None
lock = threading.Lock()
capture_thread = None
stop_event = threading.Event() # Event to signal thread to stop

# ===================== BACKGROUND CAMERA THREAD =====================
def process_yolo_detection(frame):
    """Detects objects, finds closest to center, calculates PID, draws bbox."""
    global pid_state
    if yolo_net is None:
        return frame

    height, width = frame.shape[:2]
    frame_center_x = width // 2

    # Preprocess
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, YOLO_INPUT_SIZE, (0, 0, 0),
        swapRB=True, crop=False
    )
    yolo_net.setInput(blob)
    
    # Inference
    try:
        outputs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())
    except Exception as e:
        print(f"Inference error: {e}")
        return frame

    # Parse outputs
    predictions = outputs[0]
    
    # Handle different output shapes from ONNX export
    # Expected standard for post-processing: (N, 85) or (N, 5+classes)
    # Common shapes: (1, 25200, 85) or (1, 85, 25200) or (1, 5, 2100)
    
    # If 3D array (Batch, ...), squeeze first dim if it is 1
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]

    # Now predictions could be (25200, 85) or (85, 25200) or (5, 2100)
    # We want rows to be anchors/boxes, cols to be features (x,y,w,h,conf...)
    # If cols > rows (e.g. 25200 > 85), it's likely correct.
    # If rows < cols (e.g. 85 < 25200), we probably need to transpose.
    # But checking specific dimensions is safer. 
    # Typically features are < 100 (5 + 80 classes), and anchors are > 1000.
    
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.transpose()
    
    # Now predictions should be (Anchors, Features) e.g. (25200, 85) or (2100, 5)

    class_ids = []
    confidences = []
    boxes = []

    x_factor = width / YOLO_INPUT_SIZE[0]
    y_factor = height / YOLO_INPUT_SIZE[1]

    # Only process rows with confidence > threshold to speed up
    # Note: Column 4 is object confidence
    if predictions.shape[1] >= 5:
        valid_rows = predictions[predictions[:, 4] > YOLO_CONF_THRESHOLD]
    else:
         valid_rows = []

    for row in valid_rows:
        confidence = row[4]
        # Check if we have class scores (index 5 onwards)
        if len(row) > 5:
            scores = row[5:]
            class_id = np.argmax(scores)
            score = scores[class_id]
        else:
            # If no class scores (e.g. simplified model), assume class 0 with obj conf
            class_id = 0
            score = confidence
            
        if score > 0.5:  # Class score/Object threshold
            cx, cy, w, h = row[0], row[1], row[2], row[3]

            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            w = int(w * x_factor)
            h = int(h * y_factor)

            boxes.append([left, top, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                               YOLO_CONF_THRESHOLD, YOLO_NMS_THRESHOLD)

    best_obj = None
    min_dist = float('inf')

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            obj_center_x = x + w // 2

            # 畫出所有偵測到的框（淡黃色）
            cv2.rectangle(frame, (x, y),
                          (x + w, y + h), (0, 255, 255), 1)

            # 找離畫面中心最近的物件
            dist = abs(frame_center_x - obj_center_x)
            if dist < min_dist:
                min_dist = dist
                best_obj = (obj_center_x, x, y, w, h, class_ids[i])

    # === PID 計算 & log ===
    if best_obj:
        obj_cx, x, y, w, h, cls_id = best_obj

        # 用綠框標示「選中的那一個」物件
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame,
                   (obj_cx, y + h // 2),
                   5, (0, 0, 255), -1)

        # error：畫面中心 - 物件中心 (px)
        # error > 0  → 物件在畫面左邊 → 車子應該向左修正
        # error < 0  → 物件在畫面右邊 → 車子應該向右修正
        error = frame_center_x - obj_cx

        P = PID_KP * error
        pid_state["integral"] += error
        I = PID_KI * pid_state["integral"]
        D = PID_KD * (error - pid_state["prev_error"])

        output = P + I + D
        pid_state["prev_error"] = error

        # Terminal log：清楚顯示目前狀態
        print(
            f"[PID] center={frame_center_x} obj_x={obj_cx} "
            f"err={error:+4d}  P={P:+.4f} I={I:+.4f} D={D:+.4f}  out={output:+.4f}"
        )

        # 畫在影像上
        cv2.putText(frame,
                    f"ERR:{error:+d} PID:{output:+.2f}",
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # === 之後要真的「邊前進邊修正」可以這樣做 ===
        # 現在依照你的要求「先不控制移動」，所以這段只當參考，預設不啟動。
        if AUTO_STEER_ENABLED and state.get("motion") == "forward":
            # 基礎前進速度（用目前 state["speed"]）
            base_speed = state.get("speed", PWM_SPEED_DEFAULT)

            # 這裡把 PID 輸出當作轉向量，適度縮放避免太大
            steer_gain = 1.0  # 可以視情況調小，例如 0.5
            steer = float(output) * steer_gain

            left_speed = base_speed + steer
            right_speed = base_speed - steer

            # 實際設定馬達速度（函式已經會幫你 clamp 到 [-1, 1]）
            set_motor_speeds(left_speed, right_speed)

    return frame

def capture_frames():
    """Reads frames from camera, encodes them to JPEG, and stores in a global var."""
    global outputFrame, lock
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            print("Camera read failed, stopping thread.")
            break
        
        # Rotate the frame by 90 degrees clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Run YOLO detection & PID logic
        frame = process_yolo_detection(frame)

        # Encode the frame to JPEG with a specific quality.
        # Lower quality (e.g., 75-80) means less data and lower latency.
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            continue

        with lock:
            outputFrame = buffer.tobytes()
    
    if camera.isOpened():
        camera.release()
        print("Camera released by capture thread.")

# ===================== MOTOR & SERVO HELPERS =====================
def stop_all():
    left_motor.stop()
    right_motor.stop()
    state["motion"] = "stop"

def go_forward(speed):
    left_motor.forward(speed)
    right_motor.forward(speed)
    state["motion"] = "forward"

def go_backward(speed):
    left_motor.backward(speed)
    right_motor.backward(speed)
    state["motion"] = "backward"

def turn_left(speed):
    left_motor.backward(speed)
    right_motor.forward(speed)
    state["motion"] = "left"

def turn_right(speed):
    left_motor.forward(speed)
    right_motor.backward(speed)
    state["motion"] = "right"

def set_motor_speeds(left_speed, right_speed):
    """Sets individual motor speeds. Expects values from -1.0 to 1.0."""
    # Clamp values to be safe
    left_speed = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))

    if left_speed > 0:
        left_motor.forward(left_speed)
    else:
        left_motor.backward(abs(left_speed))

    if right_speed > 0:
        right_motor.forward(right_speed)
    else:
        right_motor.backward(abs(right_speed))
    
    # Update state (this is a bit tricky with differential drive)
    if left_speed == 0 and right_speed == 0:
        state["motion"] = "stop"
    else:
        state["motion"] = "drive"
    state["left_speed"] = left_speed
    state["right_speed"] = right_speed

def set_pan_tilt(pan_val, tilt_val):
    """Moves pan/tilt servos based on input values from -1.0 to 1.0."""
    if not kit: return

    # Pan
    new_pan = state["pan"] - pan_val * SERVO_SPEED
    new_pan = max(PAN_MIN, min(new_pan, PAN_MAX))
    if abs(new_pan - state["pan"]) > 0.1: # Only update if change is significant
        state["pan"] = new_pan
        pan_servo.angle = state["pan"]

    # Tilt
    new_tilt = state["tilt"] - tilt_val * SERVO_SPEED
    new_tilt = max(TILT_MIN, min(new_tilt, TILT_MAX))
    if abs(new_tilt - state["tilt"]) > 0.1: # Only update if change is significant
        state["tilt"] = new_tilt
        tilt_servo.angle = state["tilt"]

def home_pan_tilt():
    """Resets pan/tilt servos to their home positions."""
    if not kit: return
    state["pan"] = HOME_PAN
    state["tilt"] = HOME_TILT
    pan_servo.angle = HOME_PAN
    tilt_servo.angle = HOME_TILT


def relay_on():
    relay.on()
    state["relay"] = True

def relay_off():
    relay.off()
    state["relay"] = False

def relay_toggle():
    if state["relay"]:
        relay_off()
    else:
        relay_on()

# ===================== REST API =====================
@app.post('/api/move')
def api_move():
    data = request.get_json(silent=True) or {}
    cmd = (data.get('cmd') or '').lower()
    speed = float(data.get('speed', state['speed']))
    speed = max(0.0, min(1.0, speed))
    state['speed'] = speed

    actions = {
        'forward': go_forward,
        'backward': go_backward,
        'left': turn_left,
        'right': turn_right,
        'stop': stop_all
    }
    action = actions.get(cmd)
    if action:
        if cmd != 'stop':
            action(speed)
        else:
            action()
    else:
        return jsonify({"ok": False, "error": f"unknown cmd: {cmd}"}), 400
    return jsonify({"ok": True, "state": state})

@app.post('/api/drive')
def api_drive():
    data = request.get_json(silent=True) or {}
    left_speed = float(data.get('left', 0.0))
    right_speed = float(data.get('right', 0.0))

    set_motor_speeds(left_speed, right_speed)
    return jsonify({"ok": True, "state": state})

@app.post('/api/pan_tilt')
def api_pan_tilt():
    data = request.get_json(silent=True) or {}
    pan_val = float(data.get('pan', 0.0))
    tilt_val = float(data.get('tilt', 0.0))
    set_pan_tilt(pan_val, tilt_val)
    return jsonify({"ok": True, "state": state})

@app.post('/api/pan_tilt/home')
def api_pan_tilt_home():
    home_pan_tilt()
    return jsonify({"ok": True, "state": state})


@app.post('/api/photo')
def api_photo():
    global outputFrame, lock
    
    # Ensure photos directory exists
    photo_dir = "photos"
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)

    with lock:
        if outputFrame is None:
             return jsonify({"ok": False, "error": "No camera frame available"}), 503
        frame_bytes = outputFrame

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    filepath = os.path.join(photo_dir, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(frame_bytes)
        print(f"Photo saved: {filepath}")
        return jsonify({"ok": True, "file": filename})
    except Exception as e:
        print(f"Failed to save photo: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post('/api/relay')
def api_relay():
    data = request.get_json(silent=True) or {}
    action = (data.get('state') or '').lower()
    if action == 'on': relay_on()
    elif action == 'off': relay_off()
    elif action in ('toggle', ''): relay_toggle()
    else:
        return jsonify({"ok": False, "error": f"unknown relay state: {action}"}), 400
    return jsonify({"ok": True, "state": state})

@app.get('/api/status')
def api_status():
    return jsonify({"ok": True, "state": state})

# ===================== VIDEO STREAMING =====================
def generate_frames():
    """Yields pre-encoded JPEG frames to the client."""
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                time.sleep(0.01) # Wait briefly for frame to become available
                continue
            frame_bytes = outputFrame
        
        # Yield the frame in the multipart response format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              frame_bytes + b'\r\n')
        
        # Small delay to prevent overwhelming a single client, allowing others to connect.
        # This can be adjusted or removed depending on the desired behavior.
        time.sleep(1 / CAM_FPS) # Match the camera's frame rate

@app.get('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ===================== UI (INLINE HTML) =====================
@app.get('/')
def index():
    return Response(INLINE_HTML, mimetype='text/html')

INLINE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RPI Robot Controller</title>
  <style>
    :root { --pad: 14px; --radius: 16px; --gap: 12px; }
    html, body { height: 100%; margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:#0b1220; color:#e7ecf4; }
    .wrap { max-width: 900px; margin: 0 auto; padding: 24px; display:flex; flex-direction:column; gap: var(--gap); }
    header { display:flex; align-items:center; justify-content:space-between; flex-wrap: wrap; }
    .card { background: #121a2b; border:1px solid #1f2a44; border-radius: var(--radius); padding: var(--pad); box-shadow: 0 8px 24px rgba(0,0,0,0.25); }
    .grid { display:grid; gap: var(--gap); }

    .video-container { background: #000; border-radius: calc(var(--radius) - 4px); overflow:hidden; margin-bottom: var(--gap); border:1px solid #1f2a44;}
    .video-container img { display: block; width: 100%; height: auto; }

    .dpad { grid-template-columns: repeat(3, 100px); grid-template-rows: repeat(3, 100px); justify-content:center; }
    .btn { border:1px solid #2a3a5e; background: linear-gradient(180deg,#1a2743,#0e1629); color:#e7ecf4; border-radius: 14px; font-weight:600; cursor:pointer; user-select:none; outline:none; font-size:16px; }
    .btn:active { transform: scale(0.98); }
    .btn.big { font-size:18px; }

    .up    { grid-column:2; grid-row:1; }
    .left  { grid-column:1; grid-row:2; }
    .stop  { grid-column:2; grid-row:2; background:#34202a; border-color:#6b2a3a; }
    .right { grid-column:3; grid-row:2; }
    .down  { grid-column:2; grid-row:3; }

    .relay-row { display:flex; align-items:center; gap: 12px; flex-wrap:wrap; }
    .pill { padding: 10px 16px; border-radius: 999px; background:#12213a; border:1px solid #1e365d; }
    .switch { display:inline-flex; align-items:center; gap:8px; }
    .switch input { width: 48px; height: 28px; appearance:none; background:#23365c; border-radius: 999px; position:relative; outline:none; cursor:pointer; border:1px solid #2a406e; }
    .switch input:checked { background:#2f885a; }
    .switch input::after { content:""; position:absolute; top:2px; left:2px; width:24px; height:24px; background:#e7ecf4; border-radius:50%; transition: left .15s ease; }
    .switch input:checked::after { left:22px; }

    .row { display:flex; gap:var(--gap); align-items:center; justify-content:space-between; flex-wrap:wrap; }
    .status { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 14px; opacity:0.9; }
    .footer { opacity: 0.6; font-size: 13px; text-align:center; padding-top:8px; }
    .slider { width: 220px; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h2>Raspberry Pi Robot Controller</h2>
      <div class="pill">Use arrow keys to drive · Space = Stop · R = Relay</div>
    </header>

    <div class="video-container">
        <img src="/video_feed" alt="Live Video Feed" />
    </div>

    <div class="card grid dpad">
      <button class="btn up"    data-cmd="forward">▲</button>
      <button class="btn left"  data-cmd="left">◀</button>
      <button class="btn stop big"  data-cmd="stop">■ STOP</button>
      <button class="btn right" data-cmd="right">▶</button>
      <button class="btn down"  data-cmd="backward">▼</button>
    </div>

    <div class="card">
      <div class="row">
        <div class="relay-row">
          <label class="switch">
            <input id="relay-toggle" type="checkbox"/>
            <span>Relay</span>
          </label>
          <button id="relay-on" class="btn">Relay ON</button>
          <button id="relay-off" class="btn">Relay OFF</button>
        </div>
        <div>
          <label>Speed: <input id="speed" type="range" min="0" max="1" step="0.05" class="slider"/></label>
        </div>
      </div>
      <div class="status" id="status">—</div>
      <div class="status" id="joystick-status">No joystick connected</div>
      <div class="status" id="pan-status">Pan: —</div>
      <div class="status" id="tilt-status">Tilt: —</div>
      <pre class="status" id="joystick-debug" style="display: none;"></pre>
    </div>

    <div class="footer">H-bridge control via gpiozero.Motor · Relay via gpiozero.OutputDevice</div>
  </div>

  <script>
    const statusEl = document.getElementById('status');
    const speedEl = document.getElementById('speed');
    const relayToggleEl = document.getElementById('relay-toggle');
    const joystickStatusEl = document.getElementById('joystick-status');
    const panStatusEl = document.getElementById('pan-status');
    const tiltStatusEl = document.getElementById('tilt-status');
    const joystickDebugEl = document.getElementById('joystick-debug');

    let joystickLoop = null;
    let lastJoystickData = { left: 0, right: 0, pan: 0, tilt: 0, relay: 0, home: 0, photo: 0 };

    async function post(url, body) {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {})
      });
      return await res.json();
    }

    function updateStatus(data) {
      statusEl.textContent = JSON.stringify(data.state, null, 2);
      if (typeof data.state?.relay === 'boolean') {
        relayToggleEl.checked = data.state.relay;
      }
      if (typeof data.state?.speed === 'number') {
        speedEl.value = data.state.speed;
      }
      if (typeof data.state?.pan === 'number') {
        panStatusEl.textContent = `Pan: ${data.state.pan.toFixed(0)}°`;
      }
      if (typeof data.state?.tilt === 'number') {
        tiltStatusEl.textContent = `Tilt: ${data.state.tilt.toFixed(0)}°`;
      }
    }

    async function refresh() {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.ok) updateStatus(data);
    }

    // D-pad button handlers
    document.querySelectorAll('.btn[data-cmd]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const cmd = btn.dataset.cmd;
        const speed = parseFloat(speedEl.value);
        const data = await post('/api/move', { cmd, speed });
        if (data.ok) updateStatus(data);
      });
    });

    // Relay controls
    relayToggleEl.addEventListener('change', async () => {
      const data = await post('/api/relay', { state: relayToggleEl.checked ? 'on' : 'off' });
      if (data.ok) updateStatus(data);
    });
    document.getElementById('relay-on').addEventListener('click', async () => {
      const data = await post('/api/relay', { state: 'on' });
      if (data.ok) updateStatus(data);
    });
    document.getElementById('relay-off').addEventListener('click', async () => {
      const data = await post('/api/relay', { state: 'off' });
      if (data.ok) updateStatus(data);
    });

    // Keyboard controls
    window.addEventListener('keydown', async (e) => {
      const speed = parseFloat(speedEl.value);
      let cmd = null;
      if (e.key === 'ArrowUp') cmd = 'forward';
      else if (e.key === 'ArrowDown') cmd = 'backward';
      else if (e.key === 'ArrowLeft') cmd = 'left';
      else if (e.key === 'ArrowRight') cmd = 'right';
      else if (e.key === ' ') cmd = 'stop';
      else if (e.key.toLowerCase() === 'r') {
        const data = await post('/api/relay', { state: 'toggle' });
        if (data.ok) updateStatus(data);
        return;
      }
      if (cmd) {
        const data = await post('/api/move', { cmd, speed });
        if (data.ok) updateStatus(data);
      }
    });

    // Gamepad API
    window.addEventListener("gamepadconnected", (e) => {
      joystickStatusEl.textContent = `Joystick connected: ${e.gamepad.id}`;
      joystickDebugEl.style.display = 'block';
      if (!joystickLoop) {
        joystickLoop = requestAnimationFrame(pollJoystick);
      }
    });

    window.addEventListener("gamepaddisconnected", (e) => {
      joystickStatusEl.textContent = "No joystick connected";
      joystickDebugEl.style.display = 'none';
      if (joystickLoop) {
        cancelAnimationFrame(joystickLoop);
        joystickLoop = null;
      }
    });

    function pollJoystick() {
      try {
        const gamepads = navigator.getGamepads();
        if (gamepads.length === 0 || !gamepads[0]) {
          joystickLoop = requestAnimationFrame(pollJoystick);
          return;
        }

        const gp = gamepads[0];
        const deadzone = 0.2;

        // --- Debug Display ---
        let debugTxt = "Axes:\n";
        gp.axes.forEach((axis, i) => {
          debugTxt += `  ${i}: ${axis.toFixed(3)}\n`;
        });
        debugTxt += "\nButtons:\n";
        gp.buttons.forEach((button, i) => {
          if (button.pressed) {
              debugTxt += `  ${i}: pressed (${button.value.toFixed(3)})\n`;
          }
        });
        joystickDebugEl.textContent = debugTxt;

        // --- Driving (Left stick) ---
        // Default for Xbox controller: axes[0]=LX, axes[1]=LY
        let ly = -gp.axes[1]; // Y is inverted
        let lx = gp.axes[0];
        if (Math.abs(ly) < deadzone) ly = 0;
        if (Math.abs(lx) < deadzone) lx = 0;

        let left = ly + lx;
        let right = ly - lx;
        
        const driveScale = Math.max(1, Math.abs(left), Math.abs(right));
        left /= driveScale;
        right /= driveScale;

        // --- Speed (Right trigger) ---
        // Button 6 is the right trigger (RT) on many controllers
        const speedMultiplier = (gp.buttons.length > 6 && gp.buttons[6]) ? gp.buttons[6].value : 1.0;
        left *= speedMultiplier;
        right *= speedMultiplier;

        if (Math.abs(left - lastJoystickData.left) > 0.05 || Math.abs(right - lastJoystickData.right) > 0.05) {
          post('/api/drive', { left: left, right: right }).then(data => {
            if (data.ok) updateStatus(data);
          });
          lastJoystickData.left = left;
          lastJoystickData.right = right;
        }

        // --- Pan/Tilt (Right stick) ---
        // Default for Xbox controller: axes[2]=RX, axes[3]=RY
        let ry = -gp.axes[3]; // Y is inverted
        let rx = gp.axes[2];
        if (Math.abs(ry) < deadzone) ry = 0;
        if (Math.abs(rx) < deadzone) rx = 0;

        if (Math.abs(rx) > 0.05 || Math.abs(ry) > 0.05) {
          post('/api/pan_tilt', { pan: rx, tilt: ry }).then(data => {
            if (data.ok) updateStatus(data);
          });
        }

        // --- Pan/Tilt Home (Right stick button) ---
        // Button 9 is usually the right stick button
        const homeButtonPressed = (gp.buttons.length > 9 && gp.buttons[9]) && gp.buttons[9].pressed;
        if (homeButtonPressed && !lastJoystickData.home) {
          post('/api/pan_tilt/home').then(data => {
            if (data.ok) updateStatus(data);
          });
        }
        lastJoystickData.home = homeButtonPressed;

        // --- Relay (A button) ---
        // Button 0 is usually the 'A' button
        const relayButtonPressed = (gp.buttons.length > 0 && gp.buttons[0]) && gp.buttons[0].pressed;
        if (relayButtonPressed && !lastJoystickData.relay) {
          post('/api/relay', { state: 'toggle' }).then(data => {
            if (data.ok) updateStatus(data);
          });
        }
        lastJoystickData.relay = relayButtonPressed;

        // --- Photo (Y button) ---
        // Button 3 is usually the 'Y' button
        const photoButtonPressed = (gp.buttons.length > 3 && gp.buttons[3]) && gp.buttons[3].pressed;
        if (photoButtonPressed && !lastJoystickData.photo) {
          post('/api/photo').then(data => {
             if (data.ok) {
               console.log("Photo taken:", data.file);
               // Flash effect or status update could go here
               statusEl.textContent = "Photo saved: " + data.file;
               setTimeout(() => refresh(), 2000);
             }
          });
        }
        lastJoystickData.photo = photoButtonPressed;

      } catch (e) {
        console.error("Joystick error:", e);
        joystickDebugEl.textContent = "Error: " + e.message;
        joystickDebugEl.style.display = 'block';
      }

      joystickLoop = requestAnimationFrame(pollJoystick);
    }

    refresh();
  </script>
</body>
</html>
"""

# ===================== GRACEFUL SHUTDOWN =====================
def _cleanup(*_):
    print("Cleaning up resources...")
    stop_event.set() # Signal the capture thread to stop
    try:
        stop_all()
        relay_off()
        if is_pi and kit:
            home_pan_tilt()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    time.sleep(0.5) # Give thread time to release camera
    print("Cleanup complete.")

signal(SIGINT, _cleanup)
signal(SIGTERM, _cleanup)
atexit.register(_cleanup)

# ===================== MAIN =====================
if __name__ == '__main__':
    # Start the background thread for camera capture
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    debug = (os.getenv('FLASK_DEBUG', '0') == '1')

    print(f"\nStarting Robot Controller on http://{host}:{port}")
    print(f"Left motor pins: FWD={LEFT_IN1}, BWD={LEFT_IN2}; Right motor pins: FWD={RIGHT_IN1}, BWD={RIGHT_IN2}")
    print(f"Relay pin={RELAY_PIN}, active_high={RELAY_ACTIVE_HIGH}")
    print(f"Camera device: {CAM_DEVICE_INDEX}, Resolution: {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS} FPS")
    if is_pi and kit:
        print(f"ServoKit initialized for Pan/Tilt control.")
    elif is_pi:
        print("ServoKit failed to initialize. Pan/Tilt control is disabled.")
    else:
        print("Not on a Raspberry Pi. Pan/Tilt control is disabled.")


    # Use threaded=True to handle multiple requests (e.g., UI + video stream)
    app.run(host=host, port=port, debug=debug, threaded=True)

