#!/usr/bin/env python3
"""
Paper Collector Robot - Remote Processing Version
Target Hardware: Raspberry Pi 4
Features:
- Offloads YOLO inference to a remote HTTP server
- "Step-by-Step" movement logic preserved
- Local Wall Detection (for safety speed)
"""

import os
import cv2
import time
import threading
import numpy as np
import atexit
import requests
from signal import signal, SIGINT, SIGTERM
from flask import Flask, Response

# --- Hardware Imports ---
import platform
is_pi = platform.system() == "Linux"

if is_pi:
    from gpiozero import Motor, OutputDevice
    from adafruit_servokit import ServoKit
else:
    from gpiozero.pins.mock import MockFactory, MockPWMPin
    from gpiozero import Device, Motor, OutputDevice
    Device.pin_factory = MockFactory(pin_class=MockPWMPin)
    kit = None

# ===================== CONFIGURATION =====================

# *** IMPORTANT: SET YOUR PC/SERVER IP HERE ***
SERVER_URL = "http://192.168.23.104:5000/detect" # <--- CHANGE THIS

# GPIO Pins (BCM)
LEFT_IN1 = 17
LEFT_IN2 = 27
RIGHT_IN1 = 22
RIGHT_IN2 = 23
RELAY_PIN = 24

# Servo Config
SERVO_CHANNELS = 16
PAN_CHANNEL = 0
TILT_CHANNEL = 1

# Angles
PAN_FIXED = 106     
TILT_NORMAL = 0     
TILT_DOWN_LIMIT = 42
SERVO_STEP = 5      

# Camera Config
CAM_WIDTH = 960
CAM_HEIGHT = 720
CAM_FPS = 30

# Movement Config
SPEED_SEARCH = 0.4
SPEED_TRACK = 0.4 
SPEED_BACKUP = 0.3
STEP_TURN_DURATION = 0.2
STEP_MOVE_DURATION = 0.25
BACKUP_DURATION = 0.5
LOST_TIMEOUT = 5.0

# PID Config
PID_KP = 0.003
PID_KI = 0.000
PID_KD = 0.000

# Wall Detection Config
WALL_COLOR_LOWER = np.array([0, 0, 0])
WALL_COLOR_UPPER = np.array([180, 255, 60])
WALL_AREA_THRESH = 0.3

# ===================== GLOBAL STATE =====================
state = {
    "status": "booting",
    "tilt": TILT_NORMAL,
    "pan": PAN_FIXED,
    "last_seen_time": 0,
    "object_detected": False,
    "current_action": "idle"
}

frame_lock = threading.Lock()
output_frame = None
stop_event = threading.Event()

# ===================== HARDWARE INIT =====================
try:
    left_motor = Motor(forward=LEFT_IN1, backward=LEFT_IN2, pwm=True)
    right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2, pwm=True)
except Exception as e:
    print(f"Motor Init Error: {e}")

servo_kit = None
pan_servo = None
tilt_servo = None

if is_pi:
    try:
        servo_kit = ServoKit(channels=SERVO_CHANNELS)
        pan_servo = servo_kit.servo[PAN_CHANNEL]
        tilt_servo = servo_kit.servo[TILT_CHANNEL]
    except Exception as e:
        print(f"Servo Init Error: {e}")

# ===================== HELPERS =====================

def set_servos(pan, tilt):
    if not servo_kit: return
    pan = max(0, min(180, pan))
    tilt = max(0, min(TILT_DOWN_LIMIT, tilt))
    try:
        pan_servo.angle = pan
        tilt_servo.angle = tilt
        state["pan"] = pan
        state["tilt"] = tilt
    except Exception as e:
        print(f"Servo Error: {e}")

def drive(left, right):
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    if left > 0: left_motor.forward(left)
    else: left_motor.backward(abs(left))
    if right > 0: right_motor.forward(right)
    else: right_motor.backward(abs(right))

def stop_motors():
    left_motor.stop()
    right_motor.stop()

def detect_wall(frame):
    h, w = frame.shape[:2]
    roi_y = int(h*0.75)
    roi = frame[roi_y:h, 0:w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WALL_COLOR_LOWER, WALL_COLOR_UPPER)
    ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
    return ratio, mask, roi_y

def remote_inference(frame):
    """Send frame to PC server and get detections."""
    try:
        # Encode frame to jpg
        ret, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret: return None
        
        # Post to server (Timeout is critical!)
        response = requests.post(
            SERVER_URL, 
            files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
            timeout=1.0 
        )
        
        if response.status_code == 200:
            dets = response.json().get("detections", [])
            if dets:
                print(f"Server returned {len(dets)} detections.")
            return dets
    except Exception as e:
        print(f"API Error: {e}")
    return []

def process_vision(frame):
    h, w = frame.shape[:2]
    center_x = w // 2
    best_obj = None

    # 1. Remote YOLO
    detections = remote_inference(frame)
    
    min_dist = float('inf')
    if detections:
        for det in detections:
            bx, by, bw, bh = int(det['x']), int(det['y']), int(det['w']), int(det['h'])
            
            # Draw Box
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
            
            obj_cx = bx + bw // 2
            dist = abs(center_x - obj_cx)
            
            if dist < min_dist:
                min_dist = dist
                best_obj = (bx, by, bw, bh)

    # 2. Local Wall Detection
    wall_ratio, wall_mask, roi_y = detect_wall(frame)
    if wall_ratio > 0.01:
        roi = frame[roi_y:, :]
        colored_mask = np.zeros_like(roi)
        colored_mask[wall_mask > 0] = [0, 0, 255]
        frame[roi_y:, :] = cv2.addWeighted(roi, 1.0, colored_mask, 0.5, 0)

    if wall_ratio > WALL_AREA_THRESH:
        cv2.putText(frame, "WALL!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, best_obj, wall_ratio

# ===================== LOGIC LOOP =====================

def robot_logic():
    global state
    print("Logic Thread Started...")
    set_servos(PAN_FIXED, TILT_NORMAL)
    
    # Initial Move
    drive(SPEED_TRACK, SPEED_TRACK)
    time.sleep(0.5)
    stop_motors()
    time.sleep(0.2)
    
    prev_error = 0
    integral = 0
    
    while not stop_event.is_set():
        # --- STOP & LOOK ---
        stop_motors()
        
        # Flush buffer
        for _ in range(2): camera.grab()
        
        ret, raw_frame = camera.read()
        if not ret: 
            time.sleep(0.1)
            continue
            
        # Rotate FIRST, so the server gets the correct orientation
        frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Call API (now sending rotated frame)
        processed_frame, best_obj, wall_ratio = process_vision(frame)
        
        # Update Web Stream
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            with frame_lock:
                global output_frame
                output_frame = buffer.tobytes()

        current_time = time.time()
        
        # --- DECIDE ---
        if best_obj:
            state["object_detected"] = True
            state["last_seen_time"] = current_time
            state["current_action"] = "tracking"
            
            bx, by, bw, bh = best_obj
            obj_cx = bx + bw // 2
            obj_cy = by + bh // 2
            frame_h, frame_w = frame.shape[:2]
            
            # Tilt Logic
            if obj_cy > (frame_h * 0.75):
                new_tilt = state["tilt"] + SERVO_STEP
                if new_tilt <= TILT_DOWN_LIMIT:
                    set_servos(PAN_FIXED, new_tilt)
            
            # Wall Avoidance
            if wall_ratio > WALL_AREA_THRESH:
                drive(-SPEED_BACKUP, -SPEED_BACKUP)
                time.sleep(BACKUP_DURATION)
                stop_motors()
                continue
            
            # PID Drive
            error = (frame_w // 2) - obj_cx
            P = PID_KP * error
            integral += error
            I = PID_KI * integral
            D = PID_KD * (error - prev_error)
            turn = P + I + D
            prev_error = error
            
            drive(SPEED_TRACK + turn, SPEED_TRACK - turn)
            time.sleep(STEP_MOVE_DURATION)
            stop_motors()
            
        else:
            # Lost Logic
            state["object_detected"] = False
            time_lost = current_time - state["last_seen_time"]
            
            # If never seen (last_seen_time=0) OR lost for > 5s
            if state["last_seen_time"] == 0 or time_lost > LOST_TIMEOUT:
                if state["current_action"] != "lost_search":
                    print("Action: Search Mode (Reset Tilt & Turn)")
                    set_servos(PAN_FIXED, TILT_NORMAL)
                    state["current_action"] = "lost_search"
                
                # Turn Left Step
                drive(-SPEED_SEARCH, SPEED_SEARCH)
                time.sleep(STEP_TURN_DURATION)
                stop_motors()
                time.sleep(0.2) # Reduced wait for faster scanning
            else:
                # Recently lost (waiting for re-acquire)
                if state["current_action"] != "waiting":
                    print(f"Action: Waiting for object ({time_lost:.1f}s)...")
                    state["current_action"] = "waiting"
                time.sleep(0.1)

# ===================== FLASK =====================
app = Flask(__name__)

def generate():
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            data = output_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup():
    stop_event.set()
    stop_motors()
    if camera.isOpened(): camera.release()

atexit.register(cleanup)
signal(SIGINT, lambda s, f: cleanup())

if __name__ == '__main__':
    print("Initializing Camera...")
    camera = cv2.VideoCapture(0)
    # Enable MJPEG compression for faster framerate
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAM_FPS)
    
    t = threading.Thread(target=robot_logic)
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
