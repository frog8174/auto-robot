#!/usr/bin/env python3
"""
Paper Collector Robot - Specialized Logic
Target Hardware: Raspberry Pi 4
Features:
- Circular Arena Navigation (Dark Wall Avoidance)
- Object Tracking (Paper) with YOLO
- Dynamic Camera Tilt (Tracking downwards)
- State Machine: Search, Track, Avoid, Lost

Usage:
python paper_bot.py
"""

import os
import cv2
import time
import datetime
import threading
import numpy as np
import atexit
from signal import signal, SIGINT, SIGTERM

from flask import Flask, Response

# --- Hardware Imports ---
import platform
is_pi = platform.system() == "Linux"

if is_pi:
    from gpiozero import Motor, OutputDevice
    from adafruit_servokit import ServoKit
else:
    # Mocking for non-Pi testing
    from gpiozero.pins.mock import MockFactory, MockPWMPin
    from gpiozero import Device, Motor, OutputDevice
    Device.pin_factory = MockFactory(pin_class=MockPWMPin)
    kit = None

# ===================== CONFIGURATION =====================

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
PAN_FIXED = 106     # Fixed Pan Angle
TILT_NORMAL = 0     # Walking / Searching Angle (0 is Head Up)
TILT_DOWN_LIMIT = 42 # Looking straight down (42 is lowest)
TILT_START = 75     # Initial tracking start (Deprecated/Unused?)
SERVO_STEP = 5      # Degrees to move per adjustment

# Camera Config
CAM_WIDTH = 640     # Lower res for faster processing on Pi 4
CAM_HEIGHT = 480
CAM_FPS = 30

# YOLO Config
YOLO_PATH = "yolo/yolo_paper.onnx"
YOLO_CONF_THRESH = 0.5
YOLO_NMS_THRESH = 0.4
YOLO_INPUT_SIZE = (320, 320)

# Movement Config
SPEED_SEARCH = 0.4
SPEED_TRACK = 0.4       # Slightly increased because we only move in bursts
SPEED_BACKUP = 0.3
STEP_TURN_DURATION = 0.2    # Seconds to turn left when searching
STEP_MOVE_DURATION = 0.25   # Seconds to move forward when tracking (The "Step")
BACKUP_DURATION = 0.5       # Seconds to backup when wall detected
LOST_TIMEOUT = 5.0          # Seconds before resetting tilt and searching

# PID Config (Heading)
PID_KP = 0.003  # Increased slightly since we need stronger correction per step
PID_KI = 0.000
PID_KD = 0.000

# Wall Detection Config
# HSV Thresholds for "Dark/Black" Wall
WALL_COLOR_LOWER = np.array([0, 0, 0])
WALL_COLOR_UPPER = np.array([180, 255, 60]) # Low Value (Brightness) = Dark
WALL_AREA_THRESH = 0.3 # If > 30% of bottom area is dark, it's a wall

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

# ===================== HARDWARE INITIALIZATION =====================
try:
    left_motor = Motor(forward=LEFT_IN1, backward=LEFT_IN2, pwm=True)
    right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2, pwm=True)
    print("Motors Initialized.")
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
        print("Servos Initialized.")
    except Exception as e:
        print(f"Servo Init Error: {e}")

# ===================== HARDWARE HELPERS =====================

def set_servos(pan, tilt):
    """Safely set servo angles."""
    if not servo_kit: return
    
    # Clamp angles
    pan = max(0, min(180, pan))
    # Tilt: 0 is Up, TILT_DOWN_LIMIT is Down
    tilt = max(0, min(TILT_DOWN_LIMIT, tilt))
    
    try:
        pan_servo.angle = pan
        tilt_servo.angle = tilt
        state["pan"] = pan
        state["tilt"] = tilt
    except Exception as e:
        print(f"Servo Set Error: {e}")

def drive(left, right):
    """Set motor speeds (-1.0 to 1.0)."""
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    
    if left > 0: left_motor.forward(left)
    else: left_motor.backward(abs(left))
    
    if right > 0: right_motor.forward(right)
    else: right_motor.backward(abs(right))

def stop_motors():
    left_motor.stop()
    right_motor.stop()

# ===================== VISION LOGIC =====================

def load_yolo():
    try:
        net = cv2.dnn.readNetFromONNX(YOLO_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception as e:
        print(f"YOLO Load Error: {e}")
        return None

def detect_wall(frame):
    """
    Check bottom 25% of frame for dark pixels.
    Returns: Ratio of wall pixels, Mask, ROI Y-start
    """
    h, w = frame.shape[:2]
    roi_y = int(h*0.75)
    roi = frame[roi_y:h, 0:w] # Bottom 25%
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WALL_COLOR_LOWER, WALL_COLOR_UPPER)
    
    wall_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    
    ratio = wall_pixels / total_pixels
    return ratio, mask, roi_y

def process_vision(frame, net):
    """
    1. YOLO Detection
    2. Wall Detection
    Returns: processed_frame, best_object (x, y, w, h), wall_ratio
    """
    h, w = frame.shape[:2]
    
    # --- YOLO ---
    best_obj = None
    if net:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, YOLO_INPUT_SIZE, (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        
        predictions = outputs[0]
        # Fix shape mismatch (1, 5, 2100) -> (2100, 5)
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()

        boxes = []
        confidences = []
        
        x_factor = w / YOLO_INPUT_SIZE[0]
        y_factor = h / YOLO_INPUT_SIZE[1]

        if predictions.shape[1] >= 5:
            # Filter low confidence
            valid_rows = predictions[predictions[:, 4] > YOLO_CONF_THRESH]
            
            for row in valid_rows:
                conf = row[4]
                # Assuming single class or paper is dominant
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                
                left = int((cx - bw/2) * x_factor)
                top = int((cy - bh/2) * y_factor)
                width_p = int(bw * x_factor)
                height_p = int(bh * y_factor)
                
                boxes.append([left, top, width_p, height_p])
                confidences.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONF_THRESH, 0.4)
        
        min_dist = float('inf')
        center_x = w // 2

        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes[i]
                
                # Draw Box
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                
                # Find closest to horizontal center
                obj_cx = bx + bw // 2
                dist = abs(center_x - obj_cx)
                
                if dist < min_dist:
                    min_dist = dist
                    best_obj = (bx, by, bw, bh) # Save best object

    # --- Wall Detection ---
    wall_ratio, wall_mask, roi_y = detect_wall(frame)
    
    # Overlay Wall Mask (Red semi-transparent)
    if wall_ratio > 0.01:
        # Create a red overlay
        roi = frame[roi_y:, :]
        colored_mask = np.zeros_like(roi)
        colored_mask[wall_mask > 0] = [0, 0, 255] # BGR: Red
        
        # Blend it with the original ROI
        frame[roi_y:, :] = cv2.addWeighted(roi, 1.0, colored_mask, 0.5, 0)
    
    if wall_ratio > WALL_AREA_THRESH:
        cv2.putText(frame, "WALL DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, best_obj, wall_ratio

# ===================== MAIN CONTROL LOOP =====================

def robot_logic():
    global state
    
    print("Logic Thread Started...")
    
    # 0. Initial Setup
    set_servos(PAN_FIXED, TILT_NORMAL)
    yolo_net = load_yolo()
    
    # 1. Start: Move forward a bit
    print("Action: Initial Move Forward")
    drive(SPEED_TRACK, SPEED_TRACK)
    time.sleep(0.5)
    stop_motors()
    time.sleep(0.2) # Wait for stop
    
    # PID vars
    prev_error = 0
    integral = 0
    
    while not stop_event.is_set():
        # --- STEP 1: STOP & LOOK ---
        # Ensure motors are stopped to prevent motion blur during capture
        stop_motors() 
        
        # [YOLO-Centric Safety] 
        # Flush the camera buffer to ensure we process a FRESH, STABLE frame.
        # This prevents making decisions based on old or blurry frames from when the robot was moving.
        for _ in range(4): # Flush a few frames (at 30fps, 4 frames is ~0.13s)
            camera.grab()
        
        # Get Fresh Frame
        ret, raw_frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        # Rotate
        frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Process Vision (This takes time, e.g., 200-300ms on Pi 4)
        processed_frame, best_obj, wall_ratio = process_vision(frame, yolo_net)
        
        # Update Global Stream Frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            with frame_lock:
                global output_frame
                output_frame = buffer.tobytes()

        current_time = time.time()
        
        # --- STEP 2: DECIDE & MOVE ---
        
        # 4. Check if object detected
        if best_obj:
            state["object_detected"] = True
            state["last_seen_time"] = current_time
            state["current_action"] = "tracking_step"
            
            bx, by, bw, bh = best_obj
            obj_cx = bx + bw // 2
            obj_cy = by + bh // 2
            frame_h, frame_w = frame.shape[:2]
            
            # --- Camera Tilt Logic (Priority) ---
            if obj_cy > (frame_h * 0.75):
                # Tilt down (Increase angle towards TILT_DOWN_LIMIT)
                new_tilt = state["tilt"] + SERVO_STEP
                if new_tilt <= TILT_DOWN_LIMIT:
                    print(f"Action: Tilting Down to {new_tilt}")
                    set_servos(PAN_FIXED, new_tilt)
            
            # --- Wall Avoidance ---
            if wall_ratio > WALL_AREA_THRESH:
                print("Action: Wall Close! Backing up.")
                state["current_action"] = "avoiding_wall"
                drive(-SPEED_BACKUP, -SPEED_BACKUP)
                time.sleep(BACKUP_DURATION)
                stop_motors()
                continue # Skip the rest of this cycle
            
            # --- PID Calculation ---
            error = (frame_w // 2) - obj_cx
            
            P = PID_KP * error
            integral += error
            I = PID_KI * integral
            D = PID_KD * (error - prev_error)
            
            turn_output = P + I + D
            prev_error = error
            
            # Mix with base speed
            left_s = SPEED_TRACK + turn_output
            right_s = SPEED_TRACK - turn_output
            
            # Execute the "Step"
            drive(left_s, right_s)
            time.sleep(STEP_MOVE_DURATION) # Move blindly for X seconds
            stop_motors() # Stop immediately
            
        else:
            state["object_detected"] = False
            time_since_lost = current_time - state["last_seen_time"]
            
            # 5. Lost Object Logic
            if time_since_lost > LOST_TIMEOUT:
                # > 5 Seconds lost -> Reset and Deep Search
                if state["current_action"] != "lost_search":
                    print("Action: Lost > 5s. Resetting Tilt.")
                    set_servos(PAN_FIXED, TILT_NORMAL)
                    state["current_action"] = "lost_search"
                
                print("Action: Searching (Step Turn Left)")
                drive(-SPEED_SEARCH, SPEED_SEARCH) # Turn Left
                time.sleep(STEP_TURN_DURATION)
                stop_motors()
                time.sleep(0.5) # Wait for stabilization before next Look
                
            else:
                # Recently lost (within 5 seconds) -> Stay in Tracking Mode
                # Do NOT turn blindly. Wait and look.
                # The loop will continue to grab fresh frames and try to re-acquire.
                if state["current_action"] != "waiting_for_object":
                    print(f"Action: Object lost ({time_since_lost:.1f}s). Waiting...")
                    state["current_action"] = "waiting_for_object"
                
                stop_motors()
                # We don't need a long sleep here, just enough to not hammer the CPU loop
                # The main loop already has frame capture latency.
                time.sleep(0.1)
    

# ===================== FLASK (DEBUG VIEW) =====================
app = Flask(__name__)

def generate():
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            data = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return state

# ===================== MAIN ENTRY =====================
def cleanup():
    print("Cleaning up...")
    stop_event.set()
    stop_motors()
    if camera.isOpened():
        camera.release()

atexit.register(cleanup)
signal(SIGINT, lambda s, f: cleanup())

if __name__ == '__main__':
    # Init Camera
    print("Initializing Camera...")
    camera = cv2.VideoCapture(0) # Index 0 usually
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAM_FPS)
    
    # Start Logic Thread
    t = threading.Thread(target=robot_logic)
    t.daemon = True
    t.start()
    
    # Start Web Server
    print("Starting Web Server on port 8000...")
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
