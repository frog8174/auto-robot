import cv2
import numpy as np
import time
import json
import os
import platform
import threading
import argparse
from collections import deque

# --- Hardware / GPIO Imports ---
is_pi = platform.system() == "Linux"
if not is_pi:
    from gpiozero.pins.mock import MockFactory, MockPWMPin
    from gpiozero import Device, Motor, OutputDevice
    Device.pin_factory = MockFactory(pin_class=MockPWMPin)
else:
    from gpiozero import Motor, OutputDevice

# --- Configuration ---
LEFT_IN1 = 17
LEFT_IN2 = 27
RIGHT_IN1 = 22
RIGHT_IN2 = 23
VACUUM_PIN = 24

# Servo Config
TILT_UP_ANGLE = 70
TILT_DOWN_ANGLE = 100

# Speed Settings
BASE_SPEED = 0.16
DRIFT_RIGHT_L = 0.33
DRIFT_RIGHT_R = 0.10
TURN_LEFT_L = -0.50
TURN_LEFT_R = 0.50

# Control
TARGET_DISTANCE = 150 
PID_DIST_KP = 0.0015
PID_DIST_KI = 0.000
PID_DIST_KD = 0.0005

PID_ANGLE_KP = 0.013 
PID_ANGLE_KI = 0.00
PID_ANGLE_KD = 0.001

TURN_DURATION = 1.0 

# Vision
MASK_PATH = 'mask/mask.json'
DARK_THRESHOLD_V = 80 
SMOOTHING_WINDOW = 5

# Logging
LOG_INTERVAL = 10 

# --- Globals ---
left_motor = Motor(forward=LEFT_IN1, backward=LEFT_IN2, pwm=True)
right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2, pwm=True)
vacuum = OutputDevice(VACUUM_PIN)

# Servo Init
pan_servo = None
tilt_servo= None
if is_pi:
    try:
        from adafruit_servokit import ServoKit
        kit = ServoKit(channels=16)
        pan_servo = kit.servo[1]
        pan_servo.set_pulse_width_range(500, 2000)
        pan_servo.angle = 95
        
        tilt_servo = kit.servo[0]
        tilt_servo.set_pulse_width_range(500,2000)
        tilt_servo.angle = TILT_DOWN_ANGLE
        
        print(f"[INIT] Servos initialized. Pan=95, Tilt={TILT_DOWN_ANGLE}")
    except Exception as e:
        print(f"[WARN] Servo init failed: {e}")

pid_state = {
    "dist_integral": 0,
    "dist_prev_error": 0,
    "angle_integral": 0,
    "angle_prev_error": 0
}

history_wall_x = deque(maxlen=SMOOTHING_WINDOW)
history_wall_angle = deque(maxlen=SMOOTHING_WINDOW)

# --- Camera Thread Class ---
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        actual_w = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[CAM] Resolution set to: {int(actual_w)}x{int(actual_h)}")

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- Helper Functions ---

def load_mask_polygon(width, height):
    if not os.path.exists(MASK_PATH):
        print(f"[ERROR] Mask file not found: {MASK_PATH}")
        return None, 0

    with open(MASK_PATH, 'r') as f:
        data = json.load(f)
    
    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask_right_boundary = 0

    for item in data:
        attr = item.get('shape_attributes', {})
        if attr.get('name') == 'polygon':
            all_x = attr.get('all_points_x', [])
            all_y = attr.get('all_points_y', [])
            
            if all_x and all_y:
                all_x = [min(x, width-1) for x in all_x]
                all_y = [min(y, height-1) for y in all_y]
                
                pts = np.array([list(zip(all_x, all_y))], np.int32)
                cv2.fillPoly(mask, pts, 0)
                current_max_x = max(all_x)
                if current_max_x > mask_right_boundary:
                    mask_right_boundary = current_max_x

    return mask, mask_right_boundary

def drive(left, right):
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    
    if left >= 0: left_motor.forward(left)
    else: left_motor.backward(abs(left))
        
    if right >= 0: right_motor.forward(right)
    else: right_motor.backward(abs(right))

def stop():
    left_motor.stop()
    right_motor.stop()
    vacuum.off() # Ensure vacuum is off on stop

def get_line_angle(x1, y1, x2, y2):
    if x1 == x2: return 0.0
    if y2 < y1: x1, y1, x2, y2 = x2, y2, x1, y1 
    dx = x2 - x1
    dy = y2 - y1
    if dy == 0: return 90.0 
    return np.degrees(np.arctan(dx / dy))

def check_red(frame):
    if frame is None: return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red has two ranges in HSV (0-10 and 170-180)
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    full_mask = mask1 + mask2
    
    # Dilate/Erode to remove noise
    kernel = np.ones((3,3), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    count = cv2.countNonZero(full_mask)
    h, w = frame.shape[:2]
    total_pixels = h * w
    ratio = count / total_pixels
    
    # If > 5% of screen is red, consider it "Large Area"
    return ratio > 0.05

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Wall Follower Robot')
    parser.add_argument('--show', action='store_true', help='Show video output window')
    args = parser.parse_args()

    cam = CameraStream(width=640, height=480).start()
    time.sleep(1) 

    frame = cam.read()
    if frame is None:
        print("[ERROR] No frame from camera thread.")
        cam.stop()
        return

    h, w = frame.shape[:2]
    poly_mask, mask_right_x = load_mask_polygon(w, h)
    print(f"[INIT] Mask loaded. Right Boundary X: {mask_right_x}")

    state = "TRACKING" 
    turn_end_time = 0
    frame_count = 0
    start_time = time.time()
    
    # Vacuum Control State
    vacuum_active = False
    vacuum_end_time = 0

    print(f"[START] Running... (Show Display: {args.show})")

    try:
        while True:
            # --- Global Actions ---
            # Check vacuum timer non-blocking
            if vacuum_active and time.time() > vacuum_end_time:
                print("[VACUUM] Timer expired. Turning OFF.")
                vacuum.off()
                vacuum_active = False

            frame = cam.read()
            if frame is None: continue
            
            # Prepare visualization frame only if needed
            vis_frame = frame.copy() if args.show else None
            
            if args.show:
                # Dim the masked area
                vis_frame[poly_mask == 0] = vis_frame[poly_mask == 0] // 2
                if vacuum_active:
                    cv2.putText(vis_frame, "VACUUM ON", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- State Machine ---
            if state == "TURNING":
                if time.time() > turn_end_time:
                    print("[STATE] Turn Complete. Resuming Tracking.")
                    state = "TRACKING"
                    stop() # Stop motors but NOT vacuum (handled by timer)
                    pid_state["dist_integral"] = 0
                    pid_state["angle_integral"] = 0
                    history_wall_x.clear()
                    history_wall_angle.clear()
                else:
                    drive(TURN_LEFT_L, TURN_LEFT_R)
                    if args.show:
                        cv2.putText(vis_frame, "TURNING...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("Wall Follower", vis_frame)
                        cv2.waitKey(1)
                    else:
                        time.sleep(0.01) 
                    continue

            # --- Vision Processing ---
            masked_frame = cv2.bitwise_and(frame, frame, mask=poly_mask)
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, DARK_THRESHOLD_V])
            
            color_mask = cv2.inRange(hsv, lower_black, upper_black)
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=poly_mask)

            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            color_mask = cv2.dilate(color_mask, kernel, iterations=1)

            edges = cv2.Canny(color_mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

            current_frame_lines_x = []
            current_frame_angles = []
            found_horizontal = False

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = abs(x1 - x2)
                    dy = abs(y1 - y2)

                    if dy > 2 * dx: # Vertical
                        cx = (x1 + x2) / 2
                        if cx > mask_right_x:
                            current_frame_lines_x.append(cx)
                            current_frame_angles.append(get_line_angle(x1, y1, x2, y2))
                            if args.show:
                                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    elif dx > 2 * dy: # Horizontal
                        cx = (x1 + x2) / 2
                        if w * 0.2 < cx < w * 0.8:
                            found_horizontal = True
                            if args.show:
                                cv2.line(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # --- Logic ---
            action_str = ""
            log_data = ""

            if found_horizontal:
                print("[EVENT] Horizontal Line Detected")
                stop() # Stop motors
                
                # 1. Raise Head
                if tilt_servo:
                    print(f"[SERVO] Raising Head to {TILT_UP_ANGLE}")
                    tilt_servo.angle = TILT_UP_ANGLE
                    # 2. Wait 1 second
                    time.sleep(1.0)
                else:
                    print("[WARN] No Tilt Servo configured.")
                    time.sleep(1.0) # Still wait as requested

                # 3. Take a Photo (Read fresh frame)
                check_frame = cam.read()
                
                # 4. Check for Large Red Area
                is_red = check_red(check_frame)
                print(f"[VISION] Red Check: {is_red}")
                
                # Restore Head
                if tilt_servo:
                    tilt_servo.angle = TILT_DOWN_ANGLE
                    time.sleep(0.2) # Brief pause for servo settling

                # 5. Logic
                if is_red:
                    print("[LOGIC] Red detected -> Do NOT Vacuum. Continuing.")
                else:
                    print("[LOGIC] No Red -> Vacuum ON for 3s.")
                    vacuum.on()
                    vacuum_active = True
                    vacuum_end_time = time.time() + 3.0

                # Transition to Turning
                state = "TURNING"
                turn_end_time = time.time() + TURN_DURATION
                
                if args.show:
                    status_text = "RED FOUND" if is_red else "VACUUM START"
                    cv2.putText(vis_frame, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Wall Follower", vis_frame)
                    cv2.waitKey(1)
                
                continue 

            elif len(current_frame_lines_x) > 0:
                avg_x = np.mean(current_frame_lines_x)
                avg_angle = np.mean(current_frame_angles)
                
                history_wall_x.append(avg_x)
                history_wall_angle.append(avg_angle)
                
                smooth_x = np.mean(history_wall_x)
                smooth_angle = np.mean(history_wall_angle)
                
                # PID
                current_dist = smooth_x - mask_right_x
                dist_error = TARGET_DISTANCE - current_dist
                
                P_d = PID_DIST_KP * dist_error
                pid_state["dist_integral"] += dist_error
                I_d = PID_DIST_KI * pid_state["dist_integral"]
                D_d = PID_DIST_KD * (dist_error - pid_state["dist_prev_error"])
                dist_correction = P_d + I_d + D_d
                pid_state["dist_prev_error"] = dist_error

                angle_error = smooth_angle
                P_a = PID_ANGLE_KP * angle_error
                pid_state["angle_integral"] += angle_error
                I_a = PID_ANGLE_KI * pid_state["angle_integral"]
                D_a = PID_ANGLE_KD * (angle_error - pid_state["angle_prev_error"])
                angle_correction = P_a + I_a + D_a
                pid_state["angle_prev_error"] = angle_error
                
                total_correction = dist_correction + angle_correction
                l_speed = BASE_SPEED - total_correction
                r_speed = BASE_SPEED + total_correction
                
                drive(l_speed, r_speed)
                action_str = "TRACKING"
                log_data = f"Dist={current_dist:.0f} Ang={smooth_angle:.1f}"
                
                if args.show:
                    cv2.putText(vis_frame, f"ERR:{dist_error:.0f} ANG:{smooth_angle:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                action_str = "LOST"
                log_data = "Drifting Right..."
                drive(DRIFT_RIGHT_L, DRIFT_RIGHT_R)
                if len(history_wall_x) > 0:
                    history_wall_x.popleft()
                    history_wall_angle.popleft()
                
                if args.show:
                    cv2.putText(vis_frame, "LOST TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- Display & Stats ---
            if args.show:
                cv2.imshow("Wall Follower", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % LOG_INTERVAL == 0:
                now = time.time()
                elapsed = now - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                if elapsed > 10:
                    start_time = now
                    frame_count = 0
                print(f"[FPS:{fps:.1f}] {action_str} | {log_data}")

    except KeyboardInterrupt:
        pass
    finally:
        stop()
        cam.stop()
        cv2.destroyAllWindows()
        print("[EXIT] Stopped.")

if __name__ == "__main__":
    main()