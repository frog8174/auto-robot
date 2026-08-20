#!/usr/bin/env python3
"""
IMU EKF Visualization Test
This script visualizes the difference between:
1. Pure Odometry (Green Line) - Assuming perfect motors
2. Pure IMU (Red Line) - Assuming only Gyro integration
3. EKF Fusion (White Line) - The result of fusing both

Usage:
Run the script. A window will open.
- Rotate the robot/sensor in your hand or drive it.
- Observe the lines deviating.
- 'q' to quit.
"""

import time
import math
import numpy as np
import cv2
import sys

# Import our existing config and core (EKF is inside core)
from slam_config import *
from slam_app import read_imu_gyro_z, HAS_SMBUS

if not HAS_SMBUS:
    print("Error: No IMU found (smbus failed). This test requires an IMU.")
    sys.exit(1)

class PoseTracker:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
    def update(self, v, w, dt):
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt
        # Normalize
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        return self.x, self.y, self.theta

from slam_core import ExtendedKalmanFilter

def draw_arrow(img, x, y, theta, color, length=20):
    cx = int(x)
    cy = int(y)
    ex = int(cx + length * math.cos(theta))
    ey = int(cy + length * math.sin(theta))
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.3)

def main():
    # Setup EKF
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # Setup Pure Odometry Tracker (Simulation of "Blind" driving)
    odo_tracker = PoseTracker()
    
    # Setup Pure IMU Tracker (Simulation of "Gyro only")
    imu_tracker = PoseTracker()

    print("IMU Test Started.")
    print("RED Arrow   = Pure IMU (Gyro Integration)")
    print("GREEN Arrow = Pure Motor Command (Odometry)")
    print("WHITE Arrow = EKF Fused Result (What SLAM uses)")
    print("-" * 40)
    print("Press 'w' to simulate moving Forward")
    print("Press 'a'/'d' to simulate turning Left/Right")
    print("Rotate the actual robot to see IMU effect!")
    
    w = 800
    h = 600
    center_x, center_y = w // 2, h // 2
    scale = 100 # pixels per meter
    
    dt = 0.1
    
    # Simulated commands
    cmd_v = 0.0
    cmd_w = 0.0
    
    try:
        while True:
            start_time = time.time()
            
            # 1. Create Blank Image
            img = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 2. Handle Input (Simulation)
            key = cv2.waitKey(int(dt * 1000)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'): # Forward
                cmd_v = 0.5
                cmd_w = 0.0
            elif key == ord('a'): # Left Turn Command
                cmd_v = 0.0
                cmd_w = 1.0
            elif key == ord('d'): # Right Turn Command
                cmd_v = 0.0
                cmd_w = -1.0
            else:
                cmd_v = 0.0
                cmd_w = 0.0

            # 3. Read Real IMU
            # We are testing if EKF follows IMU when cmds indicate stopped, 
            # or if it fuses when both exist.
            real_gyro_z = read_imu_gyro_z()
            
            # Filter noise
            if abs(real_gyro_z) < 0.05: real_gyro_z = 0.0

            # 4. Update Trackers
            
            # A. Pure Odometry: Trusts ONLY the keys you press. 
            # If you turn the robot by hand, this WON'T move.
            odo_x, odo_y, odo_th = odo_tracker.update(cmd_v, cmd_w, dt)
            
            # B. Pure IMU: Trusts ONLY the sensor. 
            # Assumes constant v=0 (since IMU gives no speed), only tracks rotation.
            # We assume a fake velocity just to show it moving apart from origin for visualization
            imu_v = cmd_v # Borrow velocity command just for viz, but rotation comes purely from Gyro
            imu_x, imu_y, imu_th = imu_tracker.update(imu_v, real_gyro_z, dt)

            # C. EKF: Fuses Command (v, w) with Measurement (real_gyro_z)
            # pose is [x, y, theta]
            ekf_pose = ekf.predict_update(cmd_v, cmd_w, real_gyro_z, dt)
            ekf_x, ekf_y, ekf_th = ekf_pose
            
            # 5. Draw
            
            # Green: Odometry (Command)
            draw_arrow(img, center_x + odo_x * scale, center_y + odo_y * scale, odo_th, (0, 255, 0))
            
            # Red: Real IMU
            draw_arrow(img, center_x + imu_x * scale, center_y + imu_y * scale, imu_th, (0, 0, 255))
            
            # White: EKF Fusion
            draw_arrow(img, center_x + ekf_x * scale, center_y + ekf_y * scale, ekf_th, (255, 255, 255), length=40)
            
            # Info text
            cv2.putText(img, f"Cmd W: {cmd_w:.2f} rad/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(img, f"Gyro Z: {real_gyro_z:.2f} rad/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.imshow("IMU Fusion Test", img)
            
            # Loop timing
            elapsed = time.time() - start_time
            if elapsed < dt:
                # waitKey handles sleep
                pass

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
