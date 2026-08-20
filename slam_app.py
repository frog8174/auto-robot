#!/usr/bin/env python3
"""
SLAM Robot Application
Integrates Ultrasonic SLAM logic with IMU-corrected odometry and Scanning Servo.
"""

import time
import sys
import math
import numpy as np
import cv2
import threading
from gpiozero import Motor, DistanceSensor
from slam_config import *
from slam_core import OccupancyGridSLAM

# Check for I2C support (smbus)
try:
    import smbus
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    print("Warning: 'smbus' not found. IMU data will be mocked.")

# Check for ServoKit
try:
    from adafruit_servokit import ServoKit
    # Initialize ServoKit for 16 channels (standard PCA9685)
    kit = ServoKit(channels=16)
    pan_servo = kit.servo[0] # Assuming Channel 0 is Pan
    HAS_SERVO = True
except Exception as e:
    HAS_SERVO = False
    print(f"ServoKit init failed: {e}. Servo scanning disabled.")

# ===================== HARDWARE SETUP =====================
# Motors (Reusing pin config from env or defaults in slam_config/app.py context)
import os
LEFT_IN1  = int(os.getenv("LEFT_IN1", 17))
LEFT_IN2  = int(os.getenv("LEFT_IN2", 27))
RIGHT_IN1 = int(os.getenv("RIGHT_IN1", 22))
RIGHT_IN2 = int(os.getenv("RIGHT_IN2", 23))
SPEED = 0.4

try:
    left_motor = Motor(forward=LEFT_IN1, backward=LEFT_IN2)
    right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2)
except Exception as e:
    print(f"Motor init failed (running on PC?): {e}")
    from gpiozero.pins.mock import MockFactory
    from gpiozero import Device
    Device.pin_factory = MockFactory()
    left_motor = Motor(forward=LEFT_IN1, backward=LEFT_IN2)
    right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2)

# IMU Setup
bus = None
if HAS_SMBUS:
    try:
        bus = smbus.SMBus(IMU_BUS_NUM)
        # Try address 0x68 first
        try:
            bus.write_byte_data(0x68, 0x6B, 0)
            IMU_ADDRESS = 0x68
            print("IMU found at 0x68")
        except OSError:
            # Try 0x69
            bus.write_byte_data(0x69, 0x6B, 0)
            IMU_ADDRESS = 0x69
            print("IMU found at 0x69")
    except Exception as e:
        print(f"IMU init failed: {e}")
        bus = None

# ===================== SENSOR READING =====================
def read_imu_gyro_z():
    """Reads Gyro Z, applies calibration, returns radians/sec."""
    if not bus:
        return 0.0
    
    try:
        # Read raw 16-bit value (0x47 is Gyro Z High)
        high = bus.read_byte_data(IMU_ADDRESS, 0x47)
        low = bus.read_byte_data(IMU_ADDRESS, 0x48)
        
        # Combine
        value = (high << 8) | low
        # Convert to signed 16-bit
        if value > 32768:
            value = value - 65536
            
        # Calibration
        # Formula: (Raw - Bias) / Scale
        bias = IMU_CALIBRATION['gyro_offset'][2] # Z offset
        scale = 131.0 # Default MPU6050 sensitivity
        
        corrected_dps = (value - bias) / scale
        return -math.radians(corrected_dps) # Invert Z axis for correct map orientation
        
    except Exception as e:
        # print(f"IMU read error: {e}") # Suppress repetitive error
        return 0.0

# Ultrasonic Sensor - Manual Implementation for Debugging
# Using DigitalInput/OutputDevice to manually control pins via gpiozero backend
from gpiozero import DigitalInputDevice, DigitalOutputDevice

try:
    print(f"Initializing Ultrasonic Sensor: TRIG={TRIG_PIN}, ECHO={ECHO_PIN}")
    trig = DigitalOutputDevice(TRIG_PIN)
    echo = DigitalInputDevice(ECHO_PIN)
    sensor_active = True
except Exception as e:
    print(f"Ultrasonic init failed: {e}")
    trig = None
    echo = None
    sensor_active = False

def read_ultrasonic():
    """
    Manually triggers and reads the ultrasonic sensor.
    Returns distance in meters or None on error.
    """
    if not sensor_active:
        return None

    try:
        # 1. Ensure Trigger is Low
        trig.off()
        time.sleep(0.000005)

        # 2. Check if Echo is stuck High (common issue)
        if echo.value == 1:
            print("DEBUG: Echo pin is stuck HIGH before trigger! Sensor jammed or connection bad.")
            return None

        # 3. Trigger Pulse (10us)
        trig.on()
        time.sleep(0.00001)
        trig.off()

        # 4. Wait for Echo High (Start of pulse)
        timeout = time.time() + 0.05 # 50ms timeout
        pulse_start = time.time()
        while echo.value == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                # print("DEBUG: Timeout waiting for Echo START")
                return None

        # 5. Wait for Echo Low (End of pulse)
        timeout = time.time() + 0.05 # 50ms timeout
        pulse_end = time.time()
        while echo.value == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                print("DEBUG: Timeout waiting for Echo END")
                return None

        # 6. Calculate Distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 171.50 # 343m/s / 2
        
        # print(f"DEBUG: Dist={distance:.3f}m") # Uncomment for verbose debug
        return distance

    except Exception as e:
        print(f"Ultrasonic read error: {e}")
        return None

# ===================== MAIN LOOP =====================
def main():
    slam = OccupancyGridSLAM()
    
    print("Starting SLAM Robot...")
    print("Controls: Arrow Keys to Move, 'q' to Quit.")
    print("Map is displayed in 'SLAM Map' window.")
    
    dt = 0.1 # 10Hz loop
    current_v = 0.0
    current_w = 0.0
    
    # Servo Config
    pan_angle = 90.0
    pan_step = 15.0  # Increased speed (Degrees per loop) for fast scanning
    pan_dir = 1      # 1 for increasing, -1 for decreasing
    PAN_MIN = 0      # Full range right
    PAN_MAX = 180    # Full range left
    
    window_open = True

    try:
        while True:
            start_time = time.time()
            
            # --- Servo Sweep Logic ---
            # Update Angle
            pan_angle += pan_step * pan_dir
            
            # Boundary Check & Ping-Pong
            if pan_angle >= PAN_MAX:
                pan_angle = PAN_MAX
                pan_dir = -1 # Reverse direction
            elif pan_angle <= PAN_MIN:
                pan_angle = PAN_MIN
                pan_dir = 1  # Reverse direction
            
            # Move Servo
            if HAS_SERVO:
                try:
                    pan_servo.angle = pan_angle
                except Exception:
                    pass
            
            # 1. Handle Input & Display
            img = slam.get_map_image()
            
            key = -1
            if window_open:
                try:
                    cv2.imshow("SLAM Map", img)
                    key = cv2.waitKey(int(dt * 1000)) & 0xFF
                except Exception as e:
                    print(f"Display error (running headless?): {e}")
                    window_open = False # Disable window updates
                    time.sleep(dt) # Manual sleep since waitKey failed
            else:
                time.sleep(dt)

            if key == ord('q'):
                break
            elif key == 82: # Up Arrow
                left_motor.forward(SPEED)
                right_motor.forward(SPEED)
                current_v = 0.2
                current_w = 0.0
            elif key == 84: # Down Arrow
                left_motor.backward(SPEED)
                right_motor.backward(SPEED)
                current_v = -0.2
                current_w = 0.0
            elif key == 81: # Left Arrow
                left_motor.backward(SPEED)
                right_motor.forward(SPEED)
                current_v = 0.0
                current_w = 1.0 # rad/s
            elif key == 83: # Right Arrow
                left_motor.forward(SPEED)
                right_motor.backward(SPEED)
                current_v = 0.0
                current_w = -1.0
            elif key != -1: # Only stop if a key was actually processed and wasn't one of the above
                 # If window is closed, key is always -1, so we need logic to NOT stop automatically
                 # unless we want 'dead man switch' behavior. 
                 # Current logic: if no key pressed, stop.
                 pass
            
            # If window is closed, we can't control via keyboard.
            # Just for safety in headless mode: stop motors if we can't control them.
            if not window_open:
                left_motor.stop()
                right_motor.stop()
                current_v = 0.0
                current_w = 0.0
            elif key == 255: # No key pressed
                left_motor.stop()
                right_motor.stop()
                current_v = 0.0
                current_w = 0.0

            # 2. Read Sensors
            gyro_z = read_imu_gyro_z()
            dist = read_ultrasonic()
            
            # 3. SLAM Update
            # Filter small gyro noise
            if abs(gyro_z) < 0.05: gyro_z = 0.0
            
            slam.predict(current_v, current_w, dt, gyro_z_corrected=gyro_z)
            
            # Calculate sensor angle relative to robot (radians)
            # 90 degrees is "Forward" (0 offset)
            # Inverting logic to match user's hardware setup
            # New Formula: 90 - angle
            sensor_offset = math.radians(90 - pan_angle)
            
            slam.update(dist, sensor_angle_offset=sensor_offset)
            
            # Loop timing
            elapsed = time.time() - start_time
            if elapsed < dt:
                # time.sleep is handled above if window is closed, 
                # but if window is open, waitKey handles it.
                # Just sleep remaining if any (waitKey is precise enough usually)
                pass
                
    except KeyboardInterrupt:
        pass
    finally:
        left_motor.stop()
        right_motor.stop()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()