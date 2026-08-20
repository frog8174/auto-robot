#!/usr/bin/env python3
"""
MPU-6050 Calibration Script
This script reads data from the MPU-6050 to calculate the offsets (bias).
Place the robot/sensor on a FLAT, STATIONARY surface before running.
"""

import time
import sys
import os

try:
    import smbus
except ImportError:
    print("Error: 'smbus' module not found. Please install it (e.g., sudo apt install python3-smbus).")
    sys.exit(1)

# MPU-6050 Registers
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# Search for device
bus = smbus.SMBus(1) # Bus 1 is standard on RPi
address = None

# Try 0x68 then 0x69
for addr in [0x68, 0x69]:
    try:
        bus.write_byte_data(addr, PWR_MGMT_1, 0)
        print(f"Found MPU-6050 at address 0x{addr:02X}")
        address = addr
        break
    except Exception:
        pass

if address is None:
    print("MPU-6050 not found on I2C bus 1.")
    sys.exit(1)

def read_word_2c(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg+1)
    val = (high << 8) + low
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def calibrate(samples=2000):
    print(f"\nStarting calibration with {samples} samples...")
    print("KEEP THE SENSOR COMPLETELY STILL!")
    time.sleep(2)

    ax_sum = 0
    ay_sum = 0
    az_sum = 0
    gx_sum = 0
    gy_sum = 0
    gz_sum = 0

    for i in range(samples):
        # Read Accelerometer
        ax_sum += read_word_2c(address, ACCEL_XOUT_H)
        ay_sum += read_word_2c(address, ACCEL_XOUT_H + 2)
        az_sum += read_word_2c(address, ACCEL_XOUT_H + 4)
        
        # Read Gyroscope
        gx_sum += read_word_2c(address, GYRO_XOUT_H)
        gy_sum += read_word_2c(address, GYRO_XOUT_H + 2)
        gz_sum += read_word_2c(address, GYRO_XOUT_H + 4)
        
        if i % 100 == 0:
            sys.stdout.write(f"\rProgress: {int(i/samples*100)}%")
            sys.stdout.flush()
        
        time.sleep(0.002) # Small delay

    print("\rProgress: 100%   ")
    
    # Averages
    ax_avg = ax_sum / samples
    ay_avg = ay_sum / samples
    az_avg = az_sum / samples
    gx_avg = gx_sum / samples
    gy_avg = gy_sum / samples
    gz_avg = gz_sum / samples

    # Expected values
    # We assume the sensor is flat. 
    # Accel X/Y should be 0.
    # Accel Z should be 1g. Default sensitivity +/- 2g = 16384 LSB/g.
    # If Z points UP, it reads +16384. If DOWN, -16384. We assume UP.
    EXPECTED_AZ = 16384 
    
    # Calculate Offsets (Bias)
    # Bias = Average - Expected
    ax_offset = ax_avg # Expected 0
    ay_offset = ay_avg # Expected 0
    az_offset = az_avg - EXPECTED_AZ
    
    gx_offset = gx_avg # Expected 0
    gy_offset = gy_avg # Expected 0
    gz_offset = gz_avg # Expected 0

    print("\nCalibration Complete!")
    print("-" * 30)
    print("RESULTS (Offsets to be subtracted from raw data):")
    print(f"Accel X Offset: {ax_offset:.2f}")
    print(f"Accel Y Offset: {ay_offset:.2f}")
    print(f"Accel Z Offset: {az_offset:.2f} (Assumes Z axis vertical +1g)")
    print(f"Gyro X Offset:  {gx_offset:.2f}")
    print(f"Gyro Y Offset:  {gy_offset:.2f}")
    print(f"Gyro Z Offset:  {gz_offset:.2f}")
    print("-" * 30)

    return {
        'ACC_X_BIAS': ax_offset,
        'ACC_Y_BIAS': ay_offset,
        'ACC_Z_BIAS': az_offset,
        'GYRO_X_OFFSET': gx_offset,
        'GYRO_Y_OFFSET': gy_offset,
        'GYRO_Z_OFFSET': gz_offset
    }

def save_to_file(offsets, filename="imu_data.txt"):
    print(f"Saving to {filename}...")
    try:
        with open(filename, "w") as f:
            f.write(f"ACC_X_BIAS = {offsets['ACC_X_BIAS']:.2f}, ACC_X_SCALE = 1.0\n")
            f.write(f"ACC_Y_BIAS = {offsets['ACC_Y_BIAS']:.2f}, ACC_Y_SCALE = 1.0\n")
            f.write(f"ACC_Z_BIAS = {offsets['ACC_Z_BIAS']:.2f}, ACC_Z_SCALE = 1.0\n")
            f.write(f"GYRO_X_OFFSET = {offsets['GYRO_X_OFFSET']:.2f}\n")
            f.write(f"GYRO_Y_OFFSET = {offsets['GYRO_Y_OFFSET']:.2f}\n")
            f.write(f"GYRO_Z_OFFSET = {offsets['GYRO_Z_OFFSET']:.2f}\n")
            f.write(f"MAG_X_BIAS = 0.00, MAG_X_SCALE = 1.00\n")
            f.write(f"MAG_Y_BIAS = 0.00, MAG_Y_SCALE = 1.00\n")
            f.write(f"MAG_Z_BIAS = 0.00, MAG_Z_SCALE = 1.00\n")
        print("Saved.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    offsets = calibrate()
    save_to_file(offsets)
