import os

# ===================== SENSOR CONFIG =====================
# Ultrasonic Sensor (HC-SR04)
TRIG_PIN = int(os.getenv("TRIG_PIN", 20))
ECHO_PIN = int(os.getenv("ECHO_PIN", 21))

# IMU Connection (I2C)
IMU_ADDRESS = 0x68  # Standard MPU6050/9250 address
IMU_BUS_NUM = 1

# ===================== ROBOT GEOMETRY =====================
# Units in meters
WHEEL_BASE = 0.15      # Distance between wheels
WHEEL_DIAMETER = 0.065 # Wheel diameter
TICKS_PER_REV = 20     # Encoder ticks per revolution (if available)

# ===================== MAP CONFIG =====================
MAP_SIZE_PIXELS = 500
MAP_RESOLUTION = 0.05 # Meters per pixel
MAP_CENTER_X = MAP_SIZE_PIXELS // 2
MAP_CENTER_Y = MAP_SIZE_PIXELS // 2

# ===================== IMU CALIBRATION LOADER =====================
def load_imu_calibration(filepath='imu_data.txt'):
    calib = {
        'acc_offset': [0.0, 0.0, 0.0],
        'acc_scale': [1.0, 1.0, 1.0],
        'gyro_offset': [0.0, 0.0, 0.0],
        'mag_offset': [0.0, 0.0, 0.0],
        'mag_scale': [1.0, 1.0, 1.0]
    }
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using default calibration.")
        return calib

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Helper to extract value by key
        def get_val(key):
            import re
            match = re.search(rf"{key}\s*=\s*([-0-9.]+)", content)
            return float(match.group(1)) if match else 0.0

        calib['acc_offset'] = [get_val('ACC_X_BIAS'), get_val('ACC_Y_BIAS'), get_val('ACC_Z_BIAS')]
        calib['acc_scale'] = [get_val('ACC_X_SCALE'), get_val('ACC_Y_SCALE'), get_val('ACC_Z_SCALE')]
        
        calib['gyro_offset'] = [get_val('GYRO_X_OFFSET'), get_val('GYRO_Y_OFFSET'), get_val('GYRO_Z_OFFSET')]
        
        calib['mag_offset'] = [get_val('MAG_X_BIAS'), get_val('MAG_Y_BIAS'), get_val('MAG_Z_BIAS')]
        calib['mag_scale'] = [get_val('MAG_X_SCALE'), get_val('MAG_Y_SCALE'), get_val('MAG_Z_SCALE')]
        
        print(f"Loaded IMU calibration from {filepath}")
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return calib

IMU_CALIBRATION = load_imu_calibration()

