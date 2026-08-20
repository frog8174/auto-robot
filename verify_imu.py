import smbus2
import time
import struct
import numpy as np
import math
import os

bus = smbus2.SMBus(1)

# --- 你的校正參數 (已填入) ---
ACC_BIAS = np.array([25.6100, 0.4483, -18.8117])
ACC_SCALE = np.array([258.4733, 260.0117, 257.0783])

GYRO_OFFSET = np.array([-42.8010, 28.6640, -4.9260])

MAG_BIAS = np.array([-69.00, 60.50, -1079.00])
MAG_SCALE = np.array([1.24, 1.22, 0.73])

# --- 地址設定 ---
ADDR_ACCEL = 0x53
ADDR_GYRO = 0x68
ADDR_MAG = 0x1E

def init_sensors():
    try:
        # Init Accel
        bus.write_byte_data(ADDR_ACCEL, 0x2D, 0x08)
        bus.write_byte_data(ADDR_ACCEL, 0x31, 0x0B)
        # Init Gyro
        bus.write_byte_data(ADDR_GYRO, 0x3E, 0x00)
        bus.write_byte_data(ADDR_GYRO, 0x16, 0x18 + 0x04)
        # Init Mag
        bus.write_byte_data(ADDR_MAG, 0x00, 0x70)
        bus.write_byte_data(ADDR_MAG, 0x01, 0xA0)
        bus.write_byte_data(ADDR_MAG, 0x02, 0x00)
        time.sleep(0.1)
        return True
    except:
        print("初始化失敗，請檢查接線")
        return False

def read_raw_data():
    # 讀 Accel
    data = bus.read_i2c_block_data(ADDR_ACCEL, 0x32, 6)
    ax = struct.unpack('<h', bytes(data[0:2]))[0]
    ay = struct.unpack('<h', bytes(data[2:4]))[0]
    az = struct.unpack('<h', bytes(data[4:6]))[0]
    
    # 讀 Gyro
    data = bus.read_i2c_block_data(ADDR_GYRO, 0x1D, 6)
    gx = struct.unpack('>h', bytes(data[0:2]))[0]
    gy = struct.unpack('>h', bytes(data[2:4]))[0]
    gz = struct.unpack('>h', bytes(data[4:6]))[0]
    
    # 讀 Mag (注意順序 X, Z, Y)
    try:
        data = bus.read_i2c_block_data(ADDR_MAG, 0x03, 6)
        mx = struct.unpack('>h', bytes(data[0:2]))[0]
        mz = struct.unpack('>h', bytes(data[2:4]))[0]
        my = struct.unpack('>h', bytes(data[4:6]))[0]
    except:
        mx, my, mz = 0, 0, 0
        
    return np.array([ax, ay, az]), np.array([gx, gy, gz]), np.array([mx, my, mz])

def main():
    if not init_sensors(): return

    print("開始驗證... 按 Ctrl+C 離開")
    time.sleep(1)
    
    # 用來積分計算角度 (驗證陀螺儀用)
    yaw_angle = 0.0
    last_time = time.time()

    while True:
        raw_acc, raw_gyro, raw_mag = read_raw_data()
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # --- 套用校正公式 ---
        # 1. 加速度計 (單位轉成 g)
        acc_g = (raw_acc - ACC_BIAS) / ACC_SCALE
        
        # 2. 陀螺儀 (單位轉成 deg/s)
        # ITG3205 靈敏度常數 = 14.375 LSB/deg/s
        gyro_dps = (raw_gyro - GYRO_OFFSET) / 14.375
        
        # 3. 磁力計 (單位 uT, 但這裡我們只看相對值)
        mag_cal = (raw_mag - MAG_BIAS) * MAG_SCALE

        # --- 計算輔助數值 ---
        # 合成重力 (靜止平放時應接近 1.00)
        total_g = np.sqrt(np.sum(acc_g**2))
        
        # 簡易電子羅盤方位 (0~360度)
        heading = math.degrees(math.atan2(mag_cal[1], mag_cal[0]))
        if heading < 0: heading += 360
        
        # 積分計算 Yaw 角度 (測試陀螺儀準度)
        if abs(gyro_dps[2]) > 0.5: # 簡單濾掉極小的雜訊
            yaw_angle += gyro_dps[2] * dt

        # --- 顯示儀表板 ---
        os.system('clear') # 或是 print("\033c", end="") 清除螢幕
        print("="*40)
        print("      IMU 校正成果驗收儀表板      ")
        print("="*40)
        
        print(f"【加速度計】 (靜止平放時 Z應為1.0, XY為0)")
        print(f"  X: {acc_g[0]:6.2f} g")
        print(f"  Y: {acc_g[1]:6.2f} g")
        print(f"  Z: {acc_g[2]:6.2f} g")
        print(f"  >> 合成向量: {total_g:6.3f} g (標準值: 0.98 ~ 1.02)")
        print("-" * 40)
        
        print(f"【陀螺儀】 (靜止時應為 0)")
        print(f"  X: {gyro_dps[0]:6.2f} dps")
        print(f"  Y: {gyro_dps[1]:6.2f} dps")
        print(f"  Z: {gyro_dps[2]:6.2f} dps")
        print(f"  >> 積分角度(Z): {yaw_angle:6.1f}° (試著轉動車身90度看看)")
        print("-" * 40)
        
        print(f"【磁力計】 (水平旋轉看方位)")
        print(f"  X: {mag_cal[0]:6.1f}")
        print(f"  Y: {mag_cal[1]:6.1f}")
        print(f"  >> 當前方位: {heading:6.1f}° (北=0/360, 東=90)")
        print("="*40)
        
        time.sleep(0.1)

if __name__ == '__main__':
    main()
