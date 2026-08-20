

import time
import numpy as np
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# --- 1. 硬體設定 ---
TRIG_PIN = 10
ECHO_PIN = 9

# I2C Servo 初始化
try:
    kit = ServoKit(channels=16)
    pan_servo = kit.servo[8]
    pan_servo.set_pulse_width_range(500, 2000)
    print("[系统] PCA9685 連線成功")
except Exception as e:
    print(f"[錯誤] I2C 初始化失敗: {e}")
    exit()

# GPIO 初始化
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# --- 2. 測距函式 (含 Debug) ---
def get_distance():
    # 發射超音波
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    timeout = time.time()
    pulse_start = time.time()
    
    # 等待回波開始 (若 0.1秒內沒反應，代表沒插好或壞了)
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if pulse_start - timeout > 0.1:
            return -1  # 錯誤代碼: 等不到回波

    # 等待回波結束
    pulse_end = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if pulse_end - pulse_start > 0.1: 
            return -2 # 錯誤代碼: 回波太長(超時)

    distance = (pulse_end - pulse_start) * 17150
    return round(distance, 2)

# --- 3. 主程式 ---
def radar_scan():
    print("[系统] 準備開始掃描...請觀察下方數據")
    
    try:
        plt.ion()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
        ax.set_ylim(0, 100)
        ax.set_title("Debug Mode Radar", va='bottom')
        line, = ax.plot([], [], 'ro', alpha=0.8)
        plt.show() 
        
        angles_rad = []
        distances = []
        scan_range = range(0, 181, 5) # 改成每 5 度掃一次，跑快一點比較好除錯
        
        while True:
            for angle in scan_range:
                pan_servo.angle = angle
                plt.pause(0.01) 
                
                raw_dist = get_distance()
                
                # --- Debug 輸出 ---
                if raw_dist == -1:
                    print(f"角度: {angle:3d}° | 距離: [無訊號] (請檢查 Trig/Echo 接線)")
                elif raw_dist == -2:
                    print(f"角度: {angle:3d}° | 距離: [超時] (目標太遠或感測器異常)")
                elif raw_dist > 400 or raw_dist < 2:
                    print(f"角度: {angle:3d}° | 距離: {raw_dist} cm (過濾掉)")
                else:
                    print(f"角度: {angle:3d}° | 距離: {raw_dist} cm --> [更新圖表]")
                    
                    # 只有有效距離才畫圖
                    rad = np.deg2rad(angle)
                    angles_rad.append(rad)
                    distances.append(raw_dist)
                    
                    if len(angles_rad) > 100:
                        angles_rad.pop(0)
                        distances.pop(0)
                    
                    line.set_xdata(angles_rad)
                    line.set_ydata(distances)

            # 簡化測試：這裡先不做反向掃描，直接歸零重跑，方便觀察
            # (讓馬達轉回 0 度)
            pan_servo.angle = 0
            plt.pause(0.5)

    except KeyboardInterrupt:
        print("\n程式結束")
        pan_servo.angle = 90
        time.sleep(0.5)
        GPIO.cleanup()
        plt.close()

if __name__ == "__main__":
    radar_scan()
