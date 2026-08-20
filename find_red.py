import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# ========= GPIO 腳位設定 =========
RELAY_PIN  = 24  # 要輸出高/低電位的腳位
BUZZER_PIN = 18  # 有源蜂鳴器腳位

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)  # 預設：沒有紅色 → 高電位
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)  # 蜂鳴器先關閉

# ========= Camera 設定 =========
cap = cv2.VideoCapture(0)  # 一般是 0，如果是 USB 另外一隻可能是 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)      # 降一點 FPS 減輕負擔

if not cap.isOpened():
    print("無法開啟攝影機")
    GPIO.cleanup()
    raise SystemExit

print("開始偵測紅色物件... 按 q 離開")

# 畫面中紅色佔比超過這個值就觸發
RED_RATIO_THRESHOLD = 0.01  # 1% 的畫面是紅色就觸發

# 🔴 紅色消失後延遲多久才把 GPIO24 拉回高電位（秒）
ALERT_HOLD_TIME = 1.0

# 紀錄最近一次看到紅色的時間
last_red_time = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("讀取影像失敗")
            break

        # 轉 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 紅色區間（可依實際光線調整）
        lower_red1 = np.array([0,   80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # 計算紅色區域比例
        red_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = red_pixels / float(total_pixels)

        current_time = time.time()

        # ===== 判斷是否視為「有紅色」 =====
        if red_ratio > RED_RATIO_THRESHOLD:
            # 畫面中偵測到紅色 → 更新最後看到紅色的時間
            last_red_time = current_time
            red_detected = True
        else:
            # 沒有直接偵測到紅色
            # 但如果距離最後一次看到紅色還不到 ALERT_HOLD_TIME 秒
            # 就當作「還在警戒中」，延遲解除
            if current_time - last_red_time < ALERT_HOLD_TIME:
                red_detected = True
            else:
                red_detected = False

        # ===== 根據 red_detected 控制 GPIO24 與蜂鳴器 =====
        if red_detected:
            # 有紅色（或延遲時間內）→ GPIO24 低電位、蜂鳴器響
            GPIO.output(RELAY_PIN, GPIO.LOW)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            status_text = f"RED or HOLD! ratio={red_ratio:.3f}"
        else:
            # 完全沒有紅色且超過延遲時間 → GPIO24 高電位、蜂鳴器關
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            status_text = f"No red. ratio={red_ratio:.3f}"

        # 顯示畫面與狀態（可關掉）
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera", frame)
        cv2.imshow("Red Mask", mask)

        # 按 q 離開程式
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 稍微 sleep 一下避免 CPU 100%
        time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("程式結束，GPIO 已清理乾淨")
