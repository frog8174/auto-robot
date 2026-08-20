import cv2
import numpy as np
import time
import os
from datetime import datetime

def find_wall_vertical():
    # 確保 images 資料夾存在
    if not os.path.exists('images'):
        os.makedirs('images')

    # 開啟攝影機
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("無法開啟攝影機")
        return
    CAM_WIDTH = 1000
    CAM_HEIGHT = 640
    # 設定 FPS 與 MJPG 格式以提升效能
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    CAM_WIDTH = 1000
    CAM_HEIGHT = 640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    # 讀取實際設定值確認
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS set to: {actual_fps}")

    print("操作說明:")
    print(" - 按下 'q' 鍵退出程式")
    print(" - 按下 's' 鍵拍照 (同時儲存原始圖與標記圖)")

    saved_message_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法接收畫面 (stream end?). Exiting ...")
            break

        # ---------------------------------------------------------
        # 1. 找出深色牆壁 (Color Thresholding)
        # ---------------------------------------------------------
        # 轉換到 HSV 色彩空間
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定義黑色的範圍 (參考 find_wall.py)
        # H: 0-180, S: 0-255, V: 0-60 (可視環境亮度調整 V 的上限)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80]) 

        # 建立 mask
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 形態學處理：閉運算補洞，膨脹修飾邊緣
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.dilate(mask_closed, kernel, iterations=1)

        # ---------------------------------------------------------
        # 2. 找出直線 (Line Detection on Mask)
        # ---------------------------------------------------------
        # 對 "mask_clean" 做 Canny 邊緣偵測，這樣只會抓出深色區域的輪廓
        edges = cv2.Canny(mask_clean, 50, 150)

        # Hough Line Transform 偵測直線
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

        # 準備繪圖用的影像
        output_frame = frame.copy()
        
        # 將 Mask 轉為彩色以便疊圖顯示 (半透明紅色表示偵測到的深色區域)
        mask_rgb = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        output_frame = cv2.addWeighted(output_frame, 1, mask_rgb, 0.3, 0) # 疊加 mask 視覺化

        vertical_lines_count = 0
        horizontal_lines_count = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                dx = abs(x1 - x2)
                dy = abs(y1 - y2)

                # 判斷是否為垂直線 (高度變化 > 2倍寬度變化)
                if dy > 2 * dx: 
                    # 畫出垂直線 (綠色, 粗度 3)
                    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    vertical_lines_count += 1
                # 判斷是否為水平線 (寬度變化 > 2倍高度變化)
                elif dx > 2 * dy:
                    # 畫出水平線 (藍色, 粗度 3)
                    cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    horizontal_lines_count += 1
                else:
                    # 其他線條 (黃色, 較細) - 可選擇是否顯示
                    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # ---------------------------------------------------------
        # 3. 顯示資訊與介面
        # ---------------------------------------------------------
        # 顯示垂直線數量 (綠色)
        cv2.putText(output_frame, f"V Lines: {vertical_lines_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 顯示水平線數量 (藍色)
        cv2.putText(output_frame, f"H Lines: {horizontal_lines_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 拍照回饋
        if saved_message_timer > 0:
            cv2.putText(output_frame, "Photo Saved!", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            saved_message_timer -= 1

        # 顯示視窗
        # cv2.imshow('Mask (Dark Wall)', mask_clean) # 如果想看單純的 mask 可取消註解
        cv2.imshow('Result', output_frame)

        # ---------------------------------------------------------
        # 4. 按鍵控制
        # ---------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_orig = f"images/photo_{timestamp}.jpg"
            filename_proc = f"images/photo_{timestamp}_processed.jpg"
            
            cv2.imwrite(filename_orig, frame)       # 存原始圖
            cv2.imwrite(filename_proc, output_frame) # 存結果圖 (含標記)
            
            print(f"已儲存照片:\n  - 原始: {filename_orig}\n  - 處理: {filename_proc}")
            saved_message_timer = 30

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    find_wall_vertical()
