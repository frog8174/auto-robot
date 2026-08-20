import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_black_panels(image_path):
    # 1. 讀取影像
    img = cv2.imread(image_path)
    if img is None:
        print("找不到圖片")
        return

    # 縮放一下以便觀察 (非必要)
    scale_percent = 50 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img_resized = cv2.resize(img, (width, height))
    
    # 複製一份用來畫圖
    output_img = img_resized.copy()

    # 2. 轉換到 HSV 色彩空間
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # 3. 定義黑色的範圍
    # H (色相): 0-180 (黑色通常無所謂色相，全包)
    # S (飽和度): 0-255 (黑色飽和度通常不穩定，範圍可以寬一點)
    # V (亮度): 0-80 (這是關鍵，只要亮度夠低就是黑色)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60]) # 若環境較亮，可將 80 調高至 100

    # 建立 mask
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 4. 形態學處理 (修補 mask)
    # 先用較大的 kernel 進行閉運算 (Close)，把反光的洞補起來
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 再稍微膨脹一點點，讓邊緣更滑順
    mask_clean = cv2.dilate(mask_closed, kernel, iterations=1)

    # 5. 尋找輪廓
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_count = 0
    for cnt in contours:
        # 計算面積
        area = cv2.contourArea(cnt)
        
        # 6. 篩選條件
        # 濾掉太小的雜訊 (例如地磚縫隙)
        if area > 2000:  
            # 近似多邊形，確認形狀是否接近矩形
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 雖然是矩形，但因為拍攝角度可能有透視，邊角可能是 4~8 個點
            # 這裡主要依靠面積篩選即可，若要更嚴格可加 len(approx) == 4
            
            # 畫出輪廓 (綠色)
            cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 3)
            
            # 畫出 Bounding Box (紅色框)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            found_count += 1

    # 顯示結果
    # 由於在 Jupyter/Colab 環境，這裡用 matplotlib 顯示
    # 如果是在本機跑，可以用 cv2.imshow
    img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title(f"Detection Result (Found: {found_count})")
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.title("Processed Mask")
    plt.imshow(mask_rgb)
    plt.show()

# 呼叫函式

image_path = './images/photo_20251204_184758.jpg'
detect_black_panels(image_path)