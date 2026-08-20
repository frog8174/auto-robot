def process_yolo_detection(frame):
    """Detects objects, finds closest to center, calculates PID, draws bbox."""
    global pid_state
    if yolo_net is None:
        return frame

    height, width = frame.shape[:2]
    frame_center_x = width // 2

    # Preprocess
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, YOLO_INPUT_SIZE, (0, 0, 0),
        swapRB=True, crop=False
    )
    yolo_net.setInput(blob)
    
    # Inference
    try:
        outputs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())
    except Exception as e:
        print(f"Inference error: {e}")
        return frame

    # Parse outputs (Standard YOLOv5 output: 1, 25200, 85)
    predictions = outputs[0]

    class_ids = []
    confidences = []
    boxes = []

    x_factor = width / YOLO_INPUT_SIZE[0]
    y_factor = height / YOLO_INPUT_SIZE[1]

    # 只保留 conf > YOLO_CONF_THRESHOLD 的 rows
    valid_rows = predictions[predictions[:, 4] > YOLO_CONF_THRESHOLD]

    for row in valid_rows:
        confidence = row[4]
        scores = row[5:]
        class_id = np.argmax(scores)
        if scores[class_id] > 0.5:  # Class score threshold
            cx, cy, w, h = row[0], row[1], row[2], row[3]

            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            w = int(w * x_factor)
            h = int(h * y_factor)

            boxes.append([left, top, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                               YOLO_CONF_THRESHOLD, YOLO_NMS_THRESHOLD)

    best_obj = None
    min_dist = float('inf')

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            obj_center_x = x + w // 2

            # 畫出所有偵測到的框（淡黃色）
            cv2.rectangle(frame, (x, y),
                          (x + w, y + h), (0, 255, 255), 1)

            # 找離畫面中心最近的物件
            dist = abs(frame_center_x - obj_center_x)
            if dist < min_dist:
                min_dist = dist
                best_obj = (obj_center_x, x, y, w, h, class_ids[i])

    # === PID 計算 & log ===
    if best_obj:
        obj_cx, x, y, w, h, cls_id = best_obj

        # 用綠框標示「選中的那一個」物件
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame,
                   (obj_cx, y + h // 2),
                   5, (0, 0, 255), -1)

        # error：畫面中心 - 物件中心 (px)
        # error > 0  → 物件在畫面左邊 → 車子應該向左修正
        # error < 0  → 物件在畫面右邊 → 車子應該向右修正
        error = frame_center_x - obj_cx

        P = PID_KP * error
        pid_state["integral"] += error
        I = PID_KI * pid_state["integral"]
        D = PID_KD * (error - pid_state["prev_error"])

        output = P + I + D
        pid_state["prev_error"] = error

        # Terminal log：清楚顯示目前狀態
        print(
            f"[PID] center={frame_center_x} obj_x={obj_cx} "
            f"err={error:+4d}  P={P:+.4f} I={I:+.4f} D={D:+.4f}  out={output:+.4f}"
        )

        # 畫在影像上
        cv2.putText(frame,
                    f"ERR:{error:+d} PID:{output:+.2f}",
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # === 之後要真的「邊前進邊修正」可以這樣做 ===
        # 現在依照你的要求「先不控制移動」，所以這段只當參考，預設不啟動。
        if AUTO_STEER_ENABLED and state.get("motion") == "forward":
            # 基礎前進速度（用目前 state["speed"]）
            base_speed = state.get("speed", PWM_SPEED_DEFAULT)

            # 這裡把 PID 輸出當作轉向量，適度縮放避免太大
            steer_gain = 1.0  # 可以視情況調小，例如 0.5
            steer = float(output) * steer_gain

            left_speed = base_speed + steer
            right_speed = base_speed - steer

            # 實際設定馬達速度（函式已經會幫你 clamp 到 [-1, 1]）
            set_motor_speeds(left_speed, right_speed)

    return frame
