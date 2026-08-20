import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
import io

# ================= CONFIG =================
# Path to your ONNX model on the PC
YOLO_PATH = "./yolo/yolo_paper.onnx" 
YOLO_CONF_THRESH = 0.5
YOLO_INPUT_SIZE = (320, 320)

app = FastAPI()
net = None

def load_model():
    global net
    try:
        print(f"Loading YOLO model from {YOLO_PATH}...")
        net = cv2.dnn.readNetFromONNX(YOLO_PATH)
        
        # Try to use CUDA if available (PC with NVIDIA GPU)
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("CUDA backend set!")
        except:
            print("CUDA not available, using CPU (still faster than Pi).")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print(f"Error loading model: {e}")

load_model()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if net is None:
        return {"error": "Model not loaded"}

    # Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    h, w = frame.shape[:2]

    # Preprocess
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, YOLO_INPUT_SIZE, (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Inference
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    
    # Process Outputs (Same logic as Pi)
    predictions = outputs[0]
    
    # Shape Fix
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.transpose()

    boxes = []
    confidences = []
    
    x_factor = w / YOLO_INPUT_SIZE[0]
    y_factor = h / YOLO_INPUT_SIZE[1]

    if predictions.shape[1] >= 5:
        valid_rows = predictions[predictions[:, 4] > YOLO_CONF_THRESH]
        for row in valid_rows:
            confidence = float(row[4])
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            
            left = int((cx - bw/2) * x_factor)
            top = int((cy - bh/2) * y_factor)
            width = int(bw * x_factor)
            height = int(bh * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(confidence)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONF_THRESH, 0.4)
    
    results = []
    if len(indices) > 0:
        print(f"✅ Detected {len(indices)} objects")
        for i in indices.flatten():
            # Return format: [x, y, w, h, confidence]
            bx, by, bw, bh = boxes[i]
            results.append({
                "x": bx, "y": by, "w": bw, "h": bh,
                "conf": confidences[i]
            })
    else:
        print("❌ No objects detected")

    return {"detections": results}

if __name__ == "__main__":
    # Host on 0.0.0.0 to be accessible by Pi
    uvicorn.run(app, host="0.0.0.0", port=5000)
