from ultralytics import YOLO
import cv2
import torch

# ✅ Ensure CUDA is available
if not torch.cuda.is_available():
    print("CUDA not available. Please install PyTorch with CUDA support.")
    exit()

# ✅ Load the model (YOLOv8, face-trained)
model = YOLO("yolov8l_100e.pt")  # Replace with your actual face-tuned model

# ✅ Open the video
cap = cv2.VideoCapture('../face-demographics-walking-and-pause.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Run inference on GPU
    results = model(frame, device='cuda')[0]  # Force GPU

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Face {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 Face Detection (GPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
