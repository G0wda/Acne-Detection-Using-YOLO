import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, imgsz=640, conf=0.5)

    # Annotate frame with detections
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("YOLOv11 - Acne Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
