from ultralytics import YOLO
model = YOLO('yolo11n.pt')

model.train(data='data-2/data.yaml', imgsz=640, seed=42, epochs=100)