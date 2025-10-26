from ultralytics import YOLO
model = YOLO('best.pt')
result = model.predict(source='0_before.jpg')

result[0].save("result.png")