from ultralytics import YOLO
import cv2
#Data
img_path = "bike_2.png"
model_path = "best.pt"
img = cv2.imread(img_path)
input = cv2.resize(img,(640,640))

# đưa qua model với ngưỡng conf = 0.5
model = YOLO(model_path)
results = model.predict(input, conf = 0.5)
for result in results:
    boxes = result.boxes
    result.show()
    result.save(filename="Result.jpg")  # save to disk