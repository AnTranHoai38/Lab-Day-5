from ultralytics import YOLO
import cv2
#Data
img_path = "bike_2.png"
model_path = "best.pt"

#tiền xử lí
img = cv2.imread(img_path)
img = cv2.resize(img, (640,640))
blur_img = cv2.GaussianBlur(img,(5,5), 0)
input = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)

# đưa qua model với ngưỡng conf = 0.5
model = YOLO(model_path)
results = model.predict(input, conf = 0.5)

# xử lí kết quả
for result in results[0].boxes:
    # lấy các tọa độ bbox, id, confidence
    bbox = result.xyxy[0].tolist()
    class_id = result.cls
    conf = result.conf[0]
    #vẽ lên ảnh gốc
    x,y,w,h = map(int,bbox[:4])
    label = model.names[int(class_id)]
    text = f'{label}: {conf:.2f}'
    cv2.rectangle(img,(x,y),(w,h), (0,0,255),2)
    cv2.putText(img, text,(x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255),1)
cv2.imshow("Detection",img)
cv2.imwrite("Result.jpg",img)
cv2.waitKey(0)
