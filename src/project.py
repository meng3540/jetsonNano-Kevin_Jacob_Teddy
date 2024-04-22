from ultralytics import YOLO
import cv2
import torch

camera = cv2.VideoCapture(0)

model = YOLO('yolov8n.pt')
model.export(format='engine', device='gpu')

tRT_model = YOLO('yolov8n.engine')

print (torch.cuda.is_available())
while (True):
    ret, frame = camera.read() 
    results = trt_model(frame)
    
    if cv2.waitkey(100) & 0xFF == ord('q'):
        break
vid.realease()
cv2.DestroyAllWindows()
