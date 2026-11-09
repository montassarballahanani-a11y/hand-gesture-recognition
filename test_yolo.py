


import cv2
from ultralytics import YOLO
c=cv2.VideoCapture(0)
r,img=c.read()
c.release()
if r:
    m=YOLO("yolov8n.pt")
    result=m(img,imgsz=320)
    cv2.imwrite("resultat.jpg",result[0].plot())
    print("Image OK")
