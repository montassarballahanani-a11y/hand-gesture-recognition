


import cv2,time
c=cv2.VideoCapture(0)
time.sleep(10)
for i in range(20):
    r,f=c.read()
    if not r: break
    cv2.imwrite(f"dataset/images/ok/ok_new_{i}.jpg",f)
    time.sleep(0.5)
c.release()
print("20 photos OK")
