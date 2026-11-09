import cv2

c=cv2.VideoCapture(0)
while True:
    r,f=c.read()
    if not r: break
    cv2.imshow("Camera",f)
    k=cv2.waitKey(1)
    if k==ord('s'):
        cv2.imwrite("dataset/images/stop/stop1.jpg",f)
        print("Photo OK")
    if k==ord('q'): break


c.release()
cv2.destroyAllWindows()
