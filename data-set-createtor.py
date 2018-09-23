import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
id = input("Please Inter Your Name: ")
nums = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        nums = nums+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(nums)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.waitKey(100)

    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if nums>20:
        break

cap.release()
cv2.destroyAllWindows()
