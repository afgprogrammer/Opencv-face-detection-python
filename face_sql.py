import cv2
import numpy as np
import sqlite3

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
name = ""
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(int(conf)<70):
            print(Id)
            if(int(Id) == 1):
                name="Mohammad"
            elif(int(Id) == 2):
                name="Rahmani"
        else:
            name="Unknown"

        cv2.putText(im, str(name), (x, y + h), font, 1, (255,255,255))
    cv2.imshow('im',im)
    if cv2.waitKey(30) & 0xFF == 27:
        break
cam.release()
cv2.destroyAllWindows()