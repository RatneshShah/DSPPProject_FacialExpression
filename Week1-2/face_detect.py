import numpy as np
import cv2
import os
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

img = cv2.imread('FaceDatabase/Final/5_4.JPG')
#print img
#img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 0) # (image,reduceScaleFactor,minNeighbours)
#print faces
mouth1 = None
roi_color = None
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # cv2.rectangle(image, startingcoordinates,endingcoordinatesdiagonal,(b,g,r),thickness)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    mouths = mouth_cascade.detectMultiScale(roi_gray, 2, 57)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        only_eyes = roi_color[ey:ey+eh, ex:ex+eh]
    for (mx,my,mw,mh) in mouths:
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
        mouth1 = roi_color[my:my+mh,mx:mx+mw]
cv2.imshow('img',img)
cv2.imshow('abcd',roi_color)
cv2.imshow('abcde',mouth1)
cv2.waitKey(0)
cv2.destroyAllWindows()
