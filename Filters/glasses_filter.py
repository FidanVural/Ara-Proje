# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:38:54 2022

@author: Bengi
"""

import cv2 
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor= dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img= cv2.imread("C:\\Users\\LENOVO\\Desktop\\ARA PROJE\\Filters\\ThreeFilter\\glassesImage.png")

while True:
    
    
    ret,frame = cap.read()
    
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = detector(grayFrame)
    
    for face in faces:
        
        landmarks = predictor(grayFrame,face)
        
        leftEye = landmarks.part(36).x , landmarks.part(36).y
        
        rightEye = landmarks.part(45).x , landmarks.part(45).y
        
        center =landmarks.part(28).x , landmarks.part(28).y
        
        
        """
        cv2.circle(frame,leftEye,2,(0,255,0),-1)
        cv2.circle(frame,rightEye,2,(0,255,0),-1)
        """
        
        eyeWidth = int(hypot(leftEye[0]-rightEye[0],leftEye[1]-rightEye[1])*1.9)
        
        eyeHeight = int(eyeWidth*0.4)
        
        glasses = cv2.resize(img,(eyeWidth,eyeHeight))
        
        glassesGray= cv2.cvtColor(glasses,cv2.COLOR_BGR2GRAY)
        
        _, glassesMask = cv2.threshold(glassesGray,25,255,cv2.THRESH_BINARY)
        
        #cv2.imshow("Maske",glassesMask)
        
        
        topLeft= (int(center[0]- eyeWidth/2),int(center[1]- eyeHeight/2))
        
        rightBottom = (int(center[0]+ eyeWidth/2),int(center[1]+ eyeHeight/2))
        
        #cv2.rectangle(frame,topLeft,rightBottom,(0,255,0),2)
        
        eyesArea = frame[topLeft[1]:topLeft[1]+eyeHeight,topLeft[0]:topLeft[0]+eyeWidth]
        
        #cv2.imshow("Gözler",eyesArea)
        
        noEyes =  cv2.bitwise_and(eyesArea,eyesArea,mask=glassesMask)
        
        #cv2.imshow("Filtrele",noEyes)
        
        frame[topLeft[1]:topLeft[1]+eyeHeight,topLeft[0]:topLeft[0]+eyeWidth]=noEyes
        
        
        
    
    
    cv2.imshow("Video",frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    


cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    