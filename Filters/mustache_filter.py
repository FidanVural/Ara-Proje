# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:42:49 2022

@author: FidanVural
"""

import cv2
import numpy as np
import dlib
from math import hypot
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)
mustache_img = cv2.imread("C:\\Users\\LENOVO\\Desktop\\ARA PROJE\\Filters\\ThreeFilter\\mustache_2.jpeg", -1) 

detector = dlib.get_frontal_face_detector()

# landmark için bir tahmin edici
predictor = dlib.shape_predictor("C:\\Users\\LENOVO\\Desktop\\ARA PROJE\\Filters\\shape_predictor_68_face_landmarks.dat")

#plt.imshow(mustache_img)

while True:
    
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(frame)
    
    
    for face in faces:
        
        landmarks = predictor(gray, face)
        
        left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
        right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
        
        mustache_width = int(hypot(left_mouth[0] - right_mouth[0],
                           left_mouth[1] - right_mouth[1]) * 2)
        
        mustache_height = int(mustache_width * 0.35)
        
        mustache_resize = cv2.resize(mustache_img, (mustache_width, mustache_height))
        
        #cv2.imshow("Mustache", mustache_resize)
        
        center_mustache = (landmarks.part(33).x, landmarks.part(33).y)
        
        """
        cv2.rectangle(frame, (int(center_mustache[0]-mustache_width/2), int(center_mustache[1]-mustache_height/2)),
                      (int(center_mustache[0]+mustache_width/2), int(center_mustache[1]+mustache_height/2)), (255,0,0), 2)
        """
        
        top_left = (int(center_mustache[0]-mustache_width/2), int(center_mustache[1]-mustache_height/2))
        bottom_right = (int(center_mustache[0]+mustache_width/2), int(center_mustache[1]+mustache_height/2))
        
        
        mustache_area = frame[top_left[1]: top_left[1] + mustache_height,
                          top_left[0]: top_left[0] + mustache_width]
        
        mustache_gray = cv2.cvtColor(mustache_resize, cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(mustache_gray,25,255,cv2.THRESH_BINARY)
        
        bg= cv2.bitwise_and(mustache_area,mustache_area,mask=mask)
         
        
        #cv2.imshow("Mustache area", mustache_area)
        
        frame[top_left[1]: top_left[1] + mustache_height,
                          top_left[0]: top_left[0] + mustache_width] = bg
        
        
    
        #cv2.imshow("M", mustache_resize)
        
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    
    if key == 27:
        break