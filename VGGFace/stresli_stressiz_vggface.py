# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:09:25 2022

@author: FidanVural
"""


import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

# VIDEO

cap = cv2.VideoCapture("C://Users//LENOVO//Desktop//ABO//azra.mp4")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

path = "C://Users//LENOVO//Desktop//ARA PROJE//Models//vggface_ss.h5"

model = load_model(path)


label_dict = {0 : "stresli", 1 : "stressiz"} 

while True:
    
    ret, frame = cap.read() 
    
    if ret:
        
        face_rectangle = face_cascade.detectMultiScale(frame, 1.3, 7)

        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 10)
            roi_gray = frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0) 
            
            predictions = model.predict(img_pixels)
            emotion_label = [1 * (x[0]<=0.7) for x in predictions] 
            
            emotion_prediction = label_dict[emotion_label[0]]
            
            cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

            resize_image = cv2.resize(frame, (1000,700))
            cv2.imshow('Emotion',resize_image)
            
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()