# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:30:19 2022

@author: FidanVural
"""


import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

# VIDEO

cap = cv2.VideoCapture("C://Users//LENOVO//Desktop//ABO//cocuk2.mp4")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

path = "C:\\Users\\LENOVO\\Desktop\\ARA PROJE\\Models\\vgg_face.h5"

model = load_model(path)

#label_dict = {0 : "stresli", 1 : "stressiz"}

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'} 

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        face_rectangle = face_cascade.detectMultiScale(frame, 1.4, 7)

        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 10)
            roi_gray = frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            
            predictions = model.predict(img_pixels)
            print(predictions)
            emotion_label = int(np.argmax(predictions))
            
            emotion_prediction = emotion_dict[emotion_label]
            
            cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)

            resize_image = cv2.resize(frame, (1000,700))
            cv2.imshow('Emotion',resize_image)
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()