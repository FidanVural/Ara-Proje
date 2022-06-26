# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:42:34 2022

@author: FidanVural
"""

import cv2
import numpy as np
import dlib
from math import hypot


cap = cv2.VideoCapture(0)
nose_image = cv2.imread("C:\\Users\\LENOVO\\Desktop\\ARA PROJE\\Filters\\ThreeFilter\\pig_nose.png")
#nose_image = cv2.imread("dog_nose.png")

detector = dlib.get_frontal_face_detector()

# landmark için bir tahmin edici
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
     _, frame= cap.read()
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
     faces = detector(frame)

     for face in faces:
        # print(face) # videoda yüzün koordinatlarını döner.
        # Sol üst köşe ile sağ alt köşe !
        landmarks = predictor(gray, face)
        
        # neden 29 olduğuna resimden bakabilirsin. -> facial_landmarks.png
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        # cv2.circle(frame, top_nose, 3, (255,0,0), -1)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        
        # İlk adım filtremizin boyutunu burun boyutuna göre ayarlamak olacak.
        
        # Bu nose_width ve nose_height değerleri float çıkar.
        # Ancak biz float değerler kullanamayız.
        # Bu nedenle de bunları int yaparız.
        nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1]) * 1.8)
        
        nose_height = int(nose_width * 0.63)
        # 0.63 şöyle bulundu -> Orijinal resmin boyutlarına baktık.
        # Bu boyutlar 840x535 çıktı.
        # Biz yukarıda burun için width bulmuştuk zaten.
        # Bu değeri 535/840 (0.63) ile çarparak nose_height buluruz.
        # Aslında bir oran orantı yapıyoruz.
        # 840'a 535 ise nose_width'e kaçtır? -> Kaçtırın cevabı nose_height olur.
        
        
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        
        # print(nose_width)
        
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        
        # Lets draw rectangle for the nose.
        """
        cv2.rectangle(frame, (int(center_nose[0]-nose_width/2), int(center_nose[1]-nose_height/2)),
                      (int(center_nose[0]+nose_width/2), int(center_nose[1]+nose_height/2)), (255,0,0), 2)
        """

        top_left = (int(center_nose[0]-nose_width/2), int(center_nose[1]-nose_height/2))
        bottom_right = (int(center_nose[0]+nose_width/2), int(center_nose[1]+nose_height/2))
         
        
        # Şimdi bizim filtremizi resme uygulamamız için bir adım daha var.
        # O da threshold. Biz filtre resmimizin sadece burun kısmını orijinal resmimize koymak istiyoruz.
        # Yani dikdörtgen olan filtre resminin sadece burun kısmının orijinal resmimizde olmasını isteriz.
        # Bunun için filtre resmin burun kısmını çıkarabilmek için threshold uygularız.
        
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
         
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                          top_left[0]: top_left[0] + nose_width]
         
        #cv2.imshow("Nose area", nose_area)
         
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask = nose_mask)
        
        
        final_nose = cv2.add(nose_area_no_nose, nose_pig) 
        
        frame[top_left[1]: top_left[1] + nose_height,
                           top_left[0]: top_left[0] + nose_width] = final_nose
        
        #cv2.imshow("Pig nose", nose_image)
        #cv2.imshow("Resize nose pig", nose_pig)
        #cv2.imshow("Mask", nose_mask)
        #cv2.imshow("Nose area no nose", nose_area_no_nose)
        
        # En son bunu çizdirdik şimdi bunu da yoruma alalım.
        #cv2.imshow("Final nose", final_nose)
        
     cv2.imshow("Frame", frame)
     
     key = cv2.waitKey(1)
     
     if key == 27:
         break


