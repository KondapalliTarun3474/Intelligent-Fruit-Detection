from flask import Flask, Response
import tensorflow as tf
import os
import mediapipe as mp
from keras.utils import img_to_array, load_img
import numpy as np
import cv2 as cv
import HandTrackingModule as htm
import pyttsx3
from gtts import gTTS
import io
import pygame
import time
from flask_cors import CORS
# import streamlit as st

app = Flask(__name__)
CORS(app)


# Load the model
model = tf.keras.models.load_model(r"C:\Users\K.S.TARUN\OneDrive\Desktop\swaroop_newborn\backend\FruitRecogV3.h5")

detector = htm.handDetector()

# # Load the categories
categories = os.listdir(r"C:\Users\K.S.TARUN\OneDrive\Desktop\swaroop_newborn\backend\dataset2\test")
categories.sort()

# Create a dictionary with fruit names and prices
fruit_prices = {
    "Cantalope" : 10,
    "Coconut" : 10,
    "Corn" : 10,
    "Cucumber" : 10,
    "EggPlant" : 10,
    "Guava" : 10,
    "Mango" : 10,
    "Watermelon" : 10

    # Add other fruits with their prices
}

def speech(text):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Convert the text to speech using gTTS (Google TTS)
    tts = gTTS(text=text, lang='en')

    # Create an in-memory file
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Load the speech directly from the in-memory file
    pygame.mixer.music.load(mp3_fp, 'mp3')

    # Play the audio
    pygame.mixer.music.play()

    # Keep the program running until the audio is done playing
    while pygame.mixer.music.get_busy():
        continue


# Function to prepare image
def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(100, 100))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255
    return imgResult

def detect_fruit(frame):
    
    # Resize and normalize the captured frame
    resized_frame = cv.resize(frame, (100, 100))
    # cv.imshow('Resized', resized_frame)
    imgResult = img_to_array(resized_frame)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.0

        # Predict the fruit
    resultArray = model.predict(imgResult, verbose=1)
    answers = np.argmax(resultArray, axis=1)

        # Get the predicted fruit name
    predicted_fruit = categories[answers[0]]

        # Get the price of the fruit from the dictionary
    fruit_price = fruit_prices.get(predicted_fruit, "Price not available")

    print(f"The predicted fruit is {predicted_fruit}")
    print(f"Price: {fruit_price} per unit")

    fruit_text = "The predicted fruit is " + predicted_fruit
    speech(fruit_text)
    # if len(predicted_fruit)>0:
    #     cv.putText(frame, str(f'{predicted_fruit}'), (100, 75), cv.FONT_HERSHEY_PLAIN, 2.3, (0,255,0), thickness=3)



    return predicted_fruit

def camera_start():
    ctime = 0
    ptime = 0
# Initialize video capture
    video = cv.VideoCapture(0)
    MyPrediction = "No fruit detected"
# detect = int(input())
    detect = False

    while True:
        isTrue, frame = video.read()

        if isTrue:
            frame = cv.flip(frame,1)
            original_height, original_width = frame.shape[:2]

        # Define a scale factor (e.g., 0.5 for half the size)
            scale_factor = 1.5

        # Calculate the new dimensions while maintaining the aspect ratio
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
        
            fwud=(100,100)
            fwld=(750,550)
            detectpos = (810, 50)
        # Resize the frame while maintaining the aspect ratio
            frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)


        # print(frame.shape)
            cv.rectangle(frame, fwud, fwld, (0,255,0), thickness=3)
            cv.putText(frame, "Detect", detectpos, cv.FONT_HERSHEY_PLAIN, 2.3, (0,255,0), thickness=3)
        
            
            frame = detector.findHands(frame, draw=False)
            lmlist = detector.findPositions(frame)

            if len(lmlist) != 0:
                if (lmlist[8][1] < (detectpos[0] + 120) and lmlist[8][1] > detectpos[0] and  lmlist[8][2] < detectpos[1] and lmlist[8][2] > (detectpos[1]-15)):
                    detect = True
                else:
                    detect = False
            else:
                detect = False

            if detect:
                MyPrediction = detect_fruit(frame)
            
            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime = ctime
    # Display the video feed
            cv.putText(frame, str(f'{fps}+"FPS"'), (20,20), cv.FONT_HERSHEY_PLAIN, 2.3, (0,255,0), thickness=3)
            cv.imshow('Fruit Detector', frame)
            cv.putText(frame, str(f'{MyPrediction}'), (fwud[0], fwud[1]-25), cv.FONT_HERSHEY_PLAIN, 2.3, (0,255,0), thickness=3)
            cv.imshow('Fruit Detector', frame)

            small_window = frame[fwud[0]:fwld[0], fwud[1]:fwld[1]]
            small_window = detector.findHands(small_window, draw=False)
            small_window_lmlist = detector.findPositions(small_window)

            if len(small_window_lmlist) !=0 :
                if (small_window_lmlist[8][1] > (0) and small_window_lmlist[8][1] <(fwld[0]-fwud[0]) and  small_window_lmlist[8][2] > (fwud[0]+50) and small_window_lmlist[8][2] < (fwud[0]-50)):
                    MyPrediction = "No fruit detected"

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cv.destroyAllWindows()

camera_start()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__' :
    app.run(debug=True)