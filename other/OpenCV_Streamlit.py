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
# import streamlit as st
from PIL import Image
import google.generativeai as genai
import streamlit as st

# Load the model
model = tf.keras.models.load_model(r"D:\Hackathons\Colossus\testFolder\FruitRecogV3.h5")

detector = htm.handDetector()

# # Load the categories
categories = os.listdir(r"D:\Hackathons\Colossus\testFolder\dataset3\test")
categories.sort()

st.set_page_config(page_title="Fruit Recognition")
st.header("Fruit Recognition")



# # Placeholde for image
img_placeholder = st.empty() 

#Placeholder for text
response_placeholder = st.empty()

# Create a dictionary with fruit names and prices
fruit_prices = {
    "Cantalope" : 30,
    "Coconut" : 15,
    "Corn" : 40,
    "Cucumber" : 20,
    "EggPlant" : 18,
    "Guava" : 7,
    "Mango" : 66,
    "Watermelon" :58

    # Add other fruits with their prices
}

def speech(text):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Text that you want to convert to speech
    # text = "Hello, how are you today?"

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


api_key1 = "AIzaSyCS37ro-GGbbLmOeM_l-X2j7YugpovSzrY"

if api_key1:
    genai.configure(api_key=api_key1)
    genai_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    print("API key not found. Please set the GOOGLE_API_KEY environment variable.")


def get_gemini_response(input_text, image):
    response = genai_model.generate_content([input_text, image])
    return response.text

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
    
    input_text = "From the given image detect and give me only the fruit names present in the input image, if no fruit is there give no fruit present, don't give any question mark at the end"
    
    cv.imwrite("FruitDetectionBGR.png", frame)
    image_path = "FruitDetectionBGR.png"
    image = Image.open(image_path)
    response_text = get_gemini_response(input_text, image)
    predicted_fruit = response_text
    input_text = "Count the number of each type of fruit in the image, and list each fruit by name and how many there are of them, only give names of the fruits and number only."
    response_text2 = get_gemini_response(input_text, image)
    fruit_text = "The predicted fruit is " + predicted_fruit
    # speech(fruit_text)
    speech(response_text)
    speech(response_text2)

    print(response_text)
    print(response_text2)

        # Get the price of the fruit from the dictionary
    fruit_price = fruit_prices.get(predicted_fruit, "Price not available")

    # print(f"The predicted fruit is {predicted_fruit}")
    # print(f"Price: {fruit_price} per unit")
    response_placeholder.subheader(response_text)
    response_placeholder.write(response_text2)
    # fruit_text = "The predicted fruit is " + predicted_fruit
    # speech(fruit_text)
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
            cv.putText(frame, str(f'{int(fps)} FPS'), (20,20), cv.FONT_HERSHEY_PLAIN, 2.3, (0,255,0), thickness=3)
            # cv.imshow('Fruit Detector', frame)
            cv.putText(frame, str(f'{MyPrediction}'), (fwud[0], fwud[1]-25), cv.FONT_HERSHEY_PLAIN, 2.3, (0,255,0), thickness=3)
            cv.imshow('Fruit Detector', frame)
            webImg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_placeholder.image(webImg, caption="MYImage")

            small_window = frame[fwud[0]:fwld[0], fwud[1]:fwld[1]]
            small_window = detector.findHands(small_window, draw=False)
            small_window_lmlist = detector.findPositions(small_window)

            
            if len(small_window_lmlist) !=0 :
                if (small_window_lmlist[8][1] > (0) and small_window_lmlist[8][1] <(small_window.shape[0]) and  small_window_lmlist[8][2] > (0) and small_window_lmlist[8][2] < (small_window.shape[1])):
                    # print("I am here")
                    MyPrediction = "No fruit detected"

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # detect = int(input())

    cv.destroyAllWindows()


import speech_recognition as sr
import keyboard  # Requires the 'keyboard' library

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to record and recognize speech
def recognize_speech():
    all_text = ""  # To store all recognized text

    # Use the microphone as the source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source)  # Adjusts to the surrounding noise
        print("Listening for speech... (Press 'q' to stop)")

        while True:
            if keyboard.is_pressed('q'):  # Stops listening when 'q' is pressed
                print("Stopped listening.")
                break
            # if 0xFF == ord('q'):
            #     print("Stopped listening.")
            #     break

            try:
                # Capture audio from the microphone
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing audio...")

                # Recognize speech using Google Web API
                text = recognizer.recognize_google(audio_data)
                print("Recognized text:", text)
                
                # Append recognized text to the cumulative result
                all_text += text + "\n"

                if "detect" in text.lower():
                    print("'Detect' found in speech, stopping...")
                    camera_start()
                    break  # Stop listening when the word 'Detect' is recognized

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except sr.WaitTimeoutError:
                print("Timeout: No speech detected.")

    # After the loop, save the accumulated text to a file
    if all_text:  # Only save if we have any recognized text
        with open("recognized_speech.txt", "w") as file:
            file.write(all_text)
            print("All recognized text saved to 'recognized_speech.txt'")
            

if __name__ == "__main__":
    recognize_speech()