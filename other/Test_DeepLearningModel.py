import tensorflow as tf
import os
from keras.utils import img_to_array, load_img
import numpy as np
#import cv2 as cv

#load the model
model = tf.keras.models.load_model(r"/home/tahirlinux/env/FruitRecogV3.h5")
print(model.summary())

#load the categories

# source_folder = r"C:\Users\chait\OneDrive\Desktop\Machine Learning\fruits-360_dataset\fruits-360\Test"
categories = os.listdir("/home/tahirlinux/env/dataset3/test")
categories.sort()
print(categories)
numofClasses = len(categories)
print(numofClasses)

#load and prepare image

def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(100,100))
    imgResult = img_to_array(image)
    print(imgResult.shape)
    imgResult = np.expand_dims(imgResult, axis=0)
    print(imgResult.shape)
    imgResult = imgResult/255

    return imgResult

testImagePath = "/home/tahirlinux/env/dataset2/extra/unnamed (1).jpg"
imageForModel = prepareImage(testImagePath)

resultArray = model.predict(imageForModel, verbose=1)
answers = np.argmax(resultArray, axis = 1)
print(answers[0])

text = categories[answers[0]]
print("Predicted Images is " + text)