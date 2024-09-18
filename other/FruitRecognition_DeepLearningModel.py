import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


# from keras.utils import img_to_array, load_img
import numpy as np
import cv2 as cv
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from keras.preprocessing import ImageDataGenerator

train_path = "/home/tahirlinux/env/dataset3/train"
test_path = "/home/tahirlinux/env/dataset3/test"

BatchSize = 64

# img = tf.keras.utils.load_img("/home/chaitanya/Desktop/try2/fruits-360_dataset/fruits-360/Test/Apple Braeburn/3_100.jpg")
# plt.imshow(img)
# plt.show()

# imgA = tf.keras.utils.img_to_array(img)
# print(imgA.shape)



#Build the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", input_shape=(100,100,3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5000, activation="relu"))
model.add(tf.keras.layers.Dense(1000, activation="relu"))
model.add(tf.keras.layers.Dense(12, activation="softmax"))

# print(model.summary())


#compile the model
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])



#load the data
# Load the data with corrected import and rescale value
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.3, horizontal_flip=True, vertical_flip=True, zoom_range=0.3)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_path, target_size=(100,100), batch_size=BatchSize, color_mode="rgb", class_mode="categorical", shuffle=True)

test_generator = test_datagen.flow_from_directory(test_path, target_size=(100,100), batch_size=BatchSize, color_mode="rgb", class_mode="categorical")


stepsPerEpoch = int(np.ceil(train_generator.samples / BatchSize))
ValidationSteps = int(np.ceil(test_generator.samples / BatchSize))

#Early Stopping

stop_early = EarlyStopping(monitor="val_accuracy", patience=5)

history = model.fit(train_generator, steps_per_epoch=stepsPerEpoch, epochs=50, validation_steps=ValidationSteps, callbacks=[stop_early])   #I CHANGED EPOCHS count to 5 from 50
model.save("/home/tahirlinux/env/FruitRecogV3.h5")
