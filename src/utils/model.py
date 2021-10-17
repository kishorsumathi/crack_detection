import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
import keras
from keras.layers import  Conv2D,MaxPooling2D,Dense,Dropout,Flatten,Activation
from tensorflow.keras import layers
from keras.layers import BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import logging




def efficientnetb3(input_shape,model_path):

    model = tf.keras.applications.EfficientNetB3(
    input_shape=input_shape, weights="imagenet",include_top=False
    )
    model.save(model_path)
    logging.info(f"efficient model save at: {model_path}")
    return model

def prepare_model(model,classes,freeze_all,freeze_till,learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable = False
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(classes, activation="sigmoid", name="pred")(x)
    model = tf.keras.Model(model.input, outputs, name="EfficientNetB3")
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"]
    )
    logging.info("custom model compailes and ready for training")
    print(model.summary())
    return model