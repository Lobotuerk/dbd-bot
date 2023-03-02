import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from googlenet import create_googlenet
from random import shuffle
from keras.utils.vis_utils import plot_model
import tensorflow as tf
[32073, 19, 1385, 0, 1030, 71, 0, 3, 4419]
choises = ['W', 'space', 'wctrl', 'wm1', 'wizq', 'wder']
hist = [0, 0, 0, 0, 0, 0]
FILE_I_END = 16

WIDTH = 224
HEIGHT = 224
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'FirstModel.h5'

LOAD_MODEL = False

model = create_googlenet(WIDTH, HEIGHT, 3, LR, output=6, model_name=MODEL_NAME)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model.load_weights(MODEL_NAME)

data_order = [i for i in range(1,FILE_I_END+1)]
shuffle(data_order)
for count,i in enumerate(data_order):
    try:
        histo = np.zeros((1,9))
        file_name = './datanew/training_data-{}.npy'.format(i)
        # full file info
        train_data = np.load(file_name)
        print('training_data-{}.npy'.format(i), '{}/{}'.format(count,FILE_I_END))
        X = np.array([i[0] for i in train_data])
        X = X.reshape(X.shape[0],3,WIDTH,HEIGHT)
        Y = np.array([i[1] for i in train_data])
        results = model.evaluate(X, Y, batch_size=128)
        print("test loss, test acc:", results)
        for x in X:
            prediction = model.predict(x.reshape(1,3,WIDTH,HEIGHT))
            choise = np.argmax(prediction)
            # print(choises[choise])
            hist[choise] += 1

    except Exception as ex:
        print(str(ex))
print(hist)
