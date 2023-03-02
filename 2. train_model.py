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


FILE_I_END = 142

WIDTH = 224
HEIGHT = 224
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'FullModel.h5'
PREV_MODEL = 'FullModel.h5'

LOAD_MODEL = True

model = create_googlenet(WIDTH, HEIGHT, 3, LR, output=6, model_name=MODEL_NAME)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

if LOAD_MODEL:
    model.load_weights(PREV_MODEL)
    print('We have loaded a previous model!!!!')


# iterates through the training files


for e in range(EPOCHS):
    print('EPOCH NUMBER ', e)
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):

        try:
            file_name = './datanew/training_data-{}.npy'.format(i)
            # full file info
            train_data = np.load(file_name)
            print('training_data-{}.npy'.format(i), e, '{}/{}'.format(count,FILE_I_END))
            train = train_data[:-50]
            test = train_data[-50:]
            X = np.array([cv2.resize(i[0], dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for i in train])
            X = X.reshape(X.shape[0],3,WIDTH,HEIGHT)
            Y = np.array([i[1] for i in train])

            test_x = np.array([cv2.resize(i[0], dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for i in test])
            test_x = test_x.reshape(test_x.shape[0],3,WIDTH,HEIGHT)
            test_y = np.array([i[1] for i in test])
            # print(model.summary())
            # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            model.fit(x=X, y=Y, epochs=1, validation_data=(test_x,test_y), batch_size=25)

            if count%10 == 0:
                print('SAVING MODEL!')
                model.save_weights(MODEL_NAME)

        except Exception as ex:
            print(str(ex))

# os.system('shutdown -s -t 0')
