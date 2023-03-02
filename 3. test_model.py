import numpy as np
from grabscreen import grab_screen
import cv2
import time
from pynput.keyboard import Key, Controller as keycontroller
from pynput.mouse import Button, Controller as mousecontroller
from googlenet import create_googlenet
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode,mean
import numpy as np
from motion import motion_detection
from tensorflow import keras
from pool_helper import PoolHelper
from lrn import LRN
import win32api
import win32con
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import pyautogui

keyboard = keycontroller()
mouse = mousecontroller()
GAME_WIDTH = 1920
# GAME_WIDTH = 1440
GAME_HEIGHT = 1080
# GAME_HEIGHT = 900

LR = 1e-3
WIDTH = 224
HEIGHT = 224
MODEL_NAME = 'FullModel.h5'

model = create_googlenet(WIDTH, HEIGHT, 3, LR, output=6, model_name=MODEL_NAME)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# pre_trained_model = InceptionV3(input_shape = (3, WIDTH, HEIGHT),
#                                 include_top = False,
#                                 weights = None)
# local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# pre_trained_model.load_weights(local_weights_file)
#
# for layer in pre_trained_model.layers:
#   layer.trainable = False
#
# last_layer = pre_trained_model.get_layer('mixed7')
# last_output = last_layer.output
#
# from tensorflow.keras.optimizers import RMSprop
# x = layers.Flatten()(last_output)
# x = layers.Dense(64, activation='relu')(x)
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(6, activation='sigmoid')(x)
# model = Model( pre_trained_model.input, x)
#
# model.compile(optimizer = RMSprop(lr=0.0001),
#               loss = 'categorical_crossentropy',
#               metrics = ['acc'])

model.load_weights(MODEL_NAME)

# space_t = cv2.imread('./templates/space.png')
# ctrl_t = cv2.imread('./templates/ctrl.png')

def wrout():
    keyboard.press('w')

def ctrlrout():
    keyboard.release('w')
    keyboard.press(Key.ctrl)
    time.sleep(3)
    keyboard.release(Key.ctrl)

def spacerout():
    keyboard.release('w')
    keyboard.press(Key.space)
    time.sleep(3)
    keyboard.release(Key.space)

def m1rout():
    keyboard.release('w')
    mouse.press(Button.left)
    time.sleep(1.5)
    mouse.release(Button.left)

def wctrlrout():
    keyboard.press('w')
    keyboard.press(Key.ctrl)
    time.sleep(3)
    keyboard.press('w')
    keyboard.release(Key.ctrl)

def wm1rout():
    keyboard.press('w')
    mouse.press(Button.left)
    time.sleep(0.25)
    keyboard.press('w')
    mouse.release(Button.left)

def wizqrout():
    keyboard.press('w')
    for i in range(25):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -10, 0)
        time.sleep(0.01)
        keyboard.press('w')

def wderrout():
    keyboard.press('w')
    for i in range(25):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 10, 0)
        time.sleep(0.01)
        keyboard.press('w')

def main():
    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0,0,GAME_WIDTH,GAME_HEIGHT))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen, (WIDTH,HEIGHT))
    X = np.array(screen.reshape(3,WIDTH,HEIGHT))
    X = X.reshape(1,3,WIDTH,HEIGHT)
    model.predict(X)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    while(True):

        if not paused:
            screen = grab_screen(region=(0,0,GAME_WIDTH,GAME_HEIGHT))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            if np.allclose(screen[1004,1665],[106,106,106]):
                print("Unready")
                continue
            elif (np.allclose(screen[1006,1707],[153,153,153])
            and np.allclose(screen[1006,1728],[153,153,153])
            and np.allclose(screen[1011,1751],[153,153,153])):
                print("Ready")
                pyautogui.moveTo(1728, 1006)
                time.sleep(1)
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 10, 0)
                time.sleep(1)
                pyautogui.click()
                time.sleep(1)
                pyautogui.moveTo(800, 500)
                continue
            elif (np.allclose(screen[940,1746],[153,153,153])
            and np.allclose(screen[937,1773],[153,153,153])
            and np.allclose(screen[940,1799],[153,153,153])):
                print("Searching")
                time.sleep(10)
                continue
            elif (np.allclose(screen[1006,1721],[153,153,153])
            and np.allclose(screen[1005,1758],[152,152,152])
            and np.allclose(screen[1002,1781],[152,152,152])):
                print("Continue")
                pyautogui.moveTo(1728, 1006)
                time.sleep(1)
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 10, 0)
                time.sleep(1)
                pyautogui.click()
                time.sleep(1)
                pyautogui.moveTo(800, 500)
                continue
            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            X = np.array(screen.reshape(3,WIDTH,HEIGHT))
            X = X.reshape(1,3,WIDTH,HEIGHT)
            prediction = model.predict(X)[0]
            weights += np.array([0.1, 0.05, 0.05, 0.05, 0.1, 0.1])
            prediction = np.array(prediction) * weights
            print(prediction)
            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                wrout()
                choice_picked = 'w'
                weights[mode_choice] *= 0.8
            elif mode_choice == 1:
                spacerout()
                choice_picked = 'space'
                weights[mode_choice] *= 0
            elif mode_choice == 2:
                wctrlrout()
                choice_picked = 'w+ctrl'
                weights[mode_choice] *= 0
            elif mode_choice == 3:
                wm1rout()
                choice_picked = 'w+m1'
                weights[mode_choice] *= 0.2
            elif mode_choice == 4:
                wizqrout()
                choice_picked = 'w+izq'
                weights[mode_choice] *= 0.8
            elif mode_choice == 5:
                wderrout()
                choice_picked = 'w+der'
                weights[mode_choice] *= 0.8

            print('loop took {} seconds. Choice: {}'.format( round(time.time()-last_time, 3) , choice_picked))
            print(weights)
        keys = key_check()

        if 84 in keys[0]:
            if paused:
                print("UNPAUSED")
                paused = False
                time.sleep(1)
            else:
                print("PAUSED")
                paused = True
                mouse.release(Button.left)
                keyboard.release(Key.ctrl)
                keyboard.release(Key.space)
                keyboard.release('W')
                time.sleep(1)

main()
