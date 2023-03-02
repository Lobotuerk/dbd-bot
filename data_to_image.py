import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import cv2


histo = np.zeros((1,6))
# print(Y.shape)
n = int(input("start number: "))
nmax = int(input("finish number: "))
counts = [0,0,0,0,0,0]
for i in range(n,nmax + 1):
    print(i)
    file_name = './data/training_data-{}.npy'.format(i)
    train_data = np.load(file_name , allow_pickle=True)
    X = np.array([i[0] for i in train_data])
    Y = np.array([i[1] for i in train_data])
    for i in range(len(X)):
        label = np.argmax(Y[i])
        name = './images/' + str(label) + '/{}.png'.format(counts[label])
        while os.path.isfile(name):
            counts[label] += 1
            name = './images/' + str(label) + '/{}.png'.format(counts[label])
        name = counts[label]
        imageio.imwrite('./images/' + str(label) + '/' + str(name) + '.png', X[i])
        counts[label] += 1
