import numpy as np
import time
import os
from random import shuffle
choices = ['W', 'space', 'wctrl', 'wm1', 'wizq', 'wder']
FILE_I_END = 78

WIDTH = 224
HEIGHT = 224
starting_value = 105

w       = [1,0,0,0,0,0]
space   = [0,1,0,0,0,0]
wctrl   = [0,0,1,0,0,0]
wm1     = [0,0,0,1,0,0]
wizq    = [0,0,0,0,1,0]
wder    = [0,0,0,0,0,1]
hist    = np.array([0,0,0,0,0,0])

ws = []
spaces = []
wctrls = []
wm1s = []
wizqs = []
wders = []

data_order = [i for i in range(1,FILE_I_END+1)]
shuffle(data_order)
for count,i in enumerate(data_order):
    try:
        file_name = './data/training_data-{}.npy'.format(i)
        train_data = np.load(file_name, allow_pickle=True)
        print('training_data-{}.npy'.format(i), '{}/{}'.format(count,FILE_I_END), '{}'.format(hist))
        X = np.array([i[0] for i in train_data])
        # X = X.reshape(X.shape[0],3,WIDTH,HEIGHT)
        Y = np.array([i[1] for i in train_data])
        # Y = np.delete(Y,[1,3,8],1)
        for t in range(len(Y)):
            y = Y[t]
            x = X[t]
            if np.allclose(y,w):
                ws.append([x,y])
                hist[0] += 1
            elif np.allclose(y,space):
                spaces.append([x,y])
                hist[1] += 1
            elif np.allclose(y,wctrl):
                wctrls.append([x,y])
                wctrls.append([x,y])
                wctrls.append([x,y])
                wctrls.append([x,y])
                wctrls.append([x,y])
                wctrls.append([x,y])
                hist[2] += 6
            elif np.allclose(y,wm1):
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                wm1s.append([x,y])
                hist[3] += 13
            elif np.allclose(y,wizq):
                wizqs.append([x,y])
                hist[4] += 1
            elif np.allclose(y,wder):
                wders.append([x,y])
                hist[5] += 1
            if np.amin(hist) >= 100:
                shuffle(ws)
                shuffle(spaces)
                shuffle(wctrls)
                shuffle(wm1s)
                shuffle(wizqs)
                shuffle(wders)
                training_data = ws[:100] + spaces[:100] + wctrls[:100] + wm1s[:100] + wizqs[:100] + wders[:100]
                ws = ws[100:]
                spaces = spaces[100:]
                wctrls = wctrls[100:]
                wm1s = wm1s[100:]
                wizqs = wizqs[100:]
                wders = wders[100:]
                shuffle(training_data)
                shuffle(training_data)
                file_save = 'datanew/training_data-{}.npy'.format(starting_value)
                path = os.path.join(file_save)
                np.save(path,training_data)
                print('SAVED')
                starting_value += 1
                hist -= 100
    except Exception as ex:
        print(str(ex))
print(hist)
