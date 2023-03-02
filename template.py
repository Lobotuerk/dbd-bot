import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('./images/1/76.png',0)

template = cv2.imread('./templates/space.png')
for i in range(1,2):
    print(i)
    file_name = './data/training_data-{}.npy'.format(i)
    train_data = np.load(file_name , allow_pickle=True)
    X = np.array([i[0] for i in train_data])
    Y = np.array([i[1] for i in train_data])
    for i in range(len(X)):
        cv2.rectangle(X[i], (540, 300), (623, 388), (0, 0, 0), -1)
        img = np.delete(X[i], 2, 2)
        print(img.shape)
        img = img.astype(np.float32)
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(max_val)
        if max_val >= 1000000:
            print(i)

    # Apply template Matching
