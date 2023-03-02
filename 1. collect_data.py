import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

w       = [1,0,0,0,0,0]
space   = [0,1,0,0,0,0]
wctrl   = [0,0,1,0,0,0]
wm1     = [0,0,0,1,0,0]
wizq    = [0,0,0,0,1,0]
wder    = [0,0,0,0,0,1]

starting_value = 1
while True:
    file_name = 'data/training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)

        break


def keys_to_output(keys, mouse, xpos):
    # print(keys, mouse - xpos > 50, mouse - xpos < -50)
    output = [0,0,0,0,0,0]

    if 17 in keys:
        output = wctrl
    elif 1 in keys:
        output = wm1
    elif 87 in keys:
        if mouse > xpos:
            output = wder
        elif mouse < xpos:
            output = wizq
        else:
            output = w
    else:
        #key 32
        output = space
    return output


def main(file_name, starting_value):
    xpos = 0
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = True
    print('STARTING!!!')
    while(True):

        if not paused:
            if xpos == 0:
                xpos = key_check()[1]
                continue
            screen = grab_screen(region=(0,0,1920,1080))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480,270))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()
            output = keys_to_output(keys[0], keys[1], xpos)
            # print(keys[1], xpos)
            xpos = keys[1]
            # time.sleep(1)
            # print(output)
            training_data.append([screen,output])
            # print(screen.shape)

            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
##            cv2.imshow('window',cv2.resize(screen,(640,360)))
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                cv2.destroyAllWindows()
##                break

            if len(training_data) % 100 == 0:
                # print(len(training_data))

                if len(training_data) == 500:
                    path = os.path.join(file_name)
                    np.save(path,training_data)
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = 'data/training_data-{}.npy'.format(starting_value)


        keys = key_check()
        if 84 in keys[0]:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main(file_name, starting_value)
