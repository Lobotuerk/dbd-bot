# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import time

# keyList = range(256)
keyList = (87,17,32,1,84)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(key):
            keys.append(key)
        mousex, mousey = wapi.GetCursorPos()
    return keys, mousex
