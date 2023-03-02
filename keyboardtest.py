from pynput.keyboard import Key, Controller as keycontroller
from pynput.mouse import Button, Controller as mousecontroller
import win32api, win32con
import time

keyboard = keycontroller()
mouse = mousecontroller()

time.sleep(2)
# keyboard.press('w')
for i in range(25):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 10, 0)
    time.sleep(0.01)
# keyboard.release('w')
