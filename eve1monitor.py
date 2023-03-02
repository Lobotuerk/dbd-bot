import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import cv2
from grabscreen import grab_screen
import pyautogui
from pynput.mouse import Button, Controller as mousecontroller
import win32api, win32con

def out_drones():
    pyautogui.moveTo(000 + 1111, 634)
    time.sleep(1)
    pyautogui.click(button='right')
    time.sleep(1)
    pyautogui.moveTo(000 + 1164, 645)
    time.sleep(1)
    pyautogui.click()
    time.sleep(3)

def in_drones():
    pyautogui.moveTo(000 + 1106, 657)
    time.sleep(1)
    pyautogui.click(button='right')
    time.sleep(2)
    pyautogui.moveTo(000 + 1188, 707)
    time.sleep(1)
    pyautogui.click()
    time.sleep(4)

def mine_routine(belt, screen):
    find_asteroid(belt, screen)
    out_drones()
    mine(screen)
    while True:
        screen = grab_screen(region=(000,0,1440,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        if not np.allclose(screen[863,416],[23,18,6]):
            in_drones()
            break
        elif not np.allclose(screen[106,1097],[255,255,255]):
            if not (np.allclose(screen[211,1124],[201,201,201])
            and np.allclose(screen[217,1144],[198,198,198])):
                belt += 1
                in_drones()
                find_asteroid(screen)
            else:
                mine()

def go_back(screen):
    pyautogui.moveTo(000 + 1151, 274)
    time.sleep(1)
    pyautogui.click(button='right')
    time.sleep(1)
    pyautogui.moveTo(000 + 1182, 339)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    while not (np.allclose(screen[191,1306],[172,205,216])
    and np.allclose(screen[191,1322],[172,195,204])):
        time.sleep(0.1)
        screen = grab_screen(region=(000,0,1440,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    print("docked")
    time.sleep(5)

def mine(screen):
    pyautogui.moveTo(000 + 1144, 217)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(000 + 1096, 106)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    while not (np.allclose(screen[94,1163],[229,229,229])
    and np.allclose(screen[105,1166],[191,191,191])):
        time.sleep(1)
        screen = grab_screen(region=(000,0,1440,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    print("close to asteroid")
    pyautogui.moveTo(000 + 1164, 105)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    while not (np.allclose(screen[105,1163],[0,0,255])
    and np.allclose(screen[112,1171],[42,42,170])):
        time.sleep(1)
        screen = grab_screen(region=(000,0,1440,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    print("mining")
    time.sleep(1)
    pyautogui.moveTo(000 + 840, 713)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(000 + 892, 712)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)

def undock(screen):
    pyautogui.moveTo(000 + 1328, 195)
    time.sleep(1)
    pyautogui.click()
    time.sleep(2)
    while not (np.allclose(screen[213,1245],[178,178,178])
    and np.allclose(screen[232,1299],[191,191,191])):
        time.sleep(0.1)
        screen = grab_screen(region=(000,0,1440,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    print("undocked")
    time.sleep(1)


def down_cargo(screen):
    pyautogui.moveTo(000 + 248, 137)
    time.sleep(1)
    pyautogui.mouseDown()
    time.sleep(1)
    pyautogui.moveTo(000 + 115, 172)
    time.sleep(1)
    pyautogui.mouseUp()
    time.sleep(1)
    pyautogui.moveTo(000 + 324, 137)
    time.sleep(1)
    pyautogui.mouseDown()
    time.sleep(1)
    pyautogui.moveTo(000 + 115, 172)
    time.sleep(1)
    pyautogui.mouseUp()
    time.sleep(1)
    pyautogui.moveTo(000 + 398, 137)
    time.sleep(1)
    pyautogui.mouseDown()
    time.sleep(1)
    pyautogui.moveTo(000 + 115, 172)
    time.sleep(1)
    pyautogui.mouseUp()
    time.sleep(1)
    print("finished downing cargo")
    time.sleep(1)

def go_belt(n, screen):
    time.sleep(5)
    pyautogui.moveTo(000 + 1179, 171)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(000 + 1157, 217 + n * 16)
    time.sleep(1)
    pyautogui.click(button='right')
    time.sleep(1)
    pyautogui.moveTo(000 + 1193, 227 + n * 16)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(000 + 1083, 171)
    time.sleep(1)
    pyautogui.click()
    time.sleep(8)
    while not (np.allclose(screen[98,1195],[190,190,190])
    and np.allclose(screen[109,1196],[189,189,189])):
        time.sleep(0.1)
        screen = grab_screen(region=(000,0,1440,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    print("reached belt ", n+1)
    time.sleep(8)

def find_asteroid(belt, screen):
    while True:
        go_belt(belt, screen)
        screen = grab_screen(region=(000,0,1440,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        if not (np.allclose(screen[217,1124],[192,192,192])
        and np.allclose(screen[214,1137],[194,194,194])):
            # print(screen[217,1124])
            # print(screen[214,1137])
            belt += 1
            continue
        print("found steroid")
        break


belt = 0
screen = grab_screen(region=(000,0,1440,900))
screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
while True:
    screen = grab_screen(region=(000,0,1440,900))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    undock(screen)
    mine_routine(belt, screen)
    go_back(screen)
    down_cargo(screen)
    # cv2.imshow('window',screen)
    # pyautogui.moveTo(000 + 1106, 657)
    # time.sleep(1)
    # pyautogui.click(button='right')
    # time.sleep(2)
    # cv2.imwrite("temp.jpg",screen)
    # print(screen[217,1124])
    # print(screen[214,1137])
    # break
    # cv2.destroyAllWindows()
    # elif cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
