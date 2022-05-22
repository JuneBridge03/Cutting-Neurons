import cv2
from PIL import ImageGrab
import numpy as np
from time import sleep
import threading
import random
import keyboard
from keras.models import load_model # 구동에 필요한 라이브러리들을 참조

def get_data(box): # get_data 함수, box는 캡처할 화면의 x 좌표와 y 좌표를 담은 튜플
    screen =  np.array(ImageGrab.grab(bbox=box)) # 이미지를 캡처하여서 np.ndarray 타입으로 만든 뒤 screen 변수에 저장
    img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) # screen 변수에 있는 RGB                                                                                값을 이용해서 GrayScale로 
    return img_gray #스크린 정보 반환

model = load_model("1_trex_ann_model.h5")

trex_img = cv2.cvtColor(cv2.imread("trextrex.png"), cv2.COLOR_BGR2GRAY)
cactus_img = cv2.cvtColor(cv2.imread("cactus.png"), cv2.COLOR_BGR2GRAY)
lcac_img = cv2.cvtColor(cv2.imread("little_cactus.png"), cv2.COLOR_BGR2GRAY)
bird_img = cv2.cvtColor(cv2.imread("bird.png"), cv2.COLOR_BGR2GRAY)
return_img = cv2.cvtColor(cv2.imread("return.png"), cv2.COLOR_BGR2GRAY)

h1, w1 = trex_img.shape
h2, w2 = cactus_img.shape
h3, w3 = bird_img.shape
h4, w4 = lcac_img.shape

def is_stop(screen):
    r = cv2.matchTemplate(screen, return_img, cv2.TM_CCOEFF_NORMED)
    return len(np.where(r >= 0.8)[0]) != 0

history = [450, 450, 450]

def release():
        sleep(0.3)
        keyboard.release('space')    

while True: #in night, white : 172, black: 0
    if keyboard.is_pressed('q'):
        break
    #screen = get_data((100, 170, 580, 270))
    screen = get_data((180, 170, 780, 300))
    if is_stop(screen):
        keyboard.press('space')
        keyboard.release('space')
        continue
    if screen[0][0] != 255:
        if screen[0][0] == 0:
            screen = 255 - screen
        else:
            continue
    t_l = cv2.matchTemplate(screen, trex_img, cv2.TM_CCOEFF_NORMED)
    c_l = cv2.matchTemplate(screen, cactus_img, cv2.TM_CCOEFF_NORMED)
    l_l = cv2.matchTemplate(screen, lcac_img, cv2.TM_CCOEFF_NORMED)
    b_l = cv2.matchTemplate(screen, bird_img, cv2.TM_CCOEFF_NORMED)
    
    t_y, t_x = np.where(t_l >= 0.9)
    c_y, c_x = np.where(c_l >= 0.8)
    l_y, l_x = np.where(l_l >= 0.8)
    b_y, b_x = np.where(b_l >= 0.8) 

    for i in [
        (t_x, t_y, h1, w1),
        (c_x, c_y, h2, w2),
        (b_x, b_y, h3, w3),
        (l_x, l_y, h4, w4)]:
        if len(i[0]) == 0:
            continue
        x = i[0][0]
        y = i[1][0]
        cv2.rectangle(screen, (x, y), (x + i[3], y + i[2]), (0, 0, 0), 2)
    x = 450
    for l in [(c_x, c_y), (l_x, l_y), (b_x, b_y)]:
        i = l[0]
        if len(i) == 0:
            continue
        if i[0] >= 50 and i[0] < x and l[1][0] >= 30:
            x  = i[0] - 50
    
    y = model.predict(np.expand_dims(history, axis=0))[0]
    print(y)    
    if y[1] <= 0.45:
        y = 0
    else:
        y = 1

    if y == 1:
        print("Jump")
        keyboard.press('space')
        threading.Thread(target=release).start()
    history.insert(3, x)
    del history[0]
    print("history : " + str(history))

model.save("new_trex_ann_model.h5")