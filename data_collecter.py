import cv2
from PIL import ImageGrab
import numpy as np
import random
import keyboard
import json
import datetime
#from keras.models import Sequential
#from keras.layers import Dense #필요한 라이브러리 참조

def get_data(box): # get_data 함수, box는 캡처할 화면의 x 좌표와 y 좌표를 담은 튜플
    screen =  np.array(ImageGrab.grab(bbox=box)) # 이미지를 캡처하여서 np.ndarray 타입으로 만든 뒤 screen 변수에 저장
    img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) # screen 변수에 있는 RGB 값을 이용해서 GrayScale로 
    return img_gray #스크린 정보 반환

seed = 5
np.random.seed(seed)

#model = Sequential() #모델 객체 생성
#model.add(Dense(32, input_dim=3, activation='relu')) #입력 뉴런 3개와 첫번째 은닉 층 뉴런 32개로 설정. 활성화 함수는 relu 함수로 설정
#for i in [12, 8]: #i가 순차적으로 12, 8로 바뀌게 반복
#    model.add(Dense(i, activation='relu')) #i(12, 8)을 은닉층의 뉴런 수로 순차적으로 적용 활성화 함수는 relu 함수로 설정
#model.add(Dense(2, activation='softmax')) # 출력 층의 뉴런을 2개로 설정. 활성화 함수는 softmax함수로 설정
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
wide_history = []
yide_history = []

def start_learning():
    time = datetime.datetime.now().strftime('%H %M %S')
    with open('learning_data_'+time+'.json', 'w') as f:
        data = [wide_history[:-4], yide_history[:-4]]
        json.dump(data, f)
        f.close()
    #model.fit(x=[wide_history[:-4]], y=[yide_history[:-4]], batch_size = 20, epochs = 200)
    
        
print("START")
while True: #in night, white : 172, black: 0
    if keyboard.is_pressed('q'):
        break
    screen = get_data((180, 170, 780, 300))
    if is_stop(screen):
        start_learning()
        break
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
    b_y, b_x = np.where(b_l >= 0.8) #A3 420*297
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
            x  = int(i[0] - 50)
    history.insert(3, x)
    del history[0]
    wide_history.append([history[0], history[1], history[2]])
    #print(str([history[0], history[1], history[2]]))
    yy = [0, 0]
    yy[int(keyboard.is_pressed('space'))] = 1
    yide_history.append(yy)
    #cv2.imshow("result", screen)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #        cv2.destroyAllWindows()
    #        break

#model.save("new_trex_ann_model.h5")