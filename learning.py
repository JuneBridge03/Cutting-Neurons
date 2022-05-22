import numpy as np
import json
import datetime
from keras.models import Sequential
from keras.layers import Dense #필요한 라이브러리 참조

seed = 5
np.random.seed(seed)

model = Sequential() #모델 객체 생성
model.add(Dense(8, input_dim=3, activation='relu')) #입력 뉴런 3개와 첫번째 은닉 층 뉴런 32개로 설정. 활성화 함수는 relu 함수로 설정
#for i in [9, 5]: #i가 순차적으로 12, 8로 바뀌게 반복
#    model.add(Dense(i, activation='relu')) #i(12, 8)을 은닉층의 뉴런 수로 순차적으로 적용 활성화 함수는 relu 함수로 설정
model.add(Dense(2, activation='softmax')) # 출력 층의 뉴런을 2개로 설정. 활성화 함수는 softmax함수로 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


wide_history = []
yide_history = []

def start_learning():
    time = '10 00 03'
    with open('learning_data_'+time+'.json', 'r') as f:
        data = json.load(f)
        wide_history = data[0]
        yide_history = data[1]
        f.close()
        model.fit(x=[wide_history], y=[yide_history], batch_size = 20, epochs = 200)

start_learning()

model.save("8_trex_ann_model.h5")