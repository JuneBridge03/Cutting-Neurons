import numpy as np
from keras.models import Sequential
from keras.layers import Dense #필요한 라이브러리 참조
from keras.models import load_model

#model = load_model('1_trex_ann_model.h5')
model = load_model('4_trex_ann_model.h5')

w = np.array(model.get_weights())

dense_1 = w[0]
bias_1 = w[1]
dense_2 = w[2]
bias_2 = w[3]
dense_3 = w[4]
bias_3 = w[5]
dense_4 = w[6]
bias_4 = w[7]


w[0] = dense_1[:, :21] #처음: 21까지만, 중간: 4~9 cut
w[1] = bias_1[:21]
w[2] = np.concatenate((dense_2[:21, :4], dense_2[:21, 9:]), axis=1)
w[3] = np.concatenate((bias_2[:4], bias_2[9:]), axis=0)
w[4] = np.concatenate((dense_3[:4], dense_3[9:]), axis=0)

for i in range(0, 8):
    print(w[i].shape)


model = Sequential() #모델 객체 생성
model.add(Dense(21, input_dim=3, activation='relu')) #입력 뉴런 3개와 첫번째 은닉 층 뉴런 32개로 설정. 활성화 함수는 relu 함수로 설정
for i in [9, 5]: #i가 순차적으로 12, 8로 바뀌게 반복
    model.add(Dense(i, activation='relu')) #i(12, 8)을 은닉층의 뉴런 수로 순차적으로 적용 활성화 함수는 relu 함수로 설정
model.add(Dense(2, activation='softmax')) # 출력 층의 뉴런을 2개로 설정. 활성화 함수는 softmax함수로 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




model.set_weights(w)

model.save("new_2_trex_ann_model.h5")