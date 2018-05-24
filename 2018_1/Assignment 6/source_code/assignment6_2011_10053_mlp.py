## -*- coding: utf-8 -*-
#Assignment 6
#LSTM을 사용한 악보 prediction 모형 만들기 (MLP)
#2011-10053
#Noh Ha Kyung
#Submitted: 24 April 2018
#Python v3.6.4

"""
code2idx lines:
1- octave 1, length: 1
2- octave 1, length: 2
3- octave 1, length: 3
4- octave 2, length: 1
5- octave 2, length: 2
6- octave 2, length: 3
7- octave 0, length: 1
8- octave 0, length: 2
9- octave 0, length: 3
10- octave 1, legato_start, length: 1
11- octave 1, legato_start, length: 2
12- octave 1, legato_start, length: 3
13- octave 1, legato_end, length: 1
14- octave 1, legato_end, length: 2
15- octave 1, legato_end, length: 3
16- octave 2, legato_start, length: 1
17- octave 2, legato_start, length: 2
18- octave 2, legato_start, length: 3
19- octave 2, legato_start, length: 1
20- octave 2, legato_start, length: 2
21- octave 2, legato_start, length: 3
22- pause note
"""

# 0. 패키지 불러오기
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 데이터셋 생성 함수
def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq)-window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

# 1. 데이터 준비하기
# 코드 사전 정의
code2idx = {'c_oct1_1':0, 'd_oct1_1':1, 'e_oct1_1':2, 'f_oct1_1':3, 'g_oct1_1':4, 'a_oct1_1':5, 'b_oct1_1':6, \
            'c_oct1_2':7, 'd_oct1_2':8, 'e_oct1_2':9, 'f_oct1_2':10, 'g_oct1_2':11, 'a_oct1_2':12, 'b_oct1_2':13, \
            'c_oct1_3':14, 'd_oct1_3':15, 'e_oct1_3':16, 'f_oct1_3':17, 'g_oct1_3':18, 'a_oct1_3':19, 'b_oct1_3':20, \
            'c_oct2_1':21, 'd_oct2_1':22, 'e_oct2_1':23, 'f_oct2_1':24, 'g_oct2_1':25, 'a_oct2_1':26, 'b_oct2_1':27, \
            'c_oct2_2':28, 'd_oct2_2':29, 'e_oct2_2':30, 'f_oct2_2':31, 'g_oct2_2':32, 'a_oct2_2':33, 'b_oct2_2':34, \
            'c_oct2_3':35, 'd_oct2_3':36, 'e_oct2_3':37, 'f_oct2_3':38, 'g_oct2_3':39, 'a_oct2_3':40, 'b_oct2_3':41, \
            'c_oct0_1':42, 'd_oct0_1':43, 'e_oct0_1':44, 'f_oct0_1':45, 'g_oct0_1':46, 'a_oct0_1':47, 'b_oct0_1':48, \
            'c_oct0_2':49, 'd_oct0_2':50, 'e_oct0_2':51, 'f_oct0_2':52, 'g_oct0_2':53, 'a_oct0_2':54, 'b_oct0_2':55, \
            'c_oct0_3':56, 'd_oct0_3':57, 'e_oct0_3':58, 'f_oct0_3':59, 'g_oct0_3':60, 'a_oct0_3':61, 'b_oct0_3':62, \
            'c_oct1_1_legSt':63, 'd_oct1_1_legSt':64, 'e_oct1_1_legSt':65, 'f_oct1_1_legSt':66, 'g_oct1_1_legSt':67, 'a_oct1_1_legSt':68, 'b_oct1_1_legSt':69, \
            'c_oct1_2_legSt':70, 'd_oct1_2_legSt':71, 'e_oct1_2_legSt':72, 'f_oct1_2_legSt':73, 'g_oct1_2_legSt':74, 'a_oct1_2_legSt':75, 'b_oct1_2_legSt':76, \
            'c_oct1_3_legSt':77, 'd_oct1_3_legSt':78, 'e_oct1_3_legSt':79, 'f_oct1_3_legSt':80, 'g_oct1_3_legSt':81, 'a_oct1_3_legSt':82, 'b_oct1_3_legSt':83, \
            'c_oct1_1_legEnd':84, 'd_oct1_1_legEnd':85, 'e_oct1_1_legEnd':86, 'f_oct1_1_legEnd':87, 'g_oct1_1_legEnd':88, 'a_oct1_1_legEnd':89, 'b_oct1_1_legEnd':90, \
            'c_oct1_2_legEnd':91, 'd_oct1_2_legEnd':92, 'e_oct1_2_legEnd':93, 'f_oct1_2_legEnd':94, 'g_oct1_2_legEnd':95, 'a_oct1_2_legEnd':96, 'b_oct1_2_legEnd':97, \
            'c_oct1_3_legEnd':98, 'd_oct1_3_legEnd':99, 'e_oct1_3_legEnd':100, 'f_oct1_3_legEnd':101, 'g_oct1_3_legEnd':102, 'a_oct1_3_legEnd':103, 'b_oct1_3_legEnd':104, \
            'c_oct2_1_legSt':105, 'd_oct2_1_legSt':106, 'e_oct2_1_legSt':107, 'f_oct2_1_legSt':108, 'g_oct2_1_legSt':109, 'a_oct2_1_legSt':110, 'b_oct2_1_legSt':111, \
            'c_oct2_2_legSt':112, 'd_oct2_2_legSt':113, 'e_oct2_2_legSt':114, 'f_oct2_2_legSt':115, 'g_oct2_2_legSt':116, 'a_oct2_2_legSt':117, 'b_oct2_2_legSt':118, \
            'c_oct2_3_legSt':119, 'd_oct2_3_legSt':120, 'e_oct2_3_legSt':121, 'f_oct2_3_legSt':122, 'g_oct2_3_legSt':123, 'a_oct2_3_legSt':124, 'b_oct2_2_legSt':125, \
            'c_oct2_1_legEnd':126, 'd_oct2_1_legEnd':127, 'e_oct2_1_legEnd':128, 'f_oct2_1_legEnd':129, 'g_oct2_1_legEnd':130, 'a_oct2_1_legEnd':131, 'b_oct2_1_legEnd':132, \
            'c_oct2_2_legEnd':133, 'd_oct2_2_legEnd':134, 'e_oct2_2_legEnd':135, 'f_oct2_2_legEnd':136, 'g_oct2_2_legEnd':137, 'a_oct2_2_legEnd':138, 'b_oct2_2_legEnd':139, \
            'c_oct2_3_legEnd':140, 'd_oct2_3_legEnd':141, 'e_oct2_3_legEnd':142, 'f_oct2_3_legEnd':143, 'g_oct2_3_legEnd':144, 'a_oct2_3_legEnd':145, 'b_oct2_3_legEnd':146, \
            'pause_1': 147}

idx2code = {y:x for x,y in code2idx.items()}

# 시퀀스 데이터 정의
seq = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'f_oct1_2_legSt', 'f_oct1_1_legEnd', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1_legSt', 'd_oct1_1_legEnd', 'c_oct1_1', 'b_oct0_3', 'd_oct1_2', 'e_oct1_1', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1']

# 2. 데이터셋 생성하기
dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)
print(dataset)

# 입력(X)과 출력(Y) 변수로 분리하기
x_train = dataset[:,0:4]
y_train = dataset[:,4]

max_idx_value = 147

# 입력값 정규화 시키기
x_train = x_train / float(max_idx_value)

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(350, input_dim=4, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = LossHistory() # 손실 이력 객체 생성
history.init()

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=2000, batch_size=10, verbose=2, callbacks=[history])

# 6. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 8. 모델 사용하기
pred_count = 57 # 최대 예측 개수 정의

# 한 스텝 예측
seq_out = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2']
pred_out = model.predict(x_train)

for i in range(pred_count):
    idx = np.argmax(pred_out[i]) # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx]) # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장

print("one step prediction : ", seq_out)

# 곡 전체 예측
seq_in = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4)) # batch_size, feature
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

print("full song prediction : ", seq_out)
