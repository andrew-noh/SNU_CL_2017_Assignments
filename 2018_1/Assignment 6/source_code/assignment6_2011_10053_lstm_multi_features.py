## -*- coding: utf-8 -*-
#Assignment 6
#LSTM을 사용한 악보 prediction 모형 만들기 (LSTM Multiple Features)
#2011-10053
#Noh Ha Kyung
#Submitted: 24 April 2018
#Python v3.6.4

# 0. 사용할 패키지 불러오기
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils

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
    dataset_X = []
    dataset_Y = []

    for i in range(len(seq)-window_size):

        subset = seq[i:(i+window_size+1)]

        for si in range(len(subset)-1):
            features = code2features(subset[si])
            dataset_X.append(features)

        dataset_Y.append([code2idx[subset[window_size]]])

    return np.array(dataset_X), np.array(dataset_Y)

# 속성 변환 함수
def code2features(code):
    features = []
    features.append(code2scale[code[0]]/float(max_scale_value))
    features.append(code2oct[code[1]])
    features.append(code2length[code[2]])
    features.append(code2legato[code[3]])
    return features

def codeExpander(orig):
    res = []
    for i in range(len(orig)):
        source = seq[i]
        translate = ''
        if source[0] in 'cdefgab':
            translate += source[0].upper()
            if source[1] == '0':
                translate += '_Octave0_'
            elif source[1] == '1':
                translate += '_Octave1_'
            else:
                translate += '_Octave2_'
            if source[2] == '1':
                translate += 'length1'
            elif source[2] == '2':
                translate += 'length2'
            else:
                translate += 'length3'
            if source[3] in 'se':
                if source[3] == 's':
                    translate += '_legatoStart'
                else:
                    translate += '_legatoEnd'
        else:
            translate = 'Pause_1'
        res.append(translate)
    return res

# 1. 데이터 준비하기
# 코드 사전 정의
code2scale = {'c':0, 'd':1, 'e':2, 'f':3, 'g':4, 'a':5, 'b':6, 'x':7}
code2oct = {'0':0, '1':1, '2':2}
code2length = {'1':0, '2':1, '3':2}
code2legato = {'n':0, 's':1, 'e':2} #none, start, end

max_scale_value = 7.0

# 시퀀스 데이터 정의 (c100 - note+octave+length+legato)
code2idx = {'c11n':0, 'd11n':1, 'e11n':2, 'f11n':3, 'g11n':4, 'a11n':5, 'b11n':6, \
            'c12n':7, 'd12n':8, 'e12n':9, 'f12n':10, 'g12n':11, 'a12n':12, 'b12n':13, \
            'c13n':14, 'd13n':15, 'e13n':16, 'f13n':17, 'g13n':18, 'a13n':19, 'b13n':20, \
            'c21n':21, 'd21n':22, 'e21n':23, 'f21n':24, 'g21n':25, 'a21n':26, 'b21n':27, \
            'c22n':28, 'd22n':29, 'e22n':30, 'f22n':31, 'g22n':32, 'a22n':33, 'b22n':34, \
            'c23n':35, 'd23n':36, 'e23n':37, 'f23n':38, 'g23n':39, 'a23n':40, 'b23n':41, \
            'c01n':42, 'd01n':43, 'e01n':44, 'f01n':45, 'g01n':46, 'a01n':47, 'b01n':48, \
            'c02n':49, 'd02n':50, 'e02n':51, 'f02n':52, 'g02n':53, 'a02n':54, 'b02n':55, \
            'c03n':56, 'd03n':57, 'e03n':58, 'f03n':59, 'g03n':60, 'a03n':61, 'b03n':62, \
            'c11s':63, 'd11s':64, 'e11s':65, 'f11s':66, 'g11s':67, 'a11s':68, 'b11s':69, \
            'c12s':70, 'd12s':71, 'e12s':72, 'f12s':73, 'g12s':74, 'a12s':75, 'b12s':76, \
            'c13s':77, 'd13s':78, 'e13s':79, 'f13s':80, 'g13s':81, 'a13s':82, 'b13s':83, \
            'c11e':84, 'd11e':85, 'e11e':86, 'f11e':87, 'g11e':88, 'a11e':89, 'b11e':90, \
            'c12e':91, 'd12e':92, 'e12e':93, 'f12e':94, 'g12e':95, 'a12e':96, 'b12e':97, \
            'c13e':98, 'd13e':99, 'e13e':100, 'f13e':101, 'g13e':102, 'a13e':103, 'b13e':104, \
            'c21s':105, 'd21s':106, 'e21s':107, 'f21s':108, 'g21s':109, 'a21s':110, 'b21s':111, \
            'c22s':112, 'd22s':113, 'e22s':114, 'f22s':115, 'g22s':116, 'a22s':117, 'b22s':118, \
            'c23s':119, 'd23s':120, 'e23s':121, 'f23s':122, 'g23s':123, 'a23s':124, 'b22s':125, \
            'c21e':126, 'd21e':127, 'e21e':128, 'f21e':129, 'g21e':130, 'a21e':131, 'b21e':132, \
            'c22e':133, 'd22e':134, 'e22e':135, 'f22e':136, 'g22e':137, 'a22e':138, 'b22e':139, \
            'c23e':140, 'd23e':141, 'e23e':142, 'f23e':143, 'g23e':144, 'a23e':145, 'b23e':146, \
            'x01n': 147}

idx2code = {y:x for x,y in code2idx.items()}

# 시퀀스 데이터 정의
seq = ['a13n', 'd22n', 'c21n', 'b12n', 'd21n', 'a12s', 'a11e', 'd12s','e11e', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'f12s', 'f11e', 'b12n', 'd21n', 'a12n', 'g11n', 'f13n', 'g12n', 'f11n', 'e11n', 'b11n', 'a11n', 'd13s', 'd12e', 'x01n', 'e12s', 'f11e', 'g12n', 'f11n', 'e11s', 'd11e', 'c11n', 'b03n', 'd12n', 'e11n', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'd22s', 'c21e', 'e22n', 'd21n', 'c21n', 'b11n', 'a11n', 'f13n', 'e11n', 'b11n', 'a11n', 'e11n', 'g11n', 'f11n', 'd13s', 'd12e', 'x01n']


# 2. 데이터셋 생성하기
x_train, y_train = seq2dataset(seq, window_size = 4)

# 입력을 (샘플 수, 타임스텝, 특성 수)로 형태 변환
x_train = np.reshape(x_train, (57, 4, 4))

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 3. 모델 구성하기
model = Sequential()
model.add(LSTM(350, batch_input_shape = (1, 4, 4), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습시키기
num_epochs = 2000

history = LossHistory() # 손실 이력 객체 생성
history.init()

for epoch_idx in range(num_epochs):
    print ('epochs : ' + str(epoch_idx) )
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[history]) # 50 is X.shape[0]
    model.reset_states()

# 6. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train, batch_size=1)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
model.reset_states()

# 8. 모델 사용하기
pred_count = 57 # 최대 예측 개수 정의

# 한 스텝 예측
seq_out = ['a13n', 'd22n', 'c21n', 'b12n']
pred_out = model.predict(x_train, batch_size=1)

for i in range(pred_count):
    idx = np.argmax(pred_out[i]) # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx]) # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장

print("one step prediction : ", seq_out)
print("="*80)
print("Expanded code : ", codeExpander(seq_out), '\n\n')

model.reset_states()

# 곡 전체 예측
seq_in = ['a13n', 'd22n', 'c21n', 'b12n']
seq_out = seq_in

seq_in_featrues = []

for si in seq_in:
    features = code2features(si)
    seq_in_featrues.append(features)

for i in range(pred_count):
    sample_in = np.array(seq_in_featrues)
    sample_in = np.reshape(sample_in, (1, 4, 4)) # 샘플 수, 타입스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])

    features = code2features(idx2code[idx])
    seq_in_featrues.append(features)
    seq_in_featrues.pop(0)

model.reset_states()

print("full song prediction : ", seq_out)
print("="*80)
print("Expanded code : ", codeExpander(seq_out))
