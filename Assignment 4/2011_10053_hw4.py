# coding: utf-8

#Noh Hakyung
#2011-10053
#HW4

import sys, os
sys.path.append(os.pardir)
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnistPy2 import load_mnist

start_time = datetime.now()

#Open a text file to write down results
f = open('result_output.txt', 'w')

#External functions
#----------------------------------------
#Functions.py

#1. Step Function
def step_function(x):
    return np.array(x > 0, dtype=np.int)

#2. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

#3. Sigmoid_grad
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

#4. Relu
def relu(x):
    return np.maximum(0, x)

#5. Relu_grad
def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

#6. Softmax
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

#7. Cross_entropy_error
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

#8. Softmax_loss
def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

#9. Epoch number calculation
def epoch_num(i):
    if i == 0:
        return str(1) + ' '
    else:
        epoch_num = i / iter_per_epoch
        if epoch_num < 9:
            return str(epoch_num+1) + ' '
        else:
            return str(epoch_num+1)

#----------------------------------------
#Gradient.py

#Numerical Gradient
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

#----------------------------------------
# Two_layer_net.py

class TwoLayerNet_sig:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient_sig(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

class TwoLayerNet_relu:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient_relu(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

class TwoLayerNet_step:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = step_function(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient_step(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = step_function(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

#----------------------------------------
# Main

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#Sigmoid
network_1 = TwoLayerNet_sig(input_size=784, hidden_size=50, output_size=10) #Learning Rate 1
network_2 = TwoLayerNet_sig(input_size=784, hidden_size=50, output_size=10) #Learning Rate 2 (Net1 copy)

network_3 = TwoLayerNet_relu(input_size=784, hidden_size=50, output_size=10) #Learning Rate 1
network_4 = TwoLayerNet_relu(input_size=784, hidden_size=50, output_size=10) #Learning Rate 2 (Net3 copy)

network_5 = TwoLayerNet_step(input_size=784, hidden_size=50, output_size=10) #Learning Rate 1
network_6 = TwoLayerNet_step(input_size=784, hidden_size=50, output_size=10) #Learning Rate 2 (Net5 copy)

#Network Label:
# network_1 : Sigmoid Function, Learning Rate: 0.1
# network_2 : Sigmoid Function, Learning Rate: 0.01
# network_3 : Relu Function, Learning Rate: 0.1
# network_4 : Relu Function, Learning Rate: 0.01
# network_5 : Step Function, Learning Rate: 0.1
# network_6 : Step Function, Learning Rate: 0.01


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate_1 = 0.1
learning_rate_2 = 0.01


#Gradient descent method
sig_train_loss_list_1 = []
sig_train_acc_list_1 = []
sig_test_acc_list_1 = []

sig_train_loss_list_2 = []
sig_train_acc_list_2 = []
sig_test_acc_list_2 = []

#Relu Function
relu_train_loss_list_1 = []
relu_train_acc_list_1 = []
relu_test_acc_list_1 = []

relu_train_loss_list_2 = []
relu_train_acc_list_2 = []
relu_test_acc_list_2 = []

#Step Function
step_train_loss_list_1 = []
step_train_acc_list_1 = []
step_test_acc_list_1 = []

step_train_loss_list_2 = []
step_train_acc_list_2 = []
step_test_acc_list_2 = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

#1. Sigmoid Function

print "Sigmoid Function, iter per epoch = " + str(iter_per_epoch)
print >> f, "Sigmoid Function, iter per epoch = " + str(iter_per_epoch)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad_1 = network_1.gradient_sig(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network_1.params[key] -= learning_rate_1 * grad_1[key]
        network_2.params[key] -= learning_rate_2 * grad_1[key]

    # 학습 경과 기록
    loss_1 = network_1.loss(x_batch, t_batch)
    sig_train_loss_list_1.append(loss_1)

    loss_2 = network_2.loss(x_batch, t_batch)
    sig_train_loss_list_2.append(loss_2)


    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        epoch_nmb = epoch_num(i)

        #Learning Rate 1
        train_acc_1 = network_1.accuracy(x_train, t_train)
        test_acc_1 = network_1.accuracy(x_test, t_test)
        sig_train_acc_list_1.append(train_acc_1)
        sig_test_acc_list_1.append(test_acc_1)

        #Learning Rate 2
        train_acc_2 = network_2.accuracy(x_train, t_train)
        test_acc_2 = network_2.accuracy(x_test, t_test)
        sig_train_acc_list_2.append(train_acc_2)
        sig_test_acc_list_2.append(test_acc_2)

        print "Epoch:" + epoch_nmb + " | Learning Rate: 0.1(train acc, test acc) " + str(train_acc_1) + ", " + str(test_acc_1) + "      |      " + "Learning Rate: 0.01(train acc, test acc) " + str(train_acc_2) + ", " + str(test_acc_2)
        print >> f, "Epoch:" + epoch_nmb + " | Learning Rate: 0.1(train acc, test acc) " + str(train_acc_1) + ", " + str(test_acc_1) + "      |      " + "Learning Rate: 0.01(train acc, test acc) " + str(train_acc_2) + ", " + str(test_acc_2)


#2. Relu Function

print "\nRelu Function, iter per epoch = " + str(iter_per_epoch)
print >> f, "\nRelu Function, iter per epoch = " + str(iter_per_epoch)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

# 기울기 계산
    grad_2 = network_3.gradient_relu(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network_3.params[key] -= learning_rate_1 * grad_2[key]
        network_4.params[key] -= learning_rate_2 * grad_2[key]

    # 학습 경과 기록
    loss_3 = network_3.loss(x_batch, t_batch)
    relu_train_loss_list_1.append(loss_3)

    loss_4 = network_4.loss(x_batch, t_batch)
    relu_train_loss_list_2.append(loss_4)


    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        epoch_nmb = epoch_num(i)

        #Learning Rate 1
        train_acc_3 = network_3.accuracy(x_train, t_train)
        test_acc_3 = network_3.accuracy(x_test, t_test)
        relu_train_acc_list_1.append(train_acc_3)
        relu_test_acc_list_1.append(test_acc_3)

        #Learning Rate 2
        train_acc_4 = network_4.accuracy(x_train, t_train)
        test_acc_4 = network_4.accuracy(x_test, t_test)
        relu_train_acc_list_2.append(train_acc_4)
        relu_test_acc_list_2.append(test_acc_4)

        print "Epoch:" + epoch_nmb + " | Learning Rate: 0.1(train acc, test acc) " + str(train_acc_3) + ", " + str(test_acc_3) + "      |      " + "Learning Rate: 0.01(train acc, test acc) " + str(train_acc_4) + ", " + str(test_acc_4)
        print >> f, "Epoch:" + epoch_nmb + " | Learning Rate: 0.1(train acc, test acc) " + str(train_acc_3) + ", " + str(test_acc_3) + "      |      " + "Learning Rate: 0.01(train acc, test acc) " + str(train_acc_4) + ", " + str(test_acc_4)


#3. Step Function
print "\nStep Function, iter per epoch = " + str(iter_per_epoch)
print >> f, "\nStep Function, iter per epoch = " + str(iter_per_epoch)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

# 기울기 계산
    grad_3 = network_5.gradient_step(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network_5.params[key] -= learning_rate_1 * grad_3[key]
        network_6.params[key] -= learning_rate_2 * grad_3[key]

    # 학습 경과 기록
    loss_5 = network_5.loss(x_batch, t_batch)
    step_train_loss_list_1.append(loss_5)

    loss_6 = network_6.loss(x_batch, t_batch)
    step_train_loss_list_2.append(loss_6)


    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        epoch_nmb = epoch_num(i)

        #Learning Rate 1
        train_acc_5 = network_5.accuracy(x_train, t_train)
        test_acc_5 = network_5.accuracy(x_test, t_test)
        step_train_acc_list_1.append(train_acc_5)
        step_test_acc_list_1.append(test_acc_5)

        #Learning Rate 2
        train_acc_6 = network_6.accuracy(x_train, t_train)
        test_acc_6 = network_6.accuracy(x_test, t_test)
        step_train_acc_list_2.append(train_acc_6)
        step_test_acc_list_2.append(test_acc_6)

        print "Epoch:" + epoch_nmb + " | Learning Rate: 0.1(train acc, test acc) " + str(train_acc_5) + ", " + str(test_acc_5) + "      |      " + "Learning Rate: 0.01(train acc, test acc) " + str(train_acc_6) + ", " + str(test_acc_6)
        print >> f, "Epoch:" + epoch_nmb + " | Learning Rate: 0.1(train acc, test acc) " + str(train_acc_5) + ", " + str(test_acc_5) + "      |      " + "Learning Rate: 0.01(train acc, test acc) " + str(train_acc_6) + ", " + str(test_acc_6)

print '[ Finished in ', datetime.now()-start_time, ']'
print >> f, '\n[ Finished in ', datetime.now()-start_time, ']'
f.close()


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(relu_train_acc_list_1))

plt.plot(x, sig_train_acc_list_1, label='Sigmoid: train acc(LR: 0.1)')
plt.plot(x, sig_test_acc_list_1, label='Sigmoid: test acc(LR: 0.1)', linestyle='--')

plt.plot(x, sig_train_acc_list_2, label='Sigmoid: train acc(LR: 0.01)')
plt.plot(x, sig_test_acc_list_2, label='Sigmoid: test acc(LR: 0.01)', linestyle='--')


plt.plot(x, relu_train_acc_list_1, label='Relu: train acc(LR: 0.1)')
plt.plot(x, relu_test_acc_list_1, label='Relu: test acc(LR: 0.1)', linestyle='--')

plt.plot(x, relu_train_acc_list_2, label='Relu: train acc(LR: 0.01)')
plt.plot(x, relu_test_acc_list_2, label='Relu: test acc(LR: 0.01)', linestyle='--')

plt.plot(x, step_train_acc_list_1, label='Step: train acc(LR: 0.1)')
plt.plot(x, step_test_acc_list_1, label='Step: test acc(LR: 0.1)', linestyle='--')

plt.plot(x, step_train_acc_list_2, label='Step: train acc(LR: 0.01)')
plt.plot(x, step_test_acc_list_2, label='Step: test acc(LR: 0.01)', linestyle='--')

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.suptitle('Sigmoid, Relu, Step Functions; Learning Rate: 0.1 & 0.01')
plt.show()

