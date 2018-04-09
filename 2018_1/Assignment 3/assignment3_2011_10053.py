## -*- coding: utf-8 -*-
#Assignment 3
#Simple Forward Neural Network
#2011-10053
#Noh Ha Kyung
#Submitted: 10 April 2018
#Python v3.6.4

import sys, os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from random import seed
from random import randrange


# Timer
start_time = time.clock()

#===========================func===========================

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialise parameters
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
        grads['W1'] = numerical_gradient_master(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_master(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_master(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_master(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
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

# Load a CSV file: file name, split proportion, count of columns of data
def load_csv(filename, split, data_columns):
    raw = import_csv(filename)
    train_dataset, test_dataset = train_test_split(raw, split)

    x_train, t_train = split_answer(train_dataset, data_columns)
    x_test, t_test = split_answer(test_dataset, data_columns)

    return x_train, t_train, x_test, t_test

# Read initial file
def import_csv(filename):
    dataset = list()
    with open(filename, 'rU') as file:
    	csv_reader = reader(file)
    	for row in csv_reader:
    		if not row:
    			continue
    		dataset.append(row)
    return dataset

# Randomly split data to train set and test set
def train_test_split(dataset, split):
	train_set = list()
	train_size = split * len(dataset)
	test_set = list(dataset)
	while len(train_set) < train_size:
		index = randrange(len(test_set))
		train_set.append(test_set.pop(index))
	return train_set, test_set

# Split data to dataset npArray and tests npArray
def split_answer(dataset, last_data_column):
    dataset_list = list()
    answers_list = list()

    for cell in dataset:
        dataset_list.append(cell[:last_data_column])
        answers_list.append(cell[-1])

    dataset_array = np.asarray(dataset_list, dtype=np.dtype(float))
    answers_array = np.asarray(answers_list, dtype=np.int8)

    return dataset_array, answers_array

# Convert to One Hot Encoding
def one_hot_encoding(x):
    encoded_data = np.zeros((len(x), 10))
    for i in range(len(x)):
        encoded_data[i][x[i]] = 1
    return encoded_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) #overflow solution
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient_master(f, x):
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

#===========================main===========================

filename = 'SimpleNetData.csv'

split = 0.8 # Training and test sets proportion: 80% vs 20%
data_columns = 200 # Columns with data

x_train, t_train, x_test, t_test = load_csv(filename, split, data_columns)

t_train = one_hot_encoding(t_train)
t_test = one_hot_encoding(t_test)

network = TwoLayerNet(input_size=200, hidden_size=50, output_size=10)

#Parameters
iters_num = 10000  #Iteration
train_size = x_train.shape[0]
batch_size = 100   #Mini batch size
learning_rate = 0.1
iter_per_epoch = max(train_size / batch_size, 1) #Iterations per epoch

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    #Get a mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #Calculate gradient
    grad = network.gradient(x_batch, t_batch)

    # Update parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Record history
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:

        train_acc = network.accuracy(x_train, t_train)

        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)

        test_acc_list.append(test_acc)

        print ("="*50)
        print ("Epoch #" + str(i / iter_per_epoch))
        print ("Train accuracy: "  + str(train_acc))
        print ("Test  accuracy: " +  str(test_acc))
        print ("="*50)

print ("Processing time: ", "{0:.2f}".format(time.clock() - start_time), " seconds")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig('Gradient_learning_graph.png')
plt.show()
