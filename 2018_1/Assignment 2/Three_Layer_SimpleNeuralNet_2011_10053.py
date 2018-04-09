#Assignment 2
#Simple Neural Network
#2011-10053
#Noh Ha Kyung
#Submitted: 04 April 2018
#Python v3.6.4

import sys, os
import time
import csv
import numpy as np

# Timer
start_time = time.clock()

# Load a CSV file: last column is an answer, returns two np arrays (0:last -1) & (last column)
def import_csv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")
    data_columns = 12   #last column in row with data

    x_train_list = list()
    y_train_list = list()

    for row in reader:
        x_train_list.append(row[:data_columns])
        y_train_list.append(row[-1])

    x_train = np.asarray(x_train_list, dtype=np.dtype(float))
    y_train = np.asarray(y_train_list, dtype=np.int8)

    return x_train, y_train

def one_hot_encoding_all(x):    #convert array of numbers
    encoded_data = np.zeros((len(x), 10))
    for i in range(len(x)):
        encoded_data[i][x[i]] = 1
    return encoded_data

def one_hot_encoding(x):    #convert number
    encoded_data = np.zeros((1, 10))
    encoded_data[0][x] = 1
    return encoded_data

def init_network(input_size, hidden_size, output_size):
    network = {}
    weight_int_std = 0.01
    network['W1'] = weight_int_std * np.random.randn(input_size, hidden_size)
    network['b1'] = np.zeros(hidden_size)
    network['W2'] = weight_int_std * np.random.randn(hidden_size, hidden_size)
    network['b2'] = np.zeros(hidden_size)
    network['W3'] = weight_int_std * np.random.randn(hidden_size, output_size)
    network['b3'] = np.zeros(output_size)

    return network

def predict(network, x):    #network: NN, x: source data
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def accuracy(network, train_data, answer_data):
     accuracy_cnt = 0

     for i in range(len(train_data)):
         y = predict(network, train_data[i])    #넘파이 배열 (0~9 까지 숫자들의 확률)
         p = np.argmax(y)
         predict_encoded = one_hot_encoding(p)
         if (predict_encoded == answer_data[i]).all():
             accuracy_cnt += 1

     accuracy = float(accuracy_cnt) / len(train_data)

     return accuracy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) #overflow solution
    return np.exp(x) / np.sum(np.exp(x))

#===========================main===========================

filename = 'SimpleNetData.csv'

x_train, y_train = import_csv(filename)

answer_data = one_hot_encoding_all(y_train)
network = init_network(len(x_train[1]), 100, 10)

accuracy = accuracy(network, x_train, answer_data)

print ("Accuracy: ", accuracy , "(", str("{0:.3f}".format(accuracy * 100)), "% )")
print ("Processing time: ", "{0:.2f}".format(time.clock() - start_time), " seconds")
