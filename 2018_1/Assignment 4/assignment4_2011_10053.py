## -*- coding: utf-8 -*-
#Assignment 4
#Simple Neutral Network을 이용한 IMDB movie review 데이터 감정분석
#2011-10053
#Noh Ha Kyung
#Submitted: 21 April 2018
#Python v3.6.4

import sys
from os import listdir
import time
import numpy as np
from string import punctuation
from itertools import repeat
import matplotlib.pyplot as plt

# Timer
start_time = time.clock()

#===========================func===========================
def load_doc(filename):
    # open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

def texts_to_sequence(train_docs, vocab_dict, max_count):
    original_list = []
    for review in train_docs:
        review_words = review.split()
        for i in range(len(review_words)):
            index = vocab_dict[review_words[i]] #Search word index in vocab dictionary
            normalize = index / max_count #Normalize index by dividing to number of all vocabulary
            review_words[i] = normalize
        original_list.append(review_words)
    return original_list

def pad_sequences(sequences, number, width):
    padd_seq_out = sequences
    for review_seq in padd_seq_out:
        review_seq.extend(repeat(number, width - len(review_seq)))
    return padd_seq_out

def one_hot_encoding(x, output_size):
    encoded_data = np.zeros((len(x), output_size))
    for i in range(len(x)):
        encoded_data[i][x[i]] = 1
    return encoded_data
    #1,0 = negative
    #0, 1 = positive

#===========================two_layer_net===========================
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
#========================main========================

#Load pre-processed vacab text file and convert to indexed dictionary
vocab_filename = 'vocabs.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
vocab_count = len(vocab) + 1 #Count of all words in vocab file
vocab_dict = dict((c, i) for i, c in enumerate(vocab, start=1))

#========train_text========
# load all training reviews (x_train)
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

#Encode documents
encoded_docs_train = texts_to_sequence(train_docs, vocab_dict, vocab_count)
max_length_train = max([len(s) for s in encoded_docs_train]) #Longest review word length in documents

#Define training labels (0~900 = negative; 900~1800 = positive)
t_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])
t_train = one_hot_encoding(t_train, 2)

#========test_text========
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs

#Encode documents
encoded_docs_test = texts_to_sequence(test_docs, vocab_dict, vocab_count)
max_length_test = max([len(x) for x in encoded_docs_test]) #Longest review word length in documents

#Define test labels (0~100 = negative; 100~200 = positive)
t_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])
t_test = one_hot_encoding(t_test, 2)

#Padding data
max_length = max([max_length_train, max_length_test]) #Find the longest review from train and text db
x_train = pad_sequences(encoded_docs_train, 0, max_length)
x_test = pad_sequences(encoded_docs_test, 0, max_length)

#Convert to numpy data
x_train = np.asarray(x_train, dtype=np.dtype(float))
x_test = np.asarray(x_test, dtype=np.int8)


#Two Layer Net
network = TwoLayerNet(input_size=max_length, hidden_size=50, output_size=2)

#Params
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

#1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

#Training
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Epoch # " + str(i / iter_per_epoch), ": train acc, test_acc | " + str("{0:.3f}".format(train_acc * 100)) + "%, " +str("{0:.3f}".format(test_acc * 100)) + "%")

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
#plt.savefig('Gradient_learning_graph.png')
plt.show()
