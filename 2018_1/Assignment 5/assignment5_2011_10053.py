## -*- coding: utf-8 -*-
#Assignment 5
#Vanilla Keras Simple Neutral Network
#2011-10053
#Noh Ha Kyung
#Submitted: 15 May 2018
#Python v3.6.4
#Keras 2.1.6

# Import models
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy

# fix random seed for reproducibility
numpy.random.seed(5)

# load pima indians dataset
dataset = numpy.loadtxt("SimpleNetData2.csv", delimiter=",")

# Split train and test data
split = 0.8
train_range = int(dataset.shape[0] * split)
data_range = int(dataset.shape[1]) - 1

x_train = dataset[:train_range,0:data_range]
y_train = dataset[:train_range,data_range]
x_test = dataset[train_range:,0:data_range]
y_test = dataset[train_range:,data_range]

# Neural Net
model = Sequential()
model.add(Dense(350, input_dim=200, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(200, activation='tanh'))
model.add(Dense(10, activation='softmax'))
#(Adam) All tanh: 82.70%, all relu: 84.40%, relu&tanh: 83.70%, tanh&relu: 82.75%

# Training Settings
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #acc: 82.75%
#model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) #acc: 83.55%
model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy']) #acc: 83.85%

#Model Training
model.fit(x_train, y_train, epochs=150, batch_size=80)

# Model Evaluation
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

#Final acc: 84.00%
