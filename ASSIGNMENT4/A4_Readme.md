# Assignment 4

Due date: 22 May 2017

train_neuralnet 구현하기

4장의 04-train_neuralnet.py (교재 페이지 141)는 MNIST를 미니배치를 이용하여 Gradient Descent Method로 훈련데이터와 시험데이터의 손실함수 값의 차이를 보여주는 프로그램이다.

이를 근거로 하여, 학습률을 0.1로 했을 때와 0.01로 했을 때, 그리고 각각의 학습률에 따라 step function, sigmoid, Relu에 따라 cross_entropy_error에 의한 손실함수 값의 차이를 보여주는 프로그램을 작성하라.

----------------------------------------------


**주의: 이를 위해서는 이미 구현되어 있는 two_layer_net 과 common 디렉토리의 functions에 있는 activation 함수 구현들, 그리고 gradient에 구현되어 있는 numeriacl_gradient를 import하고 그대로 사용할 수 있으나 그렇게 하지 말고, 필요한 모든 클래스와 함수를 한 프로그램에 다 모아 놓고 적절히 변형하여 구현하도록 한다. 이 과제의 목표는 지금까지 배운 neuralnetwork를 정리해 보기 위한 것임. 프로그램은 각 학습비율별로 각각의 activation 함수에 따른 훈련데이터와 시험데이터의 Accuracy를 출력. 그래프는 출력할 필요가 없음**

Filename: 2011_10053_hw4.py

Test data: MNIST

