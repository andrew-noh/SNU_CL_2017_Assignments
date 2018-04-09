# Assignment 2: 

## Simple Neural Network 구현
  
 MNIST Dataset을 응용한 간단한 Neural Network 구현
 
   **개요**:
   
1. 교재 100페이지에 있는 ch03/neuralnet_mnist.py를 이용하여 다음의 간단한 three layer SimpleNeuralNetYourID.py를 완성하라.

2. 샘플데이터 SimpleNetData.csv는 5000*13 배열로 임의의 값이 채워져 있는 파일이다. 이 파일의 마지막 열(13열)은 12열에 대한 0-9 사이의 정답 숫자로 되어 있다. 이 파일을 읽어서  x_train, y_train으로 불러들인다. x_train은 (5000, 12) 형태의 배열로 y_train은 (5000,)의 정답으로 된 배열이다. 

3. 이 simple forward network은 손실함수도 구현되지 않은 기본 단계라 성능이 낮을 것임. 앞으로 계속 발전시켜 성능을 향상 시키는 것이 목표. 우선 제시된 함수를 완성하여 그 정확도를 출력하는 프로그램을 완성하라