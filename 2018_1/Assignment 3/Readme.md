# Assignment 3: 

## Simple Neural Network - gradient 적용 업그레이드
  
Assignment2 에서 구현한 간단한  Neural Network 모델에 Gradient Descent를 적용하여 성능 향상을 시킨 버전 구현 
 
   **개요**:
   
1. 이번에 사용할 자료는 과제 2에서 좀 더 발전 시킨 SimpleNetData2.csv 로 10000*201 배열로 되어 있다. 이 행렬의 마지막 열은 이전과 마찬가지로 0-9 사이의 정답 숫자로 되어 있다. .

2. 과제2를 발전시켜 교재 137페이지에 있는 class TwoLayerNet을 그대로 사용하라. 이 class를 위한 관련 모듈과 함수는 import 하지 말고 한 프로그램에 같이 명시하라 (softmax, gradient 등)  

3. load_csv 모듈은 이제 SimpleNetData2.csv 자료를 80%는 training으로 20%는 test자료로 랜덤하게 분할하도록 한다.  

4. filename: Assginment3_yourhakbun.py

*(교재 100page, 또는 Deep Learning From Scratch source codes 의 ch04/two_layer_net.py)*