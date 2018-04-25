# Assignment 3

Due date: 10 March 2017

교차엔트로피(Cross Entropy)과제

BROWN1_D1 코퍼스를 training 코퍼스로 삼아, 영어 글자-알파벳(대소문자 구별없이)과 스페이스만-의 유니그램, 
바이그램에 기반한 글자 확률에 따른 엔트로피를 구하고 이를 BROWN1-L1 테스트 코퍼스에서 유니그램, 바이그램 기반의 언어모델을 설정하여 
교차엔트로피를 구하여 다음의 표를 출력하는 프로그램을 작성하라.


**Unigram: Entropy(), CrossEntropy(), Difference()**


Training text:


Test text: 

**Bigram: Entropy(), CrossEntropy(), Difference()**


Training text:


Test text: 

----------------------------------------------


*힌트: BROWN1_D1 trainining 코퍼스에서 유니그램의 경우 각 글자별 빈도를 구한 후 이를 전체 글자 수(알파벳+스페이스 수)로 나누면 이것이 각 글자의 확률이 된다.  
이 글자의 확률을 엔트로피 공식에 넣어 계산하면 유니그램 글자 기반의 엔트로피가 된다. 따라서 BROWN1_D1의 경우는 엔트로피와 크로스 엔트로피가 같고 그 차이도 0이 된다.
이 모델을 BROWN1_L1 테스트 코퍼스에서 테스트 하기 위해 마찬가지로 각 글자별 확률을 구하고 이를 교차 엔트로피 공식에 따라 구하면 되는데, 
이 경우 P(x)는 이 test 코퍼스의 각 글자별 확률이고 모델의 확률인 log p(m)은 training 코퍼스인 BROWN1_D1코퍼스에서 구해진 각 글자의 확률이다. 
각 글자별로 이를 다 곱해서 더 하면 교차엔트로피가 구해진다. 엔트로피와 교차엔트로피의 차이는 H(P,m) - H(p)로 그 차이가 작을수록 더 좋은 모델이 된다.
이를 각각 유니그램, 바이그램으로 나누어서 계산해 보라.*

Filename: 2011_10053_hw3.py