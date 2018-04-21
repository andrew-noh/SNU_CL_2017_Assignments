# Assignment 4: 

## Simple Neutral Network을 이용한 IMDB movie review 데이터 감정분석
  
목표: 지금까지 배운 Simple Forward Neural Network을 영화평 리뷰의 감정분석(postive, negative)에 활용할 수 있도록 모델 구축
 
   **개요**:
   
1. Keras를 활용한 [How to Develop Word Embedding Model for Predicting Movie Review Sentiment](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/) 사이트를 활용하여 keras 대신 simple neural network을 이용하여 movie review의 감정을 분석하는 모델을 개발

2. 대부분의 코드는 [How to Develop Word Embedding Model for Predicting Movie Review Sentiment](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/) 사이트에서 제공하는 것을 그대로 사용할 수 있으나  모델을 구축하고 이를 위한 텍스트를 integer로 변환하는 texts_to_sequence, 와 문서의 길이를 통일하기 위한 pad_sequences는 직접 구현

3. 영화평은 각각 1,000개의 긍정(positive), 부정(negative) 문서로 되어 있고 각각 900씩 총 1, 800문장을 training set으로 나머지 각 100 문장을 test set으로 사용한다. 

4. 기본 방법은 긍정문서에 나타나는 단어와 부정 문서에 나타나는 단어를 학습하여 문서분류를 행하는 것으로 어떤 word embedding도 사용하지 않는다.

5. 문서를 training, test set으로 분리하는 코드에서 요구되는 25,767 여개의 어휘를 추출하는 해당 사이트의 2.Data Preparation에 있으나 이를 이미 정리해놓은 vocab.txt는 여기서 제공. 따라서 3.Train Embedding Layer부터 참조할 수 있음
6. class TwoLayerNet과 관련된 모든 클래스와 함수를 포함할 것. 나중에는 MutliLayerNet을 활용하여 성능을 더 개선시킬 것임

7. Assignment4_yourhakbun.py