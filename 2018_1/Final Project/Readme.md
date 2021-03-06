# Final Project: 한국어 뉴스 기사 분류: 

### 목표: 지금까지 배운 Neural Network 및 keras를 기초로 하여, 한국어 데이터 처리, 네트웍 학습, 분류에 적합한 모델 설정 및 학습 등을   종합적으로 활용하는 프로젝트 수행

 
   **개요**:
   
- newsData.zip은 뉴스 기사를 모은 것으로 0에서 7까지 디렉토리로 이루어져 있다. 각 디렉토리에는 디렉토리 이름으로 시작하여 200개의 한국어 뉴스 기사들이 있다.
- 정치(0), 경제(1), 사회(2), 생활/문화(3), 세계(4), 기술/IT(5), 연예(6), 스포츠(7) 로 분류되어 있다.
- 각 디렉토리의 200개의 파일 중에서 160개는 학습자료로 나머지 40개는 시험자료로 삼는다. 시험자료는 각 디렉토리의 x160 - x199로 시작하는 파일로 한다.
- 한국어 데이터를 처리하기 위해서는 한국어 형태소 분석을 할 필요가 있고 형태소 분석을 위해서는 konlpy 사이트를 참조하여 형태소 분석 모듈을 설치. 실제 사용 예는 이 파일 참조
- 한국어 데이터를 어떻게 처리할 지 각자 최적의 방법을 생각해 볼 것. 특정 품사만 남기고 나머지 품사는 배제 하는 등.
- Word Embedding이 텍스트 처리에 아주 유용하게 활용될 수 있다. Keras에서 제공하는 embedding 외에 신문기사를 gensim등을 이용하여 다양한 패러미터로 학습하여 외부 embedding을 사용했을 때 성능이 향상될 수 있는지  살펴 볼 수 있다.
- Keras를 활용한 How to Develop Word Embedding Model for Predicting Movie Review Sentiment 사이트를 활용하여 데이터 준비 및 모델 구축 등을 참고할 수 있다. 이 프로젝트를 위하여 이 사이트에서처럼 별도의 vocab을 만들 경우 이를 만드는 모듈도 제시해야 하며 각자의 vocab 파일도 같이 제출해야 한다.
- 6월 11일 수업시간에 조별로 ppt를 만들어 구현한 프로그램의 특징 및 방법론 그리고 성능을 발표하도록 한다.
- 개인적으로도 할 수 있고 팀별로도 할 수 있으나 팀은 3명을 넘어서는 안된다.