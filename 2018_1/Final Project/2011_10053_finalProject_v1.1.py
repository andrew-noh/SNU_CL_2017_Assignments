# coding: utf-8
#2011-10053
#NOH HA KYUNG
#Final Project
#Korean news articles categorization
#Python v3.6.4
# Submitted: 11 June 2018
#Environment: AWS r4.8xlarge instance - ubuntu


# Import dependencies
import sys
from os import listdir
import time
import re
import numpy as np
import codecs
import collections
import hanja
from konlpy.tag import Twitter
from konlpy.tag import Komoran
from konlpy.tag import Kkma
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import csv
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(5)
np.set_printoptions(suppress=True)


# Create konlpy tags
twitter = Twitter()
komoran = Komoran()
kkma = Kkma()

#===================================================FUNCTIONS===================================================

def one_hot_encoding(x, output_size):
    encoded_data = np.zeros((len(x), output_size))
    for i in range(len(x)):
        encoded_data[i][x[i]] = 1
    return encoded_data

# Generate labels set for training and test data
# labels_count: how many categories?
# set_size: how many items are allocated for training or test data set
def generate_label_set(labels_count, set_size, one_hot):
    label_set = list()
    for i in range(labels_count):
        for x in range (set_size):
            label_set.append(i)
    if one_hot == True:
        one_hot_data = one_hot_encoding(label_set, labels_count)
        return one_hot_data
    else:
        return np.array(label_set) # No one-hot-encoding

# Preprocess raw korean article text file. Input- file name, konlpy package name, minimum syllables, remove duplicates T/F
def kr_text_preprocess(text, packageName, minSyllables, shrink):
    raw_file = codecs.open(text, "r", encoding='utf-8', errors='ignore').read()
    remove_spChar = re.sub(r'[…·]', ' ', raw_file) # Remove special characters
    remove_par = re.sub(r'\([^)]*\)', "", remove_spChar) # Remove contents inside parenthesis
    hanja2kr = hanja.translate(remove_par, 'substitution')  # Translate hanja
    if packageName == 'twitter':
        text_nouns = twitter.nouns(hanja2kr)
    elif packageName == 'komoran':
        text_nouns = komoran.nouns(hanja2kr)
    elif packageName == 'kkma':
        text_nouns = kkma.nouns(hanja2kr)
    else:
        text_nouns = ["Not a correct package name!"]
    remove_short = [x for x in text_nouns if len(x) > (minSyllables - 1)]   #Ignore words that are shorter than x syllables
    if shrink == True:
        no_duplicates = list(set(remove_short))
        return no_duplicates
    else:
        return remove_short

# Preprocessing string array for prediction
def kr_string_preprocess(article_string, packageName, minSyllables, shrink):
    raw_file = article_string
    remove_spChar = re.sub(r'[…·]', ' ', raw_file) # Remove special characters
    remove_par = re.sub(r'\([^)]*\)', "", remove_spChar) # Remove contents inside parenthesis
    hanja2kr = hanja.translate(remove_par, 'substitution')  # Translate hanja
    if packageName == 'twitter':
        text_nouns = twitter.nouns(hanja2kr)
    elif packageName == 'komoran':
        text_nouns = komoran.nouns(hanja2kr)
    elif packageName == 'kkma':
        text_nouns = kkma.nouns(hanja2kr)
    else:
        text_nouns = ["Not a correct package name!"]
    remove_short = [x for x in text_nouns if len(x) > (minSyllables - 1)]   #Ignore words that are shorter than x syllables
    if shrink == True:
        no_duplicates = list(set(remove_short))
        return no_duplicates
    else:
        return remove_short

# Preprocess all files in a given directory and create word, articles dictionary
def create_dictionary_and_backup_articles(directory, category_folders_count, minOccurance):
    articles_backup = {}    # Backup pre-processed articles made while creating dictionary
    final_dictionary = {}   # Final dictionary with all unique words in newsData folder
    total_words = list()    # Temporary list with all words (duplicates exist)

    # Go through all files in subfolders of directory, pre-process text files, save short version article, create whole dictionary
    for i in range(category_folders_count):
        for filename in listdir(directory + '/' + str(i)):
            file_index = filename[:4]
            file_address = directory + '/' + str(i) + '/' + filename
            processed_data = kr_text_preprocess(file_address, module_tag, 2, True)
            for words in processed_data:
                total_words.append(words)
            articles_backup[file_index] = processed_data    # Backup preprocessed article (dictionary; '0000 - index', '[nouns list]')
        print('Processing articles in category', article_categories[i], 'complete!')

    wordCounter = collections.Counter(total_words)
    tokens = [k for k,c in wordCounter.items() if c >= minOccurance]    # Leave only words with occurance > min occurance
    all_words_count = len(tokens)
    # Convert enum to dictionary
    numbered_dict = enumerate(tokens)
    for i, word in numbered_dict:
        final_dictionary[word] = i + 1

    return articles_backup, final_dictionary, all_words_count

# Encode article
def article_encoder(article, vocab_dict, max_count):
    article_encoded = list()

    for word in range(1, len(article)):
        word_index = vocab_dict.get(article[word])    # If word is not in dictionary, skip
        if word_index == None:
            continue
        else:
            normalize = word_index / max_count
            article_encoded.append(normalize)

    return article_encoded

# Creating np ndarray data (indexes[1~159]&[160~199], categories count(8), dictionary with tokenized articles, maximum length)
def create_np_dataset(index_list, categories, articles_dict, words_dic, max_vocab_count, max_length):
    python_list = []
    for category in range(categories):
        for i in index_list:
            data_index = str(category) + '{:03d}'.format(i)
            encoded_text = article_encoder(articles_dict[data_index], words_dic, max_vocab_count)
            if len(encoded_text) != max_length:
                difference = max_length - len(encoded_text)
                padding = np.zeros(difference).tolist() # Padding with zero, post
                padded_text = encoded_text + padding
                python_list.append(padded_text)
            else:
                python_list.append(encoded_text)

    output_data = np.array(python_list)
    return output_data

def create_dataset_nonEncoded(index_list, categories, articles_dict, max_length):
    python_list = []
    for category in range(categories):
        for i in index_list:
            data_index = str(category) + '{:03d}'.format(i)
            article_length = len(articles_dict[data_index])
            if article_length != max_length:
                difference = max_length - article_length
                padding = np.zeros(difference).tolist() # Padding with zero, post
                padded_text = articles_dict[data_index] + padding
                python_list.append(padded_text)
            else:
                python_list.append(articles_dict[data_index])
    return np.array(python_list)


def create_prediction_data(article, dictionary, max_count, max_length):
    encoded_text = article_encoder_prediction(article, dictionary, max_count)
    padded_text = list()
    if len(encoded_text) != max_length:
        difference = max_length - len(encoded_text)
        padding = np.zeros(difference).tolist() # Padding with zero, post
        padded_text = encoded_text + padding
    output_data = np.array(padded_text)
    return output_data

# Prediction data
def article_encoder_prediction(article, vocab_dictionary, max_count):
    article_encoded = list()
    for word in range(1, len(article)):
        word_index = vocab_dictionary.get(article[word])
        if word_index == None:
            continue
        else:
            normalize = word_index / max_count
            article_encoded.append(normalize)

    return article_encoded

# Plot Loss History
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

# Show image
def display_image_in_actual_size(im_path):

    dpi = 80
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

# Import CSV as dict
def import_csv(filename):
    vocab_dict = {}
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            vocab_dict[row[0]] = int(row[1])

    return vocab_dict

# Export CSV
def csv_exporter(filename, dictionary):
    name = filename + '.csv'
    with open(name, 'w') as f:
        w = csv.writer(f)
        w.writerows(dictionary.items())

article_categories = {0:'정치', 1:'경제', 2:'사회', 3:'생활/문화', 4:'세계', 5:'기술/IT', 6:'연예', 7:'스포츠'}

#===================================================VARIABLES===================================================
module_tag = 'komoran'
categories_count = 8
directory = 'newsData'
num_of_articles_in_cat = 200
split = 0.8 # Split rate
ignore_less = 2 # Ignore words with occurence less than n
encode_one_hot = True
export_csv_file = True
modelfile = "model.png"

#===================================================PREPROCESSING===================================================
# Pre-process text and generate vocab

# Timer
start_time = time.clock()
print("Pre-processing articles and creating vocabulary set. It can take a while depending on number of files. \nPlease be patient...")
articles, words_dictioinary, word_count = create_dictionary_and_backup_articles(directory, categories_count, ignore_less)

# Export vocab as csv
if export_csv_file == True:
    csv_exporter('vocab', words_dictioinary)

print ('Dictionary creation finished!')
if export_csv_file == True:
    print ('CSV file saved!')
print ('Total words in dictionary: ', word_count)
print ("Processing time: ", "{0:.2f}".format(time.clock() - start_time), " seconds")

#===================================================TRAINING===================================================

# Generate t_train and t_test
t_train = generate_label_set(categories_count, int(num_of_articles_in_cat * split), encode_one_hot)
t_test = generate_label_set(categories_count, int(num_of_articles_in_cat - num_of_articles_in_cat * split), encode_one_hot)


# Create training and test set data (File name indexes for training and test data)
training_indexes = np.arange(num_of_articles_in_cat * split).astype(np.int32).tolist()
test_indexes = np.arange(num_of_articles_in_cat * split, num_of_articles_in_cat).astype(np.int32).tolist()


# Get the length of the longest article
article_lengths = []
for k, v in articles.items():
    length = len(v)
    article_lengths.append(length)
max_length = max(article_lengths)


# Create train data
#x_train_nonEncoded = create_dataset_nonEncoded(training_indexes, categories_count, articles, max_length)
#x_test_nonEncoded = create_dataset_nonEncoded(test_indexes, categories_count, articles, max_length)

x_train = create_np_dataset(training_indexes, categories_count, articles, words_dictioinary, word_count, max_length)
x_test = create_np_dataset(test_indexes, categories_count, articles, words_dictioinary, word_count, max_length)

x_train_3d = np.expand_dims(x_train, axis=2)
x_test_3d = np.expand_dims(x_test, axis=2)

print('x_train, x_test shape:', x_train_3d.shape, x_test_3d.shape)
print('t_train, t_test shape:', t_train.shape, t_test.shape)
print('\n')
print('Sample x_train data: \n', x_train[0])

# Neural Net
print('\nLets start training network\n')

# Epochs
epochs_total = 700

# Model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(max_length, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(categories_count, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())
#plot_model(model, to_file=modelfile, show_shapes=True, show_layer_names=True)

history = model.fit(x_train_3d, t_train, batch_size=16, epochs=epochs_total)

# Show model graph
display_image_in_actual_size(modelfile)

# 학습과정 살펴보기
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(x_test_3d, t_test, batch_size=16)
print("Accuracy: %.2f%%" % (score[1]*100))



# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# 새로운 기사를 모델로 예측

#sample_prediction_article = "SK텔레콤(017670)(대표이사 사장 박정호)이 11일 모바일 방송국 메이크어스에 100억원 규모의 투자를 단행했다. 메이크어스는 음악과 음식, 여행, 뷰티 등 다양한 주제의 모바일 동영상 콘텐츠를 제작하고 페이스북 · 유튜브 채널을 운영하는 스타트업이다. 눈에 띄는 점은 자체 플랫폼을 갖지 않고 영상을 제작해 유통한다는 점이다. 10~30대의 눈높이에 맞는 감각적이고 재미있는 영상들을 내놓으며 밀레니얼 세대의 마음을 사로잡았다. 2017년 기준 메이크어스의 페이스북 · 유튜브 · 인스타그램 구독자는 3360만 명, 포스팅 조회수는 37억 회에 이른다.  특히 메이크어스의 음악채널인 딩고 뮤직은 세로가 긴 화면으로 구성된 모바일 특화 뮤직비디오를 선보이는 등 대표적인 모바일 음악 채널로 자리매김하고 있다. SK텔레콤은 하반기 출시될 새 음악 스트리밍 플랫폼의 경쟁력을 강화하는 한편, 음악 생태계를 활성화하기 위해 메이크어스 투자를 결정했다.  양사는 음악 프로그램 공동제작 등 다양한 협력 방안을 검토 중이다. 메이크어스의 주주로는 KTB네트워크, 캡스톤, 옐로 모바일 등이 있다. 연초 SK텔레콤은 AI·블록체인 등 New ICT기술을 도입한 음악 사업 진출을 선언했다. 개인 취향에 맞는 음악 추천 · 보는 음악 콘텐츠 확대 등 차별화 된 서비스 제공과 창작자 친화적인 음악 산업 생태계 조성을 천명한 바 있다. SK텔레콤은 메이크어스는 모바일 미디어를 소비하는 젊은 세대의 특성을 잘 이해하고 있는 스타트업이라며 아티스트들이 모바일 환경에서 팬들과 더 가깝게 연결될 수 있는 방법을 함께 모색해갈 것이라고 밝혔다."

prediction_preprocess = kr_string_preprocess(sample_prediction_article, module_tag, 2, True)
#prediction_preprocess = kr_text_preprocess('test_article.txt', module_tag, 2, True)
prediction_data = create_prediction_data(prediction_preprocess, words_dictioinary, word_count, max_length)
prediction_data_reshape = np.reshape(prediction_data, [1, max_length])
prediction_data_3d = np.expand_dims(prediction_data_reshape, axis=2)

print("Prediction demo\n")
print(prediction_preprocess)
print(prediction_data_3d.shape)


prediction_test = model.predict(prediction_data_3d)
prediction_index = np.argmax(prediction_test)
print("Predicted category: ", article_categories[prediction_index])
print("Accuracy: ", "{:.2%}".format(prediction_test[0][prediction_index]), '\n\n')

for i in range(categories_count):
    if i != prediction_index:
        print("Category: ", article_categories[i], "- ", "{:.2%}".format(prediction_test[0][i]))
