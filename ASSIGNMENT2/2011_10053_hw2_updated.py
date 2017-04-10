#2011-10053 Noh Hakyung
#Natural Language Processing with Python
#Assignment 2
#Sentence probability calculation

#Sentence: "I hate to say that this class is really boring."

import collections
import nltk
import nltk.data
import re

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
file_ext = ['BROWN1_A1.txt', 'BROWN1_B1.txt', 'BROWN1_C1.txt', 'BROWN1_D1.txt', 'BROWN1_E1.txt', 'BROWN1_F1.txt', 'BROWN1_G1.txt', 'BROWN1_H1.txt', 'BROWN1_J1.txt', 'BROWN1_K1.txt', 'BROWN1_L1.txt', 'BROWN1_M1.txt', 'BROWN1_N1.txt', 'BROWN1_P1.txt', 'BROWN1_R1.txt']


#Combine all text files into one corpus
with open('1_BROWN1_COMBINED.txt', 'w') as outfile:
    for fname in file_ext:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

#Tokenize sentence
text_file_raw = open('1_BROWN1_COMBINED.txt')
text = text_file_raw.read()
clean_text = text.replace('\r\n', '')
sentence_tokenizer = sent_detector.tokenize(clean_text)

with open('2_BROWN1_COMBINED_TOKENIZED.txt', 'w') as outfile:
	for sentence in sentence_tokenizer:
		outfile.write(' <s> ' + sentence + ' </s> ')


#Tokenize words
text_file_all = open('2_BROWN1_COMBINED_TOKENIZED.txt')
raw_part1 = text_file_all.read().lower()
raw_part2 = re.sub(r'\.|\?|\!', ' END_OF_SENTENCE', raw_part1)
raw = raw_part2.replace('<s>', 'START_OF_SENTENCE')
word_tokens = nltk.word_tokenize(raw)

#Generate bigrams
bgs = nltk.bigrams(word_tokens)

#Create file with all bigrams
bigram_file = open("3_BIGRAM_OUTPUT_ALL.txt", "w")

#Compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(bgs)
for k,v in fdist.items():
		bigram_file.write(str(k)+ ' ' + str(v) + '\n')

bigram_file.close()

#unigram count cidcionary
unigram_count = collections.Counter(word_tokens)
unigram_dict = dict(unigram_count)

#bigram count dictionary
bigram_dict = dict(fdist)

#unigram probability
# unigram: 'word': 1
# bigram: ('word_a', 'word_b'): 1
# probability = bigram occurence / unigram count

sentence_sample = ['START_OF_SENTENCE', 'i', 'hate', 'to', 'say', 'that', 'this', 'class', 'is', 'really', 'long', 'END_OF_SENTENCE']
word_num = 0
p_list = []

def bigram_probability(a):
	bigram_lookup = float(bigram_dict[sentence_sample[a], sentence_sample[a+1]])
	unigram_lookup = float(unigram_dict[sentence_sample[a]])
	prob = bigram_lookup / unigram_lookup
	print 'P(', sentence_sample[a+1], '|', sentence_sample[a], ') = ', bigram_lookup, '/', unigram_lookup, ' = ', prob
	return prob

for i in sentence_sample:
	while word_num < len(sentence_sample)-1:
		probability_data = bigram_probability(word_num)
		p_list.append(probability_data)
		word_num += 1

final_prob = reduce(lambda x, y: x*y, p_list)

print '\nProbability of the sentence: "', ' '.join(sentence_sample), '" is: ', final_prob


# P(hate | i) = count(i hate) / count(i)
