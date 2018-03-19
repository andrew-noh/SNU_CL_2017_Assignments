#2011-10053
#NOH HA KYUNG
#Assignment 1
#'Sentence probability calculation'
#Submitted: 19 March, 2018
#Python v3.6.4


import re
import nltk
import time
import decimal
from nltk.tokenize import sent_tokenize
from collections import Counter


#Read training text, start timer
start_time = time.clock()
textFile = open('BROWN_A1.txt')
rawText = textFile.read()


#Test sentence
#"I think I will get the best score in the class."
test_sentence_raw = 'I think I will get the best score in the class.'
#testSentence = ['<s>', 'i', 'think', 'i', 'will', 'get', 'the', 'best', 'score', 'in', 'the', 'class', '</s>']


#==========================functions==========================

#CleanUpText: (raw_text => list of unigrams(clean up text))
def cleanUpText(raw):
    #remove new line character: \n
    raw.replace('\n', '')

    #tokenize sentences (NLTK)
    sent_tokens = sent_tokenize(raw)

    #clean up special characters, add sentence boundary symbols, lower case all characters
    for a in range(len(sent_tokens)):
        clean_char = re.sub(r'[.,?!:/"<>@#$%^*~_{}()+]', '', sent_tokens[a])
        sent_tokens[a] = '<s> ' + clean_char.lower() + ' </s> '

    #join sentece tokens into one string
    clean_text = ''.join(sent_tokens)

    #split text by words
    source_text = clean_text.split()

    return source_text

#Unigram freq counting
def unigramFreqDict(wordlist):
    #wordfreq = [wordlist.count(p) for p in wordlist]
    #return dict(zip(wordlist,wordfreq))
    wordfreq = Counter(wordlist)
    return wordfreq

#Bigram freq counting
def bigramFreqDict(bigrams):
  bigram_list = []
  for i in range(len(bigrams)-1):
      bigram_list.append((bigrams[i], bigrams[i+1]))
  bigram_dict = Counter(bigram_list)
  return bigram_dict

#Bigrams list
def bigramList(bigrams):
  bigram_list = []
  for i in range(len(bigrams)-1):
      bigram_list.append((bigrams[i], bigrams[i+1]))
  return bigram_list

#Multiply all numbers in list
def multiply(numbers):
    total = 1
    for x in numbers:
        total *= x
    return total

#Formatting float to string (source from: https://goo.gl/nX1QTJ)
ctx = decimal.Context()
ctx.prec = 20

def float_to_str(f):
    #Convert the given float to a string, without resorting to scientific notation
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


#==========================main==========================

#Training text
cleanText = cleanUpText(rawText)
unigram_dict = unigramFreqDict(cleanText)
bigram_dict = bigramFreqDict(cleanText)
bigrams_total = len(bigramList(cleanText))

#Test sentence
testSentence = cleanUpText(test_sentence_raw)
testSentenceBigram = bigramList(testSentence)


#MLE probability
#No smoothing
mle_bigram_prob_list_noSmooth = []

for z in range(len(testSentenceBigram)-1):
    bi_count = bigram_dict.get(testSentenceBigram[z])
    uni_count = unigram_dict.get(testSentenceBigram[z][0])
    prob_1 = bi_count / uni_count
    mle_bigram_prob_list_noSmooth.append(prob_1)

mle_probabiliy_noSmooth = multiply(mle_bigram_prob_list_noSmooth)


#With 'plus 1' smoothing
mle_bigram_prob_list_smoothOne = []

#v = count of types
v = 0

for x in range(len(testSentence)-1):
    v += unigram_dict.get(testSentence[x])

for y in range(len(testSentenceBigram)-1):
    bi_count = bigram_dict.get(testSentenceBigram[y]) + 1
    uni_count = unigram_dict.get(testSentenceBigram[y][0]) + v
    prob_2 = bi_count / uni_count
    mle_bigram_prob_list_smoothOne.append(prob_2)

mle_probabiliy_smoothOne = multiply(mle_bigram_prob_list_smoothOne)


#All bigrams probability
#List of counts of test sentence bigrams
bigram_prob_list = []

#P(bigram) = C(bigram) / C(bigrams count all)
for b in testSentenceBigram:
    word_a_count = unigram_dict.get(b[0])
    word_b_count = unigram_dict.get(b[1])
    bigram_prob = word_b_count / word_a_count
    bigram_prob_list.append(bigram_prob)

bigram_probability = multiply(bigram_prob_list)

#==========================printing==========================

print("\n", "=" * 20, "Sentence Probability Calculation", "=" * 20, "\n")
print("Training text: BROWN_A1.txt")
print("Test sentence: ", "'", test_sentence_raw, "'")
print("=" * 74)
print("MLE probability - no smoothing: ", mle_probabiliy_noSmooth)
print("(", float_to_str(mle_probabiliy_noSmooth), ")")
print("-" * 74)
print("MLE probability - plus-one smoothing: ", mle_probabiliy_smoothOne)
print("(", float_to_str(mle_probabiliy_smoothOne), ")")
print("-" * 74)
print("Bigram probability - bigram estimate: ", bigram_probability)
print("(", float_to_str(bigram_probability), ")")
print("=" * 74)
print("Executed! Processing time: ", time.clock() - start_time, "seconds")
