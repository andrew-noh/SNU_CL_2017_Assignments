#2011-10053 Noh Hakyung
#Natural Language Processing with Python
#Assignment 3
#Cross Entropy calculation

import re
import math
from collections import Counter

#Text to character unigram convertion function
def txt_2_uni(a):
	# Read the training corpus (BROWN1_D1.txt)
	raw_text = open(a).read().lower()
	# Remove all special characters
	clean_string = re.sub(r'[^a-zA-Z ]',r'', raw_text)
	# Replace space with equal sign
	space_sub = re.sub('  ', ' ', clean_string)
	space_sub2 = re.sub(r' ', r'=', space_sub)
	uni_list = list(space_sub2)
	return uni_list
	# space character: '='

#Text to character bigram convertion function
def txt_2_bi(n):
	# Read the training corpus (BROWN1_D1.txt)
	raw_text = open(n).read().lower()
	# Remove all special characters
	clean_string = re.sub(r'[^a-zA-Z ]',r'', raw_text)
	# Replace space with equal sign
	space_sub = re.sub('  ', ' ', clean_string)
	space_sub2 = re.sub(r' ', r'=', space_sub)
	bigram_list = []
	for i in range(len(space_sub2)-1):
		new_bi = space_sub2[i:i+2]
		bigram_list.append(new_bi)
	return bigram_list
	# space character: '='

#Count unique
def count_uniq(i):
	count_all = Counter(i)
	count_unique = count_all.most_common()
	unique_dict = dict(count_unique)
	return unique_dict

#Count length(all occurences)
def count_len(p):
	total_length = len(p)
	return total_length

#Unigram probability
#Input splitted unigram text
#Output = list of probabilities of each character in given text
def uni_prob_list(k):
	unigram_count = count_uniq(k)
	unigram_length = count_len(k)
	value_list = unigram_count.values()
	key_list = unigram_count.keys()
	prob_count_list = []
	for i in value_list:
		uni_probability = float(i) / float(unigram_length)
		prob_count_list.append(uni_probability)
	return prob_count_list

#Entropy calculation
#Input probabilities list
def entr_calc(j):
	entropy = 0
	for i in j:
		if i > 0:
			entropy += -i*math.log(i, 2)
	return entropy

#Unigram cross entropy
def cross_entr(training_prob, test_prob):
	cross_entropy_pre = []
	count_index = 0
	while count_index < len(test_prob)-1:
		entropy_pre = -float(test_prob[count_index])*math.log(float(training_prob[count_index]), 2)
		cross_entropy_pre.append(entropy_pre)
		count_index += 1
	return sum(cross_entropy_pre)

#Bigram cross entropy
def cross_entr_bigram(training_prob, test_prob):
	cross_entropy_pre_2 = []
	count_index_2 = 0
	while count_index_2 < len(test_prob)-1:
		for i in test_prob.keys():
			if i in training_prob.keys():
				entropy_pre_2 = -float(test_prob[i])*math.log(float(training_prob[i]), 2)
				cross_entropy_pre_2.append(entropy_pre_2)
				count_index_2 +=1
			else:
				entropy_pre_3 = -float(test_prob[i])*math.log(zero_con / float(count_len(training_text)), 2)
				cross_entropy_pre_2.append(entropy_pre_3)
				count_index_2 += 1
	return sum(cross_entropy_pre_2)

# Update value in unique bigrams count with bigram probability(MLE)
# a = bigram dictionary, b = unigram count
def dict_probability_update(a, b):
	bigram_list = list(a)
	for i in bigram_list:
		bi_count = float(a[i])
		uni_count = float(b[i[1]])
		new_prob = bi_count / uni_count
		a[i] = new_prob


#=================MAIN FUNCTION=====================
#====================UNIGRAM========================

#Training corpus entropy calculation - BROWN1_D1.txt
training_text = txt_2_uni('BROWN1_D1.txt')
alphabet_count_t1 = count_uniq(training_text)
prob_count_list_t1 = uni_prob_list(training_text)
keys_list = count_uniq(training_text).keys()
prob_dictionary = dict(zip(keys_list, prob_count_list_t1))
unigram_entropy = entr_calc(prob_count_list_t1)

#Test corpus entropy calculation - BROWN_L1.txt
test_text = txt_2_uni('BROWN1_L1.txt')
alphabet_count_t2 = count_uniq(test_text)
prob_count_list_t2 = uni_prob_list(test_text)
keys_list_2 = count_uniq(test_text).keys()
prob_dictionary_2 = dict(zip(keys_list_2, prob_count_list_t2))
unigram_entropy_2 = entr_calc(prob_count_list_t2)

#Cross Entropy
cross_entropy = cross_entr(prob_count_list_t1, prob_count_list_t2)
entropy_diff = cross_entropy - unigram_entropy_2

#====================BIGRAM========================
#Training corpus bigram
training_bigram = txt_2_bi('BROWN1_D1.txt')
unique_bigrams_t1 = count_uniq(training_bigram)
bigrams_prob_dict_t1 = unique_bigrams_t1.copy()
dict_probability_update(bigrams_prob_dict_t1, alphabet_count_t1)

#Training corpus entropy
bigram_prob_list_t1 = bigrams_prob_dict_t1.values()
bigram_entropy_t1 = entr_calc(bigram_prob_list_t1)

#--------------------------------------------------

#Test corpus bigram
test_bigram = txt_2_bi('BROWN1_L1.txt')
unique_bigrams_t2 = count_uniq(test_bigram)
bigrams_prob_dict_t2 = unique_bigrams_t2.copy()
dict_probability_update(bigrams_prob_dict_t2, alphabet_count_t2)

#Test corpus entropy
bigram_prob_list_t2 = bigrams_prob_dict_t2.values()
bigram_entropy_t2 = entr_calc(bigram_prob_list_t2)

#--------------------------------------------------

#Cross entropy

#Zero count compensation
zero_con = float(0.5) 

#Cross entropy function
bigram_cross_entr = cross_entr_bigram(bigrams_prob_dict_t1, bigrams_prob_dict_t2)

#Difference
bigram_ent_difference = bigram_cross_entr - bigram_entropy_t2

#=================PRINT RESULT=====================
#===================================================
"""
print '='*30, 'Unigram', '='*30
print '\n1) Training text BROWN1_D1\n'
print 'Unigram length = ', count_len(training_text)
print '\nAlphabet count: \n'
print alphabet_count_t1
print'\nAlphabet probability: \n'
print '='*70
print prob_dictionary
print'='*70
print 'BROWN1_D1 text file unigram(alphabet) entropy = ', unigram_entropy
print '\n'
print '2) Test text BROWN1_L1\n'
print 'Unigram length =', count_len(test_text)
print '\nAlphabet count: \n'
print alphabet_count_t2
print '\nAlphabet probability: \n'
print '='*70
print prob_dictionary_2
print '='*70
print 'BROWN1_L1 text file unigram(alphabet) entropy = ', unigram_entropy_2
print '\n'
print 'Cross Entropy(unigram): ', cross_entropy
print 'Difference: ', entropy_diff
print '='*70
print '='*31, 'Bigram', '='*31
print 'BROWN1_D1 text file bigram(alphabet) entropy = ', bigram_entropy_t1
print '\nBROWN1_L1 text file bigram(alphabet) entropy = ', bigram_entropy_t2
print '\nCross Entropy(bigram): ', bigram_cross_entr
print '\nDifference: ', bigram_ent_difference
"""

print '='*110
print '\n'
print '                                           Entropy               CrossEntropy               Difference\n'
print 'Training corpus(BROWN1_D1) || Unigram   ', unigram_entropy, '\n'
print '                           || Bigram    ', bigram_entropy_t1
print '-'*110
print 'Test corpus(BROWN1_L1)     || Unigram   ', unigram_entropy_2, '         ', cross_entropy, '          ', entropy_diff, '\n'
print '                           || Bigram    ', bigram_entropy_t2, '          ', bigram_cross_entr, '          ', bigram_ent_difference, '\n'
print '='*110
