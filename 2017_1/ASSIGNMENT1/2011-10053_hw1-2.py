#2011-10053 Noh Hakyung
#Natural Language Processing with Python
#Exercise no.25

import sys
import re


vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y']
exception_1 = ['Qu', 'qu']
exception_2 = ['Y', 'y']
pig_suffix = 'ay'


# Main translation function
def pig_function(i):
	# Check exception case 1(qu)
	if i[:2] in exception_1:
		pig_word_exc1 = i[2:] + i[:2].lower() + pig_suffix
		return pig_word_exc1

	# Check exception case 2(y as consonant)
	else:
		if i[0] in exception_2 and i[1] in vowels:
			pig_word_exc2 = i[1:] + i[0].lower() + pig_suffix
			return pig_word_exc2

		else:
			# All other words
			# Cluster check and translation functions
			# Case 1: Word starts with vowel
			if i[0] in vowels:

				pig_word_0 = i + pig_suffix
				return pig_word_0

			# Case 2: Word starts with 1 consonant
			else:
				if i[0] != vowels and i[1] in vowels:

					if i[0].isupper(): 
						pig_word_1 = i[1].upper() + i[2:] + i[0].lower() + pig_suffix
						return pig_word_1
					else:
						pig_word_1 = i[1:] + i[0] + pig_suffix
						return pig_word_1

			# Case 3: Word starts with 2 consonants
				else:
					if i[0] != vowels and i[1] != vowels and i[2] in vowels:

						if i[0].isupper(): 
							pig_word_2 = i[2].upper() + i[3:] + i[:2].lower() + pig_suffix
							return pig_word_2
						else:
							pig_word_2 = i[2:] + i[:2] + pig_suffix
							return pig_word_2

			# Case 4: Word starts with 3 consonants
					else:
						if i[0] != vowels and i[1] != vowels and i[2] != vowels and i[3] in vowels:

							if i[0].isupper(): 
								pig_word_3 = i[3].upper() + i[4:] + i[:3].lower() + pig_suffix
								return pig_word_3
							else:
								pig_word_3 = i[3:] + i[:3] + pig_suffix
								return pig_word_3

						else:
							print 'Not a standard English word'
							return None


# Print function
def print_new_word(a, b):
	print "Pig Latin translation: ", a, '==>', b


# ==================================================================
# Main code

print '='* 60, '\n'
print 'Pig Latin Translator by Hakyung Noh, 2011-10053, v.1.0 \n'
print 'What type of data do you want to translate into Pig Latin? \n Press 1 to type in a word or a sentence. \n Press 2 to enter a text document. \n\n * The file should be in the same location with this code. \n * Do not specify file extension(only file name)'
print '\n'
print '='* 60
a = raw_input()


if a == str(1):
	word = raw_input('Please type in a word or a sentence : ')

# 1. Check wheather the input text is word or sentence
# 2. Put the sentence into list
# 3. If not an alphabetical character, then do nothing
# 4. If is a word, then translate to Pig Latin

	if ' ' in word: 
		sentence = re.findall(r'\w+|\S\w*', word)
		word_count = 0
		new_text = []

		for words in sentence:
			a = re.match(r'[^A-Za-z]', sentence[word_count])
			if a:
				new_text = new_text + words.split()
				word_count += 1
			else: 
				new_text = new_text + pig_function(sentence[word_count]).split()
				word_count += 1

		print '\nOriginal text: ', word
		print 'Translated text: ', ' '.join(new_text)

	else:
		if word.isalpha() == True:
			new_pig_word = pig_function(word)
			print_new_word(word, new_pig_word)
		else:
			word = raw_input('Please type only alphabetical characters : ')
			new_pig_word = pig_function(word)
			print_new_word(word, new_pig_word)
else: 
	if a == str(2):
		source_name = raw_input('Please specify a file name : ')
		print 'File name: ', source_name
		
		source_file = source_name + '.txt'
		target_name = source_name + '_pig_ver.txt'
		source_text = open(source_file, 'rU')
		target_loc = open(target_name, 'w')

		source_text_read = source_text.read()
		source_text_str = str(source_text_read)
		original_data = re.findall(r'\w+|\S\w*', source_text_str)

		word_count_f = 0
		new_text_w = []
		
		for z in original_data:
			a = re.match(r'[^A-Za-z]', original_data[word_count_f])
			if a:
				new_text_w = new_text_w + z.split()
				word_count_f += 1
			else: 
				new_text_w = new_text_w + pig_function(original_data[word_count_f]).split()
				word_count_f += 1

		print '\nTranslated text: ', ' '.join(new_text_w)
		target_loc.write (' '.join(new_text_w))
		target_loc.close()
		print '\nTask complete!'


	else:
		print 'Input not valid!'






# ==================================================================


# Notes:
#
#In English, the longest possible initial cluster is three consonants
# Source: https://en.wikipedia.org/wiki/Consonant_cluster
#
#The letter y is a consonant when it is the first letter of a syllable 
#that has more than one letter(yes, yogurt, yell). If y is anywhere else in the syllable, it is a vowel.
# Source: http://www.phonicsontheweb.com/y-roles.php
