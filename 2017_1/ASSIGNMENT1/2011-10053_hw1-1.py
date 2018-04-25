#2011-10053 Noh Hakyung
#Natural Language Processing with Python
#Exercise no.18

import re
from collections import Counter

# Read text from raw file, make all characters lower case
raw_text = open('BROWN_A1.txt').read().lower()

# Remove all special characters
clean_string = re.sub(r'[?|$|.|!]',r'', raw_text)

# Find 'wh' words
# Words pool: who, what, how, where, when, why, which, whom, whose
words = re.findall(r'\b(who|what|how|where|when|why|which|whom|whose)', clean_string)

# Count words and sort
w_count = Counter(words)

# Convert result to list
sort_words = w_count.most_common()

# Separator 1
print '=' *40

# Print results
for word, num in sort_words:
	print "%r => %r" % (word, num)

# Sum all occurences number
w_sum = sum(w_count.values())

# Separator 2
print '=' *40

# Print total 'wh' words occurances
print 'Total %s WH words found in given text!' %w_sum
