import numpy as np    #numpy is a package for scientific computing
from collections import Counter
vocab = Counter()
text = "Hi"
#Get all words
for word in text.split(' '):
    vocab[word]+=1

#Convert words to indexes
def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word] = i

    return word2index
#Now we have an index
word2index = get_word_2_index(vocab)
total_words = len(vocab)
#This is how we create a numpy array (our matrix)
matrix = np.zeros((total_words),dtype=float)
#Now we fill the values
for word in text.split():
    matrix[word2index[word]] += 1
print(matrix)

y = np.zeros((3),dtype=float)
if category == 0:
    y[0] = 1.        # [ 1.  0.  0.]
elif category == 1:
    y[1] = 1.        # [ 0.  1.  0.]
else:
     y[2] = 1.       # [ 0.  0.  1.]
