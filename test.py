'''
File to test the word embeddings generated
'''

from gensim.models import Word2Vec
from keras.layers.embeddings import Embedding
import numpy as np
from keras.layers import Input, merge
from keras.models import Model

def get_sim(valid_word_idx, vocab_size):
    sim = np.zeros((vocab_size,))
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    in_arr1[0,] = valid_word_idx
    for i in range(vocab_size):
        in_arr2[0,] = i
        out = k_model.predict_on_batch([in_arr1, in_arr2])
        sim[i] = out
    return sim

model=Word2Vec.load('static/model.bin')
embedding_dim = 50
embedding_matrix = np.zeros((len(model.wv.vocab), embedding_dim))

for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# input words - in this case we do sample by sample evaluations of the similarity
valid_word = Input((1,), dtype='int32')
other_word = Input((1,), dtype='int32')
# setup the embedding layer
embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix])
embedded_a = embeddings(valid_word)
embedded_b = embeddings(other_word)
similarity = merge([embedded_a, embedded_b], mode='cos', dot_axes=2)
# create the Keras model
k_model = Model(input=[valid_word, other_word], output=similarity)

# now run the model and get the closest words to the valid examples
for i in range(valid_size):
    valid_word = model.wv.index2word[valid_examples[i]]
    top_k = 8  # number of nearest neighbors
    sim = get_sim(valid_examples[i], len(model.wv.vocab))
    nearest = (-sim).argsort()[1:top_k + 1]
    log_str = 'Nearest to %s:' % valid_word
    for k in range(top_k):
        close_word = model.wv.index2word[nearest[k]]
        log_str = '%s %s,' % (log_str, close_word)
    print(log_str)
