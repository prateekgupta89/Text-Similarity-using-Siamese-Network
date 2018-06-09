from config import network_config
from gensim.models import Word2Vec
import numpy as np

def get_training_data(path):
    '''
    Function to read the training data file
    Arguments:
        path: Path of training data file
    Returns:
        data: List of training examples
    '''

    # Read the training dataset
    data = []

    with open(path) as fp:
        lines = fp.readlines()

    for line in lines:
        data.append(line.split('\t'))

    return data

def get_embedding_vectors(documents, embedding_dim):
    '''
    Function to get word vectors
    Arguments:
        document: List of sentences
        embedding_dim: dimensions of word vector
    Returns:
        word_vectors: word vectors
    '''

    model = Word2Vec(documents, min_count=1, size=embedding_dim)
    model.save('static/model.bin')
    return model.wv

def get_embedding_matrix(word_to_idx, documents):
    '''
    Function to generate word2vec embedding matrix
    Arguments:
        word_to_idx: word to index dictionary
        documents: List of sentences
    Returns:
        embedding matrix
    '''

    embedding_dim = network_config['embedding_dim']
    word_vectors = get_embedding_vectors(documents, embedding_dim)
    vocab_size = len(word_to_idx)
    embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
    print 'vocab size = %d' % vocab_size
    print 'Length of word vectors = %d' % len(word_vectors.vocab)
    print 'Dimensions of embedding matrix = %s' % str(embedding_matrix.shape)
 
    for word, i in word_to_idx.items():
        embedding_vector = word_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix 
