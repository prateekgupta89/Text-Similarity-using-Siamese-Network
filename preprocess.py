from keras.preprocessing.sequence import pad_sequences
from config import network_config
from gensim.models import Word2Vec
import numpy as np
import os.path
import random

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
    if os.path.isfile('static/model.bin'):
        model=Word2Vec.load('static/model.bin')
        word_vectors = model.wv
    else: 
        word_vectors = get_embedding_vectors(documents, embedding_dim)
    vocab_size = len(word_to_idx)
    embedding_matrix = np.zeros((len(word_vectors.vocab), embedding_dim))
    print 'vocab size = %d' % vocab_size
    print 'Number of word vectors = %d' % len(word_vectors.vocab)
    print 'Dimensions of embedding matrix = %s' % str(embedding_matrix.shape)
 
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def create_train_dev_test_set(tokenizer, sentences1, sentences2, sim_score):
    '''
    Function to split sentences into train, validation and test set
    Arguments:
        tokenizer: keras tokenizer object
        sentences1: List of first pair of sentences
        sentences2: List of second pair of sentences
        sim_Score: similarity score for sentences
    Returns:
        dictionary containing training set, validation set and test set
    '''
   
    maxLen = network_config['maxLen']
    train_split_ratio = network_config['train_split_ratio']
    validation_split_ratio = network_config['validation_split_ratio']
    num_training_examples = len(sentences1)

    train_sentences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sentences_2 = tokenizer.texts_to_sequences(sentences2)

    padded_sentences_1 = pad_sequences(train_sentences_1, maxlen=maxLen)
    padded_sentences_2 = pad_sequences(train_sentences_2, maxlen=maxLen)
    training_labels = np.asarray(sim_score)

    max_train_idx = train_split_ratio*num_training_examples
    max_validation_idx = validation_split_ratio*num_training_examples

    shuffle_indices = range(0, num_training_examples)
    random.shuffle(shuffle_indices)

    padded_sentences_1 = padded_sentences_1[shuffle_indices]
    padded_sentences_2 = padded_sentences_2[shuffle_indices]
    training_lables = training_labels[shuffle_indices]

    training_set_1 = padded_sentences_1[0:max_train_idx]
    training_set_2 = padded_sentences_2[0:max_train_idx]
    train_labels = training_lables[0:max_train_idx]

    validation_set_1 = padded_sentences_1[max_train_idx:max_validation_idx]
    validation_set_2 = padded_sentences_2[max_train_idx:max_validation_idx]
    validation_labels = training_lables[max_train_idx:max_validation_idx]

    test_set_1 = padded_sentences_1[max_validation_idx:]
    test_set_2 = padded_sentences_2[max_validation_idx:]
    test_labels = training_lables[max_validation_idx:]

    train_dev_test_dict = {}
    train_dev_test_dict['training_set_1'] = training_set_1
    train_dev_test_dict['training_set_2'] = training_set_2
    train_dev_test_dict['training_lables'] = training_lables
    train_dev_test_dict['validation_set_1'] = validation_set_1
    train_dev_test_dict['validation_set_2'] = validation_set_2
    train_dev_test_dict['validation_labels'] = validation_labels
    train_dev_test_dict['test_set_1'] = test_set_1
    train_dev_test_dict['test_set_2'] = test_set_2
    train_dev_test_dict['test_labels'] = test_labels

    return train_dev_test_dict
    
    
     
