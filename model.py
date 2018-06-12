from config import network_config
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM, Merge 
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from time import time
import keras.backend as K
import datetime

class BiLSTMNetwork(object):

    def __init__(self):
        
        self.embedding_dim = network_config['embedding_dim']
        self.num_lstm_units = network_config['num_lstm_units']
        self.maxLen = network_config['maxLen']
        selftrain_split_ratio = network_config['train_split_ratio']
        self.validation_split_ratio = network_config['validation_split_ratio']

    def exponent_neg_manhattan_distance(self, left, right):
        '''
        Helper function for the similarity estimate of the LSTMs outputs
        Arguments:
            left: Left LSTM output
            right: Right LSTM output
        Returns:
            Manhattan distance
        '''
        
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

    def train_model(self, train_dev_test_dict, embedding_matrix):
        '''
        Function to train the model
        Arguments:
            train_dev_test_dict: dictionary containing training, validation and test set
            embedding_matrix: embedding matrix containing word vectors
        Returns:
            model: trained Bidirectional LSTM model
        '''

        training_set_1 = train_dev_test_dict['training_set_1']
        training_set_2 = train_dev_test_dict['training_set_2']
        training_labels = train_dev_test_dict['training_lables']
        validation_set_1 = train_dev_test_dict['validation_set_1']
        validation_set_2 = train_dev_test_dict['validation_set_2']
        validation_labels = train_dev_test_dict['validation_labels']
        test_set_1 = train_dev_test_dict['test_set_1']
        test_set_2 = train_dev_test_dict['test_set_2']
        test_labels = train_dev_test_dict['test_labels']

        # Define input to computational graph
        x1 = Input(shape=(self.maxLen,), dtype='int32') 
        x2 = Input(shape=(self.maxLen,), dtype='int32') 
        
        # Create embeddings layer
        embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=self.maxLen, trainable=False)

        # Create lstm encoder layer
        lstm_encoder = LSTM(self.num_lstm_units)

        # Propagate input through embedded layer
        embedded_sequence_1 = embeddings(x1)
        embedded_sequence_2 = embeddings(x2)
        
        # Propagate embeddings through LSTM layer
        encoded_sequence_1 = lstm_encoder(embedded_sequence_1)
        encoded_sequence_2 = lstm_encoder(embedded_sequence_2)
       
        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Merge(mode=lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([encoded_sequence_1, encoded_sequence_2]) 

        # Create the model
        model = Model(inputs=[x1, x2], outputs=malstm_distance)
        
        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

        # Dump the model architecture to a file
        model_architecture = model.to_yaml()
        with open('model_architecture.yaml', 'a') as model_file:
            model_file.write(model_architecture)

        file_path='weights.{epoch:02d}-{acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(file_path, monitor="acc", verbose=1, save_weights_only=True, save_best_only=True, mode="max")
        callbacks = [checkpoint]
    
        # Start training
        training_start_time = time()
        
        model.fit([training_set_1, training_set_1], training_labels, validation_data=([validation_set_1, validation_set_1], validation_labels), epochs=100, batch_size=128, shuffle=True, callbacks=callbacks)

        print "Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)) 
