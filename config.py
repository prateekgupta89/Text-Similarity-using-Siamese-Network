EMBEDDING_DIM = 300
NUM_LSTM_UNITS = 50
MAXLEN = 20
TRAIN_VALIDATION_SPLIT_RATIO = 0.98
PRE_TRAINED_VECTOR_FLAG = True
TRAIN_DATA_FILE_PATH = './data/train.csv'
TEST_DATA_FILE_PATH = './data/test.csv'

network_config = {
    'embedding_dim': EMBEDDING_DIM,
    'num_lstm_units': NUM_LSTM_UNITS,
    'maxLen': MAXLEN,
    'train_validation_split_ratio': TRAIN_VALIDATION_SPLIT_RATIO,
    'pre_trained_vector_flag': PRE_TRAINED_VECTOR_FLAG,
    'train_data_file_path': TRAIN_DATA_FILE_PATH,
    'test_data_file_path': TEST_DATA_FILE_PATH
}
