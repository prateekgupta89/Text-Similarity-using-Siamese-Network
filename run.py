from preprocess import get_training_data, get_embedding_matrix, create_train_dev_test_set, text_to_word_list
from config import network_config
from keras.preprocessing.text import Tokenizer
from model import BiLSTMNetwork

if __name__ == '__main__':
   
    # Training data path 
    train_data_path = network_config['train_data_file_path']
    
    # Get training data
    df = get_training_data(train_data_path)
    questions1 = list(df['question1'])
    questions2 = list(df['question2'])
    sim_score = list(df['is_duplicate'])
    
    # Check if lengths of the lists are the same 
    assert(len(questions1)==len(questions2)==len(sim_score)) 
   
    # Preprocess the sentences
    m = len(questions1)   
    sentences1 = list()
    sentences2 = list()

    for i in range(0, m):
        sentences1.append(text_to_word_list(questions1[i]))
        sentences2.append(text_to_word_list(questions2[i]))

    print 'Corpus length = %d' % m

    documents = sentences1 + sentences2 
 
    # Create the tokenizer
    tokenizer = Tokenizer()

    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(documents)

    # Get the word to index dictionary
    word_to_idx = tokenizer.word_index
    print 'Vocabulary size = %d' % len(word_to_idx)

    # Generate the embedding matrix
    embedding_matrix, word2vec = get_embedding_matrix(word_to_idx, documents, network_config['pre_trained_vector_flag'])

    # Create training, validation and test set
    train_validation_dict = create_train_dev_test_set(tokenizer, sentences1, sentences2, sim_score)        
    
    # Train the model
    lstm_network = BiLSTMNetwork()
    lstm_network.train_model(train_validation_dict, embedding_matrix)
