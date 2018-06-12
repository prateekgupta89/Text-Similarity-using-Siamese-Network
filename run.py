from preprocess import get_training_data, get_embedding_matrix, create_train_dev_test_set, text_to_word_list
from config import network_config
from keras.preprocessing.text import Tokenizer
from model import BiLSTMNetwork

if __name__ == '__main__':
   
    # Training data path 
    path = 'data/train_data.txt'
    
    # Get training data
    data = get_training_data(path)

    sentences1, sentences2, sim_score = [], [], []
    count = 0

    for sentence1, sentence2, score in data:
        sentence1 = text_to_word_list(sentence1)
        sentence2 = text_to_word_list(sentence2)
        if len(sentence1) <= 20 and len(sentence2) <= 20:
            sentences1.append(sentence1)
            sentences2.append(sentence2)
            sim_score.append(int(score.strip('\n')))
            count += 1

    print 'Corpus length = %d' % count

    sentences = sentences1 + sentences2 
 
    # create documents list
    documents = []
    for sentence in sentences:
        documents.append(sentence)

    # Create the tokenizer
    tokenizer = Tokenizer()

    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(documents)

    # Get the word to index dictionary
    word_to_idx = tokenizer.word_index    

    # Generate the embedding matrix
    embedding_matrix = get_embedding_matrix(word_to_idx, documents)

    # Create training, validation and test set
    train_dev_test_dict = create_train_dev_test_set(tokenizer, sentences1, sentences2, sim_score)        
    # Train the model
    lstm_network = BiLSTMNetwork()
    lstm_network.train_model(train_dev_test_dict, embedding_matrix) 
