from preprocess import get_training_data, get_embedding_matrix, create_train_dev_test_set
from config import network_config
from keras.preprocessing.text import Tokenizer

if __name__ == '__main__':
   
    # Training data path 
    path = 'data/train_data.txt'
    
    # Get training data
    data = get_training_data(path)

    sentences1 = [x[0] for x in data] 
    sentences2 = [x[1] for x in data]
    sentences = sentences1 + sentences2 
    sim_score = [int(x[2].strip('\n')) for x in data]

    # create documents list
    documents = []
    maxLen = 0
    for sentence in sentences:
        documents.append(sentence.rstrip('.').split(" "))

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
