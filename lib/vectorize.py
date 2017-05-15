from lib.file_utils import deserialize
from lib.data_utils import vectorize_data

import os


data_dir='data/'


def load_data():
    train_data = deserialize(os.path.join(data_dir, 'snli_1.0_train.bin'))
    valid_data = deserialize(os.path.join(data_dir, 'snli_1.0_dev.bin'))
    test_data = deserialize(os.path.join(data_dir, 'snli_1.0_test.bin'))
    return train_data, valid_data, test_data

def load_dict():
    word2index = deserialize(os.path.join(data_dir, 'word2index.bin'))
    word_embeddings = deserialize(os.path.join(data_dir, 'word_embeddings.bin'))
    return word2index, word_embeddings


def vectorize_all_data(train_data, valid_data, test_data, word2index, sent_size):
    train_data = vectorize_data(train_data, word2index, max_sent_size=sent_size)
    valid_data = vectorize_data(valid_data, word2index, max_sent_size=sent_size)
    test_data = vectorize_data(test_data, word2index, max_sent_size=sent_size)
    return train_data, valid_data, test_data


def vectorize(sent_size=30):
    train_data, valid_data, test_data = load_data()
    word2index, word_embeddings = load_dict()
    train_data, valid_data, test_data = vectorize_all_data(train_data, valid_data, test_data,
                                                               word2index, sent_size)
    return { 
            'train_data' : train_data,
            'valid_data' : valid_data,
            'test_data' : test_data,
            'word2index' : word2index,
            'word_embeddings' : word_embeddings
           }
