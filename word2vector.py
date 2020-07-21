from utils import load_training_data, load_testing_data, load_stop_words
from data_preprocessing import preprocess_x, preprocess_y, sens_to_ids
from gensim.models import word2vec
import numpy as np
# gensim接受语；料的格式为[[w,w,w],[w,w,w,w]]


def train_word2vector(x):
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=5, sg=1)
    return model


if __name__ == '__main__':
    stop_words = load_stop_words()
    raw_train_x, raw_train_y = load_training_data()
    raw_test_x = load_testing_data()

    seg_train_x = preprocess_x(raw_train_x, stop_words)
    seg_train_y = preprocess_y(raw_train_y)
    seg_test_x = preprocess_x(raw_test_x, stop_words)

    w2v_model = train_word2vector(seg_train_x + seg_train_y + seg_test_x)
    w2v_model.save('./w2v.model')

    index2word = w2v_model.wv.index2word
    word2index = {word: index for index, word in enumerate(index2word)}

    train_X = sens_to_ids(seg_train_x, word2index)
    train_Y = sens_to_ids(seg_train_y, word2index)
    test_X = sens_to_ids(seg_test_x, word2index)

    np.savetxt('/data/train_X.txt', train_X, fmt='%d')
    np.savetxt('/data/train_Y.txt', train_Y, fmt='%d')
    np.savetxt('/data/test_X.txt', test_X, fmt='%d')
