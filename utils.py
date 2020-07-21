import pandas as pd
import numpy as np
from gensim.models import word2vec


def load_stop_words():
    with open('/data/stopwords.txt', 'r') as f:
        words = f.readlines()
    return [word.strip('\n') for word in words]


def load_training_data():
    train_df = pd.read_csv('/data/AutoMaster_TrainSet.csv')
    # 去除空值
    train_df = train_df.dropna(subset=['Question', 'Dialogue', 'Report'])
    train_x = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x).strip(), axis=1).tolist()
    train_y = train_df['Report'].tolist()

    return train_x, train_y


def load_testing_data():
    test_df = pd.read_csv('/data/AutoMaster_TestSet.csv')
    test_x = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x).strip(), axis=1).tolist()

    return test_x


def load_vocab_embedding_matrix():
    w2v_model = word2vec.Word2Vec.load('/data/w2v.model')
    # 加载word2vector
    embedding_dim = w2v_model.vector_size
    index2word = w2v_model.wv.index2word
    word2index = {word: index for index, word in enumerate(index2word)}
    embedding_matrix = w2v_model.wv.vectors

    # 添加<BOS><EOS><PAD>到vocab和embedding matrix
    index2word = index2word + ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
    word2index['<BOS>'] = len(index2word) - 4
    word2index['<EOS>'] = len(index2word) - 3
    word2index['<PAD>'] = len(index2word) - 2
    word2index['<UNK>'] = len(index2word) - 1
    vector = np.random.randn(4, embedding_dim)  # 生成随机的四个个vector
    embedding_matrix = np.vstack((embedding_matrix, vector))

    return index2word, word2index, embedding_matrix


def list_to_txt(data, path):
    with open(path, 'w') as f:
        for s in data:
            f.write(s + '\n')


def txt_to_list(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return [d.strip('\n').strip() for d in data]


def save_seg_data(data, path):
    lines = []
    for l in data:
        line = ' '.join(l)
        lines.append(line)
    list_to_txt(lines, path)


def load_seg_data(path):
    sens = txt_to_list(path)
    return [sen.split() for sen in sens]
