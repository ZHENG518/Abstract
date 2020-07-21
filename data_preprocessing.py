import utils
import re
import numpy as np
import jieba

jieba.load_userdict('/data/user_dict.txt')


def preprocess_x(data, stop_words):
    x = []
    for sen in data:
        # 去除特殊字符
        sen = re.sub(r'[\s+\-\|\!\/\[\]\{\}_.$^*(+\"\')]+|[:：+——()【】、~@#￥%……&*（）]+|车主说|技师说|语音|图片',
                     '', sen)
        # jieba分词
        words = jieba.lcut(sen)
        # 去除停用词
        words = [word for word in words if word not in stop_words]
        # 去除空的句子
        if len(words) > 0:
            x.append(words)
    return x


def preprocess_y(data):
    y = []
    for sen in data:
        words = jieba.lcut(sen)
        y.append(words)
    return y


def sens_to_ids(data, word2index):
    # 计算统一的长度
    len_list = [len(sen) for sen in data]
    max_len = int(np.mean(len_list) + 2 * np.std(len_list))

    # 获取特殊符号index
    BOS = word2index['<BOS>']
    EOS = word2index['<EOS>']
    UNK = word2index['<UNK>']
    PAD = word2index['<PAD>']

    id_list = []
    for line in data:
        # 截断长句子
        line = line[:max_len]
        # 句首添加<BOS>
        line_ids = [BOS]
        # 转换为id,添加<UNK>
        line_ids += [word2index.get(word, UNK) for word in line]
        # 添加<EOS>
        line_ids.append(EOS)
        # 统一长度，添加<PAD>
        line_ids += [PAD] * (max_len + 2 - len(line_ids))
        id_list.append(line_ids)

    return id_list
