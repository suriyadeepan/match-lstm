# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict
from lib.embed_utils import PAD_ID, NULL_ID
from lib.file_utils import deserialize, DATA_DIR


def vectorize_data(data, word2idx: Dict, max_sent_size):
    new_data = []
    for d in data:
        sent1 = d[0]
        sent1 = [word2idx.get(s, -1) for s in sent1]
        sent1.append(NULL_ID)  # premise需要在结尾加一个null word
        pad_length = max_sent_size - len(sent1)
        sent1.extend([PAD_ID] * pad_length)

        sent2 = d[1]
        sent2 = [word2idx.get(s, -1) for s in sent2]
        pad_length = max_sent_size - len(sent2)
        sent2.extend([PAD_ID] * pad_length)

        line = [sent1, sent2, d[2], d[3]]
        new_data.append(line)
    return new_data

def cnt2ratio(label_list, num_class):
    ratio_list = [0.0] * num_class
    for label in label_list:
        ratio_list[label] += 1
    ratio_list = [x / 5 for x in ratio_list]
    return ratio_list


def unpack_data(data, num_class):
    new_data = list(zip(*data))

    labels = list(new_data[-1])
    labels = [cnt2ratio(l, num_class) for l in labels]
    new_data[-1] = labels
    
    return new_data