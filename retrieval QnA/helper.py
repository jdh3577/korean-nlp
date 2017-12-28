# -*- encoding:utf-8 -*-
import os
import json
import pickle

import numpy as np
import pandas as pd
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from konlpy.tag import Twitter
twiter = Twitter()

TRAIN_PICKLE = "sequences.train.pickle"
TEST_PICKLE = "sequences.test.pickle"
DEV_PICKLE = "sequences.dev.pickle"
TOKENIZER_PICKLE = "tokenizer.pickle"

NB_CLASSES = 2
label2class = {'False': 0, 'True': 1}

def get_normalized_data(sentence):
    # original_sentence = mecab.pos(sentence, norm=True, stem=True)
    original_sentence = twiter.pos(sentence, norm=True)
    inputData = []
    for w, t in original_sentence:
        if t not in ['Number','Punctuation', 'KoreanParticle']:
            inputData.append(w)
    return (' '.join(inputData)).strip()

def load_data(max_seq, num_words):
#     if (os.path.exists(TRAIN_PICKLE)
#         and os.path.exists(TEST_PICKLE)
#         and os.path.exists(DEV_PICKLE)):

#         with open(TRAIN_PICKLE, 'rb') as fp:
#             X_train_1, X_train_2, Y_train = pickle.load(fp)
#         with open(TEST_PICKLE, 'rb') as fp:
#             X_test_1, X_test_2, Y_test = pickle.load(fp)
#         with open(DEV_PICKLE, 'rb') as fp:
#             X_dev_1, X_dev_2, Y_dev = pickle.load(fp)
# 
#     else:
    x_train_1, x_train_2, y_train = [], [], []
    x_test_1, x_test_2, y_test = [], [], []
    x_dev_1, x_dev_2, y_dev = [], [], []

    train_df = pd.read_csv("dataset/retrieval_train_data.csv", encoding='CP949')

    x_1 = train_df['question'].values
    x_2 = train_df['answer'].values
    y = train_df['label'].values

    normalized_x_1 = []
    normalized_x_2 = []
    [normalized_x_1.append(get_normalized_data(i)) for i in x_1]
    [normalized_x_2.append(get_normalized_data(i)) for i in x_2]

    data_len = len(normalized_x_1)
    train_len = int(data_len * 0.7)+1
    dev_len = int(data_len * 0.85)+1
    test_len = int(data_len)

    x_train_1 = normalized_x_1[:train_len]
    x_dev_1 = normalized_x_1[train_len:dev_len]
    x_test_1 = normalized_x_1[dev_len:]

    x_train_2 = normalized_x_2[:train_len]
    x_dev_2 = normalized_x_2[train_len:dev_len]
    x_test_2 = normalized_x_2[dev_len:]

    y_train = y[:train_len]
    y_dev = y[train_len:dev_len]
    y_test = y[dev_len:]

    
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train_1)
    tokenizer.fit_on_texts(x_train_2)
    tokenizer.fit_on_texts(x_test_1)
    tokenizer.fit_on_texts(x_test_2)
    tokenizer.fit_on_texts(x_dev_1)
    tokenizer.fit_on_texts(x_dev_2)

    X_train_1 = tokenizer.texts_to_sequences(x_train_1)
    X_train_2 = tokenizer.texts_to_sequences(x_train_2)
    X_test_1 = tokenizer.texts_to_sequences(x_test_1)
    X_test_2 = tokenizer.texts_to_sequences(x_test_2)
    X_dev_1 = tokenizer.texts_to_sequences(x_dev_1)
    X_dev_2 = tokenizer.texts_to_sequences(x_dev_2)

    MAX_SEQUENCE_LENGTH = max([len(seq) for seq in X_train_1 + X_train_2
                                                 + X_test_1 + X_test_2
                                                 + X_dev_1 + X_dev_2])
    MAX_SEQUENCE_LENGTH = max_seq
    # print(X_train_1 + X_train_2 + X_test_1 + X_test_2 + X_dev_1 + X_dev_2)
    MAX_NB_WORDS = len(tokenizer.word_index) + 1

    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))

    X_train_1 = pad_sequences(X_train_1, maxlen=MAX_SEQUENCE_LENGTH)
    X_train_2 = pad_sequences(X_train_2, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_1 = pad_sequences(X_test_1, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_2 = pad_sequences(X_test_2, maxlen=MAX_SEQUENCE_LENGTH)
    X_dev_1 = pad_sequences(X_dev_1, maxlen=MAX_SEQUENCE_LENGTH)
    X_dev_2 = pad_sequences(X_dev_2, maxlen=MAX_SEQUENCE_LENGTH)

    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
    Y_dev = np_utils.to_categorical(y_dev, NB_CLASSES)

    with open(TRAIN_PICKLE, 'wb') as fp:
        pickle.dump((X_train_1, X_train_2, Y_train), fp)
    with open(TEST_PICKLE, 'wb') as fp:
        pickle.dump((X_test_1, X_test_2, Y_test), fp)
    with open(DEV_PICKLE, 'wb') as fp:
        pickle.dump((X_dev_1, X_dev_2, Y_dev), fp)

    with open(TOKENIZER_PICKLE, 'wb') as fp:
        pickle.dump(tokenizer, fp)

    return (X_train_1, X_train_2, Y_train,
            X_test_1, X_test_2, Y_test,
            X_dev_1, X_dev_2, Y_dev)


def load_tokenizer():
    if not os.path.exists(TOKENIZER_PICKLE):
        load_data()
    with open(TOKENIZER_PICKLE, 'rb') as fp:
        tokenizer = pickle.load(fp)
    return tokenizer