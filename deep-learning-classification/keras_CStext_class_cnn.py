# -*- coding: utf-8 -*-


from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import itertools
import random
import numpy as np
import pandas as pd

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Merge, Dropout, Concatenate, GlobalAveragePooling1D
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Model

import os
import sys

# ---------------Data Load Starts------------------

train_df = pd.read_csv("dataset/binary_train_data_all.csv", encoding='CP949')

corpus = train_df['QAContent_1'].values
domain = train_df['label'].values

from konlpy.tag import Twitter

mecab = Twitter()

def get_normalized_data(sentence):
    # original_sentence = mecab.pos(sentence, norm=True, stem=True)
    original_sentence = mecab.pos(sentence, norm=True)
    inputData = []
    for w, t in original_sentence:
        if t not in ['Number', 'Punctuation', 'KoreanParticle']:
            inputData.append(w)
    return (' '.join(inputData)).strip()


docs = [get_normalized_data(i) for i in corpus]

stopWords = []
with open('./dataset/stopwords.txt', encoding='cp949') as f:
    for i in f:
        stopWords.append(i.replace('\n', ''))
stopWords.extend(['안녕', '감사'])

from keras.preprocessing.text import Tokenizer

# ------------------Param. Starts ---------------------

# set parameters:
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

# Preprecessing param
# max_document_length 계산하기
# nb_words= 3 #extract only top word
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

x = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# For Domain, 0 / 1
# convert domain into with
def conv_str_list(x):
    df_cont = list(map(lambda i: int(i), x))
    return df_cont


df_domain = conv_str_list(domain)
# df_idx = conv_str_list(idx)

"""Multiclass"""
# make them into categorical
from keras.utils import to_categorical

df_domain = to_categorical(df_domain, num_classes=2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, df_domain, test_size=0.2, random_state=42)

# -----------------Data Preprocessing Ends -----------

# ------Create Model--------

# Import EarlyStopping
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=5)


### Pretrained Word Embedding
pretrained_word = True

import fasttext

INPUT_TXT = 'dataset/word-vector-preprocessing-noun.txt'
OUTPUT_PATH = 'skipgram-noun'

print('FastText Training...')
# Learn the word representation using skipgram model
skipgram = fasttext.skipgram(INPUT_TXT, OUTPUT_PATH, lr=0.02, dim=EMBEDDING_DIM, ws=5,
        epoch=1, min_count=5, neg=5, loss='ns', bucket=2000000, minn=1, maxn=6, word_ngrams=2,
        thread=6, t=1e-4, lr_update_rate=100)


if pretrained_word:
    embeddings_index = {}
    f = open(OUTPUT_PATH+'.vec', encoding='cp949')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors in FastText.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
else:
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

convs = []
filter_sizes = [3, 4, 5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)

l_merge = Concatenate(axis=1)(convs)
l_cov1 = Conv1D(64, 3, activation='relu')(l_merge)
l_cov2 = Conv1D(64, 3, activation='relu')(l_cov1)
l_pool1 = MaxPooling1D(3)(l_cov2)
# l_drop1 = Dropout(0.5)(l_pool1)
l_cov3 = Conv1D(128, 3, activation='relu')(l_pool1)
l_cov4 = Conv1D(128, 3, activation='relu')(l_cov3)
l_pool2 = GlobalAveragePooling1D()(l_cov4)
l_drop1 = Dropout(0.5)(l_pool2)
preds = Dense(2, activation='softmax')(l_drop1)

model = Model(sequence_input, preds)

print("model fitting - more complex convolutional neural network")
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("model fitting - more complex convolutional neural network")
history = model.fit(x_train, y_train, validation_split=0.2,
                    epochs=20, batch_size=32, callbacks=[early_stopping_monitor])
model.save('cs_classify_cnn.h5')


# Create the plot
import matplotlib.pyplot as plt

fig = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
fig = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Save model
from keras.models import load_model


my_model = load_model('cs_classify_cnn.h5')
# my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
my_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

scores = my_model.evaluate(x_test, y_test, verbose=0)
print("Load model Test Accuracy: %.2f%%" % (scores[1] * 100))