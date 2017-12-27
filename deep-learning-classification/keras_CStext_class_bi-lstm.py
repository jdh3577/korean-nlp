#-*- coding: utf-8 -*-

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import itertools
import random
import numpy as np

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Merge, Bidirectional
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

corpus_fname = './dataset/train_data.txt'

from konlpy.tag import Twitter
mecab = Twitter()

def get_normalized_data(sentence):
    # original_sentence = mecab.pos(sentence, norm=True, stem=True)
    original_sentence = mecab.pos(sentence)
    inputData = []
    for w, t in original_sentence:
        # if t in ['Number']:
        #     w = '0'
        if t not in ['Number','Punctuation', 'KoreanParticle']:
            inputData.append(w)
    return (' '.join(inputData)).strip()


def get_text(fname):
    with open(fname, encoding='utf-8') as f:
        docs = [doc.replace('\n', '').split('\t') for doc in f]
        df_docs = docs  # 에러로 인한 수정

    texts, domain = zip(*df_docs)
    normalized_text = []
    for i in texts:
        normalized_text.append(get_normalized_data(i))
    return normalized_text, domain

docs, domain = get_text(corpus_fname)


from keras.preprocessing.text import Tokenizer

# ------------------Param. Starts ---------------------

# set parameters:
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
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
df_domain= to_categorical(df_domain, num_classes=2)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, df_domain, test_size=0.2, random_state=42)


# -----------------Data Preprocessing Ends -----------

# ------Create Model--------

# Import EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience = 3)

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
preds = Dense(2, activation='softmax')(l_lstm)
model = Model(sequence_input, preds)

print("model fitting - more complex convolutional neural network")
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("model fitting - more complex convolutional neural network")
model.summary()
history = model.fit(x_train, y_train, validation_split=0.2,
                    epochs=20, batch_size=32, callbacks=[early_stopping_monitor])
model.save('cs_classify_biLSTM.h5')



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


my_model = load_model('cs_classify_biLSTM.h5')
# my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

scores = my_model.evaluate(x_test, y_test, verbose=0)
print("Load model Test Accuracy: %.2f%%" % (scores[1] * 100))
