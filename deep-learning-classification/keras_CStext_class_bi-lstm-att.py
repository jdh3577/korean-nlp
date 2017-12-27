#-*- coding: utf-8 -*-

import os
import sys

# os.environ['KERAS_BACKEND']='theano'

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import itertools
import random
import numpy as np

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
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K

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
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# Preprecessing param
# max_document_length 계산하기
# nb_words= 3 #extract only top word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs, num_words=MAX_NB_WORDS)
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


# # -----------------Data Preprocessing Ends -----------

# ------Create Model--------

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

# Attention GRU network
class AttLayer(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

# Import EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience = 5)

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer()(l_lstm)
preds = Dense(2, activation='softmax')(l_att)

model = Model(sequence_input, preds)

print("model fitting - attention GRU network")
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("model fitting - more complex convolutional neural network")
model.summary()
history = model.fit(x_train, y_train, validation_split=0.2,
                    epochs=20, batch_size=32, callbacks=[early_stopping_monitor])
model.save('cs_classify_biLSTM_att.h5')

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


my_model = load_model('cs_classify_biLSTM_att.h5', custom_objects={'AttLayer':AttLayer})
# my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
my_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

scores = my_model.evaluate(x_test, y_test, verbose=0)
print("Load model Test Accuracy: %.2f%%" % (scores[1] * 100))


