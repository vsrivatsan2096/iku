# Reference
## https://www.kaggle.com/zhihang/an-ensemble-approach-cnn-and-timedistributed


import re
import nltk
import numpy as np
import pandas as pd
import datetime, time, json
from string import punctuation
from keras import initializers
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Sequential

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D, concatenate 


class  CNNAndTimeDistributed(object):
    def __init__(self, sentences1, sentences2, is_duplicate):
        self.sentences = np.append(sentences1, sentences2, axis=0)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.length = self.sentences1.shape[0]
        self.embedding_dim = 300
        self.word_index = None
        self.max_words_len = 0
        self.is_duplicate = is_duplicate
    
    def do_preprocess(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.sentences.tolist())
        word_sequences_1 = tokenizer.texts_to_sequences(self.sentences1.tolist())
        word_sequences_2 = tokenizer.texts_to_sequences(self.sentences2.tolist())
        self.word_index = tokenizer.word_index
        self.max_words_len = 0
        for each in range(self.length):
            self.max_words_len = max(self.max_words_len, len(word_sequences_1[each]), len(word_sequences_2[each]))
        
        word_sequences_1 = pad_sequences(word_sequences_1, maxlen = self.max_words_len)
        word_sequences_2 = pad_sequences(word_sequences_2, maxlen = self.max_words_len)

        return word_sequences_1, word_sequences_2
    
    def get_embeddings(self, model_path):
        embeddings_index = {}
        with open(model_path, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding
        print('Word embeddings:', len(embeddings_index))

        nb_words = len(self.word_index)
        word_embedding_matrix = np.zeros((nb_words + 1, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                #words not found in embedding index will be all-zeros.
                word_embedding_matrix[i] = embedding_vector

        print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

        return word_embedding_matrix
    
    def get_model(self, modal_path):

        units = 128 # Number of nodes in the Dense layers
        dropout = 0.25 # Percentage of nodes to drop
        nb_filter = 32 # Number of filters to use in Convolution1D
        filter_length = 3 # Length of filter for Convolution1D

        weights = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2)
        bias = bias_initializer='zeros'

        word_embedding_matrix = self.get_embeddings(modal_path)
        nb_words = len(self.word_index)

        # CNN for sentences set 1
        model_1_input = Input(shape = (self.max_words_len,), dtype = 'int32', name = 'model_1_input')
        model_1_embedding = Embedding(nb_words + 1,
                     self.embedding_dim,
                     weights = [word_embedding_matrix], 
                     input_length = self.max_words_len,
                     trainable = False)(model_1_input)
        model_1_conv_a = Convolution1D(filters = nb_filter, 
                         kernel_size = filter_length, 
                         padding = 'same')(model_1_embedding)
        model_1_batch_a = BatchNormalization()(model_1_conv_a)
        model_1_act = Activation('relu')(model_1_batch_a)
        model_1_drop_a = Dropout(dropout)(model_1_act)
        model_1_conv_b = Convolution1D(filters = nb_filter, 
                         kernel_size = filter_length, 
                         padding = 'same')(model_1_drop_a)
        model_1_batch_b = BatchNormalization()(model_1_conv_b)
        model_1_act_b = Activation('relu')(model_1_batch_b)
        model_1_drop_b = Dropout(dropout)(model_1_act_b)
        model_1_flat = Flatten()(model_1_drop_b)

        # CNN for sentences set 2
        model_2_input = Input(shape = (self.max_words_len,), dtype = 'int32', name = 'model_2_input')
        model_2_embedding = Embedding(nb_words + 1,
                     self.embedding_dim,
                     weights = [word_embedding_matrix], 
                     input_length = self.max_words_len,
                     trainable = False)(model_2_input)
        model_2_conv_a = Convolution1D(filters = nb_filter, 
                         kernel_size = filter_length, 
                         padding = 'same')(model_2_embedding)
        model_2_batch_a = BatchNormalization()(model_2_conv_a)
        model_2_act = Activation('relu')(model_2_batch_a)
        model_2_drop_a = Dropout(dropout)(model_2_act)
        model_2_conv_b = Convolution1D(filters = nb_filter, 
                         kernel_size = filter_length, 
                         padding = 'same')(model_2_drop_a)
        model_2_batch_b = BatchNormalization()(model_2_conv_b)
        model_2_act_b = Activation('relu')(model_2_batch_b)
        model_2_drop_b = Dropout(dropout)(model_2_act_b)
        model_2_flat = Flatten()(model_2_drop_b)

        # TimeDistribute for sentences set 1
        model_3_input = Input(shape = (self.max_words_len,), dtype = 'int32', name = 'model_3_input')
        model_3_embedding = Embedding(nb_words + 1,
                     self.embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = self.max_words_len,
                     trainable = False)(model_3_input)
        model_3_time_distributed = TimeDistributed(Dense(self.embedding_dim))(model_3_embedding)
        model_3_batch = BatchNormalization()(model_3_time_distributed)
        model_3_act = Activation('relu')(model_3_batch)
        model_3_drop = Dropout(dropout)(model_3_act)
        model_3_lambda = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.embedding_dim, ))(model_3_drop)

        # TimeDistribute for sentences set 2
        model_4_input = Input(shape = (self.max_words_len,), dtype = 'int32', name = 'model_4_input')
        model_4_embedding = Embedding(nb_words + 1,
                     self.embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = self.max_words_len,
                     trainable = False)(model_4_input)
        model_4_time_distributed = TimeDistributed(Dense(self.embedding_dim))(model_4_embedding)
        model_4_batch = BatchNormalization()(model_4_time_distributed)
        model_4_act = Activation('relu')(model_4_batch)
        model_4_drop = Dropout(dropout)(model_4_act)
        model_4_lambda = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.embedding_dim, ))(model_4_drop)

        # Merging all the layers
        merge_layer = concatenate([model_1_flat, model_2_flat, model_3_lambda, model_4_lambda], name = 'merge_layer')

        t = Dense(200, activation = 'relu', name = 'dense1')(merge_layer)
        t = Dropout(0.3)(t)
        t = BatchNormalization()(t)

        t = Dense(200, activation = 'relu', name  ='dense2')(t)
        t = Dropout(0.3)(t)
        t = BatchNormalization()(t)

        t = Dense(100, activation= 'relu',name = 'dense3')(t)
        t = Dropout(0.3)(t)
        t = BatchNormalization()(t)

        final_output = Dense(1, activation = 'sigmoid')(t)

        model = Model(inputs = [model_1_input, model_2_input, model_3_input, model_4_input], outputs = final_output)
        
        return model


    def execute(self, modal_path):
        model = self.get_model(modal_path)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        save_best_weights = 'question_pairs_weights.h5'
        
        train_1, train_2 = self.do_preprocess()

        t0 = time.time()
        callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]
        history = model.fit([train_1, train_2, train_1, train_2],
                    self.is_duplicate,
                    batch_size=256,
                    epochs=1, #Use 100, I reduce it for Kaggle,
                    validation_split=0.15,
                    verbose=True,
                    shuffle=True,
                    callbacks=callbacks)
        t1 = time.time()
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                              'train_acc': history.history['acc'],
                              'valid_acc': history.history['val_acc'],
                              'train_loss': history.history['loss'],
                              'valid_loss': history.history['val_loss']})

        print(summary_stats)

        plt.plot(summary_stats.train_loss) # blue
        plt.plot(summary_stats.valid_loss) # green
        plt.show()

        min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
        print('Minimum loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_loss))
        min_loss = round(min_loss, 4)

        return min_loss
