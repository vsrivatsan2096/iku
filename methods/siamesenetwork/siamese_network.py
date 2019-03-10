# Reference
## https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07

import numpy as np
from keras.models import Model
import keras.backend as backend
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Lambda, GRU, Dropout


class SiameseNetworkRNN(object):
    def __init__(self, sentences, sentences1, sentences2, is_duplicate):
        self.sentences = sentences
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.is_duplicate = is_duplicate
        self.vocabulary = None
        self.embedding_dim = None
        self.embeddings = None
        self.length = self.sentences1.shape[0]
        self.max_seq_length = None
    
    def get_model(self, model_name):
        model = None
        if model_name == 'google':
            model = KeyedVectors.load_word2vec_format("E:\Models\pre_trained\word2vec\google\google.300d.bin", binary=True)
        elif model_name == 'wiki':
            model = KeyedVectors.load_word2vec_format("E:\Models\pre_trained\glove\wiki\wiki.300d.txt", binary=False)
        elif model_name == 'common crawl':
            model = KeyedVectors.load_word2vec_format("E:\Models\pre_trained\glove\commoncrawl\common_crawl.300d.txt", binary=False)
        else:
            assert('Model does not exist')
        return model
    
    def do_preprocess(self):
        self.vocabulary = dict()
        inverse_vocabulary = ['<unk>']

        sentences_left = []
        for sentence in self.sentences1.tolist():
            temp_sentence = []
            for word in sentence.split():
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(inverse_vocabulary)
                    temp_sentence.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    temp_sentence.append(self.vocabulary[word])
            sentences_left.append(temp_sentence)

        sentences_right = []
        for sentence in self.sentences2.tolist():
            temp_sentence = []
            for word in sentence.split():
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(inverse_vocabulary)
                    temp_sentence.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    temp_sentence.append(self.vocabulary[word])
            sentences_right.append(temp_sentence)

        self.max_seq_length = 0
        for each in range(self.length):
            self.max_seq_length = max(self.max_seq_length, len(sentences_left[each]), len(sentences_right[each]))
        
        sentences_left = pad_sequences(sentences_left, maxlen=self.max_seq_length)
        sentences_right = pad_sequences(sentences_right, maxlen=self.max_seq_length)

        return sentences_left, sentences_right

    def get_embeddings(self):
        self.embedding_dim = 300
        embeddings = np.zeros((len(self.vocabulary) + 1, self.embedding_dim))
        embeddings[0] = 0
            
        model = self.get_model('google')
        for word, index in self.vocabulary.items():
            if word in model.vocab:
                embeddings[index] = model.word_vec(word)
        
        del model
            
        return embeddings
        
    def get_rnn_model(self):
        n_hidden1 = 512
        n_hidden2 = 384
        n_hidden3 = 256
        n_hidden4 = 128

        embeddings = self.get_embeddings()

        left_input = Input(shape=(self.max_seq_length, ), dtype='int32')
        right_input = Input(shape=(self.max_seq_length, ), dtype='int32')

        embedding_layer = Embedding(len(embeddings), self.embedding_dim, weights=[embeddings], 
                            input_length=self.max_seq_length, trainable=False)
        
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        shared_lstm1 = LSTM(n_hidden1, return_sequences=True)
        shared_dropout1 = Dropout(0.3)
        shared_gru1 = GRU(n_hidden2, return_sequences=True)
        shared_dropout2 = Dropout(0.4)
        shared_gru2 = GRU(n_hidden3, return_sequences=True)
        shared_dropout3 = Dropout(0.3)
        shared_lstm2 = LSTM(n_hidden4, return_sequences=False)

        left_lstm1 = shared_lstm1(encoded_left)
        left_dropout1 = shared_dropout1(left_lstm1)
        left_gru1 = shared_gru1(left_dropout1)
        left_dropout2 = shared_dropout2(left_gru1)
        left_gru2 = shared_gru2(left_dropout2)
        left_dropout3 = shared_dropout3(left_gru2)
        left_lstm2 = shared_lstm2(left_dropout3)

        right_lstm1 = shared_lstm1(encoded_right)
        right_dropout1 = shared_dropout1(right_lstm1)
        right_gru1 = shared_gru1(right_dropout1)
        right_dropout2 = shared_dropout2(right_gru1)
        right_gru2 = shared_gru2(right_dropout2)
        right_dropout3 = shared_dropout3(right_gru2)
        right_lstm2 = shared_lstm2(right_dropout3)

        manhattan_distance_for_lstm = Lambda(function=lambda x: backend.exp(-backend.sum(backend.abs(x[0]-x[1]), axis=1, keepdims=True)),
                                     output_shape=lambda x: (x[0][0], 1))([left_lstm2, right_lstm2])

        model = Model([left_input, right_input], manhattan_distance_for_lstm)

        return model
    
    def execute(self):
        model = self.get_rnn_model()

        dataset_left, dataset_right = self.do_preprocess()
        model.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy'])
        model.fit([dataset_left, dataset_right], self.is_duplicate, batch_size=128, epochs=128)
