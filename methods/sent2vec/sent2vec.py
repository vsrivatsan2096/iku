# Reference
## https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


class Sentence2Vector(object):
    def __init__(self, sentences_1, sentences_2, model_name):
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2
        self.model = self.get_model(model_name)
    
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
    
    def sentence_vectorizer(self, model, sentence):
        vectors =[]
        num = 0
        for i in sentence.split():
            try:
                if num == 0:
                    vectors = model[i]
                else:
                    vectors = np.add(vectors, model[i])
                num += 1
            except:
                pass
        return np.array(vectors) / num

    def get_vectors(self):
        sent_vec1 = []
        for each in self.sentences_1:
            temp = self.sentence_vectorizer(self.model, each)
            if temp.shape[0] != 0:
                sent_vec1.append(temp)
            else:
                sent_vec1.append(np.zeros((300,)))
        sent_vec1 = np.asarray(sent_vec1)

        sent_vec2 = []
        for each in self.sentences_2:
            temp = self.sentence_vectorizer(self.model, each)
            if temp.shape[0] != 0:
                sent_vec1.append(temp)
            else:
                sent_vec1.append(np.zeros((300,)))
        sent_vec2 = np.asarray(sent_vec2)

        return sent_vec1, sent_vec2
