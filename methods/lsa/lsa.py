# Reference
## http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



class LatentSemanticAnalysis(object):
    def __init__(self, sentences1, sentences2):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.sentences = np.append(sentences1, sentences2, axis=0)

    def get_count_vectorizer(self):
        count_vectorizer = CountVectorizer(input='content', encoding='utf-8', decode_error='strict', 
                    strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, 
                    stop_words=None, ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, 
                    max_features=None, vocabulary=None, binary=False)
        return count_vectorizer
    
    def get_tf_idf_transformer(self):
        tf_idf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
        return tf_idf_transformer

    def get_svd(self):
        svd = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
        return svd
    
    def get_lsa(self):
        count_vectorizer = self.get_count_vectorizer()
        tf_idf_transformer = self.get_tf_idf_transformer()
        svd = self.get_svd()
        lsa = Pipeline([('count vectorizer', count_vectorizer), 
                        ('tfidf', tf_idf_transformer), 
                        ('svd', svd)])
        return lsa
    
    def fit_transform(self):
        lsa = self.get_lsa()
        lsa.fit(self.sentences)
        lsa_sentences_1 = lsa.transform(self.sentences1)
        lsa_sentences_2 = lsa.transform(self.sentences2)
        return lsa_sentences_1, lsa_sentences_2
