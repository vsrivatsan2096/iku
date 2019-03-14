# Reference
## https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

import numpy as np


def sentence_vectorizer(model, sentence):
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
