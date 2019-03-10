# Reference
## https://tfhub.dev/google/universal-sentence-encoder/2

import tensorflow as tf
import tensorflow_hub as hub


class UniversalSentenceEncoder2(object):
    def __init__(self, sentences1, sentences2):
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(self.module_url)
        self.sentences1 = sentences1
        self.sentences2 = sentences2

    def encode_sentences(self):
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sentences_embeddings_1 = session.run(self.embed(self.sentences1))
        
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sentences_embeddings_2 = session.run(self.embed(self.sentences2))
        
        return sentences_embeddings_1, sentences_embeddings_2
