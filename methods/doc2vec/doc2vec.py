
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
# numpy
import numpy

from sklearn.metrics.pairwise import cosine_similarity
# random
from random import shuffle

import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import pandas as pd
nltk.download('punkt')
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def preprocess(model,sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    stop_words= set(stopwords.words('english'))
    filtered_words = [w for w in tokens if not w in stop_words]
    #w = filter(lambda x: x in model.vocab, filtered_words)
    return " ".join(filtered_words)





def review_to_wordlist(review, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords.
    
    # Convert words to lower case and split them
    words = review.lower().split()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    
    words = review_text.split()
    
    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    review_text = " ".join(stemmed_words)
    
    # Return a list of words
    return(review_text)


def process_questions(model,question_list, questions, question_list_name):
# function to transform questions and display progress
    for question in questions:
        question_list.append(preprocess(model,question))
        if len(question_list) % 10000 == 0:
            progress = len(question_list)/len(df) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))


def performance_report(value,df, score_list):
    # the value (0-1) is the cosine similarity score to determine if a pair of questions
    # have the same meaning or not.
    scores = []
    for score in score_list:
        if score >= value:
            scores.append(1)
        else:
            scores.append(0)
    X=df.is_duplicate
    accuracy = accuracy_score(X, scores) * 100
    print("Accuracy score is {}%.".format(round(accuracy),1))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(X, scores))
    print()
    print("Classification Report:")
    print(classification_report(X, scores))

def main():
    model = Doc2Vec.load('./doc2vec.bin')
    df = pd.read_csv("questions.csv")
    df= df[15:20]
    questions1 = []     
    process_questions(model,questions1, df.question1, "questions1")
    print()
    questions2 = []     
    process_questions(model,questions2, df.question2, "questions2")


    # Split questions for computing similarity and determining the lengths of the questions.
    questions1_split = []
    for question in questions1:
        questions1_split.append(question.split())
        
    questions2_split = []
    for question in questions2:
        questions2_split.append(question.split())


    # Determine the length of questions to select more optimal parameters.
    lengths = []
    for i in range(len(questions1_split)):
        lengths.append(len(questions1_split[i]))
        lengths.append(len(questions2_split[i]))
    lengths = pd.DataFrame(lengths, columns=["count"])




    doc2vec_scores = []
    for i in range(len(questions1_split)):
        # n_similarity computes the cosine similarity in Doc2Vec
        score = model.n_similarity(questions1_split[i],questions2_split[i])
        doc2vec_scores.append(score)
        if i % 10000 == 0:
            progress = i/len(questions1_split) * 100
            print("{}% complete.".format(round(progress,2)))

    performance_report(0.92, df,doc2vec_scores)
    



if __name__ == "__main__":
    main()
