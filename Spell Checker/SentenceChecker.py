import json
from WordChecker import *

with open("words_dictionary.json") as words_dictionary_file:
    word_dict = json.load(words_dictionary_file)

def sentences(sentence):
    sentence = sentence.lower().split()
    combination_sentences = []
    combination_probabilities = []
    meta_data = {}
    for each in sentence:
        if not word_dict.get(each, None):
            possible_words = candidates(each)
            probabilities = []
            for each_word in possible_words:
                probabilities.append(P(each_word))
            meta_data[each] = [list(possible_words), list(probabilities)]
    for i in range(len(sentence)):
        if meta_data.get(sentence[i], None):
            for each in meta_data[sentence[i]][0]:
                combination_sentences.append(" ".join(sentence[:i]) + " " + each + " ".join(sentence[i+1:]))
    return combination_sentences


print(sentences("Every thing comes with a pric"))
