from math import *


class Similarity(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def _euclidean_distance(x, y):
        return sqrt(sum(pow(x-y,2) for a, b in zip(x, y)));

    @staticmethod
    def _manhattan_distance(x, y):
        return sum(abs(a-b) for a,b in zip(x,y));

    @staticmethod
    def __square_rooted(x):
        return round(sqrt(sum([a*a for a in x])),3)

    @staticmethod
    def _cosine_similarity(x,y):
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = square_rooted(x)*square_rooted(y)
        return round(numerator/float(denominator),3)

    @staticmethod
    def _jaccard_similarity(x, y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

    @staticmethod
    def _similarity_helper(x, y):
        similarity_scores = []
        similarity_scores.append(Similarity._euclidean_distance(x, y))
        similarity_scores.append(Similarity._manhattan_distance(x, y))
        similarity_scores.append(Similarity._cosine_similarity(x, y))
        similarity_scores.append(Similarity._jaccard_similarity(x, y))
        return sum(similarity_scores)/len(similarity_scores)

    def get_similarity_score(self):
        similarity_score = []
        for x, y in zip(self.x, self.y):
            score = self._similarity_helper(x, y)
            if not(isnan(score)):
                similarity_score.append(score)
            else:
                similarity_score.append(0)
        return similarity_score
