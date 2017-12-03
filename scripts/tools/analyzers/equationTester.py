## Syntactic Analogy Tests
## Written by R. Thomas McCoy

import numpy as np
from numpy.linalg import norm
import sys

def get_closest(result, K): # vecs (a, b, c) a+b = c?
    scores = []
    for name, vector in name_to_vec.items():
        scores.append((name, vector.dot(result)/(norm(vector)*norm(result))))
    scores.sort(key=lambda x: -x[1])

    selected_scores = [score for score in scores if score[0] in top_K]
    return (scores[:K], selected_scores[:K])



## set up 
if __name__ == '__main__':
    embedding_file = sys.argv[1]
    fhand = open(embedding_file)

    binaryCorrect = []
    binaryCorrectTopN = []
    positionCorrect = []
    positionCorrectTopN = []

    name_to_vec = {}
    top_K = []
    K = int(sys.argv[3])
    eqFile = sys.argv[2]
    for line in fhand:
        words = line.split()
        if len(top_K)<K:
            top_K.append(words[0])
        name_to_vec[words[0]] = np.array(words[1:]).astype(float)
    fhand.close()
    with open(eqFile) as fhand:
        numLine = 0
        for line in fhand:
    #	print(str(numLine))
            numLine += 1
            words = line.split()
            vecs = map(lambda x: name_to_vec[x], words[:-1])
            if len(words) == 3:
                result = vecs[0]+vecs[1]
                raw_scores, common_scores = get_closest(result, K)
               # print('{}+{}?={}'.format(*words))
               # print(raw_scores)
               # print('out of top {}'.format(K))
               # print(common_scores)
            if len(words) == 4:
                result = vecs[0]-vecs[1]+vecs[2]
                raw_scores, common_scores = get_closest(result, K)
               # print('{}-{}+{}?={}'.format(*words))
               # print(raw_scores)
               # print('out of top {}'.format(K))
               # print(common_scores)
            if raw_scores[0][0] == words[-1]:
                    binaryCorrect.append(1)
            else:
                    binaryCorrect.append(0)
            if common_scores[0][0] == words[-1]:
                    binaryCorrectTopN.append(1)
            else:
                    binaryCorrectTopN.append(0)
            position = 1
            for score in raw_scores:
                    if score[0] == words[-1]:
                            positionCorrect.append(position)
                            break
                    position += 1

            position = 1
            for score in common_scores:
                    if score[0] == words[-1]:
                            positionCorrectTopN.append(position)
                            break
                    position += 1

    print(numLine)
    print('The correct percentage is ' + str(sum(binaryCorrect) * 1.0 /len(binaryCorrect)))
    print('The correct percentage of the top ' + str(K) + ' is ' + str(sum(binaryCorrectTopN) * 1.0 /len(binaryCorrectTopN)))
    print('The average position of the correct answer is ' + str(sum(positionCorrect) * 1.0 /len(positionCorrect)))
    print('The average position of the correct answer of the top ' + str(K) + ' is ' + str(sum(positionCorrectTopN) * 1.0 /len(positionCorrectTopN)))

