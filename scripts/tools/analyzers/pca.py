import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import sys

def get_closest(result, K = 5): # vecs (a, b, c) a+b = c?
    scores = []
    for name, vector in name_to_vec.items():
        scores.append((name, vector.dot(result)/(norm(vector)*norm(result))))
    scores.sort(key=lambda x: -x[1])

    selected_scores = [score for score in scores if score[0] in top_K]
    return (scores[:K], selected_scores[:K])

#fhand = open('results/stag_embedding.txt')
#fhand = open('results/stag_embedding.txt')
#fhand = open('results/stag_embedding_srole.txt')
embedding_file = sys.argv[1]
fhand = open(embedding_file)

## set up 
stag_to_idx = {}
top_K = []
K = 300
idx = 0
vec_list = []
for line in fhand:
    words = line.split()
    stag_to_idx[words[0]] = idx
    vec_list.append(np.array(words[1:]).astype(float))
    idx+=1

fhand.close()
A = np.vstack(vec_list).T
u,s,v=np.linalg.svd(A,full_matrices=False)
A_prime = u[:, :2].transpose().dot(A)
A_reduced = A_prime.T
analogies = ['t27', 't81', 't220', 't112'
,'t27' ,'t81' ,'t347' ,'t141' 
,'t27' ,'t81' ,'t375' ,'t168'
,'t27' ,'t81' ,'t343' ,'t182'
,'t27' ,'t81' ,'t332' ,'t301'
,'t27' ,'t81' ,'t404' ,'t423'
,'t27' ,'t81' ,'t315' ,'t456'
,'t27' ,'t81' ,'t440' ,'t59'
,'t27' ,'t81' ,'t191' ,'t68'
,'t27' ,'t81' ,'t262' ,'t784'
,'t27' ,'t81' ,'t374' ,'t97'
,'t27' ,'t81' ,'t109' ,'t99', 't27' ,'t81' ,'t81' ,'t27']
#print(A_reduced[stag_to_idx['t182']] - A_reduced[stag_to_idx['t343']])
#print(A_reduced[stag_to_idx['t27']] - A_reduced[stag_to_idx['t81']])
#print(A_reduced[stag_to_idx['t99']] - A_reduced[stag_to_idx['t109']])
#print(A_reduced[stag_to_idx['t332']] - A_reduced[stag_to_idx['t301']])
trans = []
trans_stags = []
intrans = []
intrans_stags = []
for i in xrange(len(analogies)/2):
    if i % 2==0:
        continue
        #print(A_reduced[stag_to_idx[analogies[2*i+1]]]-A_reduced[stag_to_idx[analogies[2*i]]])
    else:
        intrans.append(A_reduced[stag_to_idx[analogies[2*i]]])
        intrans_stags.append(analogies[2*i])
        trans.append(A_reduced[stag_to_idx[analogies[2*i+1]]])
        trans_stags.append(analogies[2*i+1])
       
#subst = A_reduced[stag_to_idx['subst1']]

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111)
# red dashes, blue squares and green triangles
for j in xrange(len(intrans)):
    ax.plot([intrans[j][0], trans[j][0]], [intrans[j][1], trans[j][1]], 'rx--')
#ax.plot([0, subst[0]], [0, subst[1]], 'b->')
#ax.annotate("", xy=(subst[0], subst[1]), xycoords='data', xytext = (0, 0), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = 'b'))
#ax.annotate('substitution1', xy=(-0.20, -0.25))

for j in xrange(len(intrans)):
    #if trans_stags[j] == 't301':
    #    ax.annotate(intrans_stags[j], xy=(intrans[j][0]-0.1, intrans[j][1]-0.01))
    #    print('change')
    #elif intrans_stags[j] == 't81':
    #    ax.annotate(intrans_stags[j], xy=(intrans[j][0]-0.1, intrans[j][1]))
    #    print('change')
    #elif intrans_stags[j] == 't220':
    #    ax.annotate(intrans_stags[j], xy=(intrans[j][0]-0.01, intrans[j][1]))
    #    print('change')
    #else:
    ax.annotate(intrans_stags[j], xy=(intrans[j][0], intrans[j][1]))
for j in xrange(len(trans)):
    ax.annotate(trans_stags[j], xy=(trans[j][0], trans[j][1]))
plt.savefig('pca.png')
plt.close(fig)
