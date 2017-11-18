import numpy as np

n = 10
def output_mica_nbest(probs, idx2stag):
    ordered_idxes = (-probs).argsort(axis=-1)[:,:n]
    orders = ordered_idxes.argsort(axis=-1)
    #nbest_probs = probs[orders<n].reshape(-1, n)
    nb_tokens = probs.shape[0]
    #nb_stags = probs.shape[1]
    stags_file = '{}best_stags.txt'.format(n)
    probs_file = '{}best_probs.txt'.format(n)
    with open(stags_file, 'wt') as f_stags:
        with open(probs_file, 'wt') as f_probs:
            for token_idx in xrange(nb_tokens):
                f_stags.write(' '.join(map(lambda x: idx2stag[x], ordered_idxes[token_idx])))
                f_stags.write('\n')
                f_probs.write(' '.join(map(str, probs[token_idx][ordered_idxes[token_idx]])))
                f_probs.write('\n')

#if __name__ == '__main__':
#    nb_tokens = 5
#    nb_stags = 10
#    idx2stag = {}
#    filename = 'test'
#    probs = np.arange(nb_tokens*nb_stags).reshape(nb_tokens, nb_stags)
#    probs[3,3] = -3
#    output_mica_nbest(probs, filename, idx2stag)
