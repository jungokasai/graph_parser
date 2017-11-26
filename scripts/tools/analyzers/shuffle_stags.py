import sys
import numpy as np

def get_stags(file_name):
    stags = []
    with open(file_name) as fhand:
        for line in fhand:
            stags_sent = line.split()
            stags.extend(stags_sent)
    return stags

def shuffle(stags):
    np.random.seed(0)
    np.random.shuffle(stags)
    return stags
    

def output_stags(stags, gold, filename):
    token_idx = 0
    with open(gold) as fin:
        with open(filename, 'wt') as fout:
            for line in fin:
                start_idx = token_idx
                nb_tokens = len(line.split())
                token_idx += nb_tokens
                fout.write(' '.join(stags[start_idx:token_idx]))
                fout.write('\n')

if __name__ == '__main__':
    gold = sys.argv[1]
    gold_stags = get_stags(gold)
    shuffled_stags = shuffle(gold_stags)
    filename = 'shuffled_stags.txt'
    output_stags(shuffled_stags, gold, filename)
