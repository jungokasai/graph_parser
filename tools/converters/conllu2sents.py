#with open('data/conllu/en-ud-dev.conll16') as fhand:
import sys

#idx = 1 #word
def conllu2sents(idx, input_dir, output_dir):
    #idx = 4 #pos
    #idx = 10 #stag
    #idx =  7#relation
#    idx = 7 #parent
#    count = 1
    #with open('data/conllu/en-ud-dev.conll16') as fhand: 
    with open(input_dir) as fhand: 
    #with open('train.txt') as fhand: 
        words = []
        words_sent = []
        for line in fhand:
            tokens = line.split()
            if len(tokens) == 0: 
                if len(words_sent)>0: ## avoid multiple empty lines that sometimes happen
                    #if idx in [1, 4, 10]: ## words, pos, or stags
                    #    words.append(['-ROOT-'] + words_sent)
                    #else:
                    words.append(words_sent)
                    words_sent = []
                continue
            word = tokens[idx]
            words_sent.append(word)

    #with open('dev.txt', 'wt') as fhand:
    with open(output_dir, 'wt') as fhand:
        for words_sent in words:
            fhand.write(' '.join(words_sent))
            fhand.write('\n')
#idx = 4 #pos
#idx = 10 #stag
#idx =  7#relation
#idx = 6 #parent
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: python tools/conllu2sents.py ...')
    idx = int(sys.argv[1])
    conllu_file = sys.argv[2]
    output_file = sys.argv[2]

#    conllu2sents(idx, 'data/conllu/en-ud-train_parsey.txt', 'train.txt') 
    conllu2sents(idx, conllu_file, output_file) 
#    conllu2sents(4, 'data/conllu/en-ud-dev_parsey.txt', 'dev.txt') 
#    conllu2sents(4, 'dev', 'dev.txt') 
#    conllu2sents(4, 'train_long', 'train.txt') 
