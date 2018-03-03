def convert(data_type):
    filename = 'sents/{}.txt'.format(data_type)
    with open(filename) as fin:
        with open(filename+'new', 'wt') as fout:
            for line in fin:
                words = line.split()
                words = map(converter, words)
                fout.write(' '.join(words))
                fout.write('\n')
def converter(word):
    if word == '(':
        return '-LRB-'
    elif word == ')':
        return '-RRB-'
    else:
        return word
if __name__ == '__main__':
    convert('test')
