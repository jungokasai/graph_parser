import pickle

def main(file_name):
    with open('emnlp/test16_ss3_bs3.pkl') as fin:
        parses = pickle.load(fin)
    sents = read_stags('sents/{}.txt'.format(file_name))
    arcs = []
    rels = []
    for sent, parse in zip(sents, parses):
        parse = sorted(parse, key = lambda x: x[0])
        print(parse)
        new_parse = []
        i = 1
        for dep in parse:
            if i < dep[0]:
                for j in range(i, dep[0]):
                    new_parse.append((j, j, 'NA'))
            new_parse.append(dep)
            i = dep[0] + 1
        arcs_sent = []
        rels_sent = []
        for dep in new_parse:
            arcs_sent.append(dep[1])
            rels_sent.append(dep[2].replace('adj', 'ADJ').replace('root', 'ROOT'))
        arcs.append(arcs_sent)
        rels.append(rels_sent)
    output_arcs_rels(arcs, 'predicted_arcs/{}.txt'.format(file_name))
    output_arcs_rels(rels, 'predicted_rels/{}.txt'.format(file_name))
                
        
def read_stags(path_to_sents):
    ## create a list of lists
    stags = []
    with open(path_to_sents) as f_stags:
        for stags_sent in f_stags:
            stags_sent = stags_sent.split()
            stags.append(stags_sent)
    return stags

def output_arcs_rels(arcs, file_name):
    with open(file_name, 'wt') as fout:
        for arcs_sent in arcs:
            fout.write(' '.join(map(str, arcs_sent)))
            fout.write('\n')

if __name__ == '__main__':
    main('test_emnlp')
