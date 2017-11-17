import sys
sys.path.append('./')
from utils.tarjan import Tarjan
from tools.converters.sents2conllustag import read_sents
import numpy as np

def check_cycles(file_name):
    arcs = read_sents(file_name)
    nb_sents = len(arcs)
    cycle_sents = []
    cycle_sents_conllu = []  ## keep conllu idx for reference
    conllu_idx = 1
    nb_cycles = 0
    nb_tokens = 0
    for sent_idx in xrange(nb_sents):
        arcs_sent = map(int, arcs[sent_idx])
        nb_tokens += len(arcs_sent)
        arcs_sent = np.hstack([0] + arcs_sent)
        graph = Tarjan(arcs_sent, np.arange(1, len(arcs_sent))) 
	cycles = graph._SCCs
	iscycle = False
	for cycle in cycles:
	    if len(cycle) > 1:
		iscycle = True
                nb_cycles += 1
        if iscycle:
            cycle_sents.append(sent_idx)
            cycle_sents_conllu.append(conllu_idx)
        conllu_idx += len(arcs_sent)
    return cycle_sents, cycle_sents_conllu, nb_cycles, nb_tokens

def check_root(file_name):
    arcs = read_sents(file_name)
    nb_sents = len(arcs)
    wrong_root_sents = []
    wrong_root_sents_conllu = []
    conllu_idx = 1
    nb_wrong = 0
    for sent_idx in xrange(nb_sents):
        arcs_sent = np.array(map(int, arcs[sent_idx]))
        nb_roots = np.sum(arcs_sent == 0)
        nb_wrong += abs(nb_roots-1)
        if nb_roots != 1:
            wrong_root_sents.append(sent_idx)
            wrong_root_sents_conllu.append(conllu_idx)
        conllu_idx += len(arcs_sent)+1
    return wrong_root_sents, wrong_root_sents_conllu, nb_wrong

def main(file_name):
    cycle_sents, cycle_sents_conllu, nb_cycles, nb_tokens = check_cycles(file_name)
    wrong_root_sents, wrong_root_sents_conllu, nb_wrong = check_root(file_name)
    print(zip(cycle_sents, cycle_sents_conllu))
#    print(len(cycle_sents))
##    print(len(wrong_root_sents))
##    print(len(wrong_root_sents))
#    print(nb_cycles)
#    print(nb_tokens)
#    print(wrong_root_sents)
#    print(wrong_root_sents_conllu)
#    print(len(wrong_root_sents))
#    print(nb_wrong)

if __name__ == '__main__':
    file_name = 'data/tag_wsj/predicted_arcs/dev.txt'
    main(file_name)
    file_name = 'data/tag_wsj/predicted_arcs_greedy/dev.txt'
    main(file_name)
