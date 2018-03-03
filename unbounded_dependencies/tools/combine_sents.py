### Do not use the raw file. It's corrupted and there is some punctuation disagreement.
def combine_raw(data_type):
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    ## in the order presented in the Rimell et al. 2009 paper
    with open('sents/{}.txt'.format(data_type), 'wt') as fout:
        for construction in constructions:
            with open('{}/sents/{}.txt'.format(construction, data_type), 'wt') as fout_construction:
                with open('longrange-distrib/{}/{}.{}'.format(construction, data_type, construction)) as fin: 
                    flag = 0 
                    for line in fin:
                        words = line.split()
                        if len(words) == 0:
                            flag = 0 
                        else:
                            flag += 1
                        if flag == 2:
                            line = ' '.join(words)
                            fout.write(line)
                            fout.write('\n')
                            fout_construction.write(line)
                            fout_construction.write('\n')

if __name__ == '__main__':
    combine_raw('dev')
    combine_raw('test')
