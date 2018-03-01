def combine_raw(data_type):
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    ## in the order presented in the Rimell et al. 2009 paper
    with open('sents/{}.txt'.format(data_type), 'wt') as fout:
        for construction in constructions:
            with open('longrange-distrib/{}/{}.raw.{}'.format(construction, data_type, construction)) as fin:
                for line in fin:
                    fout.write(line)

if __name__ == '__main__':
    combine_raw('dev')
    combine_raw('test')
