import os
def results2conllu(base_dir, data_type, construction):
    dependencies = read_unbounded(construction, data_type)
    sent_file = os.path.join(base_dir, 'sents', data_type+'.txt') 
    pos_file = os.path.join(base_dir, 'predicted_pos', data_type+'.txt')
    stag_file = os.path.join(base_dir, 'predicted_stag', data_type+'.txt') 
    arc_file = os.path.join(base_dir, 'predicted_arcs', data_type+'.txt') 
    rel_file = os.path.join(base_dir, 'predicted_rels', data_type+'.txt') 
    sents = read_file(sent_file)
    pos = read_file(pos_file)
    stags = read_file(stag_file)
    arcs = read_file(arc_file)
    rels = read_file(rel_file)
    filename = os.path.join(base_dir, data_type+'.conllu')
    output_conllu(filename, sents, pos, stags, arcs, rels, dependencies)

def read_file(filename):
    all_tokens = []
    with open(filename) as fin:
        for line in fin:
            tokens = line.split()
            all_tokens.append(tokens)
    return all_tokens

def output_conllu(filename, sents, pos, stags, arcs, rels, dependencies):
    with open(filename, 'wt') as fwrite:
        for sent_idx in xrange(len(sents)):
            sent = sents[sent_idx]
            pos_sent = pos[sent_idx]
            stags_sent = stags[sent_idx]
            arcs_sent = arcs[sent_idx]
            rels_sent = rels[sent_idx]
            deps_sent = dependencies[sent_idx]
            colors = ['blue', 'red', 'green', 'orange']
            for dep, color in zip(deps_sent, colors):
                fwrite.write('# visual-style {} bgColor:{}\n'.format(dep[0], color))
                fwrite.write('# visual-style {} bgColor:{}\n'.format(dep[1], color))
            for token_idx in xrange(len(sent)):
                output_list = [str(token_idx+1), sent[token_idx], '_', stags_sent[token_idx], pos_sent[token_idx], '_', str(arcs_sent[token_idx]), rels_sent[token_idx], '_', '_']
                fwrite.write('\t'.join(output_list))
                fwrite.write('\n')
            fwrite.write('\n')

def read_unbounded(construction, data_type):
    with open('longrange-distrib/{}/{}.{}'.format(construction, data_type, construction)) as fin: 
        dependencies = []
        flag = 0 
        dependencies_sent = []
        for line in fin:
            words = line.split()
            if len(words) == 0:
                ## the end of a line
                dependencies.append(dependencies_sent)
                dependencies_sent = []
                flag = 0 
            else:
                flag += 1
            if flag >= 3:
                rel = words[0]
                parent_id = int(words[1].split('_')[-1])+1 ## start from 1. Zero is ROOT.
                child_id = int(words[2].split('_')[-1])+1
                ## convert it to our notation
                dependencies_sent.append((child_id, parent_id, rel))
        return dependencies

if __name__ == '__main__':
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    for construction in constructions:
        base_dir = '{}'.format(construction)
        data_type = 'dev'
        results2conllu(base_dir, data_type, construction)
        data_type = 'test'
        results2conllu(base_dir, data_type, construction)
