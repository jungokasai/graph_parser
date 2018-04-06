from nltk.parse.dependencygraph import DependencyGraph
import os
from read_data import read_data, read_stags
from read_unbounded import read_unbounded
from transform import transform
from get_treeprops import get_t2props_dict, get_t2topsub_dict


def results2conllu(base_dir, data_type, construction, new_edges):
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
    output_dir = os.path.join(base_dir, 'images', data_type)
    result_file = os.path.join(base_dir, 'results/test/results.txt')
    output_conllu(filename, sents, pos, stags, arcs, rels, dependencies, new_edges, output_dir, result_file)

def read_file(filename):
    all_tokens = []
    with open(filename) as fin:
        for line in fin:
            tokens = line.split()
            all_tokens.append(tokens)
    return all_tokens

def output_conllu(filename, sents, pos, stags, arcs, rels, dependencies, new_edges, output_dir, result_file):
    scores = {}
    with open(result_file) as fin:
        for line in fin:
            line = line.split()
            scores[(int(line[0]), int(line[1]))] = int(line[2])
    tree_prop_file = 'd6.treeproperties'
    t2props_dict = get_t2props_dict(tree_prop_file)
    t2topsub_dict = get_t2topsub_dict(tree_prop_file)
    for sent_idx in xrange(len(sents)):
        deps_sent = dependencies[sent_idx]
        for dep_idx, dep in enumerate(deps_sent):
            unbounded_dep = dep
            start = min(int(dep[0]), int(dep[1]))-1
            end = max(int(dep[0]), int(dep[1]))+1
            conllu = ''
            sent = sents[sent_idx]
            pos_sent = pos[sent_idx]
            stags_sent = stags[sent_idx]
            arcs_sent = arcs[sent_idx]
            rels_sent = rels[sent_idx]
            token_idx = int(dep[1])
            output_list = [str(token_idx), sent[token_idx-1]+'_'+stags_sent[token_idx-1], '_', stags_sent[token_idx-1], pos_sent[token_idx-1], '_', str(dep[0]), dep[2], '_', '_']
            conllu += '\t'.join(output_list)
            conllu += '\n'
            for token_idx in xrange(len(sent)):
                if token_idx >= start and token_idx <= end:
                    #if  arcs_sent[token_idx] >= start and arcs_sent[token_idx] <= end:
                    output_list = [str(token_idx+1), sent[token_idx]+'_'+stags_sent[token_idx], '_', stags_sent[token_idx], pos_sent[token_idx], '_', str(arcs_sent[token_idx]), rels_sent[token_idx], '_', '_']
                    conllu += '\t'.join(output_list)
                    conllu += '\n'
            for new_idx, dep in enumerate(new_edges[sent_idx]):
                if dep[0] >= start and dep[0] <= end:
                    #if  dep[1] >= start and dep[1] <= end:
                        token_idx = int(dep[0])
                        output_list = [str(token_idx), sent[token_idx-1]+'_'+stags_sent[token_idx-1], '_', stags_sent[token_idx-1], pos_sent[token_idx-1], '_', str(dep[1]), dep[2], '_', '_']
                        conllu += '\t'.join(output_list)
                        conllu += '\n'
            graph = DependencyGraph(conllu)
            if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
            output_file = os.path.join(output_dir, 'sent{}_dep{}_correct{}.gv'.format(sent_idx, dep_idx, scores[(sent_idx, dep_idx)]))
            dot_string = graph.to_dot()
            ## add colors
            new_dot_string = ''
            new_lines = ['{} -> {} [label="{}"]'.format(dep[1], dep[0], dep[2]) for dep in new_edges[sent_idx]]
            for line in dot_string.split('\n'):
                line = line.strip()
                if line == '{} -> {} [label="{}"]'.format(unbounded_dep[0], unbounded_dep[1], unbounded_dep[2]):
                    line = '{} -> {} [label="{}", color="red"]'.format(unbounded_dep[1], unbounded_dep[0], unbounded_dep[2])
                elif line in new_lines:
                    line = line[:-1] + ', color="blue"]' 
                new_dot_string += line
                new_dot_string += '\n'
            with open(output_file, 'wt') as fout:
                fout.write(new_dot_string)
            

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

def get_new_edges(data_type, construction):
    tree_prop_file = 'd6.treeproperties'
    t2props_dict = get_t2props_dict(tree_prop_file)
    t2topsub_dict = get_t2topsub_dict(tree_prop_file)
    ## get predicted_dependencies and apply transformations
    predicted_dependencies = read_data(construction, data_type)
    unbounded_dependencies = read_unbounded(construction, data_type)
    sents = read_stags(construction, data_type, 'sents')
    predicted_stags = read_stags(construction, data_type)
    predicted_pos = read_stags(construction, data_type, 'predicted_pos')
    new_edges =[]
    for sent_idx in xrange(len(unbounded_dependencies)):
    #for sent_idx in [0]:
        sent = sents[sent_idx]
        ## TAG analysis
        predicted_dependencies_sent = predicted_dependencies[sent_idx]
        predicted_stags_sent = predicted_stags[sent_idx]
        predicted_pos_sent = predicted_pos[sent_idx]
        transformed_sent = transform(t2props_dict, t2topsub_dict, sent, predicted_dependencies_sent, predicted_stags_sent, predicted_pos_sent)
        new_edges_sent = list(set(transformed_sent)-set(predicted_dependencies_sent))
        new_edges_sent = [x for x in new_edges_sent if x[0] != x[1]]
        #print(new_edges_sent)
        new_edges.append(new_edges_sent)
    return new_edges
if __name__ == '__main__':
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    #constructions = ['sbj_extract_rel_clause']
    for construction in constructions:
        base_dir = '{}'.format(construction)
        #data_type = 'dev'
        #new_edges = get_new_edges(data_type, construction)
        #results2conllu(base_dir, data_type, construction, new_edges)
        data_type = 'test'
        new_edges = get_new_edges(data_type, construction)
        results2conllu(base_dir, data_type, construction, new_edges)
