import os
from read_data import read_data, read_stags
from read_unbounded import read_unbounded
from transform import transform
from get_treeprops import get_t2props_dict, get_t2topsub_dict

def evaluate(corpus_data_type, debug=False, input_data_type=None):
    if input_data_type is None:
        input_data_type = corpus_data_type
    tree_prop_file = 'd6.treeproperties'
    t2props_dict = get_t2props_dict(tree_prop_file)
    t2topsub_dict = get_t2topsub_dict(tree_prop_file)
    if debug:
        #constructions = ['sbj_embedded']
        #constructions = ['obj_qus']
        #constructions = ['obj_extract_red_rel']
        constructions = ['right_node_raising']
    else:
        constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    #constructions = ['obj_qus']
    all_total = 0
    all_correct = 0
    nb_constructions = 0
    total_scores = 0
    for construction in constructions:
        ## get predicted_dependencies and apply transformations
        result_dir = os.path.join(construction, 'results', 'test')
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        predicted_dependencies = read_data(construction, input_data_type)
        unbounded_dependencies = read_unbounded(construction, corpus_data_type)
        sents = read_stags(construction, input_data_type, 'sents')
        predicted_stags = read_stags(construction, input_data_type)
        predicted_pos = read_stags(construction, input_data_type, 'predicted_pos')
        #assert(len(predicted_dependencies) == len(unbounded_dependencies))
        total = 0
        correct = 0
        if debug:
            sent_idxes = [70] 
        else:
            sent_idxes = range(len(unbounded_dependencies))
        with open(os.path.join(result_dir, 'results.txt'), 'wt') as fout:
            for sent_idx in sent_idxes:
            #for sent_idx in [73]:
                sent = sents[sent_idx]
                ## TAG analysis
                predicted_dependencies_sent = predicted_dependencies[sent_idx]
                predicted_stags_sent = predicted_stags[sent_idx]
                predicted_pos_sent = predicted_pos[sent_idx]
                transformed_sent = transform(t2props_dict, t2topsub_dict, sent, predicted_dependencies_sent, predicted_stags_sent, predicted_pos_sent)
                #transformed_sent = predicted_dependencies_sent
                #print(transformed_sent)
                assert(len(sent) == len(predicted_stags_sent))
                unbounded_dependencies_sent = unbounded_dependencies[sent_idx]
                for dep_idx, dep in enumerate(unbounded_dependencies_sent):
                    total += 1
                    all_total += 1
                    if 'nsubj' == dep[2]:
                        new_dep = (dep[0], dep[1], '0')
                        if construction == 'sbj_embedded':
                            if (sent_idx, dep_idx) in [(77, 0), (42, 0)]:
                                new_dep = tuple([dep[0], dep[1], '1']) ## causative-inchoative
                    elif 'dobj' == dep[2]:
                        new_dep = tuple([dep[0], dep[1], '1'])
                        if construction == 'obj_qus':
                            if sent[0].lower() in ['where']:
                                new_dep = tuple([dep[0], dep[1], '-unk-'])
                    elif 'pobj' == dep[2]:
                        new_dep = tuple([dep[0], dep[1], '1'])
                    elif 'nsubjpass' in dep[2]:
                        new_dep = (dep[0], dep[1], '1')
                    elif 'advmod' in dep[2]:
                        if sent[dep[0]-1] == 'out':
                            new_dep = (dep[0], dep[1], 'ADJ')
                        else:
                            new_dep = (dep[0], dep[1], '-unk-')
                    elif 'prep' in dep[2]:
                        new_dep = (dep[0], dep[1], 'ADJ')
                    elif 'infmod' in dep[2]:
                        new_dep = (dep[0], dep[1], 'ADJ')
                    elif 'obj2' in dep[2]:
                        new_dep = (dep[0], dep[1], '1')
                    else:
                        new_dep = (dep[0], dep[1], 'ADJ')
                    if new_dep in transformed_sent:
                        correct += 1
                        all_correct += 1
                        success = 1
                    else:
                        success = 0
                    fout.write(' '.join([str(sent_idx), str(dep_idx), str(success)]))
                    fout.write('\n')
        print('Construction: {}'.format(construction))
        print('# total: {}'.format(total))
        print('# correct: {}'.format(correct))
        print('Accuracy: {}'.format(float(correct)/total))
        total_scores += float(correct)/total
        nb_constructions += 1
        #print(predicted_dependencies[0])
        #print(unbounded_dependencies[0])
        #for predicted_dependencies_sent in predicted_dependencies:
        #    predicted_dependencies_sent = transform(predicted_dependencies_sent)
    print('All constructions')
    print('# total: {}'.format(all_total))
    print('# correct: {}'.format(all_correct))
    print('Macro Accuracy: {}'.format(float(all_correct)/all_total))
    print('Overall Accuracy: {}'.format(float(total_scores)/nb_constructions))
if __name__ == '__main__':
    evaluate('test', False)
    #evaluate('test', True)
    #evaluate('test', False, 'test_emnlp')
