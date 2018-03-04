from read_data import read_data, read_stags
from read_unbounded import read_unbounded
from transform import transform
from get_treeprops import get_t2props_dict, get_t2topsub_dict

def evaluate(data_type):
    tree_prop_file = 'd6.treeproperties'
    t2props_dict = get_t2props_dict(tree_prop_file)
    t2topsub_dict = get_t2topsub_dict(tree_prop_file)
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    all_total = 0
    all_correct = 0
    for construction in constructions:
        ## get predicted_dependencies and apply transformations
        predicted_dependencies = read_data(construction, data_type)
        unbounded_dependencies = read_unbounded(construction, data_type)
        sents = read_stags(construction, data_type, 'sents')
        predicted_stags = read_stags(construction, data_type)
        predicted_pos = read_stags(construction, data_type, 'predicted_pos')
        #assert(len(predicted_dependencies) == len(unbounded_dependencies))
        total = 0
        correct = 0
        for sent_idx in xrange(len(unbounded_dependencies)):
            sent = sents[sent_idx]
            ## TAG analysis
            predicted_dependencies_sent = predicted_dependencies[sent_idx]
            predicted_stags_sent = predicted_stags[sent_idx]
            predicted_pos_sent = predicted_stags[sent_idx]
            transformed_sent = transform(t2props_dict, t2topsub_dict, sent, predicted_dependencies_sent, predicted_stags_sent)
            print(sent_idx)
            print(construction)
            print(sent)
            print(predicted_stags_sent)
            assert(len(sent) == len(predicted_stags_sent))
            unbounded_dependencies_sent = unbounded_dependencies[sent_idx]
            for dep in unbounded_dependencies_sent:
                total += 1
                all_total += 1
                if 'subj' in dep[2]:
                    new_dep = tuple([dep[0], dep[1], '0'])
                if 'obj' in dep[2]:
                    new_dep = tuple([dep[0], dep[1], '1'])
                if new_dep in transformed_sent:
                    correct += 1
                    all_correct += 1
        print('Construction: {}'.format(construction))
        print('# total: {}'.format(total))
        print('# correct: {}'.format(correct))
        print('Accuracy: {}'.format(float(correct)/total))
        #print(predicted_dependencies[0])
        #print(unbounded_dependencies[0])
        #for predicted_dependencies_sent in predicted_dependencies:
        #    predicted_dependencies_sent = transform(predicted_dependencies_sent)
    print('All constructions')
    print('# total: {}'.format(all_total))
    print('# correct: {}'.format(all_correct))
    print('Accuracy: {}'.format(float(all_correct)/all_total))
if __name__ == '__main__':
    evaluate('test')
    evaluate('dev')
