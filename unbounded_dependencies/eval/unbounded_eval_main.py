from read_data import read_data, read_stags
from read_unbounded import read_unbounded
from transform import transform
from get_treeprops import get_t2props_dict, get_t2topsub_dict

def evaluate(data_type):
    tree_prop_file = 'd6.treeproperties'
    t2props_dict = get_t2props_dict(tree_prop_file)
    t2topsub_dict = get_t2topsub_dict(tree_prop_file)
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    for construction in constructions:
        ## get predicted_dependencies and apply transformations
        predicted_dependencies = read_data(construction, data_type)
        unbounded_dependencies = read_unbounded(construction, data_type)
        sents = read_stags(construction, data_type, 'sents')
        predicted_stags = read_stags(construction, data_type)
        predicted_pos = read_stags(construction, data_type, 'predicted_pos')
        #assert(len(predicted_dependencies) == len(unbounded_dependencies))
        for sent_idx in xrange(len(unbounded_dependencies)):
            sent = sents[sent_idx]
            ## TAG analysis
            predicted_dependencies_sent = predicted_dependencies[sent_idx]
            predicted_stags_sent = predicted_stags[sent_idx]
            predicted_pos_sent = predicted_stags[sent_idx]
            transformed_sent = transform(t2props_dict, t2topsub_dict, sent, predicted_dependencies_sent, predicted_stags_sent)
            if sent_idx == 4:
                print(transformed_sent)
            unbounded_dependencies_sent = unbounded_dependencies[sent_idx]
        #print(predicted_dependencies[0])
        #print(unbounded_dependencies[0])
        #for predicted_dependencies_sent in predicted_dependencies:
        #    predicted_dependencies_sent = transform(predicted_dependencies_sent)
            
if __name__ == '__main__':
    evaluate('test')
