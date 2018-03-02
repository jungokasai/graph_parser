from read_data import read_data
from read_unbounded import read_unbounded
from transform import transform
from get_treeprops import get_t2props_dict, get_t2topsub_dict

def evaluate():
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    for construction in constructions:
        ## get predicted_dependencies and apply transformations
        predicted_dependencies = read_data(construction, 'dev')
        unbounded_dependencis = read_unbounded(construction, 'dev')
        #for predicted_dependencies_sent in predicted_dependencies:
        #    predicted_dependencies_sent = transform(predicted_dependencies_sent)
            
if __name__ == '__main__':
    evaluate()
