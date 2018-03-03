import os
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

def total(nested_list):
    total = 0
    for x in nested_list:
        total += len(x)
    return total
if __name__ == '__main__':
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    #constructions = ['sbj_extract_rel_clause']
    data_type = 'dev'
    for construction in constructions:
        dependencies = read_unbounded(construction, data_type)
        print(dependencies[0])
        #print(dependencies)
        #print(total(dependencies))
        #print(len(dependencies))
