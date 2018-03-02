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
                dependencies_sent.append(tuple(words))
        return dependencies

def total(nested_list):
    total = 0
    for x in nested_list:
        total += len(x)
    return total
if __name__ == '__main__':
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    data_type = 'dev'
    for construction in constructions:
        dependencies = read_unbounded(construction, data_type)
        print(total(dependencies))
