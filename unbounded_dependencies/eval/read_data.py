import os
def read_data(base_dir, data_type):
    ## create a list of lists of tuples
    dependencies = []
    with open(os.path.join(base_dir, 'predicted_arcs', data_type+'.txt')) as f_arcs:
        with open(os.path.join(base_dir, 'predicted_rels', data_type+'.txt')) as f_rels:
            for arcs_sent, rels_sent in zip(f_arcs, f_rels):
                arcs_sent = map(int, arcs_sent.split())
                rels_sent = rels_sent.split()
                dependencies_sent = zip(xrange(1, len(arcs_sent)+1), arcs_sent, rels_sent)
                dependencies.append(dependencies_sent)
    return dependencies


if __name__ == '__main__':
    read_data('obj_extract_rel_clause', 'dev')
