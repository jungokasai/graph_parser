import os
import subprocess

def main(base_dir):
    dot_files = os.listdir(base_dir)
    for dot_file in dot_files:
        if dot_file[-2:]  == 'gv':
            convert(os.path.join(base_dir, dot_file))
def convert(dot_file):
    new_file = dot_file[:-2] + 'ps'
    final_file = dot_file[:-2] + 'pdf'
    command = 'dot -Tps {} -o {}'.format(dot_file, new_file)
    subprocess.check_call(command, shell=True)
    command = 'convert --density 500 {} {}'.format(new_file, final_file)
    subprocess.check_call(command, shell=True)
if __name__ == '__main__':
    constructions = ['obj_extract_rel_clause', 'obj_extract_red_rel', 'sbj_extract_rel_clause', 'obj_free_rels', 'obj_qus', 'right_node_raising', 'sbj_embedded']
    for construction in constructions:
        base_dir = os.path.join(construction, 'images', 'test')
        main(base_dir)
