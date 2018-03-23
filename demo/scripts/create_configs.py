import subprocess
import os
import sys
import json 
from argparse import ArgumentParser 

parser = ArgumentParser()
parser.add_argument('config_file', metavar='N', help='an integer for the accumulator')
opts = parser.parse_args()


def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict, os.path.dirname(os.path.abspath(config_file))

def test_parser(config, best_model, data_types, base_dir):
    ## Setting up
    #base_dir = config['data']['base_dir'] 
    base_command = 'python demo/scripts/graph_parser_main.py test --base_dir {}'.format(base_dir)
    base_command += ' --pretrained'
    model_type = config['parser']['model_options']['model']
    features = ['sents', 'sents', 'sents', 'sents', 'sents', 'punc']
    model_info = ' --model {}'.format(os.path.join(base_dir, best_model))
    for data_type in data_types:
        output_file = os.path.join(base_dir, 'predicted_stag', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info = ' --predicted_stags_file {}'.format(output_file)
        output_file = os.path.join(base_dir, 'predicted_pos', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info += ' --predicted_pos_file {}'.format(output_file)
        output_file = os.path.join(base_dir, 'predicted_arcs', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info += ' --predicted_arcs_file {}'.format(output_file)
        output_file = os.path.join(base_dir, 'predicted_rels', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info += ' --predicted_rels_file {}'.format(output_file)
        output_file = os.path.join(base_dir, 'predicted_arcs_greedy', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        model_config_dict = config['parser']
        for param_type in model_config_dict.keys():
            if param_type == 'scores':
                for option, value in model_config_dict[param_type].items():
                    model_config_info = ' --{} {}'.format(option, value)
        output_info += ' --predicted_arcs_file_greedy {}'.format(output_file)
        output_file = os.path.join(base_dir, 'predicted_rels_greedy', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info += ' --predicted_rels_file_greedy {}'.format(output_file)
        test_data_dirs = map(lambda x: os.path.join(base_dir, x, '{}.txt'.format(data_type)), features)
        test_data_info = ' --text_test {} --jk_test {} --tag_test {} --arc_test {} --rel_test {} --punc_test {}'.format(*test_data_dirs)
        complete_command = base_command + model_info + output_info + test_data_info + model_config_info
        subprocess.check_call(complete_command, shell=True)
######### main ##########

if __name__ == '__main__':
    config_file = opts.config_file
    config_file, base_dir = read_config(config_file)
    best_model = 'Pretrained_Parser/best_model'
    data_types = ['test']
    test_parser(config_file, best_model, data_types, base_dir)
