import subprocess
import os
import sys
import json 
import tools
from tools.converters.conllu2sents import conllu2sents
from tools.converters.sents2conllustag import output_conllu

def converter(config):
    data_types = config['data']['split'].keys()
    features = ['sents', 'predicted_pos', 'predicted_stag', 'arcs', 'rels']
    for feature in features:
        for data_type in data_types:
            input_file = os.path.join(config['data']['base_dir'], config['data']['split'][data_type])
            output_file = os.path.join(config['data']['base_dir'], feature, data_type+'.txt')
            if not os.path.isdir(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            if feature == 'sents':
                index = 1
            elif feature == 'predicted_pos':
                index = 4
            elif feature == 'predicted_stag':
                index = 10
            elif feature == 'rels':
                index = 7
            elif feature == 'arcs':
                index = 6
            conllu2sents(index, input_file, output_file)

def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict

def get_best_model(config):
    base_dir = config['data']['base_dir']
    with open(os.path.join(base_dir, 'checkpoint.txt')) as fhand:
        for line in fhand:
            best_model = line.split()[0]
    return best_model


def train_pos_tagger(config):
    base_dir = config['data']['base_dir']
    base_command = 'python graph_parser_main.py train --task POS_models --base_dir {}'.format(base_dir)
    train_data_info = ' --text_train {} --jk_train {} --tag_train {}'.format(os.path.join(base_dir, 'sents', 'train.txt'), os.path.join(base_dir, 'gold_pos', 'train.txt'), os.path.join(base_dir, 'gold_pos', 'train.txt'))
    dev_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', 'dev.txt'), os.path.join(base_dir, 'gold_pos', 'dev.txt'), os.path.join(base_dir, 'gold_pos', 'dev.txt'))
    model_config_dict = config['pos_parameters']
    model_config_info = ''
    for option, value in model_config_dict.items():
        model_config_info += ' --{} {}'.format(option, value)
    complete_command = base_command + train_data_info + dev_data_info + model_config_info
    subprocess.check_call(complete_command, shell=True)

def train_parser(config):
    base_dir = config['data']['base_dir']
    features = ['sents', 'predicted_pos', 'predicted_stag', 'arcs', 'rels']
    base_command = 'python graph_parser_main.py train --base_dir {}'.format(base_dir)
    train_data_dirs = map(lambda x: os.path.join(base_dir, x, 'train.txt'), features)
    train_data_info = ' --text_train {} --jk_train {} --tag_train {} --arc_train {} --rel_train {}'.format(*train_data_dirs)
    features = ['sents', 'predicted_pos', 'predicted_stag', 'arcs', 'rels', 'punc']
    dev_data_dirs = map(lambda x: os.path.join(base_dir, x, 'dev.txt'), features)
    dev_data_info = ' --text_test {} --jk_test {} --tag_test {} --arc_test {} --rel_test {} --punc_test'.format(*dev_data_dirs)
    model_config_dict = config['parser']
    model_config_info = ''
    for param_type in model_config_dict.keys():
        for option, value in model_config_dict[param_type].items():
            model_config_info += ' --{} {}'.format(option, value)
    complete_command = base_command + train_data_info + dev_data_info + model_config_info
#    complete_command += ' --max_epochs 1' ## for debugging
    subprocess.check_call(complete_command, shell=True)

def test_parser(config, best_model, data_types):
    base_dir = config['data']['base_dir'] 
    features = ['sents', 'predicted_pos', 'predicted_stag', 'arcs', 'rels', 'punc']
    base_command = 'python graph_parser_main.py test'
    model_info = ' --model {}'.format(best_model)
    for data_type in data_types:
        output_file = os.path.join(base_dir, 'predicted_arcs', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info = ' --save_tags {} --get_accuracy'.format(output_file)
        test_data_dirs = map(lambda x: os.path.join(base_dir, x, '{}.txt'.format(data_type)), features)
        test_data_info = ' --text_test {} --jk_test {} --tag_test {} --arc_test {} --rel_test {} --punc_test {}'.format(*test_data_dirs)
        complete_command = base_command + model_info + output_info + test_data_info
        subprocess.check_call(complete_command, shell=True)
        #output_conllu(output_file, os.path.join(base_dir, config['data']['split'][data_type]), os.path.join(base_dir, config['data']['split'][data_type]+'_stag'))
######### main ##########

if __name__ == '__main__':
    config_file = sys.argv[1]
    config_file = read_config(config_file)
    print('Convert conllu+stag file to sentences, pos, stag, arcs, and rels')
    converter(config_file)
#    print('Train POS-tagger')
#    train_pos_tagger(config_file)
#    print('Run Jackknife Training of POS tagging for Supertagging')
    print('Train Parser')
#    train_parser(config_file)
    print('Training is done. Run the parser.')
    best_model = get_best_model(config_file)
#    data_types = config_file['data']['split'].keys()
    data_types = ['dev']
    test_parser(config_file, best_model, data_types)
#    test_stagger(config_file, best_model, ['train'])
