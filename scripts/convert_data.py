import os
import sys
import tools
from tools.converters.conllu2sents import conllu2sents

def converter(config):
    data_types = config['data']['split'].keys()
    features = ['sents', 'arcs', 'rels', 'pos', 'cpos', 'stag']
    for feature in features:
        for data_type in data_types:
            input_file = os.path.join(config['data']['base_dir'], config['data']['split'][data_type])
            predicted_list = config['data']['split'][data_type].split('_')
            if feature in predicted_list:
            else:
            output_file = os.path.join(config['data']['base_dir'], feature, data_type+'.txt')
            if not os.path.isdir(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            if feature == 'sents':
                index = 1
            elif feature == 'predicted_pos':
                index = 4
            elif feature == 'predicted_cpos':
                index = 3
            elif feature == 'predicted_stag':
                index = 10
            elif feature == 'rels':
                index = 7
            elif feature == 'arcs':
                index = 6
            conllu2sents(index, input_file, output_file) 
######### main ##########

if __name__ == '__main__':
    print('The input file name tells you which is predicted')
    converter(config_file)
