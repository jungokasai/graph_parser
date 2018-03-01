"""
read in sentences and parses.

Usage: 
from read_data import *

"""
import pickle

# the parse triples in a pickle file
parser_versions = ['parse_stag_pos.pkl', 'parse_pos.pkl']

# quick notes: 
# used_dataset: a value in ['test', 'dev']
# parser_version_id: an index for parser_versions, file_prepend
def get_data(used_dataset='test', parser_version_id=0):
    from collections import defaultdict

    global parser_versions

    # possible parse names: 
    parser_version = parser_versions[parser_version_id]


    top_folder = "data/pete_Nov_27/"

    # datasets, with text-hypothesis distinctions
    t_folder = used_dataset + "/"


    # file prepends by version_id
    file_prepend = parser_version[:-4]
    if file_prepend != "":
        file_prepend += '_'
    # stags prepend
    if parser_version in ["Joint", "parse_stag_pos"]:
        stags_prepend = parser_version + '_'
    else:
        stags_prepend = "Pipeline_"
    # pos prepend
    pos_prepend = ''
    

    # filenames
    pos_file = "pos.txt"
    stags_file = "stags.txt"
    text_file = "texts.txt"


    # file paths
    sents_txt_t = top_folder + t_folder + text_file

    parses_pkl_t = top_folder + t_folder + parser_version

    pos_path_t = top_folder + t_folder + pos_prepend + pos_file

    stags_path_t = top_folder + t_folder + stags_prepend + stags_file



    """ read in sents for t"""
    # load original sentences for t
    with open(sents_txt_t, "r") as f:
        sents_t = f.read().split('\n')
        # remove the last ''
        if sents_t[-1] == '':
            sents_t.pop()

    print("read sents")
    

    """ read in parses for t"""
    # load TAG parses for t
    with open(parses_pkl_t, "rb") as f:
        parses_t = pickle.load(f)

    print("read parses")



    """ read in POS tags"""
    all_pos_t = open(pos_path_t, "r").read().split('\n')

    print("read all_pos")


    """ read in stags"""
    stags_t = open(stags_path_t, "r").read().split('\n')

    print("read stags")


    """ read in d6.treeproperties as a dictionary"""
    from get_treeprops import get_t2props_dict

    t2props_dict = get_t2props_dict()

    print("read t2props_dict")

    
    """ read in d6.clean2.f.str as a dictionary"""
    t2topsub_dict = {}



    """ wrap all in a dictionary """
    data = defaultdict(dict)
    data['sents']['t'] = sents_t
    data['sents']['h'] = {}

    data['all_pos']['t'] = all_pos_t
    data['all_pos']['h'] = {}

    data['stags']['t'] = stags_t
    data['stags']['h'] = {}

    data['parses']['t'] = parses_t
    data['parses']['h'] = {}

    data['t2props_dict'] = t2props_dict
    data['t2topsub_dict'] = t2topsub_dict
    
    data['version_info']['dataset'] = used_dataset
    data['version_info']['parser'] = parser_version
    data['version_info']['parser_id'] = parser_version_id

    
    return(data)


""" read stags for word 'and', based on stag gold train and dev sets """

def read_and_stags(fn = "data/supertag_info/and_stags"):
    with open(fn, 'r') as f:
        stags = f.read().split()
    return(stags)
