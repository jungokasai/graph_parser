from bilstm_stagger_model import run_model, run_model_test
import os
from argparse import ArgumentParser 
import pickle
import sys

parser = ArgumentParser()
subparsers = parser.add_subparsers(title='different modes', dest = 'mode', description='train or test')
train_parser=subparsers.add_parser('train', help='train parsing')

## train options
## data information
train_parser.add_argument("--base_dir", dest="base_dir", help="base directory for data")
train_parser.add_argument("--text_train", dest="text_train", help="text data for training")
train_parser.add_argument("--jk_train", dest="jk_train", help="jk data for training")
train_parser.add_argument("--tag_train", dest="tag_train", help="tag data for training")
train_parser.add_argument("--text_test", dest="text_test", help="text data for testing")
train_parser.add_argument("--jk_test", dest="jk_test", help="jk data for testing")
train_parser.add_argument("--tag_test", dest="tag_test", help="tag data for testing")

## model configuration
train_parser.add_argument("--lstm", dest="lstm", help="rnn architecutre", type = int, default = 1)
train_parser.add_argument("--capitalize", dest="cap", help="head capitalization", type = int, default = 1)
train_parser.add_argument("--num_indicator", dest="num", help="number indicator", type = int, default = 1)
train_parser.add_argument("--bidirectional", dest="bi", help="bidirectional LSTM", type = int, default = 1)
train_parser.add_argument("--max_epochs",  dest="max_epochs", help="max_epochs", type=int, default = 100)
train_parser.add_argument("--num_layers",  dest="num_layers", help="number of layers", type=int, default = 2)
train_parser.add_argument("--units", dest="units", help="hidden units size", type=int, default = 64)
train_parser.add_argument("-E", "--seed", dest="seed", help="set seed", type= int, default = 0)
train_parser.add_argument("--jk_dim", dest="jk_dim", help="jakcknife dimension", type=int, default = 5)
train_parser.add_argument("--embedding_dim", dest="embedding_dim", help="embedding dim", type=int, default = 100)
train_parser.add_argument("--early_stopping", dest="early_stopping", help="early stopping", type=int, default = 2)
train_parser.add_argument("--suffix_dim", dest="suffix_dim", help="suffix_dim", type=int, default = 10)
train_parser.add_argument("--lrate", dest="lrate", help="lrate", type=float, default = 0.01)
train_parser.add_argument("--dropout_p", dest="dropout_p", help="keep fraction", type=float, default = 1.0)
train_parser.add_argument("--hidden_p", dest="hidden_p", help="keep fraction of hidden units", type=float, default = 1.0)
train_parser.add_argument("--input_p", dest="input_dp", help="keep fraction for input", type=float, default = 1.0)
train_parser.add_argument("--task", dest="task", help="supertagging or tagging", default='Super_models', choices=['POS_models', 'Super_models'])

## test options
test_parser=subparsers.add_parser('test', help='test tagging')
## data information
test_parser.add_argument("--base_dir", dest="base_dir", help="base directory for data")
test_parser.add_argument("--text_test", dest="text_test", help="text data for testing")
test_parser.add_argument("--jk_test", dest="jk_test", help="jk data for testing")
test_parser.add_argument("--tag_test", dest="tag_test", help="tag data for testing")
## Model Information
test_parser.add_argument("--model", dest="modelname", help="model name")
## Output Options
test_parser.add_argument("--get_accuracy",  help="compute tag accuracy", action="store_true", default=False)
test_parser.add_argument("--save_tags", dest="save_tags", help="save 1-best tags")

opts = parser.parse_args()

if opts.mode == "train":
    model_dir = '{}/cap{}_num{}_bi{}_numlayers{}_embeddim{}_seed{}_units{}_dropout{}_inputdp{}_hp{}_suffix{}_jkdim{}'.format(opts.task, opts.cap, opts.num, opts.bi, opts.num_layers, opts.embedding_dim, opts.seed, opts.units, opts.dropout_p, opts.input_dp, opts.hidden_p, opts.suffix_dim, opts.jk_dim)
    opts.model_dir = os.path.join(opts.base_dir, model_dir)
    print('Model Dirctory: {}'.format(opts.model_dir))
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)
    with open(os.path.join(opts.model_dir, 'options.pkl'), 'wt') as fhand:
        pickle.dump(opts, fhand)
    run_model(opts)
    
if opts.mode == "test":
    with open(os.path.join(os.path.dirname(opts.modelname), 'options.pkl')) as foptions:
        options=pickle.load(foptions)
    run_model_test(options, opts)
