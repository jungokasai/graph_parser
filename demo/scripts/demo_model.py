from __future__ import print_function
import tensorflow as tf
import sys, os, pickle
sys.path.insert(0, os.path.abspath('.'))
import utils
from nltk.tokenize import sent_tokenize, word_tokenize
        
def demo_model(opts, test_opts):
    g = tf.Graph()
    with g.as_default():
        Model = getattr(utils, opts.model) ## Choose model type
        model = Model(opts, test_opts)
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as session: 
            session.run(tf.global_variables_initializer())
            saver.restore(session, test_opts.modelname)
            scores = model.run_epoch(session, True)
def output_conllu(test_opts):
    pass
def output_sents(sents, test_opts):
    sents = sent_tokenize(sents)
    sents = map(word_tokenize, sents)
    with open(os.path.join(test_opts.base_dir, 'sents', 'test.txt'), 'wt') as fout:
        for sent in sents:
            fout.write(' '.join(sent))
            fout.write('\n')
if __name__ == '__main__':
    sents = 'TAG is the best formalism. We should all learn it.'
    with open('demo/configs/config_demo.pkl') as fin:
        opts = pickle.load(fin)
    with open('demo/configs/config_demo_test.pkl') as fin:
        test_opts = pickle.load(fin)
    output_sents(sents, test_opts)
#    demo_model(opts, test_opts)
    output_conllu(test_opts)
