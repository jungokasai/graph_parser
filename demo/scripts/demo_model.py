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
def read_sents(sents_file):
    sents = []
    with open(sents_file) as fhand:
        for line in fhand:
            sent = line.split()
            sents.append(sent)
    return sents

def output_conllu(test_opts):
    sents = read_sents(os.path.join(test_opts.base_dir, 'sents', 'test.txt'))
    stags = read_sents(os.path.join(test_opts.base_dir, 'predicted_stag', 'test.txt'))
    pos = read_sents(os.path.join(test_opts.base_dir, 'predicted_pos', 'test.txt'))
    arcs = read_sents(os.path.join(test_opts.base_dir, 'predicted_arcs', 'test.txt'))
    rels = read_sents(os.path.join(test_opts.base_dir, 'predicted_rels', 'test.txt'))
    with open(os.path.join(test_opts.base_dir, 'predicted_conllu', 'test.conllu'), 'wt') as fout:
        for sent_idx in xrange(len(sents)):
            sent = sents[sent_idx]
            stags_sent = stags[sent_idx]
            pos_sent = pos[sent_idx]
            arcs_sent = arcs[sent_idx]
            rels_sent = rels[sent_idx]
            for word_idx in xrange(len(sent)):
                line = [str(word_idx+1)]
                line.append(sent[word_idx])
                line.append('_')
                line.append(stags_sent[word_idx])
                line.append(pos_sent[word_idx])
                line.append('_')
                line.append(arcs_sent[word_idx])
                line.append(rels_sent[word_idx])
                line.append('_')
                fout.write('\t'.join(line))
                fout.write('\n')
            fout.write('\n')
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
    demo_model(opts, test_opts)
    output_conllu(test_opts)
