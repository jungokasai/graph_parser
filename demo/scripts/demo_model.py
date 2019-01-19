from __future__ import print_function
import sys, os
sys.path.append('./')
from utils.models.demo import Demo_Parser
import tensorflow as tf
import sys, os, pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from argparse import ArgumentParser 

parser = ArgumentParser()
parser.add_argument("--tokenize",  help="tokenizes. False if it is already tokenized.", action="store_true", default=False)
parser.add_argument("--infile",  help="input sentence text file")
opts = parser.parse_args()
        
def demo_model():
    pretrained_model = 'demo/Pretrained_Parser/best_model'
    base_dir = 'demo/'
    g = tf.Graph()
    batch_size = 80
    if opts.tokenize:
        with open(opts.infile) as fin:
            sents = fin.read()
        sents = sent_tokenize(sents)
    else:
        with open(opts.infile) as fin:
            sents = fin.readlines()
    sents = map(word_tokenize_period, sents)
    with g.as_default():
        model = Demo_Parser(base_dir)
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as session: 
            session.run(tf.global_variables_initializer())
            saver.restore(session, pretrained_model)
            nb_baches = len(sents)//batch_size + 1
            for i in xrange(nb_baches):
                sents_batch = sents[i*batch_size:(i+1)*batch_size]
                results, arc_probs, rel_probs = model.run_on_sents(session, sents_batch)
                conllu = get_conllu(results, sents_batch)
                print(conllu)

def get_conllu(results, sents):
    start_idx = 0
    conllu = ''
    for sent_idx in xrange(len(sents)):
        sent = sents[sent_idx]
        for word_idx in xrange(len(sent)):
            output = []
            output.append(str(word_idx+1))
            output.append(sent[word_idx])
            output.append('_')
            output.append(results['stags'][start_idx])
            output.append(results['jk'][start_idx])
            output.append('_')
            output.append(results['arcs'][start_idx])
            output.append(results['rels'][start_idx])
            output.append('_')
            output.append('_')
            conllu += '\t'.join(output)
            conllu += '\n'
            start_idx += 1
        conllu += '\n'
    return conllu
def word_tokenize_period(sent):
    words = word_tokenize(sent)
    if words[-1] not in ['.', '?']:
        words.append('.')
    return words
    
def main():
    demo_model()
if __name__ == '__main__':
    main()
