from __future__ import print_function
import sys, os
sys.path.append('./')
from utils.models.demo import Demo_Parser
import tensorflow as tf
import sys, os, pickle
from nltk.tokenize import sent_tokenize, word_tokenize
        
def demo_model():
    pretrained_model = 'demo/Pretrained_Parser/best_model'
    base_dir = 'demo/'
    glove_dir = 'glovevector/'
    g = tf.Graph()
    with g.as_default():
        model = Demo_Parser(base_dir, glove_dir)
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as session: 
            session.run(tf.global_variables_initializer())
            saver.restore(session, pretrained_model)
            sents = 'TAG is the best formalism. We should all learn it.'
            sents = sent_tokenize(sents)
            sents = map(word_tokenize, sents)
            results = model.run_on_sents(session, sents)
            conllu = get_conllu(results, sents)
            print(conllu)
            sents = 'Can you parse this? We willll see!'
            sents = sent_tokenize(sents)
            sents = map(word_tokenize, sents)
            results = model.run_on_sents(session, sents)
            conllu = get_conllu(results, sents)
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
    
def main():
    demo_model()
if __name__ == '__main__':
    sents = 'TAG is the best formalism. We should all learn it.'
    sents = sent_tokenize(sents)
    sents = map(word_tokenize, sents)
    sents = map(lambda x: ' '.join(x), sents)
    print(sents)
    main()