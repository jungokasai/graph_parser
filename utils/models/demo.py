from __future__ import print_function
import os, sys
sys.path.append(os.getcwd())
from utils.models.parsing_model_joint_both import Parsing_Model_Joint_Both
from utils.decoders.predict import predict_arcs_rels
from utils.decoders.predict import predict_arcs_rels
from utils.data_loader.demo_data_loader import Demo_Dataset
from utils.data_loader.options import Opts
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss


class Demo_Parser(Parsing_Model_Joint_Both):
    def __init__(self, base_dir):
        self.opts = Opts()
        ## build a model
        self.loader = Demo_Dataset(base_dir, self.opts.embedding_dim)
        #self.batch_size = 100
	#self.get_features()
        self.features = ['words', 'chars']
        self.add_placeholders()
        self.inputs_dim = self.opts.embedding_dim + self.opts.jk_dim + self.opts.stag_dim + self.opts.nb_filters
        self.outputs_dim = (1+self.opts.bi)*self.opts.units
        inputs_list = [self.add_word_embedding()]
        if self.opts.jk_dim > 0:
            inputs_list.append(self.add_jackknife_embedding())
        if self.opts.stag_dim > 0:
            inputs_list.append(self.add_stag_embedding())
        if self.opts.chars_dim > 0:
            inputs_list.append(self.add_char_embedding())
        inputs_tensor = tf.concat(inputs_list, 2) ## [seq_len, batch_size, inputs_dim]
        inputs_tensor = self.add_dropout(inputs_tensor, self.input_keep_prob)
        inputs_shape = tf.shape(self.inputs_placeholder_dict['words'])
        ## no need to worry about the heads of <-root-> and zero-pads
        ## Let's get those non-padding places so we can reinitialize hidden states after each padding in the backward path
        ### because the backward path starts with zero pads.
        self.weight = tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.zeros(inputs_shape, tf.int32)), tf.float32) ## [batch_size, seq_len]
        for i in xrange(self.opts.num_layers):
            forward_outputs_tensor = self.add_dropout(self.add_lstm_hw(inputs_tensor, i, 'Forward'), self.keep_prob) ## [seq_len, batch_size, units]
            if self.opts.bi:
                backward_outputs_tensor = self.add_dropout(self.add_lstm_hw(tf.reverse(inputs_tensor, [0]), i, 'Backward', True), self.keep_prob) ## [seq_len, batch_size, units]
                inputs_tensor = tf.concat([forward_outputs_tensor, tf.reverse(backward_outputs_tensor, [0])], 2)
            else:
                inputs_tensor = forward_outputs_tensor
        self.weight = self.weight*tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.ones(inputs_shape, tf.int32)*self.loader.word_index['<-root->']), tf.float32) ## [batch_size, seq_len]
        lstm_outputs = inputs_tensor ## [seq_len, batch_size, outputs_dim]

        self.arc_outputs, rel_outputs, self.rel_scores, joint_output, joint_output_jk = self.add_biaffine(lstm_outputs)
#        projected_outputs = tf.map_fn(lambda x: self.add_projection(x), lstm_outputs) #[seq_len, batch_size, nb_tags]
#        projected_outputs = tf.transpose(projected_outputs, perm=[1, 0, 2]) # [batch_size, seq_len, nb_tags]
        #self.loss = self.add_loss_op(self.arc_outputs, self.inputs_placeholder_dict['arcs']) + self.add_loss_op(rel_outputs, self.inputs_placeholder_dict['rels']) + self.add_loss_op(joint_output, self.inputs_placeholder_dict['stags']) + self.add_loss_op(joint_output_jk, self.inputs_placeholder_dict['jk'])
        #self.predicted_arcs, self.UAS = self.add_accuracy(self.arc_outputs, self.inputs_placeholder_dict['arcs'])
        #self.predicted_rels, self.rel_acc = self.add_accuracy(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.predicted_stags = self.add_predictions(joint_output)
        self.predicted_jk = self.add_predictions(joint_output_jk)
        #self.train_op = self.add_train_op(self.loss)

    def run_on_sents(self, session, sents):
        input_sents = map(lambda x: ' '.join(x), sents)
        self.loader.run_on_sents(input_sents)
        feed = {}
        predictions_batch = {}
        for feat in self.inputs_placeholder_dict.keys():
            feed[self.inputs_placeholder_dict[feat]] = self.loader.inputs_test_batch[feat]
        feed[self.keep_prob] = 1.0
        feed[self.hidden_prob] = 1.0
        feed[self.input_keep_prob] = 1.0
        feed[self.mlp_prob] = 1.0
        weight, arc_outputs, rel_scores, predicted_stags, predicted_jk, arc_probs, rel_probs = session.run([self.weight, self.arc_outputs, self.rel_scores, self.predicted_stags, self.predicted_jk, self.arc_probs, self.rel_probs], feed_dict=feed)
        weight = weight.astype(bool)
        predicted_stags = predicted_stags[weight]
        predicted_jk = predicted_jk[weight]
        predictions_batch['stags'] = predicted_stags
        predictions_batch['jk'] = predicted_jk
        non_padding = weight.astype(bool)
        non_padding[:, 0] = True ## take the dummy root nodes
        predicted_arcs, predicted_rels = predict_arcs_rels(arc_outputs, rel_scores, non_padding)
        predictions_batch['arcs'] = predicted_arcs
        predictions_batch['rels'] = predicted_rels
        results = self.loader.get_results(predictions_batch)
        ## nbest
        arc_probs = arc_probs[weight]
        rel_probs = rel_probs[weight]
        return results, arc_probs, rel_probs
    def add_predictions(self, output):
        predictions = tf.cast(tf.argmax(output, 2), tf.int32) ## [batch_size, seq_len]
        return predictions

if __name__ == '__main__':
    base_dir = 'demo/'
    model = Demo_Parser(base_dir)
