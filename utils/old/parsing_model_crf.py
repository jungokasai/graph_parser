from __future__ import print_function
#import matplotlib
from data_process_secsplit import Dataset
from parsing_model import Parsing_Model
from lstm import get_lstm_weights, lstm
from mlp import get_mlp_weights, mlp
from arc import get_arc_weights, arc_equation
from rel import get_rel_weights, rel_equation
from predict import predict_arcs_rels
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
import os
import sys


class Parsing_Model_Global(Parsing_Model):
    
    def __init__(self, opts, test_opts=None):
       
        self.opts = opts
        self.test_opts = test_opts
        self.loader = Dataset(opts, test_opts)
        self.batch_size = 100
        self.add_placeholders()
        self.inputs_dim = self.opts.embedding_dim + self.opts.jk_dim + self.opts.stag_dim
        self.outputs_dim = (1+self.opts.bi)*self.opts.units
        inputs_list = [self.add_word_embedding()]
        if self.opts.jk_dim:
            inputs_list.append(self.add_jackknife_embedding())
        if self.opts.stag_dim > 0:
            inputs_list.append(self.add_stag_embedding())
        inputs_tensor = tf.concat(inputs_list, 2) ## [seq_len, batch_size, inputs_dim]
        forward_inputs_tensor = self.add_dropout(inputs_tensor, self.input_keep_prob)
        for i in xrange(self.opts.num_layers):
            forward_inputs_tensor = self.add_dropout(self.add_lstm(forward_inputs_tensor, i, 'Forward'), self.keep_prob) ## [seq_len, batch_size, units]
        lstm_outputs = forward_inputs_tensor
        if self.opts.bi:
            backward_inputs_tensor = self.add_dropout(tf.reverse(inputs_tensor, [0]), self.input_keep_prob)
            for i in xrange(self.opts.num_layers):
                backward_inputs_tensor = self.add_dropout(self.add_lstm(backward_inputs_tensor, i, 'Backward'), self.keep_prob) ## [seq_len, batch_size, units]
            backward_inputs_tensor = tf.reverse(backward_inputs_tensor, [0])
            lstm_outputs = tf.concat([lstm_outputs, backward_inputs_tensor], 2) ## [seq_len, batch_size, outputs_dim]
        self.arc_outputs, rel_outputs, self.rel_scores = self.add_biaffine(lstm_outputs)
#        projected_outputs = tf.map_fn(lambda x: self.add_projection(x), lstm_outputs) #[seq_len, batch_size, nb_tags]
#        projected_outputs = tf.transpose(projected_outputs, perm=[1, 0, 2]) # [batch_size, seq_len, nb_tags]
        inputs_shape = tf.shape(self.inputs_placeholder_dict['words'])
        self.weight = tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.zeros(inputs_shape, tf.int32)), tf.float32)*tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.ones(inputs_shape, tf.int32)*self.loader.word_index['<-root->']), tf.float32) ## [batch_size, seq_len]
        ## no need to worry about the heads of <-root-> and zero-pads
        self.loss = self.add_loss_op(self.arc_outputs, self.inputs_placeholder_dict['arcs']) + self.add_loss_op(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.predicted_arcs, self.UAS = self.add_accuracy(self.arc_outputs, self.inputs_placeholder_dict['arcs'])
        self.predicted_rels, self.rel_acc = self.add_accuracy(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.train_op = self.add_train_op(self.loss)

