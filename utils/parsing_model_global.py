from __future__ import print_function
#import matplotlib
from data_process_secsplit import Dataset
from parsing_model import Parsing_Model
from lstm import get_lstm_weights, lstm
from mlp import get_mlp_weights, mlp
from global_tools import get_global_weights, global_equation
from arc import get_arc_weights, arc_equation
from rel import get_rel_weights, rel_equation
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
import os
import sys

class Parsing_Model_Global(Parsing_Model):

    def add_biaffine_global(self, inputs):
        ## inputs [seq_len, batch_size, units] = [n, b, d]
        ## first define four different MLPs
        arc_roles = ['arc-dep', 'arc-head']
        rel_roles = ['rel-dep', 'rel-head']
        vectors = {}
        for arc_role in arc_roles:
            for i in xrange(self.opts.mlp_num_layers):
                if i == 0:
                    inputs_dim = self.outputs_dim
                    vector_mlp = inputs
                else:
                    inputs_dim = self.opts.arc_mlp_units
                weights = get_mlp_weights('{}_MLP_Layer{}'.format(arc_role, i), inputs_dim, self.opts.arc_mlp_units)
                vector_mlp = self.add_dropout(tf.map_fn(lambda x: mlp(x, weights), vector_mlp), self.mlp_prob)
            vectors[arc_role] = vector_mlp
            ## [seq_len, batch_size, arc_mlp_units]
        arc_weights = get_arc_weights('arc', self.opts.arc_mlp_units)
        for rel_role in rel_roles:
            for i in xrange(self.opts.mlp_num_layers):
                if i == 0:
                    inputs_dim = self.outputs_dim
                    vector_mlp = inputs
                else:
                    inputs_dim = self.opts.rel_mlp_units
                weights = get_mlp_weights('{}_MLP_Layer{}'.format(rel_role, i), inputs_dim, self.opts.rel_mlp_units)
                vector_mlp = self.add_dropout(tf.map_fn(lambda x: mlp(x, weights), vector_mlp), self.mlp_prob)
            vectors[rel_role] = vector_mlp
            ## [seq_len, batch_size, rel_mlp_units]
        rel_weights = get_rel_weights('rel', self.opts.rel_mlp_units, self.loader.nb_rels)
        global_weights = get_global_weights('global', self.opts.arc_mlp_units, self.loader.nb_rels)
        arc_output, rel_output, rel_scores = global_equation(vectors, arc_weights, rel_weights, global_weights)
        return arc_output, rel_output, rel_scores

    
    def __init__(self, opts, test_opts=None):
       
        self.opts = opts
        self.test_opts = test_opts
        self.loader = Dataset(opts, test_opts)
        self.batch_size = 50
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
        self.arc_outputs, rel_outputs, self.rel_scores = self.add_biaffine_global(lstm_outputs)
#        projected_outputs = tf.map_fn(lambda x: self.add_projection(x), lstm_outputs) #[seq_len, batch_size, nb_tags]
#        projected_outputs = tf.transpose(projected_outputs, perm=[1, 0, 2]) # [batch_size, seq_len, nb_tags]
        inputs_shape = tf.shape(self.inputs_placeholder_dict['words'])
        self.weight = tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.zeros(inputs_shape, tf.int32)), tf.float32)*tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.ones(inputs_shape, tf.int32)*self.loader.word_index['<-root->']), tf.float32) ## [batch_size, seq_len]
        ## no need to worry about the heads of <-root-> and zero-pads
        self.loss = self.add_loss_op(self.arc_outputs, self.inputs_placeholder_dict['arcs']) + self.add_loss_op(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.predicted_arcs, self.UAS = self.add_accuracy(self.arc_outputs, self.inputs_placeholder_dict['arcs'])
        self.predicted_rels, self.rel_acc = self.add_accuracy(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.train_op = self.add_train_op(self.loss)

