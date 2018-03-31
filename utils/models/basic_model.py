from __future__ import print_function
import os, sys
sys.path.append(os.getcwd())
from utils.data_loader.data_process_secsplit import Dataset
from utils.equations.lstm import get_lstm_weights, lstm
from utils.equations.lstm_hw import get_lstm_hw_weights, lstm_hw
from utils.equations.char_encoding import get_char_weights, encode_char
from utils.equations.mlp import get_mlp_weights, mlp
from utils.equations.arc import get_arc_weights, arc_equation
from utils.equations.rel import get_rel_weights, rel_equation
from utils.decoders.predict import predict_arcs_rels
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss


class Basic_Model(object):
    def add_placeholders(self):
        self.inputs_placeholder_dict = {}
        for feature in self.features:
            if feature == 'chars':
                self.inputs_placeholder_dict[feature] = tf.placeholder(tf.int32, shape = [None, None, None])
            else:
                self.inputs_placeholder_dict[feature] = tf.placeholder(tf.int32, shape = [None, None])

        self.keep_prob = tf.placeholder(tf.float32)
        self.input_keep_prob = tf.placeholder(tf.float32)
        self.hidden_prob = tf.placeholder(tf.float32)
        self.mlp_prob = tf.placeholder(tf.float32)  

    def add_word_embedding(self): 
        with tf.device('/cpu:0'):
            with tf.variable_scope('word_embedding') as scope:
                embedding = tf.get_variable('word_embedding_mat', self.loader.word_embeddings.shape, initializer=tf.constant_initializer(self.loader.word_embeddings))

            inputs = tf.nn.embedding_lookup(embedding, self.inputs_placeholder_dict['words']) ## [batch_size, seq_len, embedding_dim]
            inputs = tf.transpose(inputs, perm=[1, 0, 2]) # [seq_length, batch_size, embedding_dim]
        return inputs 

    def add_jackknife_embedding(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('jk_embedding') as scope:
                embedding = tf.get_variable('jk_embedding_mat', [self.loader.nb_jk+1, self.opts.jk_dim]) # +1 for padding
            inputs = tf.nn.embedding_lookup(embedding, self.inputs_placeholder_dict['jk']) ## [batch_size, seq_len, embedding_dim]
            inputs = tf.transpose(inputs, perm=[1, 0, 2]) # [seq_length, batch_size, embedding_dim]
        return inputs 

    def add_stag_embedding(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('stag_embedding') as scope:
                embedding = tf.get_variable('stag_embedding_mat', [self.loader.nb_stags+1, self.opts.stag_dim]) # +1 for padding
            inputs = tf.nn.embedding_lookup(embedding, self.inputs_placeholder_dict['stags']) ## [batch_size, seq_len, embedding_dim]
            inputs = tf.transpose(inputs, perm=[1, 0, 2]) # [seq_length, batch_size, embedding_dim]
        self.stag_embeddings = embedding
        return inputs 

    def add_char_embedding(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('char_embedding') as scope:
                embedding = tf.get_variable('char_embedding_mat', [self.loader.nb_chars+1, self.opts.chars_dim]) # +1 for padding

            inputs = tf.nn.embedding_lookup(embedding, self.inputs_placeholder_dict['chars']) ## [batch_size, seq_len-1, word_len, embedding_dim] 
            ## -1 because we don't have ROOT
            inputs = tf.transpose(inputs, perm=[1, 0, 2, 3])
            ## [seq_len-1, batch_size, word_len, embedding_dim]
            inputs = self.add_dropout(inputs, self.input_keep_prob)
            weights = get_char_weights(self.opts, 'char_encoding')
            inputs = encode_char(inputs, weights) ## [seq_len-1, batch_size, nb_filters]
            shape = tf.shape(inputs)
            ## add 0 vectors for <-root->
            inputs = tf.concat([tf.zeros([1, shape[1], shape[2]]), inputs], 0)
        return inputs 


    def add_lstm(self, inputs, i, name, backward=False):
        prev_init = tf.zeros([2, tf.shape(inputs)[1], self.opts.units])  # [2, batch_size, num_units]
        #prev_init = tf.zeros([2, 100, self.opts.units])  # [2, batch_size, num_units]
        if i == 0:
            inputs_dim = self.inputs_dim
        else:
            inputs_dim = self.opts.units*2 ## concat after each layer
        weights = get_lstm_weights('{}_LSTM_layer{}'.format(name, i), inputs_dim, self.opts.units, tf.shape(inputs)[1], self.hidden_prob)
        if backward:
            ## backward: reset states after zero paddings
            non_paddings = tf.transpose(self.weight, [1, 0]) ## [batch_size, seq_len] => [seq_len, batch_size]
            non_paddings = tf.reverse(non_paddings, [0])
            cell_hidden = tf.scan(lambda prev, x: lstm(prev, x, weights, backward=backward), [inputs, non_paddings], prev_init)
        else:
            cell_hidden = tf.scan(lambda prev, x: lstm(prev, x, weights), inputs, prev_init)
         #cell_hidden [seq_len, 2, batch_size, units]
        h = tf.unstack(cell_hidden, 2, axis=1)[1] #[seq_len, batch_size, units]
        return h

    def add_lstm_hw(self, inputs, i, name, backward=False):
        prev_init = tf.zeros([2, tf.shape(inputs)[1], self.opts.units])  # [2, batch_size, num_units]
        #prev_init = tf.zeros([2, 100, self.opts.units])  # [2, batch_size, num_units]      
        if i == 0:
            inputs_dim = self.inputs_dim
        else:   
            inputs_dim = self.opts.units*2 ## concat after each layer
        weights = get_lstm_hw_weights('{}_LSTM_layer{}'.format(name, i), inputs_dim, self.opts.units, tf.shape(inputs)[1], self.hidden_prob)
        if backward:
            ## backward: reset states after zero paddings
            non_paddings = tf.transpose(self.weight, [1, 0]) ## [batch_size, seq_len] => [seq_len, batch_size] 
            non_paddings = tf.reverse(non_paddings, [0])
            cell_hidden = tf.scan(lambda prev, x: lstm_hw(prev, x, weights, backward=backward), [inputs, non_paddings], prev_init) 
        else:
            cell_hidden = tf.scan(lambda prev, x: lstm_hw(prev, x, weights), inputs, prev_init)
         #cell_hidden [seq_len, 2, batch_size, units]
        h = tf.unstack(cell_hidden, 2, axis=1)[1] #[seq_len, batch_size, units]
        return h

    def add_dropout(self, inputs, keep_prob):
        ## inputs [seq_len, batch_size, inputs_dims/units]
        dummy_dp = tf.ones(tf.shape(inputs)[1:])
        dummy_dp = tf.nn.dropout(dummy_dp, keep_prob)
        return tf.map_fn(lambda x: dummy_dp*x, inputs)

    def add_projection(self, inputs): 
        with tf.variable_scope('Projection') as scope:
            proj_U = tf.get_variable('weight', [self.outputs_dim, self.loader.nb_tags]) 
            proj_b = tf.get_variable('bias', [self.loader.nb_tags])
            outputs = tf.matmul(inputs, proj_U)+proj_b 
            return outputs

    def add_loss_op(self, output, gold):
        cross_entropy = sequence_loss(output, gold, self.weight)
        loss = tf.reduce_sum(cross_entropy)
        return loss

    def add_accuracy(self, output, gold):
        predictions = tf.cast(tf.argmax(output, 2), tf.int32) ## [batch_size, seq_len]
        correct_predictions = self.weight*tf.cast(tf.equal(predictions, gold), tf.float32)
        accuracy = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))/tf.reduce_sum(tf.cast(self.weight, tf.float32))
        return predictions, accuracy

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
        return train_op

    def get_features(self):
        self.features = ['words', 'arcs', 'rels']
        if self.opts.jk_dim > 0 or self.opts.model in ['Parsing_Model_Joint_Both']:
            self.features.append('jk')
        if self.opts.stag_dim > 0 or self.opts.model in ['Parsing_Model_Joint', 'Parsing_Model_Joint_Both']:
            self.features.append('stags')
        if self.opts.chars_dim > 0:
            self.features.append('chars')
    def add_probs(self, output):
        self.probs = tf.nn.softmax(output)

    def add_biaffine(self, inputs):
        ## inputs [seq_len, batch_size, units]
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
                ## [seq_len, batch_size, 2*mlp_units]
            vectors[arc_role] = vector_mlp
        weights = get_arc_weights('arc', self.opts.arc_mlp_units)
        arc_output = arc_equation(vectors['arc-head'], vectors['arc-dep'], weights) # [batch_size, seq_len, seq_len] dim 1: deps, dim 2: heads
#        arc_predictions = get_arcs(arc_output, self.test_opts) # [batch_size, seq_len]
        arc_predictions = tf.argmax(arc_output, 2) # [batch_size, seq_len]
        for rel_role in rel_roles:
            for i in xrange(self.opts.mlp_num_layers):
                if i == 0:
                    inputs_dim = self.outputs_dim
                    vector_mlp = inputs
                else:
                    inputs_dim = self.opts.rel_mlp_units
                weights = get_mlp_weights('{}_MLP_Layer{}'.format(rel_role, i), inputs_dim, self.opts.rel_mlp_units)
                vector_mlp = self.add_dropout(tf.map_fn(lambda x: mlp(x, weights), vector_mlp), self.mlp_prob)
                ## [seq_len, batch_size, 2*mlp_units]
            vectors[rel_role] = vector_mlp
        weights = get_rel_weights('rel', self.opts.rel_mlp_units, self.loader.nb_rels)
        rel_output, rel_scores = rel_equation(vectors['rel-head'], vectors['rel-dep'], weights, arc_predictions)  #[batch_size, seq_len, nb_rels]
        return arc_output, rel_output, rel_scores
