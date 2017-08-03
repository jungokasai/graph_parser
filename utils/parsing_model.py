from __future__ import print_function
#import matplotlib
from data_process_secsplit import Dataset
from lstm import get_lstm_weights, lstm
from mlp import get_mlp_weights, mlp
from arc import get_arc_weights, arc_equation
from rel import get_rel_weights, rel_equation
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
import os
import sys


class Parsing_Model(object):
    def add_placeholders(self):
        features = ['words', 'jk', 'stags', 'arcs', 'rels']
        self.inputs_placeholder_dict = {}
        for feature in features:
            self.inputs_placeholder_dict[feature] = tf.placeholder(tf.int32, shape = [None, None])
        self.keep_prob = tf.placeholder(tf.float32)  
        self.input_keep_prob = tf.placeholder(tf.float32)  
        self.hidden_prob = tf.placeholder(tf.float32)  

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
                embedding = tf.get_variable('stag_embedding_mat', [self.loader.nb_jk+1, self.opts.stag_dim]) # +1 for padding
            inputs = tf.nn.embedding_lookup(embedding, self.inputs_placeholder_dict['stag']) ## [batch_size, seq_len, embedding_dim]
            inputs = tf.transpose(inputs, perm=[1, 0, 2]) # [seq_length, batch_size, embedding_dim]
        return inputs 

    def add_lstm(self, inputs, i, name):
        prev_init = tf.zeros([2, tf.shape(inputs)[1], self.opts.units])  # [2, batch_size, num_units]
        #prev_init = tf.zeros([2, 100, self.opts.units])  # [2, batch_size, num_units]
        if i == 0:
            inputs_dim = self.inputs_dim
        else:
            inputs_dim = self.opts.units
        weights = get_lstm_weights('{}_LSTM_layer{}'.format(name, i), inputs_dim, self.opts.units, tf.shape(inputs)[1], self.hidden_prob)
        cell_hidden = tf.scan(lambda prev, x: lstm(prev, x, weights), inputs, prev_init)
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

    def add_loss_op(self, output):
        cross_entropy = sequence_loss(output, self.inputs_placeholder_list[5], self.weight)
        loss = tf.reduce_sum(cross_entropy)
        return loss

    def add_accuracy(self, output):
        self.predictions = tf.cast(tf.argmax(output, 2), tf.int32) ## [batch_size, seq_len]
        correct_predictions = self.weight*tf.cast(tf.equal(self.predictions, self.inputs_placeholder_list[5]), tf.float32)
        self.accuracy = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))/tf.reduce_sum(tf.cast(self.weight, tf.float32))

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
        return train_op

    def add_biaffine(self, inputs):
        ## inputs [seq_len, batch_size, units]
        ## first define four different MLPs
        roles = ['arc-dep', 'arc-head', 'rel-dep', 'rel-head']
        vectors = {}
        batch_size = tf.shape(inputs)[1]
        for i in xrange(self.opts.num_mlp_layers):
            if i == 0:
                inputs_dim = self.outputs_dim
            else:
                inputs_dim = self.opts.units
            weights = get_mlp_weights('MLP_Layer{}'.format(i), inputs_dim, self.opts.mlp_units, batch_size, self.opts.mlp_prob)
            vectors_mlps = tf.map_fn(lambda x: mlp(x, weights), inputs)
            ## [seq_len, batch_size, 4*mlp_units]
            vectors_mlps = tf.split(vectors, 4, axis=2)
            for role, vectors_mlp in zip(roles, vectors_mlps):
                vectors[role] = vectors_mlp
        weights = get_arc_weights
        arc_output = arc_equation(vectors['arc-head'], vectors['arc-dep'], weights) # [batch_size, seq_len, seq_len] dim 1: deps, dim 2: heads
        weights = get_rel_weights
        rel_output = rel_equation(vectors['rel-head'], vectors['rel-dep'], weights) 
        return arc_output, rel_output
    
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
#        projected_outputs = tf.map_fn(lambda x: self.add_projection(x), lstm_outputs) #[seq_len, batch_size, nb_tags]
#        projected_outputs = tf.transpose(projected_outputs, perm=[1, 0, 2]) # [batch_size, seq_len, nb_tags]
        inputs_shape = tf.shape(self.inputs_placeholder_dict['words'])
        self.weight = tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.zeros(inputs_shape, tf.int32)), tf.float32)*tf.cast(tf.not_equal(self.inputs_placeholder_list[0], tf.ones(inputs_shape, tf.int32)*self.loader.word_index['<-root->']), tf.float32) ## [batch_size, seq_len]
        ## no need to worry about the heads of <-root-> and zero-pads
        self.loss = self.add_loss_op(projected_outputs)
        self.train_op = self.add_train_op(self.loss)
        self.add_accuracy(projected_outputs)

    def run_batch(self, session, testmode = False):
        if not testmode:
            feed = {}
            for placeholder, data in zip(self.inputs_placeholder_list, self.loader.inputs_train_batch):
                feed[placeholder] = data
            feed[self.keep_prob] = self.opts.dropout_p
            feed[self.hidden_prob] = self.opts.hidden_p
            feed[self.input_keep_prob] = self.opts.input_dp
            train_op = self.train_op
            _, loss, accuracy = session.run([train_op, self.loss, self.accuracy], feed_dict=feed)
            return loss, accuracy
        else:
            feed = {}
            for placeholder, data in zip(self.inputs_placeholder_list, self.loader.inputs_test_batch):
                feed[placeholder] = data
            feed[self.keep_prob] = 1.0
            feed[self.hidden_prob] = 1.0
            feed[self.input_keep_prob] = 1.0
            loss, accuracy, predictions, weight = session.run([self.loss, self.accuracy, self.predictions, self.weight], feed_dict=feed)
            weight = weight.astype(bool)
            predictions = predictions[weight]
            return loss, accuracy, predictions

    def run_epoch(self, session, testmode = False):

        if not testmode:
            epoch_start_time = time.time()
            next_batch = self.loader.next_batch
            epoch_incomplete = next_batch(self.batch_size)
            while epoch_incomplete:
                loss, accuracy = self.run_batch(session)
                print('{}/{}, loss {:.4f}, accuracy {:.4f}'.format(self.loader._index_in_epoch, self.loader.nb_train_samples, loss, accuracy), end = '\r')
                epoch_incomplete = next_batch(self.batch_size)
            print('\nEpoch Training Time {}'.format(time.time() - epoch_start_time))
            return loss, accuracy
        else: 
            next_test_batch = self.loader.next_test_batch
            test_incomplete = next_test_batch(self.batch_size)
            predictions = []
            while test_incomplete:
                loss, accuracy, predictions_batch = self.run_batch(session, True)
                predictions.append(predictions_batch)
                #print('Testmode {}/{}, loss {}, accuracy {}'.format(self.loader._index_in_test, self.loader.nb_validation_samples, loss, accuracy), end = '\r')
                print('Test mode {}/{}'.format(self.loader._index_in_test, self.loader.nb_validation_samples), end = '\r')
                test_incomplete = next_test_batch(self.batch_size)
            predictions = np.hstack(predictions)
            if self.test_opts is not None:
                self.loader.output_stags(predictions, self.test_opts.save_tags)
                        
            accuracy = np.mean(predictions == self.loader.test_gold)
            return accuracy
