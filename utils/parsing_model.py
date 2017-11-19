from __future__ import print_function
#import matplotlib
from data_process_secsplit import Dataset
from lstm import get_lstm_weights, lstm
from char_encoding import get_char_weights, encode_char
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


class Parsing_Model(object):
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
        if self.opts.jk_dim > 0:
            self.features.append('jk')
        if self.opts.stag_dim > 0:
            self.features.append('stags')
        if self.opts.chars_dim > 0:
            self.features.append('chars')

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
    
    def __init__(self, opts, test_opts=None):
       
        self.opts = opts
        self.test_opts = test_opts
        self.loader = Dataset(opts, test_opts)
        self.batch_size = 100
	self.get_features()
        self.add_placeholders()
        self.inputs_dim = self.opts.embedding_dim + self.opts.jk_dim + self.opts.stag_dim + self.opts.nb_filters
        self.outputs_dim = (1+self.opts.bi)*self.opts.units
        inputs_list = [self.add_word_embedding()]
        if self.opts.jk_dim:
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
            forward_outputs_tensor = self.add_dropout(self.add_lstm(inputs_tensor, i, 'Forward'), self.keep_prob) ## [seq_len, batch_size, units]
            if self.opts.bi:
                backward_outputs_tensor = self.add_dropout(self.add_lstm(tf.reverse(inputs_tensor, [0]), i, 'Backward', True), self.keep_prob) ## [seq_len, batch_size, units]
                inputs_tensor = tf.concat([forward_outputs_tensor, tf.reverse(backward_outputs_tensor, [0])], 2)
            else:
                inputs_tensor = forward_outputs_tensor
        self.weight = self.weight*tf.cast(tf.not_equal(self.inputs_placeholder_dict['words'], tf.ones(inputs_shape, tf.int32)*self.loader.word_index['<-root->']), tf.float32) ## [batch_size, seq_len]
        lstm_outputs = inputs_tensor ## [seq_len, batch_size, outputs_dim]

        self.arc_outputs, rel_outputs, self.rel_scores = self.add_biaffine(lstm_outputs)
#        projected_outputs = tf.map_fn(lambda x: self.add_projection(x), lstm_outputs) #[seq_len, batch_size, nb_tags]
#        projected_outputs = tf.transpose(projected_outputs, perm=[1, 0, 2]) # [batch_size, seq_len, nb_tags]
        self.loss = self.add_loss_op(self.arc_outputs, self.inputs_placeholder_dict['arcs']) + self.add_loss_op(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.predicted_arcs, self.UAS = self.add_accuracy(self.arc_outputs, self.inputs_placeholder_dict['arcs'])
        self.predicted_rels, self.rel_acc = self.add_accuracy(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.train_op = self.add_train_op(self.loss)

    def run_batch(self, session, testmode = False):
        if not testmode:
            feed = {}
            for feat in self.inputs_placeholder_dict.keys():
                feed[self.inputs_placeholder_dict[feat]] = self.loader.inputs_train_batch[feat]
            feed[self.keep_prob] = self.opts.dropout_p
            feed[self.hidden_prob] = self.opts.hidden_p
            feed[self.input_keep_prob] = self.opts.input_dp
            feed[self.mlp_prob] = self.opts.mlp_prob
            train_op = self.train_op
            _, loss, UAS, rel_acc = session.run([train_op, self.loss, self.UAS, self.rel_acc], feed_dict=feed)
            return loss, UAS, rel_acc
        else:
            feed = {}
            predictions_batch = {}
            for feat in self.inputs_placeholder_dict.keys():
                feed[self.inputs_placeholder_dict[feat]] = self.loader.inputs_test_batch[feat]
            feed[self.keep_prob] = 1.0
            feed[self.hidden_prob] = 1.0
            feed[self.input_keep_prob] = 1.0
            feed[self.mlp_prob] = 1.0
#            loss, accuracy, predictions, weight = session.run([self.loss, self.accuracy, self.predictions, self.weight], feed_dict=feed)
            loss, predicted_arcs, predicted_rels, UAS, weight, arc_outputs, rel_scores = session.run([self.loss, self.predicted_arcs, self.predicted_rels, self.UAS, self.weight, self.arc_outputs, self.rel_scores], feed_dict=feed)
            weight = weight.astype(bool)
            predicted_arcs_greedy = predicted_arcs[weight]
            predicted_rels_greedy = predicted_rels[weight]
            predictions_batch['arcs_greedy'] = predicted_arcs_greedy
            predictions_batch['rels_greedy'] = predicted_rels_greedy
            non_padding = weight.astype(bool)
            non_padding[:, 0] = True ## take the dummy root nodes
            predicted_arcs, predicted_rels = predict_arcs_rels(arc_outputs, rel_scores, non_padding)
            predictions_batch['arcs'] = predicted_arcs
            predictions_batch['rels'] = predicted_rels
#            print(predicted_greedy_arcs.shape)
#            print(predicted_arcs.shape)
            #print(arc_outputs.shape)
            return loss, predictions_batch, UAS

    def run_epoch(self, session, testmode = False):

        if not testmode:
            epoch_start_time = time.time()
            next_batch = self.loader.next_batch
            epoch_incomplete = next_batch(self.batch_size)
            while epoch_incomplete:
                loss, UAS, rel_acc = self.run_batch(session)
                print('{}/{}, loss {:.4f}, Raw UAS {:.4f}, Rel Acc {:.4f}'.format(self.loader._index_in_epoch, self.loader.nb_train_samples, loss, UAS, rel_acc), end = '\r')
                epoch_incomplete = next_batch(self.batch_size)
            print('\nEpoch Training Time {}'.format(time.time() - epoch_start_time))
            return loss, UAS
        else: 
            next_test_batch = self.loader.next_test_batch
            test_incomplete = next_test_batch(self.batch_size)
            output_types = ['arcs', 'rels', 'arcs_greedy', 'rels_greedy']
            predictions = {output_type: [] for output_type in output_types}
            while test_incomplete:
                loss, predictions_batch, UAS = self.run_batch(session, True)
                for name, pred in predictions_batch.items():
                    predictions[name].append(pred)
                #print('Testmode {}/{}, loss {}, accuracy {}'.format(self.loader._index_in_test, self.loader.nb_validation_samples, loss, accuracy), end = '\r')
                print('Test mode {}/{}, Raw UAS {:.4f}'.format(self.loader._index_in_test, self.loader.nb_validation_samples, UAS), end='\r') #, end = '\r')
                test_incomplete = next_test_batch(self.batch_size)
            for name, pred in predictions.items():
                predictions[name] = np.hstack(pred)
            if self.test_opts is not None:
                self.loader.output_arcs(predictions['arcs'], self.test_opts.predicted_arcs_file)
                self.loader.output_rels(predictions['rels'], self.test_opts.predicted_rels_file)
                self.loader.output_arcs(predictions['arcs_greedy'], self.test_opts.predicted_arcs_file_greedy)
                self.loader.output_rels(predictions['rels_greedy'], self.test_opts.predicted_rels_file_greedy)
            scores = self.loader.get_scores(predictions, self.opts, self.test_opts)
            if self.test_opts.get_weight:
                stag_embeddings = session.run(self.stag_embeddings)
                self.loader.output_weight(stag_embeddings)
            #scores['UAS'] = np.mean(predictions['arcs'][self.loader.punc] == self.loader.gold_arcs[self.loader.punc])
            #scores['UAS_greedy'] = np.mean(predictions['arcs_greedy'][self.loader.punc] == self.loader.gold_arcs[self.loader.punc])
            return scores
