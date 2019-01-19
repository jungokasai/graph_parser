from __future__ import print_function
import os, sys
sys.path.append(os.getcwd())
from utils.models.basic_model import Basic_Model
from utils.decoders.predict import predict_arcs_rels
from utils.data_loader.data_process_secsplit import Dataset
import numpy as np
import time
import pickle
import tensorflow as tf


class Parsing_Model(Basic_Model):
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
            if self.test_opts is not None:
                if self.test_opts.get_weight:
                    stag_embeddings = session.run(self.stag_embeddings)
                    self.loader.output_weight(stag_embeddings)
            #scores['UAS'] = np.mean(predictions['arcs'][self.loader.punc] == self.loader.gold_arcs[self.loader.punc])
            #scores['UAS_greedy'] = np.mean(predictions['arcs_greedy'][self.loader.punc] == self.loader.gold_arcs[self.loader.punc])
            return scores
