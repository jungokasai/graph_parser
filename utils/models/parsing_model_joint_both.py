from __future__ import print_function
import os, sys
#sys.path.append(os.getcwd())
from utils.models.basic_model import Basic_Model
from utils.decoders.predict import predict_arcs_rels
from utils.data_loader.data_process_secsplit import Dataset
from utils.equations.mlp import get_mlp_weights, mlp 
from utils.equations.arc import get_arc_weights, arc_equation
from utils.equations.rel import get_rel_weights, rel_equation
from utils.equations.joint import get_joint_weights, joint_equation
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss


class Parsing_Model_Joint_Both(Basic_Model):

    def add_biaffine(self, inputs):
        arc_roles = ['arc-dep', 'arc-head']
        rel_roles = ['rel-dep', 'rel-head']
        joint_roles = ['jk', 'stag']
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
        ## joint stagging
        for joint_role in joint_roles:
            for i in xrange(self.opts.mlp_num_layers):
                if i == 0:
                    inputs_dim = self.outputs_dim
                    vector_mlp = inputs
                else:
                    inputs_dim = self.opts.joint_mlp_units
                weights = get_mlp_weights('{}_MLP_Layer{}'.format(joint_role, i), inputs_dim, self.opts.joint_mlp_units)
                vector_mlp = self.add_dropout(tf.map_fn(lambda x: mlp(x, weights), vector_mlp), self.mlp_prob)
                ## [seq_len, batch_size, 2*mlp_units]
            vectors[joint_role] = vector_mlp
        weights = get_joint_weights('stag', self.opts.joint_mlp_units, self.loader.nb_stags)
        self.stag_embeddings = tf.transpose(weights['W-joint'], [1,0])
        joint_output = joint_equation(vectors['stag'], weights) # [batch_size, seq_len, nb_stags]
        weights = get_joint_weights('jk', self.opts.joint_mlp_units, self.loader.nb_jk)
        joint_output_jk = joint_equation(vectors['jk'], weights) # [batch_size, seq_len, nb_stags]
        return arc_output, rel_output, rel_scores, joint_output, joint_output_jk
    
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
        self.loss = self.add_loss_op(self.arc_outputs, self.inputs_placeholder_dict['arcs']) + self.add_loss_op(rel_outputs, self.inputs_placeholder_dict['rels']) + self.add_loss_op(joint_output, self.inputs_placeholder_dict['stags']) + self.add_loss_op(joint_output_jk, self.inputs_placeholder_dict['jk'])
        self.add_probs(joint_output)
        self.predicted_arcs, self.UAS = self.add_accuracy(self.arc_outputs, self.inputs_placeholder_dict['arcs'])
        self.predicted_rels, self.rel_acc = self.add_accuracy(rel_outputs, self.inputs_placeholder_dict['rels'])
        self.predicted_stags, self.stag_acc = self.add_accuracy(joint_output, self.inputs_placeholder_dict['stags'])
        self.predicted_jk, self.jk_acc = self.add_accuracy(joint_output_jk, self.inputs_placeholder_dict['jk'])
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
            _, loss, UAS, rel_acc, stag_acc = session.run([train_op, self.loss, self.UAS, self.rel_acc, self.stag_acc], feed_dict=feed)
            return loss, UAS, rel_acc, stag_acc
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
            loss, predicted_arcs, predicted_rels, UAS, weight, arc_outputs, rel_scores, stag_acc, predicted_stags, probs, predicted_jk = session.run([self.loss, self.predicted_arcs, self.predicted_rels, self.UAS, self.weight, self.arc_outputs, self.rel_scores, self.stag_acc, self.predicted_stags, self.probs, self.predicted_jk], feed_dict=feed)
            weight = weight.astype(bool)
            predicted_arcs_greedy = predicted_arcs[weight]
            predicted_rels_greedy = predicted_rels[weight]
            predicted_stags = predicted_stags[weight]
            predicted_jk = predicted_jk[weight]
            predictions_batch['arcs_greedy'] = predicted_arcs_greedy
            predictions_batch['rels_greedy'] = predicted_rels_greedy
            predictions_batch['stags'] = predicted_stags
            predictions_batch['jk'] = predicted_jk
            non_padding = weight.astype(bool)
            non_padding[:, 0] = True ## take the dummy root nodes
            predicted_arcs, predicted_rels = predict_arcs_rels(arc_outputs, rel_scores, non_padding)
            predictions_batch['arcs'] = predicted_arcs
            predictions_batch['rels'] = predicted_rels
            probs = probs[weight]
#            print(predicted_greedy_arcs.shape)
#            print(predicted_arcs.shape)
            #print(arc_outputs.shape)
            return loss, predictions_batch, UAS, probs

    def run_epoch(self, session, testmode = False):

        if not testmode:
            epoch_start_time = time.time()
            next_batch = self.loader.next_batch
            epoch_incomplete = next_batch(self.batch_size)
            while epoch_incomplete:
                loss, UAS, rel_acc, stag_acc = self.run_batch(session)
                print('{}/{}, loss {:.4f}, Raw UAS {:.4f}, Rel Acc {:.4f}, Stag Acc {:.4f}'.format(self.loader._index_in_epoch, self.loader.nb_train_samples, loss, UAS, rel_acc, stag_acc), end = '\r')
                epoch_incomplete = next_batch(self.batch_size)
            print('\nEpoch Training Time {}'.format(time.time() - epoch_start_time))
            return loss, UAS
        else: 
            next_test_batch = self.loader.next_test_batch
            test_incomplete = next_test_batch(self.batch_size)
            output_types = ['arcs', 'rels', 'arcs_greedy', 'rels_greedy', 'stags', 'jk']
            predictions = {output_type: [] for output_type in output_types}
            probs = []
            while test_incomplete:
                loss, predictions_batch, UAS, probs_batch = self.run_batch(session, True)
                    
                for name, pred in predictions_batch.items():
                    predictions[name].append(pred)
                #print('Testmode {}/{}, loss {}, accuracy {}'.format(self.loader._index_in_test, self.loader.nb_validation_samples, loss, accuracy), end = '\r')
                probs.append(probs_batch)
                print('Test mode {}/{}, Raw UAS {:.4f}'.format(self.loader._index_in_test, self.loader.nb_validation_samples, UAS), end='\r') #, end = '\r')
                test_incomplete = next_test_batch(self.batch_size)
            for name, pred in predictions.items():
                predictions[name] = np.hstack(pred)
            if self.test_opts is not None:
                self.loader.output_arcs(predictions['arcs'], self.test_opts.predicted_arcs_file)
                self.loader.output_rels(predictions['rels'], self.test_opts.predicted_rels_file)
                self.loader.output_arcs(predictions['arcs_greedy'], self.test_opts.predicted_arcs_file_greedy)
                self.loader.output_rels(predictions['rels_greedy'], self.test_opts.predicted_rels_file_greedy)
                self.loader.output_stags(predictions['stags'], self.test_opts.predicted_stags_file)
                self.loader.output_pos(predictions['jk'], self.test_opts.predicted_pos_file)
                if self.test_opts.save_probs:
                    self.loader.output_probs(np.vstack(probs))
                if self.test_opts.get_weight:
                    stag_embeddings = session.run(self.stag_embeddings)
                    self.loader.output_weight(stag_embeddings)
            scores = self.loader.get_scores(predictions, self.opts, self.test_opts)
            #scores['UAS'] = np.mean(predictions['arcs'][self.loader.punc] == self.loader.gold_arcs[self.loader.punc])
            #scores['UAS_greedy'] = np.mean(predictions['arcs_greedy'][self.loader.punc] == self.loader.gold_arcs[self.loader.punc])
            return scores
