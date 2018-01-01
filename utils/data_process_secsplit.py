import numpy as np
from preprocessing import Tokenizer, pad_sequences, arcs2seq
from mica.nbest import output_mica_nbest
import os 
import sys
import pickle
import random
import io

np.random.seed(1234)

class Dataset(object):

    def __init__(self, opts, test_opts=None):
        path_to_text = opts.text_train
        path_to_tag = opts.tag_train
        path_to_jk = opts.jk_train
        path_to_arc = opts.arc_train
        path_to_rel = opts.rel_train
        if test_opts is None:
            path_to_text_test = opts.text_test
            path_to_tag_test = opts.tag_test
            path_to_jk_test = opts.jk_test
            path_to_arc_test = opts.arc_test
            path_to_rel_test = opts.rel_test
            path_to_punc_test = opts.punc_test
        else:
            path_to_text_test = test_opts.text_test
            path_to_tag_test = test_opts.tag_test
            path_to_jk_test = test_opts.jk_test
            path_to_arc_test = test_opts.arc_test
            path_to_rel_test = test_opts.rel_test
            path_to_punc_test = test_opts.punc_test

        self.inputs_train = {}
        self.inputs_test = {}

        ## indexing sents files
        f_train = open(path_to_text)
        texts = f_train.readlines()
        self.nb_train_samples = len(texts)
        f_train.close()
        tokenizer = Tokenizer(lower=True)
        tokenizer.fit_on_texts(texts)
        #print(tokenizer.word_index['-unseen-'])
        self.word_index = tokenizer.word_index
        self.nb_words = len(self.word_index)
        print('Found {} unique lowercased words including -unseen- and <-root->.'.format(self.nb_words))

        # lookup the glove word embeddings
        # need to reserve indices for testing file. 
        glove_size = opts.embedding_dim
        self.embeddings_index = {}
        print('Indexing word vectors.')
        f = open(opts.word_embeddings_file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found {} word vectors.'.format(len(self.embeddings_index)))

        unseens = list(set(self.embeddings_index.keys()) - set(self.word_index.keys())) ## list of words that appear in glove but not in the training set
        nb_unseens = len(unseens)
        print('Found {} words not in the training set but in the glove data'.format(nb_unseens))

        self.word_embeddings = np.zeros((self.nb_words+1+nb_unseens, glove_size)) ## +1 for padding (idx 0)
        for word, i in self.word_index.items(): ## first index the words in the training set
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None: ## otherwise zero vector
                self.word_embeddings[i] = embedding_vector
        for unseen in unseens:
            self.word_index[unseen] = len(self.word_index) + 1 ## add unseen words to the word_index dictionary
            self.word_embeddings[self.word_index[unseen]] = self.embeddings_index[unseen]
        self.idx_to_word = invert_dict(self.word_index)
        print('end glove indexing')
        f_test = open(path_to_text_test)
        texts = texts +  f_test.readlines()
        self.nb_validation_samples = len(texts) - self.nb_train_samples
        f_test.close()
        text_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_word[x], text_sequences[self.nb_train_samples]))
        self.inputs_train['words'] = text_sequences[:self.nb_train_samples]
        self.inputs_test['words'] = text_sequences[self.nb_train_samples:]
        ## indexing sents files ends
        ## indexing char files
        if opts.chars_dim > 0:
            f_train = io.open(path_to_text, encoding='utf-8')
            texts = f_train.readlines()
            f_train.close()
            tokenizer = Tokenizer(lower=False,char_encoding=True, root=False) 
            ## char embedding for <-root-> does not make sense
            tokenizer.fit_on_texts(texts) ## char embedding for <-root-> does not make sense
            self.char_index = tokenizer.word_index
            self.nb_chars = len(self.char_index)
            self.idx_to_char = invert_dict(self.char_index)
            print('Found {} unique characters including -unseen-. NOT including <-root->.'.format(self.nb_chars))
            f_test = io.open(path_to_text_test, encoding='utf-8')
            texts = texts + f_test.readlines() ## do not lowercase tCO
            f_test.close()
            char_sequences = tokenizer.texts_to_sequences(texts)
            #print(map(lambda x: self.idx_to_jk[x], jk_sequences[self.nb_train_samples]))
            self.inputs_train['chars'] = char_sequences[:self.nb_train_samples]
            self.inputs_test['chars'] = char_sequences[self.nb_train_samples:]
            ## indexing char files ends

        ## indexing jackknife files
        if (opts.jk_dim > 0) or (opts.model in ['Parsing_Model_Joint_Both']):
            f_train = open(path_to_jk)
            texts = f_train.readlines()
            f_train.close()
            tokenizer = Tokenizer(lower=False) 
            tokenizer.fit_on_texts(texts, zero_padding=False)
            self.jk_index = tokenizer.word_index
            self.nb_jk = len(self.jk_index)
            self.idx_to_jk = invert_dict(self.jk_index)
            print('Found {} unique POS tags including -unseen- and <-root->.'.format(self.nb_jk))
            f_test = open(path_to_jk_test)
            texts = texts + f_test.readlines() ## do not lowercase tCO
            f_test.close()
            jk_sequences = tokenizer.texts_to_sequences(texts)
            self.inputs_train['jk'] = jk_sequences[:self.nb_train_samples]
            self.inputs_test['jk'] = jk_sequences[self.nb_train_samples:]
            self.gold_jk = np.hstack(map(lambda x: x[1:], jk_sequences[self.nb_train_samples:]))
            ## indexing jackknife files ends
        ## indexing stag files
        if (opts.stag_dim > 0) or (opts.model in ['Parsing_Model_Joint', 'Parsing_Model_Shuffle', 'Parsing_Model_Joint_Both']):
            f_train = open(path_to_tag)
            texts = f_train.readlines()
            f_train.close()
            tokenizer = Tokenizer(lower=False) ## for tCO
            tokenizer.fit_on_texts(texts, zero_padding=False)
            ## if zero_padding is True, index 0 is reserved, never assigned to an existing word
            self.tag_index = tokenizer.word_index
            self.nb_stags = len(self.tag_index)
            self.idx_to_tag = invert_dict(self.tag_index)
            print('Found {} unique supertags including -unseen- and <-root->.'.format(self.nb_stags))
            f_test = open(path_to_tag_test)
            texts = texts + f_test.readlines() ## do not lowercase tCO
            f_test.close()
            tag_sequences = tokenizer.texts_to_sequences(texts)
            #print(map(lambda x: self.idx_to_tag[x], tag_sequences[self.nb_train_samples+8]))
            self.inputs_train['stags'] = tag_sequences[:self.nb_train_samples]
            self.inputs_test['stags'] = tag_sequences[self.nb_train_samples:]
            self.gold_stags = np.hstack(map(lambda x: x[1:], tag_sequences[self.nb_train_samples:]))
            ## indexing stag files ends

        ## indexing rel files
        f_train = open(path_to_rel)
        texts = f_train.readlines()
        f_train.close()
        tokenizer = Tokenizer(lower=False)
        tokenizer.fit_on_texts(texts, zero_padding=False)
        self.rel_index = tokenizer.word_index
        self.nb_rels = len(self.rel_index)
        self.idx_to_rel = invert_dict(self.rel_index)
        print('Found {} unique rels including -unseen-, NOT including <-root->.'.format(self.nb_rels))
        f_test = open(path_to_rel_test)
        texts = texts + f_test.readlines() ## do not lowercase tCO
        f_test.close()
        rel_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_tag[x], tag_sequences[self.nb_train_samples+8]))
        self.inputs_train['rels'] = rel_sequences[:self.nb_train_samples]
        self.inputs_test['rels'] = rel_sequences[self.nb_train_samples:]
        self.gold_rels = np.hstack(map(lambda x: x[1:], rel_sequences[self.nb_train_samples:]))
        ## indexing rel files ends

        ## indexing arc files
        ## Notice arc sequences are already integers
        f_train = open(path_to_arc)
        arc_sequences = f_train.readlines()
        f_train.close()
        f_test = open(path_to_arc_test)
        arc_sequences = arcs2seq(arc_sequences + f_test.readlines())
        f_test.close()
        self.inputs_train['arcs'] = arc_sequences[:self.nb_train_samples]
        self.inputs_test['arcs'] = arc_sequences[self.nb_train_samples:]
        ## indexing arc files ends
        self.gold_arcs = np.hstack(arc_sequences[self.nb_train_samples:])
        if path_to_punc_test is not None:
            self.punc = arc_sequences[self.nb_train_samples:]
            with open(path_to_punc_test) as fhand:
                for sent_idx, line in zip(xrange(len(self.punc)), fhand):
                    self.punc[sent_idx] = [True for _ in xrange(len(self.punc[sent_idx]))]
                    for punc_idx in map(int, line.split()):
                        self.punc[sent_idx][punc_idx-1] = False
            self.punc = np.hstack(self.punc)#.astype(bool)

        ## padding the train inputs and test inputs
        self.inputs_train = {key: pad_sequences(x, key) for key, x in self.inputs_train.items()}
        self.inputs_train['arcs'] = np.hstack([np.zeros([self.inputs_train['arcs'].shape[0], 1]).astype(int), self.inputs_train['arcs']])
        ## dummy parents for the roots
        random.seed(0)
        perm = np.arange(self.nb_train_samples)
        random.shuffle(perm)
        self.inputs_train = {key: x[perm] for key, x in self.inputs_train.items()}

        self.inputs_test = {key: pad_sequences(x, key) for key, x in self.inputs_test.items()}
        ## dummy parents for the roots
        self.inputs_test['arcs'] = np.hstack([np.zeros([self.inputs_test['arcs'].shape[0], 1]).astype(int), self.inputs_test['arcs']])

        ## padding ends

        ## setting the current indices
        self._index_in_epoch = 0
        self._epoch_completed = 0
        self._index_in_test = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if self._index_in_epoch >= self.nb_train_samples:
                # iterate until the very end do not throw away
            self._index_in_epoch = 0
            self._epoch_completed+=1
            perm = np.arange(self.nb_train_samples)
            random.shuffle(perm)
            self.inputs_train = {key: x[perm] for key, x in self.inputs_train.items()}
            return False
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        self.inputs_train_batch = {}
        x = self.inputs_train['words']
        x_batch = x[start:end]
        max_len = np.max(np.sum(x_batch!=0, axis=1))
        for key, x in self.inputs_train.items():
            x_batch = x[start:end]
            if key == 'chars':
                max_word_len = np.max(np.sum(x_batch!=0, axis=2))
                self.inputs_train_batch[key] = x_batch[:, :max_len-1, :max_word_len] ## max_len-1 because chars do not have <-root->
            else:
                self.inputs_train_batch[key] = x_batch[:, :max_len]
        return True

    def next_test_batch(self, batch_size):
        start = self._index_in_test
        if self._index_in_test >= self.nb_validation_samples:
                # iterate until the very end do not throw away
            self._index_in_test = 0
            return False
        self._index_in_test += batch_size
        end = self._index_in_test
        self.inputs_test_batch = {}
        x = self.inputs_test['words']
        x_batch = x[start:end]
        max_len = np.max(np.sum(x_batch!=0, axis=1))
        for key, x in self.inputs_test.items():
            x_batch = x[start:end]
            if key == 'chars':
                max_word_len = np.max(np.sum(x_batch!=0, axis=2))
                self.inputs_test_batch[key] = x_batch[:, :max_len-1, :max_word_len] ## max_len-1 because chars do not have <-root->
            else:
                self.inputs_test_batch[key] = x_batch[:, :max_len]
        return True

    def output_rels(self, predictions, filename):
        if filename is not None:
            stags = map(lambda x: self.idx_to_rel[x], predictions)
            ## For formatting, let's calculate sentence lengths. np.sum is also faster than a for loop
            sents_lengths = np.sum(self.inputs_test['words']!=0, 1) - 1 ## dummy ROOT
            stag_idx = 0
            with open(filename, 'wt') as fwrite:
                for sent_idx in xrange(len(sents_lengths)):
                    fwrite.write(' '.join(stags[stag_idx:stag_idx+sents_lengths[sent_idx]]))
                    fwrite.write('\n')
                    stag_idx += sents_lengths[sent_idx]

    def output_arcs(self, predictions, filename):
        if filename is not None:
            stags = map(str, predictions)
            ## For formatting, let's calculate sentence lengths. np.sum is also faster than a for loop
            sents_lengths = np.sum(self.inputs_test['words']!=0, 1) - 1 ## dummy ROOT
            stag_idx = 0
            with open(filename, 'wt') as fwrite:
                for sent_idx in xrange(len(sents_lengths)):
                    fwrite.write(' '.join(stags[stag_idx:stag_idx+sents_lengths[sent_idx]]))
                    fwrite.write('\n')
                    stag_idx += sents_lengths[sent_idx]

    def output_stags(self, predictions, filename): ## output stags for joint
        stags = map(lambda x: self.idx_to_tag[x], predictions)
        ## For formatting, let's calculate sentence lengths. np.sum is also faster than a for loop
        ## To Do: allow for the CoNLL format
        sents_lengths = np.sum(self.inputs_test['words']!=0, 1) - 1 ## dummy ROOT
        stag_idx = 0
        with open(filename, 'wt') as fwrite:
            for sent_idx in xrange(len(sents_lengths)):
                fwrite.write(' '.join(stags[stag_idx:stag_idx+sents_lengths[sent_idx]]))
                fwrite.write('\n')
                stag_idx += sents_lengths[sent_idx]

    def output_pos(self, predictions, filename): ## output stags for joint
        stags = map(lambda x: self.idx_to_jk[x], predictions)
        ## For formatting, let's calculate sentence lengths. np.sum is also faster than a for loop
        ## To Do: allow for the CoNLL format
        sents_lengths = np.sum(self.inputs_test['words']!=0, 1) - 1 ## dummy ROOT
        stag_idx = 0
        with open(filename, 'wt') as fwrite:
            for sent_idx in xrange(len(sents_lengths)):
                fwrite.write(' '.join(stags[stag_idx:stag_idx+sents_lengths[sent_idx]]))
                fwrite.write('\n')
                stag_idx += sents_lengths[sent_idx]
    def output_probs(self, probs):
        output_mica_nbest(probs, self.idx_to_tag)

    def output_weight(self, stag_embeddings):
        filename = 'stag_embeddings.txt'
        with open(filename, 'wt') as fout:
            for i in xrange(stag_embeddings.shape[0]):
                if i in self.idx_to_tag.keys():
                    output_row = [self.idx_to_tag[i]]+map(str, stag_embeddings[i])
                    fout.write(' '.join(output_row))
                    fout.write('\n')

    def get_scores(self, predictions, opts, test_opts):
        if test_opts is None:
            metrics = opts.metrics ## use train opts
        else:
            metrics = test_opts.metrics
        scores = {}
        for metric in metrics:
            if metric == 'NoPunct_UAS':
                scores[metric] = np.mean(predictions['arcs'][self.punc] == self.gold_arcs[self.punc])
            elif metric == 'NoPunct_LAS':
                scores[metric] = np.mean((predictions['arcs'][self.punc] == self.gold_arcs[self.punc])*(predictions['rels'][self.punc] == self.gold_rels[self.punc]))
            elif metric == 'UAS':
                scores[metric] = np.mean(predictions['arcs_greedy'] == self.gold_arcs)
            elif metric == 'LAS':
                scores[metric] = np.mean((predictions['arcs_greedy'] == self.gold_arcs)*(predictions['rels_greedy'] == self.gold_rels))
            elif metric == 'CUAS':
                scores[metric] = np.mean((predictions['arcs_greedy'][self.punc] == self.gold_arcs[self.punc])*self.content)
            elif metric == 'CLAS':
                scores[metric] = np.mean(((predictions['arcs_greedy'][self.punc] == self.gold_arcs[self.punc])*(predictions['rels_greedy'][self.punc] == self.gold_rels[self.punc]))*self.content)
            elif metric == 'Stagging':
                scores[metric] = np.mean((predictions['stags'] == self.gold_stags))
            elif metric == 'POS':
                scores[metric] = np.mean((predictions['jk'] == self.gold_jk))
            elif metric == 'NoPunct_LAS_Stagging':
                scores[metric] = np.mean((predictions['arcs_greedy'][self.punc] == self.gold_arcs[self.punc])*(predictions['rels_greedy'][self.punc] == self.gold_rels[self.punc])*(predictions['stags'][self.punc] == self.gold_stags[self.punc]))
            elif metric == 'NoPunct_LAS_Both':
                scores[metric] = np.mean((predictions['arcs_greedy'][self.punc] == self.gold_arcs[self.punc])*(predictions['rels_greedy'][self.punc] == self.gold_rels[self.punc])*(predictions['stags'][self.punc] == self.gold_stags[self.punc])*(predictions['jk'][self.punc] == self.gold_jk[self.punc]))
        return scores

def invert_dict(index_dict): 
    return {j:i for i,j in index_dict.items()}

        

if __name__ == '__main__':
    class Opts(object):
        def __init__(self):
            self.jackknife = 1
            self.embedding_dim = 100
#            self.text_train = 'sample_data/sents/train.txt'
#            self.tag_train = 'sample_data/predicted_stag/train.txt'
#            self.jk_train = 'sample_data/predicted_pos/train.txt'
#            self.arc_train = 'sample_data/arcs/train.txt'
#            self.rel_train = 'sample_data/rels/train.txt'
#            self.text_test = 'sample_data/sents/dev.txt'
#            self.tag_test = 'sample_data/predicted_stag/dev.txt'
#            self.jk_test = 'sample_data/predicted_pos/dev.txt'
#            self.arc_test = 'sample_data/arcs/dev.txt'
#            self.rel_test = 'sample_data/rels/dev.txt'
            self.text_train = 'data/tag_wsj/sents/train.txt'
            self.tag_train = 'data/tag_wsj/predicted_stag/train.txt'
            self.jk_train = 'data/tag_wsj/predicted_pos/train.txt'
            self.arc_train = 'data/tag_wsj/arcs/train.txt'
            self.rel_train = 'data/tag_wsj/rels/train.txt'
            self.text_test = 'data/tag_wsj/sents/dev.txt'
            self.tag_test = 'data/tag_wsj/predicted_stag/dev.txt'
            self.jk_test = 'data/tag_wsj/predicted_pos/dev.txt'
            self.arc_test = 'data/tag_wsj/arcs/dev.txt'
            self.rel_test = 'data/tag_wsj/rels/dev.txt'
            self.punc_test = 'data/tag_wsj/punc/dev.txt'
            self.word_embeddings_file = 'glovevector/glove.6B.100d.txt'
            self.chars_dim = 30
            self.chars_window_size = 30
            self.nb_filters = 30
    opts = Opts()
    data_loader = Dataset(opts)
    #print(data_loader.inputs_train)
    print(data_loader.inputs_train['chars'].shape)
    print(data_loader.inputs_train['words'].shape)
    data_loader.next_batch(10)
    print(data_loader.inputs_train_batch['chars'].shape)
    print(data_loader.inputs_train_batch['words'].shape)
#
