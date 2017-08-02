import numpy as np
from preprocessing import Tokenizer
from preprocessing import pad_sequences 
import os
import sys
import pickle
import random

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
        else:
            path_to_text_test = test_opts.text_test
            path_to_tag_test = test_opts.tag_test
            path_to_jk_test = test_opts.jk_test
            path_to_arc_test = test_opts.arc_test
            path_to_rel_test = test_opts.rel_test

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
        print('Found {} unique lowercased words including -unseen-.'.format(self.nb_words))

        # lookup the glove word embeddings
        # need to reserve indices for testing file. 
        glove_size = opts.embedding_dim
        self.embeddings_index = {}
        print('Indexing word vectors.')
        f = open('glovevector/glove.6B.{}d.txt'.format(glove_size))
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
        ## indexing jackknife files
        f_train = open(path_to_jk)
        texts = f_train.readlines()
        f_train.close()
        tokenizer = Tokenizer(lower=False) 
        tokenizer.fit_on_texts(texts)
        self.jk_index = tokenizer.word_index
        self.nb_jk = len(self.jk_index)
        self.idx_to_jk = invert_dict(self.jk_index)
        print('Found {} unique tags including -unseen-.'.format(self.nb_jk))
        f_test = open(path_to_jk_test)
        texts = texts + f_test.readlines() ## do not lowercase tCO
        f_test.close()
        jk_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_jk[x], jk_sequences[self.nb_train_samples]))
        self.inputs_train['jk'] = jk_sequences[:self.nb_train_samples]
        self.inputs_test['jk'] = jk_sequences[self.nb_train_samples:]
        ## indexing jackknife files ends
        ## indexing stag files
        f_train = open(path_to_tag)
        texts = f_train.readlines()
        f_train.close()
        tokenizer = Tokenizer(lower=False) ## for tCO
        tokenizer.fit_on_texts(texts, zero_padding=False)
        #print(tokenizer.word_index['-unseen-'])
        self.tag_index = tokenizer.word_index
        self.nb_tags = len(self.tag_index)
        self.idx_to_tag = invert_dict(self.tag_index)
        print('Found {} unique tags including -unseen-.'.format(self.nb_tags))
        f_test = open(path_to_tag_test)
        texts = texts + f_test.readlines() ## do not lowercase tCO
        f_test.close()
        tag_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_tag[x], tag_sequences[self.nb_train_samples+8]))
        self.inputs_train['tags'] = tag_sequences[:self.nb_train_samples]
        self.inputs_test['tags'] = tag_sequences[self.nb_train_samples:]
        ## indexing stag files ends

        ## padding the train inputs and test inputs
        self.inputs_train = {key: pad_sequences(x) for key, x in self.inputs_train.itmes()}
        random.seed(0)
        perm = np.arange(self.nb_train_samples)
        random.shuffle(perm)
        self.inputs_train = {key: x[perm] for key, x in self.inputs_train.items()}

        self.inputs_test = {key: pad_sequences(x) for key, x in self.inputs_test.items()}

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
            self.inputs_train = [x[perm] for x in self.inputs_train]
            return False
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        self.inputs_train_batch = []
        for i, x in enumerate(self.inputs_train):
            x_batch = x[start:end]
            if i == 0:
                max_len = np.max(np.sum(x_batch!=0, axis=-1))
            self.inputs_train_batch.append(x_batch[:, :max_len])
        return True

    def next_test_batch(self, batch_size):

        start = self._index_in_test
        if self._index_in_test >= self.nb_validation_samples:
                # iterate until the very end do not throw away
            self._index_in_test = 0
            return False
        self._index_in_test += batch_size
        end = self._index_in_test
        self.inputs_test_batch = []
        for i, x in enumerate(self.inputs_test):
            x_batch = x[start:end]
            if i == 0:
                max_len = np.max(np.sum(x_batch!=0, axis=-1))
            self.inputs_test_batch.append(x_batch[:, :max_len])
        return True

    def output_stags(self, predictions, filename):
        stags = map(lambda x: self.idx_to_tag[x], predictions)
        ## For formatting, let's calculate sentence lengths. np.sum is also faster than a for loop
        ## To Do: allow for the CoNLL format
        sents_lengths = np.sum(self.inputs_test[0]!=0, 1)
        stag_idx = 0
        with open(filename, 'wt') as fwrite:
            for sent_idx in xrange(len(sents_lengths)):
                fwrite.write(' '.join(stags[stag_idx:stag_idx+sents_lengths[sent_idx]]))
                fwrite.write('\n')
                stag_idx += sents_lengths[sent_idx]

def invert_dict(index_dict): 
    return {j:i for i,j in index_dict.items()}


#if __name__ == '__main__':
#    class Opts(object):
#        def __init__(self):
#            self.task = 'Super_models'
#            self.jackknife = 1
#            self.embedding_dim = 100
#    opts = Opts()
#    data_loader = Dataset(opts)
#    data_loader.next_batch(2)
#    print(data_loader.inputs_train_batch[0])
#    data_loader.next_test_batch(3)
#    print(data_loader.inputs_test_batch[0])
#
