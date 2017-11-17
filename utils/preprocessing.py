# -*- coding: utf-8 -*-
'''These preprocessing utilities would greatly benefit
from a fast Cython rewrite.
'''
from __future__ import absolute_import
from __future__ import division

import string
import sys
import numpy as np
from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f


def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()

    #text = text.translate(maketrans(filters, split*len(filters)))
    seq = text.split()
    return seq
    #return [_f for _f in seq if _f]


def one_hot(text, n, filters=base_filter(), lower=True, split=" "):
    seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]

def pad_sequences(sequences, feature, window = False, dtype='int32',
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''

    if feature == 'chars':
        maxlen=None
        lengths = [len(s) for s in sequences]

        window_size = 0

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        max_word_len = 0
        for s in sequences:
            for word in s:
                if max_word_len < len(word):
                    max_word_len = len(word)
        x = np.zeros((nb_samples, maxlen + 2*window_size, max_word_len)).astype(dtype)

        for sent_idx, s in enumerate(sequences):
            for word_idx, word in enumerate(s):
                if truncating == 'pre':
                    trunc = word[-max_word_len:]
                elif truncating == 'post':
                    trunc = word[:max_word_len]
                else:
                    raise ValueError('Truncating type "%s" not understood' % truncating)

                # check `trunc` has expected shape
                trunc = np.asarray(trunc, dtype=dtype)

                if padding == 'post':
                    x[sent_idx, word_idx, :len(trunc)] = trunc
                elif padding == 'pre':
                    x[sent_idx, word_idx, -len(trunc):] = trunc
                else:
                    raise ValueError('Padding type "%s" not understood' % padding)
        return x
    else:
        maxlen=None
        lengths = [len(s) for s in sequences]

        window_size = 0


        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
        x = (np.zeros((nb_samples, maxlen + 2*window_size) + sample_shape)).astype(dtype)

        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
           
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x

class Tokenizer(object):
    def __init__(self, nb_words=None, filters=base_filter(),
                 lower=True, split=' ', char_level=False, char_encoding=False, root=True):
        '''The class allows to vectorize a text corpus, by turning each
        text into either a sequence of integers (each integer being the index
        of a token in a dictionary) or into a vector where the coefficient
        for each token could be binary, based on word count, based on tf-idf...
        # Arguments
            nb_words: the maximum number of words to keep, based
                on word frequency. Only the most common `nb_words` words will
                be kept.
            filters: a string where each element is a character that will be
                filtered from the texts. The default is all punctuation, plus
                tabs and line breaks, minus the `'` character.
            lower: boolean. Whether to convert the texts to lowercase.
            split: character or string to use for token splitting.
            char_level: if True, every character will be treated as a word.
        By default, all punctuation is removed, turning the texts into
        space-separated sequences of words
        (words maybe include the `'` character). These sequences are then
        split into lists of tokens. They will then be indexed or vectorized.
        `0` is a reserved index that won't be assigned to any word.
        '''
        self.word_counts = {}
        self.word_docs = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.nb_words = nb_words
        self.document_count = 0
        self.char_level = char_level
        self.char_encoding = char_encoding
        self.root = root

    def fit_on_texts(self, texts, zero_padding = True, non_split = False):
        '''Required before using texts_to_sequences or texts_to_matrix
        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        '''
        self.document_count = 0
        if self.char_encoding:
            for text in texts:
                self.document_count += 1
                seq = text if self.char_level or non_split else text_to_word_sequence(text, self.filters, self.lower, self.split)
                for w in seq:
                    for char in w:
                        if char in self.word_counts:
                            self.word_counts[char] += 1
                        else:
                            self.word_counts[char] = 1
	else:
	    for text in texts:
		self.document_count += 1
		seq = text if self.char_level or non_split else text_to_word_sequence(text, self.filters, self.lower, self.split)
		for w in seq:
		    if w in self.word_counts:
			self.word_counts[w] += 1
		    else:
			self.word_counts[w] = 1
		for w in set(seq):
		    if w in self.word_docs:
			self.word_docs[w] += 1
		    else:
			self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
            
        wcounts.sort(key=lambda x: x[1], reverse=True)
        #print(wcounts)
        sorted_voc = [wc[0] for wc in wcounts]
        # if zero_padding is true, note that index 0 is reserved, never assigned to an existing word
        zero_padding = int(zero_padding)
        self.word_index = dict(list(zip(sorted_voc, list(range(zero_padding, len(sorted_voc) + zero_padding)))))

        self.word_index['-unseen-'] = len(self.word_index) + zero_padding
        if self.root:
            self.word_index['<-root->'] = len(self.word_index) + zero_padding

        self.index_docs = {}
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def fit_on_sequences(self, sequences):
        '''Required before using sequences_to_matrix
        (if fit_on_texts was never called)
        '''
        self.document_count = len(sequences)
        self.index_docs = {}
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                if i not in self.index_docs:
                    self.index_docs[i] = 1
                else:
                    self.index_docs[i] += 1
    def nbest_to_sequences(self, data):
        nb_words = self.nb_words
        output_data = []
        nbest_stags_sent = []
        for line in data:
            tags = line.split()
            if tags[0] == '...EOS...':
                output_data.append(nbest_stags_sent)
                nbest_stags_sent = []
            else:
                tag_indecies = tuple(map(lambda w: self.word_index.get(w, self.word_index['-unseen-']), tags)) ## unseeen word
                nbest_stags_sent.append(tag_indecies)
        return output_data

    def texts_to_sequences(self, texts, non_split=False):
        '''Transforms each text in texts in a sequence of integers.
        Only top "nb_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        Returns a list of sequences.
        '''
        res = []
        for vect in self.texts_to_sequences_generator(texts, non_split):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, texts, non_split):
        '''Transforms each text in texts in a sequence of integers.
        Only top "nb_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        Yields individual sequences.
        # Arguments:
            texts: list of strings.
        '''
        nb_words = self.nb_words
        if self.char_encoding:
            for text in texts:
                seq = text if self.char_level or non_split else text_to_word_sequence(text, self.filters, self.lower, self.split)
                vects = []
                for w in seq:
                    vect = []
                    for char in w:
                        i = self.word_index.get(char, self.word_index['-unseen-']) ## unseeen word
                        vect.append(i)
                    vects.append(vect)
                yield vects
        else:
            for text in texts:
                seq = text if self.char_level or non_split else text_to_word_sequence(text, self.filters, self.lower, self.split)
                if self.root:
                    vect = [self.word_index['<-root->']]
                else:
                    vect = []
                for w in seq:
                    i = self.word_index.get(w, self.word_index['-unseen-']) ## unseeen word
                   # if i is not None:
                   #     if nb_words and i >= nb_words:
                   #         continue
                   #     else:
                    vect.append(i)
                yield vect

    def cap_indicator(self, texts):
        vects =[]
        
        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, False, self.split) # don't decapitalize words in this case to see capitalization

            vect = map(lambda x: int(x[0].isupper())+1, seq)  # add one so pad is zero
         
            vects.append(vect) 
        return vects

    def num_detect(self, word):
        try:
            num=float(word)
            if num == 1:
                return 2 
            return 1
        except ValueError:
            return 0 

    def num_indicator(self, texts):
        vects =[]

        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, False, self.split) # don't decapitalize words in this case to see capitalization

            vect = map(lambda x: self.num_detect(x), seq)  # Non-number is 0

            vects.append(vect)
        return vects


    def suffix_extract(self, texts):
        vects =[]
        
        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, True, self.split) #lowercase suffix

            vect = map(lambda x: x[-2:], seq)  # add one so pad is zero
         
            vects.append(vect) 
        return vects
    def suffix_extract(self, texts):
        vects =[]
        
        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, True, self.split) #lowercase suffix

            vect = map(lambda x: x[-2:], seq)  # add one so pad is zero
         
            vects.append(vect) 
        return vects


    def texts_to_matrix(self, texts, mode='binary'):
        '''Convert a list of texts to a Numpy matrix,
        according to some vectorization mode.
        # Arguments:
            texts: list of strings.
            modes: one of "binary", "count", "tfidf", "freq"
        '''
        sequences = self.texts_to_sequences(texts)
        return self.sequences_to_matrix(sequences, mode=mode)

    def sequences_to_matrix(self, sequences, mode='binary'):
        '''Converts a list of sequences into a Numpy matrix,
        according to some vectorization mode.
        # Arguments:
            sequences: list of sequences
                (a sequence is a list of integer word indices).
            modes: one of "binary", "count", "tfidf", "freq"
        '''
        if not self.nb_words:
            if self.word_index:
                nb_words = len(self.word_index) + 1
            else:
                raise Exception('Specify a dimension (nb_words argument), '
                                'or fit on some text data first.')
        else:
            nb_words = self.nb_words

        if mode == 'tfidf' and not self.document_count:
            raise Exception('Fit the Tokenizer on some data '
                            'before using tfidf mode.')

        X = np.zeros((len(sequences), nb_words))
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            counts = {}
            for j in seq:
                if j >= nb_words:
                    continue
                if j not in counts:
                    counts[j] = 1.
                else:
                    counts[j] += 1
            for j, c in list(counts.items()):
                if mode == 'count':
                    X[i][j] = c
                elif mode == 'freq':
                    X[i][j] = c / len(seq)
                elif mode == 'binary':
                    X[i][j] = 1
                elif mode == 'tfidf':
                    # Use weighting scheme 2 in
                    #   https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                    tf = 1 + np.log(c)
                    idf = np.log(1 + self.document_count / (1 + self.index_docs.get(j, 0)))
                    X[i][j] = tf * idf
                else:
                    raise Exception('Unknown vectorization mode: ' + str(mode))
        return X

def get_scores(tag_file):
    scores_dict = {}
    with open(tag_file) as fhand:
        sent_idx = 0
        word_number = 1
        for line in fhand:
            words = line.split()
            if words[0] == '...EOS...':
               sent_idx += 1
               word_number = 1
            else:
                scores = map(lambda x: np.log(float(x)), words)
                #scores = map(lambda x: float(x), words)
                for i, score in enumerate(scores):
                    scores_dict[(sent_idx, word_number, i)] = score
                word_number += 1
    return scores_dict

def arcs2seq(texts):
    seq = []
    for text in texts:
        vect = []
        for w in text.split():
            try:
                vect.append(int(w))
            except:
                vect.append(0) ## dummy for no_gold
        seq.append(vect)
    return seq
