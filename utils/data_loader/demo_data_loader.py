import os, sys
sys.path.append(os.getcwd())
import numpy as np
from utils.data_loader.preprocessing import Tokenizer, pad_sequences, arcs2seq
from utils.mica.nbest import output_mica_nbest
import pickle
import random
import io


class Demo_Dataset(object):

    def __init__(self, base_dir, glove_dir, glove_size):

        #self.inputs_train = {}
        #self.inputs_test = {}

        ## indexing sents files
        tokenizer_dir = os.path.join(base_dir, 'tokenizers')
        with open(os.path.join(tokenizer_dir, 'word_tokenizer.pkl')) as fin:
            self.word_tokenizer = pickle.load(fin)
        self.nb_train_samples = 0
        self.word_index = self.word_tokenizer.word_index
        self.nb_words = len(self.word_index)
        # lookup the glove word embeddings
        # need to reserve indices for testing file. 
        self.embeddings_index = {}
        print('Indexing word vectors.')
        f = open(os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(glove_size)))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found {} word vectors.'.format(len(self.embeddings_index)))

        unseens = list(set(self.embeddings_index.keys()) - set(self.word_index.keys())) ## list of words that appear in glove but not in the training set
        nb_unseens = len(unseens)

        self.word_embeddings = np.zeros((self.nb_words+1+nb_unseens, glove_size)) ## +1 for padding (idx 0)
        for word, i in self.word_index.items(): ## first index the words in the training set
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None: ## otherwise zero vector
                self.word_embeddings[i] = embedding_vector
        for unseen in unseens:
            self.word_index[unseen] = len(self.word_index) + 1 ## add unseen words to the word_index dictionary
            self.word_embeddings[self.word_index[unseen]] = self.embeddings_index[unseen]
        self.idx_to_word = invert_dict(self.word_index)
        print('End glove indexing')

        #f_test = open(path_to_text_test)
        #texts = texts +  f_test.readlines()
        #self.nb_validation_samples = len(texts) - self.nb_train_samples
        #f_test.close()
        #text_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_word[x], text_sequences[self.nb_train_samples]))
        #self.inputs_train['words'] = text_sequences[:self.nb_train_samples]
        #self.inputs_test['words'] = text_sequences[self.nb_train_samples:]
        ## indexing sents files ends
        ## indexing char files
        #if opts.chars_dim > 0:
        with open(os.path.join(tokenizer_dir, 'char_tokenizer.pkl')) as fin:
            self.char_tokenizer = pickle.load(fin)
        self.char_index = self.char_tokenizer.word_index
        self.nb_chars = len(self.char_index)
        self.idx_to_char = invert_dict(self.char_index)
        print('Found {} unique characters including -unseen-. NOT including <-root->.'.format(self.nb_chars))
        #f_test = io.open(path_to_text_test, encoding='utf-8')
        #f_test.close()
        #char_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_jk[x], jk_sequences[self.nb_train_samples]))
        #self.inputs_train['chars'] = char_sequences[:self.nb_train_samples]
        #self.inputs_test['chars'] = char_sequences[self.nb_train_samples:]
        ## indexing char files ends

    ## indexing jackknife files
        with open(os.path.join(tokenizer_dir, 'pos_tokenizer.pkl')) as fin:
            self.pos_tokenizer = pickle.load(fin)
        texts = []
        self.jk_index = self.pos_tokenizer.word_index
        self.nb_jk = len(self.jk_index)
        self.idx_to_jk = invert_dict(self.jk_index)
        #print('Found {} unique POS tags including -unseen- and <-root->.'.format(self.nb_jk))
        #f_test = open(path_to_jk_test)
        #texts = texts + f_test.readlines() ## do not lowercase tCO
        #f_test.close()
        #jk_sequences = tokenizer.texts_to_sequences(texts)
        #self.inputs_train['jk'] = jk_sequences[:self.nb_train_samples]
        #self.inputs_test['jk'] = jk_sequences[self.nb_train_samples:]
        ## indexing jackknife files ends
        ## indexing stag files
        #if (opts.stag_dim > 0) or (opts.model in ['Parsing_Model_Joint', 'Parsing_Model_Shuffle', 'Parsing_Model_Joint_Both']):
        with open(os.path.join(tokenizer_dir, 'stag_tokenizer.pkl')) as fin:
            self.stag_tokenizer = pickle.load(fin)
        #texts = []
        ## if zero_padding is True, index 0 is reserved, never assigned to an existing word
        self.tag_index = self.stag_tokenizer.word_index
        self.nb_stags = len(self.tag_index)
        self.idx_to_tag = invert_dict(self.tag_index)
        print('Found {} unique supertags including -unseen- and <-root->.'.format(self.nb_stags))
        #f_test = open(path_to_tag_test)
        #texts = texts + f_test.readlines() ## do not lowercase tCO
        #f_test.close()
        #tag_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_tag[x], tag_sequences[self.nb_train_samples+8]))
#            self.inputs_train['stags'] = tag_sequences[:self.nb_train_samples]
        #self.inputs_test['stags'] = tag_sequences[self.nb_train_samples:]
        #self.gold_stags = np.hstack(map(lambda x: x[1:], tag_sequences[self.nb_train_samples:]))
        ## indexing stag files ends

        ## indexing rel files
        with open(os.path.join(tokenizer_dir, 'rel_tokenizer.pkl')) as fin:
            self.rel_tokenizer = pickle.load(fin)
        #texts = []
        self.rel_index = self.rel_tokenizer.word_index
        self.nb_rels = len(self.rel_index)
        self.idx_to_rel = invert_dict(self.rel_index)
        print('Found {} unique rels including -unseen-, NOT including <-root->.'.format(self.nb_rels))

    def run_on_sents(self, sents):
        self.inputs_test_batch = {}
        self.inputs_test_batch['words'] = self.word_tokenizer.texts_to_sequences(sents)
        self.inputs_test_batch['chars'] = self.char_tokenizer.texts_to_sequences(map(unicode, sents)) ## unicode for char. Otherwise will screw Chinese.
        self.inputs_test_batch = {key: pad_sequences(x, key) for key, x in self.inputs_test_batch.items()}
    def get_results(self, predictions):
        new_predictions = {}
        for key, item in predictions.items():
            if key == 'rels':
                new_predictions[key] = map(lambda x: self.idx_to_rel[x], item)
            elif key == 'arcs':
                new_predictions[key] = map(str, item)
            elif key == 'stags':
                new_predictions[key] = map(lambda x: self.idx_to_tag[x], item)
            elif key == 'jk':
                new_predictions[key] = map(lambda x: self.idx_to_jk[x], item)
        return new_predictions

def invert_dict(index_dict): 
    return {j:i for i,j in index_dict.items()}

        

if __name__ == '__main__':
    data_loader = Demo_Dataset(os.path.join(os.getcwd(), 'demo'), 'glovevector', 100)
    data_loader.run_on_sents(['TAG is fun .', 'We should study it'])
    print(data_loader.inputs_test_batch['words'].shape)
    print(data_loader.inputs_test_batch['chars'])
