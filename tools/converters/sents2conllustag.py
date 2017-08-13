def read_sents(sents_file):
    stags = []
    with open(sents_file) as fhand:
        for line in fhand:
            stags_sent = line.split()
            stags.append(stags_sent)
    return stags
    
## inputs is a dictionry {index_in_conllu: sents_file}
def output_conllu(input_conllu_file, output_conllu_file, inputs):
    sent_idx = 0 
    word_idx = 0
    for idx, sents_file in inputs.items():
        inputs[idx] = read_sents(sents_file)
    with open(output_conllu_file, 'wt') as fwrite:
        with open(input_conllu_file) as fhand:
            for line in fhand:
                tokens = line.split()
                if len(tokens) >= 10: 
                    for idx in inputs.keys():
                        if idx <= len(tokens)-1:
                            tokens[idx] = inputs[idx][sent_idx][word_idx]
                        else:
                            tokens.append(inputs[idx][sent_idx][word_idx])
                    word_idx += 1
                else:
                    sent_idx += 1
                    word_idx = 0 
                fwrite.write('\t'.join(tokens))
                fwrite.write('\n')

if __name__ == '__main__':
#    sents_file = '/data/lily/jk964/Dropbox/dev.txt'
#    input_conllu_file = '../../ud/stag_extraction/new_data/WSJ/conllu/wsj.dev.conllu1'
#    output_conllu_file = 'wsj.dev.conllu_stag'
#    output_conllu(sents_file, input_conllu_file, output_conllu_file)
    data_types = ['dev', 'test', 'train']
    for data_type in data_types:
        inputs = {}
        inputs[10] = 'data/tag_wsj/predicted_stag/{}.txt'.format(data_type)
        inputs[4] = 'data/tag_wsj/predicted_pos/{}.txt'.format(data_type)
        input_conllu_file = 'data/tag_wsj/conllu/{}.conllu'.format(data_type)
#        output_conllu_file = 'data/tag_wsj/conllu/{}.conllu0_pos_stag'.format(data_type)
        output_conllu(input_conllu_file, 'test', inputs)


