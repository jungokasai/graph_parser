def get_punc(file_name, output_file):
    punc = []
    with open(file_name) as fhand:
        for line in fhand:
            punc_sent = []
            tokens = line.split()
            word_idx = 1
            for token in tokens:
                if token == 'PUNC':
                    word_idx += 1
                    punc_sent.append(word_idx)
            punc.append(punc_sent)
    output_punc(punc, ouput_file)

def output_punc(punc, output_file):
    with open(output_file, 'wt') as fwrite:
        for punc_sent in punc:
            fwrite.write(map(str, punc_sent))
            fwrite.write('\n')

