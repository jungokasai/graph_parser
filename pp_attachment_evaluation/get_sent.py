with open('pp_attachment_evaluation/test_set.txt') as fhand:
    with open('pp_attachment_evaluation/test.txt', 'wt') as fwrite:
        for line in fhand:
            tokens = line.split()
            fwrite.write(' '.join(tokens[3:]))
            fwrite.write('\n')
