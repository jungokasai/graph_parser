punc_nums = []
with open('eval/punc/test.txt') as fin:
    for line in fin:
        tokens = line.split()
        punc_nums.append([int(token) for token in tokens])
print(len(punc_nums))
print(punc_nums[:3])

sent_idx = 0
with open('data/tag_wsj/predicted_conllu/test.conllu') as fin:
    with open('predicted_test.conllu', 'wt') as fout:
        for line in fin:
            line = line.split()
            if len(line) == 0:
                sent_idx += 1
                fout.write('\n')
            else:
                if int(line[0]) in punc_nums[sent_idx]:
                    line[7] = 'punct'
                fout.write('\t'.join(line))
                fout.write('\n')
