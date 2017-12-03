import sys
def read_sents(file_name):
    stags = []
    unique_stags = set()
    with open(file_name) as fhand:
        for line in fhand:
            stags_sent = line.split()
            stags.append(stags_sent)
            unique_stags = unique_stags.union(set(stags_sent))
    print(len(unique_stags))
    return stags

def read_n_best(file_name):
    stags = []
    stags_sent = []
    with open(file_name) as fhand:
        for line in fhand:
            stags_token = line[1:].split()
            if len(stags_token) == 1:
                ## EOS
                stags.append(stags_sent)
                stags_sent = []
            else:
                stags_sent.append(map(lambda x: x.split(':')[0][:-2], stags_token))
    return stags

def sentwise_acc(gold_stags, predicted_stags):
    count = 0
    nb_correct = 0
    assert(len(gold_stags) == len(predicted_stags))
    for sent_idx in xrange(len(gold_stags)):
        count += 1
        gold_stags_sent = gold_stags[sent_idx]
        predicted_stags_sent = predicted_stags[sent_idx] 
        if ' '.join(gold_stags_sent) == ' '.join(predicted_stags_sent):
            nb_correct += 1
    return float(nb_correct)/count*100

def tokenwise_acc(gold_stags, predicted_stags, n):
    count = 0
    nb_correct = 0
    assert(len(gold_stags) == len(predicted_stags))
    for sent_idx in xrange(len(gold_stags)):
        gold_stags_sent = gold_stags[sent_idx]
        predicted_stags_sent = predicted_stags[sent_idx]
        for word_idx in xrange(len(gold_stags_sent)):
            count += 1
            if gold_stags_sent[word_idx] in predicted_stags_sent[word_idx][:n]:
                nb_correct += 1
    return float(nb_correct)/count*100

if __name__ == '__main__':
    gold = sys.argv[1]
    gold_stags = read_sents(gold)
    predicted = sys.argv[2]
    n = int(sys.argv[3])
    predicted_stags = read_n_best(predicted)
    for n in [1, 2, 3, 5, 10]:
        acc = tokenwise_acc(gold_stags, predicted_stags, n)
        print('{} Best Raw Accuracy {}'.format(n, round(acc, 2)))
