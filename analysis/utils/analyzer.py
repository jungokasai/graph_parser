import pickle
import os
import numpy as np

def main(model_type):
    ## load data
    sents = read_stags(os.path.join('data', 'gold', 'sents', 'dev.txt'))
    gold_arcs = read_stags(os.path.join('data', 'gold', 'arcs', 'dev.txt'))
    gold_rels = read_stags(os.path.join('data', 'gold', 'rels', 'dev.txt'))
    predicted_arcs = read_stags(os.path.join('data', model_type, 'predicted_arcs', 'dev.txt'))
    predicted_rels = read_stags(os.path.join('data', model_type, 'predicted_rels', 'dev.txt'))
    rec_total, rec_correct, prec_total, prec_correct = dep_length(gold_arcs, gold_rels, predicted_arcs, predicted_rels)
    #accuracies = dep_length(gold_arcs, gold_rels, gold_arcs, predicted_rels)
    print('Dep Length')
    print('Precision')
    prec_accuracies = convert2latex(prec_total, prec_correct, 10)
    print('Recall')
    rec_accuracies = convert2latex(rec_total, rec_correct, 10)
    print('F1')
    get_f1(prec_accuracies, rec_accuracies)

    print('Root Length')
        
    rec_total, rec_correct, prec_total, prec_correct = root_length(gold_arcs, gold_rels, predicted_arcs, predicted_rels)
    #accuracies = dep_length(gold_arcs, gold_rels, gold_arcs, predicted_rels)
    print('Precision')
    prec_accuracies = convert2latex(prec_total, prec_correct, 10)
    print('Recall')
    rec_accuracies = convert2latex(rec_total, rec_correct, 10)
    print('F1')
    get_f1(prec_accuracies, rec_accuracies)

    print('Rel Type')
    rec_total, rec_correct, prec_total, prec_correct = rel_type(gold_arcs, gold_rels, predicted_arcs, predicted_rels)
    print(rec_correct)
    print(prec_correct)
def read_stags(path_to_sents):
    ## create a list of lists
    stags = []
    with open(path_to_sents) as f_stags:
        for stags_sent in f_stags:
            stags_sent = stags_sent.split()
            stags.append(stags_sent)
    return stags

def dep_length(gold_arcs, gold_rels, predicted_arcs, predicted_rels):
    rec_correct = {}
    rec_total = {}
    prec_correct = {}
    prec_total = {}
    for sent_idx in range(len(gold_arcs)):
        gold_arcs_sent = gold_arcs[sent_idx]
        gold_rels_sent = gold_rels[sent_idx]
        predicted_arcs_sent = predicted_arcs[sent_idx]
        predicted_rels_sent = predicted_rels[sent_idx]
        for word_idx in range(len(gold_arcs_sent)):
            gold_arc = int(gold_arcs_sent[word_idx])
            gold_rel = gold_rels_sent[word_idx]
            predicted_rel = predicted_rels_sent[word_idx]
            predicted_arc = int(predicted_arcs_sent[word_idx])
            rec_distance = abs((word_idx+1)-gold_arc)
            prec_distance = abs((word_idx+1)-predicted_arc)
            if rec_distance in rec_correct.keys():
                rec_total[rec_distance] += 1
                if gold_arc == predicted_arc:
                    rec_correct[rec_distance] += 1
            else:
                rec_total[rec_distance] = 1
                rec_correct[rec_distance] = 0
                if gold_arc == predicted_arc:
                    rec_correct[rec_distance] += 1
            if prec_distance in prec_correct.keys():
                prec_total[prec_distance] += 1
                if gold_arc == predicted_arc:
                    prec_correct[prec_distance] += 1
            else:
                prec_total[prec_distance] = 1
                prec_correct[prec_distance] = 0
                if gold_arc == predicted_arc:
                    prec_correct[prec_distance] += 1
    #accuracies = {}
    #for length in correct.keys():
    #    accuracies[length] = float(correct[length])/total[length]
    return rec_total, rec_correct, prec_total, prec_correct

def root_length(gold_arcs, gold_rels, predicted_arcs, predicted_rels):
    rec_correct = {}
    rec_total = {}
    prec_correct = {}
    prec_total = {}
    for sent_idx in range(len(gold_arcs)):
        gold_arcs_sent = gold_arcs[sent_idx]
        gold_rels_sent = gold_rels[sent_idx]
        predicted_arcs_sent = predicted_arcs[sent_idx]
        predicted_rels_sent = predicted_rels[sent_idx]
        rec_dict = {}
        for word_idx, arc in enumerate(gold_arcs_sent):
            rec_dict[word_idx+1] = int(arc)
        prec_dict = {}
        for word_idx, arc in enumerate(predicted_arcs_sent):
            prec_dict[word_idx+1] = int(arc)
        for word_idx in range(len(gold_arcs_sent)):
            gold_arc = int(gold_arcs_sent[word_idx])
            gold_rel = gold_rels_sent[word_idx]
            predicted_rel = predicted_rels_sent[word_idx]
            predicted_arc = int(predicted_arcs_sent[word_idx])
            rec_distance = 0 
            child = word_idx+1
            for _ in range(len(gold_arcs_sent)):
                parent = rec_dict[child]
                rec_distance += 1
                child = parent
                if parent == 0:
                    break
            prec_distance = 0
            child = word_idx+1
            for _ in range(len(gold_arcs_sent)):
                parent = prec_dict[child]
                prec_distance += 1
                child = parent
                if parent == 0:
                    break
            if rec_distance in rec_correct.keys():
                rec_total[rec_distance] += 1
                if gold_arc == predicted_arc:
                    rec_correct[rec_distance] += 1
            else:
                rec_total[rec_distance] = 1
                rec_correct[rec_distance] = 0
                if gold_arc == predicted_arc:
                    rec_correct[rec_distance] += 1
            if prec_distance in prec_correct.keys():
                prec_total[prec_distance] += 1
                if gold_arc == predicted_arc:
                    prec_correct[prec_distance] += 1
            else:
                prec_total[prec_distance] = 1
                prec_correct[prec_distance] = 0
                if gold_arc == predicted_arc:
                    prec_correct[prec_distance] += 1
    #accuracies = {}
    #for length in correct.keys():
    #    accuracies[length] = float(correct[length])/total[length]
    return rec_total, rec_correct, prec_total, prec_correct

def rel_type(gold_arcs, gold_rels, predicted_arcs, predicted_rels):
    rec_correct = {}
    rec_total = {}
    prec_correct = {}
    prec_total = {}
    for sent_idx in range(len(gold_arcs)):
        gold_arcs_sent = gold_arcs[sent_idx]
        gold_rels_sent = gold_rels[sent_idx]
        predicted_arcs_sent = predicted_arcs[sent_idx]
        predicted_rels_sent = predicted_rels[sent_idx]
        for word_idx in range(len(gold_arcs_sent)):
            gold_arc = int(gold_arcs_sent[word_idx])
            gold_rel = gold_rels_sent[word_idx]
            predicted_rel = predicted_rels_sent[word_idx]
            predicted_arc = int(predicted_arcs_sent[word_idx])
            rec_distance = gold_rel
            prec_distance = predicted_rel
            if rec_distance in rec_correct.keys():
                rec_total[rec_distance] += 1
                if gold_arc == predicted_arc:
                    rec_correct[rec_distance] += 1
            else:
                rec_total[rec_distance] = 1
                rec_correct[rec_distance] = 0
                if gold_arc == predicted_arc:
                    rec_correct[rec_distance] += 1
            if prec_distance in prec_correct.keys():
                prec_total[prec_distance] += 1
                if gold_arc == predicted_arc:
                    prec_correct[prec_distance] += 1
            else:
                prec_total[prec_distance] = 1
                prec_correct[prec_distance] = 0
                if gold_arc == predicted_arc:
                    prec_correct[prec_distance] += 1
    #accuracies = {}
    #for length in correct.keys():
    #    accuracies[length] = float(correct[length])/total[length]
    return rec_total, rec_correct, prec_total, prec_correct
def convert2latex(total, correct, threshold):
    total_long = 0
    correct_long = 0
    accuracies = [0 for _ in range(threshold+1)]
    for length in correct.keys():
        if length <= threshold:
            accuracies[length-1] = round(float(correct[length])/total[length]*100, 2)
        else:
            total_long += total[length]
            correct_long += correct[length]
    accuracies[threshold] = round(correct_long/float(total_long)*100, 2)
    output = ''
    for i, acc in enumerate(accuracies):
        x_ = 10+i*20
        output += '({},{})'.format(x_, acc)
    #print(output)
    return accuracies
def get_f1(prec_accuracies, rec_accuracies):
    f1 = []
    for prec, rec in zip(prec_accuracies, rec_accuracies):
        f1.append((prec+rec)/2)
    output = ''
    for i, acc in enumerate(f1):
        x_ = 10+i*20
        output += '({},{})'.format(x_, acc)
    print(output)

    

if __name__ == '__main__':
    main('naacl')
    print('-'*50)
    main('emnlp')
