from edmonds import get_mst
import numpy as np

def predict_arcs_rels(arc_outputs, rel_scores, non_padding):
    ## arc_outputs non_padding [b, n (dep), n (parent)]
    ## rel_scores [b, n (dep), n (parent), r]
    MIN = -10000
    predicted_arcs = []
    predicted_rels = []
    for sent_idx in xrange(arc_outputs.shape[0]):
        scores_sent = arc_outputs[sent_idx][non_padding[sent_idx], :][:, non_padding[sent_idx]] 
        #non_roots[best_root] = False
        #scores_sent[non_roots, 0] = MIN
        #scores_sent[best_root] = -MIN
        predicted_arcs_sent = get_mst(scores_sent)
        predicted_arcs.append(predicted_arcs_sent[1:]) ## skip ROOT
        rel_scores_sent = rel_scores[sent_idx][np.arange(predicted_arcs_sent.shape[0]), predicted_arcs_sent, :] ## [n, r]
        predicted_rels_sent = np.argmax(rel_scores_sent, axis=1)
        predicted_rels.append(predicted_rels_sent[1:]) ## skip ROOT
    predicted_arcs = np.hstack(predicted_arcs)
    predicted_rels = np.hstack(predicted_rels)
    return predicted_arcs, predicted_rels
