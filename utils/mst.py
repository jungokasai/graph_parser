from tarjan import Tarjan
import numpy as np

def chu_liu_edmonds(scores_sent, dep2vertex, parent2vertex, committed_edges):
    ## scores_sent [sent_len, sent_len]
    MIN = -10000
    greedy_sent = np.argmax(scores_sent, axis=1)
    tokens = np.arange(1, greedy_sent.shape[0]) 
    graph = Tarjan(greedy_sent, tokens)
    cycles = graph._SCCs
    istree = True
    for cycle in cycles:
        if len(cycle) > 1:
            istree = False
            non_cycle = sorted(list(graph._vertices - cycle))
            cycle = sorted(list(cycle))
            scores_cycle = scores_sent[cycle, greedy_sent[cycle]]
            new_scores_sent = MIN*np.ones([len(non_cycle)+1, len(non_cycle)+1])
            ## add a new vertex representing the cycle
            new_scores_sent[:len(non_cycle), :][:, :len(non_cycle)] = scores_sent[non_cycle, :][:, non_cycle]
            ## leave the scores outside the cycle as they are
            new_dep2vertex = {idx: vertex for idx, vertex in enumerate(map(lambda x: dep2vertex[x], non_cycle))}
            new_parent2vertex = {idx: vertex for idx, vertex in enumerate(map(lambda x: parent2vertex[x], non_cycle))}
            ## for the vertices outside the cycle, dependent and parent indices are the same, inheriting from the original

            ## arcs going away from the cycle
            scores_out_cycle = scores_sent[non_cycle, :][:, cycle]
            ## arcs going out from the cycle i.e. a member in the cycle is a parent of such arcs
            out_max_idx = np.argmax(scores_out_cycle)
            out_score = scores_out_cycle.reshape(-1)[out_max_idx]
            out_dep = out_max_idx//scores_out_cycle.shape[1]
            out_parent = cycle[out_max_idx - out_dep*scores_out_cycle.shape[1]] 
#            out_dep = non_cycle[out_dep]
            out_dep_non_cycle = out_dep
            new_parent2vertex[len(new_parent2vertex)] = parent2vertex[out_parent]
            ## arcs coming into the cycle
            scores_into_cycle = scores_sent[cycle, :][:, non_cycle]
            scores_into_cycle = scores_into_cycle + np.sum(scores_cycle) - scores_cycle.reshape([len(cycle), 1])
            ## deciding a dependent node in the cycle costs the score assigned to the original edge: trade-off
            in_max_idx = np.argmax(scores_into_cycle)
            in_score = scores_into_cycle.reshape(-1)[in_max_idx]
            in_dep = in_max_idx//scores_into_cycle.shape[1]
#            in_parent = non_cycle[in_max_idx - in_dep*scores_out_cycle.shape[1]] 
            in_parent_non_cycle = in_max_idx - in_dep*scores_into_cycle.shape[1]
            in_dep = cycle[in_dep]
            new_dep2vertex[len(new_dep2vertex)] = dep2vertex[in_dep]
            for idx in cycle:
                if idx != in_dep: ## if the incoming arrow points to the node, add the original edge in the cycle
                    committed_edges[dep2vertex[idx]] = parent2vertex[greedy_sent[idx]]
            new_scores_sent[out_dep_non_cycle, -1] = out_score ## parent is the cycle
            new_scores_sent[-1, in_parent_non_cycle] = in_score ## dep is the cycle 

            break
    if istree:
        return greedy_sent, dep2vertex, parent2vertex, committed_edges
    else:
        return chu_liu_edmonds(new_scores_sent, new_dep2vertex, new_parent2vertex, committed_edges)
        
def get_mst(scores_sent):
    predictions = np.zeros([scores_sent.shape[0]]).astype(int)
    dep2vertex = {i: i for i in xrange(scores_sent.shape[0])}
    parent2vertex = {i: i for i in xrange(scores_sent.shape[0])}
    greedy_sent, dep2vertex, parent2vertex, committed_edges =  chu_liu_edmonds(scores_sent, dep2vertex, parent2vertex, {})
    for child, parent in committed_edges.items():
        predictions[child] = parent
    for child, parent in enumerate(greedy_sent):
        predictions[dep2vertex[child]] = parent2vertex[parent]
    return predictions

if __name__ == '__main__':
    b = 1
    n = 4  ## including ROOT
    print('John saw Mary')
    print('from McDonald et al. EMNLP 2017')
    scores = np.zeros([b, n, n])
    weight = np.ones([b, n]).astype(bool)
    scores[0] = np.array([[0, 0, 0, 0], [9, 0, 30, 11], [10, 20, 0, 0], [9, 3, 30, 0]])
#    scores_sent = scores[0]
    for i in xrange(200):
        predictions = get_mst(scores[0])
        print(predictions)
#    get_mst(scores_sent, dep2vertex, parent2vertex, {})

        
