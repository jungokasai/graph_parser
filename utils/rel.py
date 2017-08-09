import tensorflow as tf 

def get_rel_weights(name, units, nb_rels): # no dropout
    weights = {}
    with tf.variable_scope(name) as scope:
	weights['U-rel'] = tf.get_variable('U-rel', [units, nb_rels, units])
	weights['W-rel'] = tf.get_variable('W-rel', [units, nb_rels])
	weights['b-rel'] = tf.get_variable('b-rel', [nb_rels])
    return weights

## rel_equation is fairly complicated, so let's use this shorthand
## batch_size: b
## units: d
## nb_rels: r 
## seq_len (including root): n

def rel_equation(H_rel_head, H_rel_dep, weights, predictions, test_opts=None): 
    ## H_rel_head: [n, b, d]
    ## H_rel_dep: [n, b, d]
    ## predictions: [b, n]
    shape = tf.shape(H_rel_head)
    n, b, d = shape[0], shape[1], shape[2]
    r = tf.shape(weights['U-rel'])[1]
    H_rel_head = tf.transpose(H_rel_head, [1, 0, 2]) 
    H_rel_dep = tf.transpose(H_rel_dep, [1, 0, 2]) 
    ## H_rel_head: [b, n, d]
    ## H_rel_dep: [b, n, d]
    if test_opts is None:
        one_hot_pred = tf.one_hot(predictions, n)
        ## [b, n] => [b, n, n]
        H_rel_head = tf.matmul(one_hot_pred, H_rel_head) ## filtered through predictions
        ## [b, n, n] x [b, n, d] => [b, n, d]
        U_rel = tf.reshape(weights['U-rel'], [d, r*d]) ## [d, rd]
        interactions = tf.reshape(tf.matmul(tf.reshape(H_rel_head, [b*n, d]), U_rel), [b*n, r, d])
        ## [bn, d] x [d, rd] => [bn, rd] => [bn, r, d]
        interactions = tf.reshape(tf.matmul(interactions, tf.reshape(H_rel_dep, [b*n, d, 1])), [b, n, r])
        ## [bn, r, d] x [bn, d, 1] => [bn, r, 1] => [b, n, r]
        sums = tf.reshape(tf.matmul(tf.reshape(H_rel_head+H_rel_dep, [b*n, d]), weights['W-rel']) + weights['b-rel'], [b, n, r])
        ## [b*n, d] x [d, r] + [r] => [b*n, r] (broadcast) => [b, n, r]
        output = interactions + sums
        return output


#    predictions = tf.reshape(predictions, [b*n, 1])
#    output = tf.map_fn(lambda x: get_useful_rels(x[0], x[1]) [interactions, predictions])
#    ## [bn, r]
#    output = tf.reshape(output, [b, n, r])
#    return output
#
#def get_useful_rels(interactions, predictions):
#    # interactions: [n, r]
#    # predictions: [1]
#    with tf.device('/cpu:0'):
#	inputs = tf.nn.embedding_lookup(interactions, predictions) ## [batch_size, seq_len, embedding_dim
#    return inputs
#
#
