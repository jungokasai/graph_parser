import tensorflow as tf 

def get_rel_weights(name, units): # no dropout
    weights = {}
    with tf.variable_scope(name) as scope:
	weights['W-arc'] = tf.get_variable('W-arc', [units, units])
	weights['b-arc'] = tf.get_variable('b-arc', [units])
    return weights

def rel_equation(H_arc_head, H_arc_dep, weights): 
    ## H_arc_head: [seq_len, batch_size, units]
    ## H_arc_dep: [seq_len, batch_size, units]
    H_arc_head = tf.transpose(H_arc_head, [1, 2, 0]) 
    H_arc_dep = tf.transpose(H_arc_dep, [1, 0, 2]) 
    ## H_arc_head: [batch_size, units, seq_len]
    ## H_arc_dep: [batch_size, seq_len, units]
    output = tf.matmul(tf.matmul(H_arc_dep, weights['W-arc']), H_arc_head)
    # output: [batch_size, seq_len, seq_len]
    bias = tf.matmul(weights['b-arc'], H_arc_head) ## [batch_size, seq_len]
    bias = tf.expand_dims(bias, 1) ## [batch_size, 1, seq_len] for broadcasting
    ## bias applies uniformly over different h_arc_dep's.
    output += bias
    return output


