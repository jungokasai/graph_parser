import tensorflow as tf 

def get_arc_weights(name, units): # no dropout
    weights = {}
    with tf.variable_scope(name) as scope:
	weights['W-arc'] = tf.get_variable('W-arc', [units, units])
	weights['b-arc'] = tf.get_variable('b-arc', [units])
    return weights

def arc_equation(H_arc_head, H_arc_dep, weights): 
    ## H_arc_head: [seq_len, batch_size, units]
    ## H_arc_dep: [seq_len, batch_size, units]
    shape = tf.shape(H_arc_head)
    n, b, d = shape[0], shape[1], shape[2]
    H_arc_head = tf.transpose(H_arc_head, [1, 0, 2]) 
    H_arc_dep = tf.transpose(H_arc_dep, [1, 0, 2]) 
    ## H_arc_head: [batch_size, seq_len, units]= [b, n, d]
    ## H_arc_dep: [batch_size, seq_len, units]= [b, n, d]
    output = tf.reshape(tf.matmul(tf.reshape(H_arc_dep, [b*n, d]), weights['W-arc']), [b, n, d])
    ## [bn, d] x [d, d] => [bn, d] => [b, n, d]
    output = tf.matmul(output, tf.transpose(H_arc_head, [0, 2, 1]))
    ## [b, n, d] x [b, d, n] => [b, n, n]
    # output: [batch_size, seq_len, seq_len]
    bias = tf.matmul(tf.reshape(H_arc_head, [b*n, d]), tf.expand_dims(weights['b-arc'], 1)) ## [batch_size, seq_len]
    ## [bn, d] x [d, 1] => [bn, 1]
    bias = tf.transpose(tf.reshape(bias, [b, n, 1]), [0, 2, 1]) ## [batch_size, 1, seq_len] for broadcast
    ## bias applies uniformly over different h_arc_dep's.
    output += bias
    return output


