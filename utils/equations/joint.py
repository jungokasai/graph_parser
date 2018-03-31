import tensorflow as tf 

def get_joint_weights(name, units, nb_stags): # no dropout
    weights = {}
    with tf.variable_scope(name) as scope:
        weights['W-joint'] = tf.get_variable('W-joint', [units, nb_stags])
        weights['b-joint'] = tf.get_variable('b-joint', [nb_stags])
    return weights

def joint_equation(H_joint, weights): 
    output = tf.map_fn(lambda x: tf.matmul(x, weights['W-joint'])+weights['b-joint'], H_joint)
    ## [n, b, nb_stags]
    output = tf.transpose(output, perm=[1, 0, 2])
    ## [b, n, nb_stags]
    return output

