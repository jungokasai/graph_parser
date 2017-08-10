import tensorflow as tf 

def get_mlp_weights(name, inputs_dim, units):
    weights = {}
    with tf.variable_scope(name) as scope:
	weights['weight'] = tf.get_variable('weight', [inputs_dim, units])
        ## run all MLPs at the same time
	weights['bias'] = tf.get_variable('bias', [units])
    return weights

def mlp(x, weights, activation = 'relu'):
    output = tf.matmul(x, weights['weight']+weights['bias'])
    if activation == 'relu':
        output = tf.nn.relu(output)
    return output


