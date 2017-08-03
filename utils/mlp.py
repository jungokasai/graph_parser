import tensorflow as tf 

def get_mlp_weights(name, inputs_dim, units, batch_size, hidden_prob):
    weights = {}
    concat_units = 4*units
    with tf.variable_scope(name) as scope:
	weights['weight'] = tf.get_variable('weight', [inputs_dim, concat_units])
        ## run all MLPs at the same time
	weights['bias'] = tf.get_variable('bias', [concat_units])
        dummy_dp = tf.ones([batch_size, concat_units])
        weights['dropout'] = tf.nn.dropout(dummy_dp, hidden_prob)
    return weights

def mlp(x, weights, activation = 'relu'):
    output = tf.matmul(x, weights['weight']+weights['bias'])
    if activation == 'relu':
        output = tf.nn.relu(output)
    output = output*weights['dropout']
    return output


