import tensorflow as tf 

def get_char_weights(opts, name):
    weights = {}
    with tf.variable_scope(name) as scope:
        weights['W-char'] = tf.get_variable('W-char', [opts.chars_window_size, opts.chars_dim, opts.nb_filters])
        weights['b-char'] = tf.get_variable('b-char', [opts.nb_filters])
    return weights

def encode_char(inputs, weights): 
    ## NEED TO CHANGE IT TO MAP_FN IF THERE OCCURS A MEMORY ISSUE
    ## inputs: [seq_len, batch_size, word_len, embedding_dim] = [n, b, m, d]
    shape = tf.shape(inputs)
    n, b, m, d = shape[0], shape[1], shape[2], shape[3]
    inputs = tf.reshape(inputs, [n*b, m, d])
    #inputs = tf.map_fn(lambda x: sent_conv(x, weights), inputs) ##[n*b, nb_filters]
    inputs = char_conv(inputs, weights)
    inputs = tf.reshape(inputs, [n, b, -1])
    return inputs

def char_conv(inputs_word, weights):
    ## inputs_word: [n*b, m, d]
    h_conv1 = tf.nn.relu(tf.nn.conv1d(inputs_word, weights['W-char'], stride = 1, padding = 'SAME')+weights['b-char'])
    ## h_conv1 [n*b, m, nb_filteres]
    h_conv1 = tf.reduce_max(h_conv1, axis=1)  ##[n*b, nb_filters]
    return h_conv1
