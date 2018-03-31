import tensorflow as tf 
def get_global_weights(name, units, nb_rels): # no dropout weights = {}
    weights = {}
    with tf.variable_scope(name) as scope:
	weights['W-global'] = tf.get_variable('W-global', [nb_rels, units]) 
    return weights

def global_equation(vectors,  arc_weights, rel_weights, global_weights): 
    ## H_arc_head: [seq_len, batch_size, units]
    ## H_arc_dep: [seq_len, batch_size, units]
    H_arc_head = vectors['arc-head']
    shape = tf.shape(H_arc_head)
    n, b, arc_d = shape[0], shape[1], shape[2]
    r = tf.shape(global_weights['W-global'])[0]
    H_arc_dep = vectors['arc-dep']
    H_rel_head = vectors['rel-head'] ##[n, b, rel_d]
    H_rel_dep = vectors['rel-dep']
    rel_d = tf.shape(H_rel_head)[2]
    H_arc_head = tf.transpose(H_arc_head, [1, 0, 2]) ## [b, n, arc_d]
    H_rel_head = tf.transpose(H_rel_head, [1, 0, 2]) ## [b, n, rel_d]
    prev_init = [H_arc_head, tf.zeros([b,n]), tf.zeros([b], dtype=tf.int32), tf.zeros([b, n, r]), tf.zeros([b, r]), tf.zeros([b], dtype=tf.int32)]
    ## new_state = [H_arc_head, arc_output, arc_predictions, rel_all_output, rel_predicted_output, rel_predictions]
    input_list = [H_arc_dep, H_rel_dep]
    ## We need the following items in memory
    ### 1. H_arc_head we update this
    all_states = tf.scan(lambda prev, x: global_equation_word(prev, x, arc_weights, rel_weights, global_weights, H_rel_head), input_list, prev_init)
    arc_output = all_states[1] ## [n, b, n]
    arc_output = tf.transpose(arc_output, [1, 0, 2])
    rel_predicted_output = all_states[4]  ## [n, b, r]
    rel_predicted_output = tf.transpose(rel_predicted_output, [1, 0, 2])
    rel_all_output = all_states[3] ## [n, b, n, r]
    rel_all_output = tf.transpose(rel_all_output, [1, 0, 2, 3])
    return arc_output, rel_predicted_output, rel_all_output

def global_equation_word(prev_list, input_list, arc_weights, rel_weights, global_weights, H_rel_head):
    H_arc_head = prev_list[0] ## [n, b, arc_d]
    shape = tf.shape(H_arc_head)
    n, b, arc_d = shape[0], shape[1], shape[2]
    H_arc_dep = input_list[0] ## [b, arc_d] 
    H_rel_dep = input_list[1] ## [b, rel_d] 
    arc_output = tf.map_fn(lambda x: arc_equation(x, arc_weights), [H_arc_head, H_arc_dep], dtype=tf.float32) ## [b, n] 
    arc_predictions = tf.cast(tf.argmax(arc_output, axis=1), tf.int32) ## [b]
    rel_all_output, rel_predicted_output = tf.map_fn(lambda x: rel_equation(x, rel_weights), [H_rel_head, H_rel_dep, tf.expand_dims(arc_predictions, 1)], dtype=(tf.float32, tf.float32)) 
    rel_predictions = tf.cast(tf.argmax(rel_predicted_output, axis=1), tf.int32) ## [b]
    ## rel_all_output [b, n, r], rel_predicted_output [b, r]
    H_arc_head = update_H_arc_head(H_arc_head, arc_predictions, rel_predictions, global_weights)
    new_state = [H_arc_head, arc_output, arc_predictions, rel_all_output, rel_predicted_output, rel_predictions]
    return new_state
    
def arc_equation(input_list, arc_weights):
    H_arc_head_sent = input_list[0] ## [n, arc_d]
    H_arc_dep_sent = input_list[1] ## [arc_d] 
    output = tf.squeeze(tf.matmul(tf.matmul(H_arc_head_sent, arc_weights['W-arc']), tf.expand_dims(H_arc_dep_sent, 1)), 1) ## [n, arc_d] x [arc_d, arc_d] => [n, arc_d]
    ## [n, arc_d] x [arc_d, 1] => [n, 1]
    ## [n, 1] => [n]
    bias = tf.squeeze(tf.matmul(H_arc_head_sent, tf.expand_dims(arc_weights['b-arc'], 1)), 1)
    ## [n, arc_d] x [arc_d, 1] => [n, 1]
    ## [n, 1] => [n]
    output = output + bias # [n]
    return output
def rel_equation(input_list, rel_weights):
    H_rel_head = input_list[0] # [n, rel_d]
    n = tf.shape(H_rel_head)[0]
    U_rel = rel_weights['U-rel']
    shape = tf.shape(U_rel)
    rel_d, r = shape[0], shape[1]
    H_rel_dep = input_list[1] # [rel_d]
    arc_prediction = input_list[2] # [1]
    interactions = tf.reshape(tf.matmul(H_rel_head, tf.reshape(U_rel, [rel_d, r*rel_d])), [n*r, rel_d])
    ## [n, rel_d] x [rel_d, r*rel_d] => [n, r*rel_d] => [nr, rel_d]
    interactions = tf.reshape(tf.matmul(interactions, tf.expand_dims(H_rel_dep, 1)), [n, r])
    ## [nr, rel_d] x [rel_d, 1] => [n, r]
    sums = tf.matmul(tf.reshape(tf.reshape(H_rel_head, [n, rel_d])+tf.reshape(H_rel_dep, [1, rel_d]), [n, rel_d]), rel_weights['W-rel']) + rel_weights['b-rel']
    ## [n(parent), rel_d] x [rel_d, r] + [r] => [n, r] 
    rel_all_output = interactions + sums
    rel_predicted_output = tf.reshape(tf.matmul(tf.reshape(tf.one_hot(arc_prediction, n), [1, n]), rel_all_output), [r])
    ## [1, n] x [n, r] => [1, r] => [r]
    return rel_all_output, rel_predicted_output

def update_H_arc_head(H_arc_head, arc_predictions, rel_predictions, global_weights):
    shape = tf.shape(H_arc_head)
    b, n, arc_d = shape[0], shape[1], shape[2]
    nb_rels = tf.shape(global_weights['W-global'])[0]
    #addition = tf.reshape(tf.matmul(tf.reshape(tf.one_hot(rel_predictions, nb_rels), [b, nb_rels]), global_weights['W-global']), [b, 1, arc_d])
    ## [b, nb_rels] x [nb_rels, arc_d] => [b, arc_d] => [b, 1, arc_d]
    #addition = tf.reshape(tf.one_hot(arc_predictions, n), [b, n, 1])*addition 
    with tf.device('/cpu:0'):
        addition = tf.nn.embedding_lookup(global_weights['W-global'], rel_predictions) ## [b, arc_d]
    addition = tf.reshape(tf.one_hot(arc_predictions, n), [b, n, 1])*tf.reshape(addition, [b, 1, arc_d])
    ## [b, n, 1] x [b, 1, arc_d] => [b, n, arc_d]
    H_arc_head += addition
    return H_arc_head
