from tarjan import Tarjan

def get_arcs(probs, test_opts):
    ## probs [batch_size, seq_len, seq_len]
    predictions = tf.argmax(probs, 2)
    ## [batch_size, seq_len, seq_len] => [batch_size, seq_len]
    if test_opts is None: ## training
        return predictions 
    else: ## test, need to fix cycles
        Tarjan(predictions)
        probs = tf.nn.log_softmax(probs)
        
