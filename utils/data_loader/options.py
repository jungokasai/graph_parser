class Opts(object):
    def __init__(self):
        self.arc_mlp_units=500
        self.bi=1
        self.chars_dim=30
        self.chars_window_size=3
        self.dropout_p=0.67
        self.early_stopping=5
        self.embedding_dim=100
        self.hidden_p=0.67
        self.input_dp=0.67
        self.jk_dim=0
        self.joint_mlp_units=500
        self.lrate=0.01
        self.lstm=1
        self.max_epochs=100.0
        self.mlp_num_layers=1
        self.mlp_prob=0.67
        self.nb_filters=30
        self.num_layers=4
        self.rel_mlp_units=100
        self.seed=0
        self.stag_dim=0
        self.units=400
if __name__ == '__main__':
    opts = Opts()
    print(opts.jk_dim)
