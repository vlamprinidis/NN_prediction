class maps:
    def __init__(self):
        self.fws = ['tflow', 'ptorch']
        self.known_models = ['cnn', 'rnn']
        self.batches = [64, 128, 512]
        self.ranks = [0, 1, 2]
        self.nodes = [1, 2, 3]
        
        self.batches_str = [str(b) for b in self.batches]
        self.ranks_str = [str(r) for r in self.ranks]
        self.nodes_str = [str(n) for n in self.nodes]