
class IC:
    def __init__(self, estimator, alpha=0.05, k=None):
        self.estimator = estimator
        self.alpha = alpha
        self.separating_sets_ = None
        self.graph_ = None
        self.max_k = k

    def fit(self, X):