from sklearn.decomposition import IncrementalPCA
import numpy as np
class IPCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(n_components, whiten=self.whiten, batch_size=max(100, 5*n_components))
        self.batch_support = True

    def get_param_str(self):
        return "ipca_c{}{}".format(self.n_components, '_w' if self.whiten else '')

    def fit(self, X):
        self.transformer.fit(X)

    def fit_partial(self, X):
        try:
            self.transformer.partial_fit(X)
            self.transformer.n_samples_seen_ = \
                self.transformer.n_samples_seen_.astype(np.int64) # avoid overflow
            return True
        except ValueError as e:
            print(f'\nIPCA error:', e)
            return False

    def get_components(self):
        stdev = np.sqrt(self.transformer.explained_variance_) # already sorted
        var_ratio = self.transformer.explained_variance_ratio_
        return self.transformer.components_, stdev, var_ratio # PCA outputs are normalized