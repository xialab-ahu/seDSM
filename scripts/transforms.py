
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class RandomFeatureSelector(TransformerMixin, BaseEstimator):
    """Select a subset of features from a given data"""
    def __init__(self, n_features=None, p=None):
        super().__init__()
        self.n_features = int(n_features)
        self.random_state = None
        self.columns = None
        self.p = p

    def fit(self, X, y=None):
        columns_index = np.arange(X.shape[1]).tolist()
        if self.p is not None:
            self.columns = np.random.choice(columns_index, size=self.n_features, p=self.p, replace=False)
        else:
            self.columns = np.random.choice(columns_index, size=self.n_features, replace=False)
        return self

    def transform(self, X):
        return X[:, self.columns]


if __name__ == "__main__":
    a = np.arange(100).reshape(5, 20)
    b = np.arange(5)

    rfs = RandomFeatureSelector(n_features=5)
    a_ = rfs.fit_transform(a, b)

    from sklearn.preprocessing import MinMaxScaler
    pipeline = Pipeline([("rfs", rfs), ("mn", MinMaxScaler())])
    print(pipeline.fit_transform(a, b))
