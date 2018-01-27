from sklearn.base import TransformerMixin, BaseEstimator
from utils import NoFitMixin
import pandas as pd


class DFQuantile(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, col, q, suffix='_quantile', copy=True):
        self.copy = copy
        self.col = col
        self.suffix = suffix
        self.q = q

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_[self.col + '_' + self.suffix] = pd.qcut(X_[self.col],
                                                   self.q, labels=False)
        return X_


class DFArrayExplodePivot(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, copy=True):
        self.copy = copy

    def __explode__(self, X):
        rows = []
        ind = 0
        for row in X:
            for element in row:
                rows.append([ind, element])
            ind += 1
        return pd.DataFrame(data=rows, columns=['old_index', 'exploded'])

    def __pivot__(self, X):
        return X.pivot(index='old_index', columns='exploded', values='val')

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_ = self.__explode__(X_)
        X_['val'] = 1
        return self.__pivot__(X_).fillna(int(0))
