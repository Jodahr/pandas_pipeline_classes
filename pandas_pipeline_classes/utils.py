from sklearn.base import TransformerMixin, BaseEstimator


class NoFitMixin():
    '''Dummy Class to inherit from if you do not need a fit function.'''
    def fit(self, X, y=None):
        return self


class DFtoMat(TransformerMixin, BaseEstimator, NoFitMixin):
    def transform(self, X):
        return X.as_matrix()


class DFTransform(TransformerMixin, BaseEstimator, NoFitMixin):
    '''Class which can be used to apply any kind of function on a DataFrame
    inside a Pipeline. Note that used Parameters in the function cannot
    be changed in a gridSearch.'''
    def __init__(self, func, copy=True):
        self.copy = copy
        self.func = func

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)
