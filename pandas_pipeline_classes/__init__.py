# load classes to inherit from
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# load useful classes
import pandas as pd
from functools import reduce


# define classes
class NoFitMixin():
    def fit(self, X, y=None):
        return self

    
class DFTransform(TransformerMixin, BaseEstimator, NoFitMixin):
    def __init__(self, func, copy=True):
        self.copy = copy
        self.func = func

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)


class DFDummyTransformer(TransformerMixin):
    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        XDict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(XDict)
        return self

    def transform(self, X):
        XDict = X.to_dict('records')
        XT = self.dv.transform(XDict)
        cols = self.dv.get_feature_names()
        XDum = pd.DataFrame(XT, index=X.index, columns=cols)
        return XDum

    
class DFScaler(TransformerMixin):
    def __init__(self, scaler, col=None):
        self.s = None
        self.scaler = scaler
        self.col = col
        
    def fit(self, X, y=None):
        if self.scaler == 'MinMaxScaler':
            self.s = MinMaxScaler().fit(X[[self.col]])
        elif self.scaler == 'StandardScaler':
            self.s = StandardScaler().fit(X[[self.col]])
        else:
            print('{} does not exists'.format(self.scaler))
        return self

    def transform(self, X):
        Xs = self.s.transform(X[[self.col]])
        Xs_df = pd.DataFrame(Xs, index=X.index, columns=['scaled'])
        X[self.col + '_scaled'] = Xs_df['scaled']
        return X


class DFImputer_withDict(TransformerMixin, NoFitMixin):
    def __init__(self, replaceDict, copy=True):
        self.replaceDict = replaceDict
        self.copy = copy

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        for key, value in self.replaceDict.iter():
            X_.loc[X_[key].isnull(), key] = value
        return X_


class DFImputer_groupByLabel(TransformerMixin):
    def __init__(self, col, method, copy=True):
        self.copy = copy
        self.method = method
        self.col = col
        self.conDF = None
        self.replaceDict = None
        self.y_col = None

    def fit(self, X, y):
        self.y_col = y.name
        self.conDF = pd.concat((X, y), axis=1)
        if self.method == 'median':
            self.replaceDict = self.conDF.groupby(self.y_col)[self.col]\
                                         .median().to_dict()
        if self.method == 'mean':
            self.replaceDict = self.conDF.groupby(self.y_col)[self.col]\
                                         .mean().to_dict()
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        for key, value in self.replaceDict.items():
            X_.loc[(X_[self.col].isnull()) &
                   (X_[self.y_col] == key),
                   self.col] = self.replaceDict[key]
        return X_


class DFImputer(TransformerMixin):
    def __init__(self, col, method, copy=True):
        self.copy = copy
        self.method = method
        self.col = col
        self.value = None

    def fit(self, X, y=None):
        if self.method == 'median':
            self.value = X[self.col].median()
        if self.method == 'mean':
            self.value = X[self.col].mean()
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_.loc[X_[self.col].isnull(), self.col] = self.value
        return X_


class DFQuantile(TransformerMixin, NoFitMixin):
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


class DFApply(TransformerMixin, NoFitMixin):
    def __init__(self, func, col, resCol, copy=True):
        self.func = func
        self.copy = copy
        self.resCol = resCol
        self.col = col

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_[self.resCol] = X_[self.col].apply(self.func)
        return X_
        

class ColumnSelector(TransformerMixin, NoFitMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    
class DropColumns(TransformerMixin, NoFitMixin):
    def __init__(self, cols, copy=True):
        self.copy = copy
        self.cols = cols

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_.drop(self.cols, inplace=True, axis=1)
        return X_

    
class DataTypeTransformer(TransformerMixin, NoFitMixin):
    def __init__(self, dataType, cols=None, copy=True):
        self.copy = copy
        self.dataType = dataType
        self.cols = cols

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        if self.cols is None:
            self.cols = X.columns.tolist()
        for element in self.cols:
            if self.dataType == 'category':
                X_[element] = X_[element].astype('category')
            elif self.dataType == 'str':
                X_[element] = X_[element].astype(str)
            elif self.dataType == 'dateTime':
                X_[element] = pd.to_datetime(X_[element])
            else:
                print('{} not implemented.'.format(self.dataType))
        return X_

                
class DFFeatureUnion(TransformerMixin):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2:
                        pd.merge(X1, X2, left_index=True,
                                 right_index=True), Xts)
        return Xunion


class DFAddMissingCatCols(TransformerMixin):
    def __init__(self, copy=True):
        self.copy = copy
        self.cols = None

    def fit(self, X, y=None):
        self.cols = X.columns.values.tolist()
        return self
        
    def __add_missing_dummy_columns__(self, X):
        missing_cols = set(self.cols) - set(X.columns)
        if len(missing_cols) != 0:
            for column in missing_cols:
                X[column] = 0
        return X

    def __rem_additional_dummy_columns__(self, X):
        add_cols = set(self.cols) - set(X.columns)
        if len(add_cols) != 0:
            for columns in add_cols:
                del X[columns]
        return X

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        self.__add_missing_dummy_columns__(X_)
        self.__rem_additional_dummy_columns__(X_)
        return X_
        

class DFStrintToList(TransformerMixin, NoFitMixin):
    def __init__(self, separator, copy=True):
        self.copy = copy
        self.separator = separator

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return X_.apply(self.__wrapper__)

    def __wrapper__(self, record):
        return record.split(self.separator)
        

class DFArrayExplodePivot(TransformerMixin, NoFitMixin):
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
