# load classes to inherit from
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# load useful classes
import pandas as pd
from functools import reduce

# define classes
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


class DFDummyTransformer(TransformerMixin, BaseEstimator):
    '''One Hot Encoder. Fixes automatically a mismatch between
    categorical variables in training and test dataset'''
    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # creates a list of record dicts, e.g.
        # [{'col': value1, col2: value1, ...}, {col: value2, col2: value2},...]
        # first dict respresents index 0 (first record/row), and so on...
        XDict = X.to_dict('records')

        # fit kind of binary one hot encoder
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(XDict)
        return self

    def transform(self, X):
        XDict = X.to_dict('records')
        # transform dict to a one hot encoded numpy array
        XT = self.dv.transform(XDict)
        # get features names and create DF
        cols = self.dv.get_feature_names()
        XDum = pd.DataFrame(XT, index=X.index, columns=cols)
        return XDum


class DFScaler(TransformerMixin, BaseEstimator):
    '''Takes an sklearn scaler object, applies it and return a DF.'''
    def __init__(self, scaler, copy=True):
        self.scaler = scaler
        self.copy = copy

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        columns = X_.columns.tolist()
        index = X_.index
        X_ = self.scaler.transform(X_)
        Xs_df = pd.DataFrame(X_, index=index, columns=columns)
        return Xs_df

    
class DFImputer(TransformerMixin, BaseEstimator):
    def __init__(self, imputer, copy=True):
        self.copy = copy
        self.imputer = imputer

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        columns = X_.columns.tolist()
        index = X_.index
        X_ = self.imputer.transform(X_)
        Ximp_df = pd.DataFrame(X_, index=index, columns=columns)
        return Ximp_df


class DFImputer_withDict(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, replaceDict, copy=True):
        self.replaceDict = replaceDict
        self.copy = copy

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        for key, value in self.replaceDict.iter():
            X_.loc[X_[key].isnull(), key] = value
        return X_

    
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


class DFApply(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, func, col, resCol, copy=True):
        self.func = func
        self.copy = copy
        self.resCol = resCol
        self.col = col

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_[self.resCol] = X_[self.col].apply(self.func)
        return X_
        

class ColumnSelector(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    
class DropColumns(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, cols, copy=True):
        self.copy = copy
        self.cols = cols

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_.drop(self.cols, inplace=True, axis=1)
        return X_

    
class DataTypeTransformer(TransformerMixin, NoFitMixin, BaseEstimator):
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

                
class DFFeatureUnion(TransformerMixin, BaseEstimator):
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


class DFAddMissingCatCols(TransformerMixin, BaseEstimator):
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
        

class DFStrintToList(TransformerMixin, NoFitMixin, BaseEstimator):
    def __init__(self, separator, copy=True):
        self.copy = copy
        self.separator = separator

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return X_.apply(self.__wrapper__)

    def __wrapper__(self, record):
        return record.split(self.separator)
        

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


class HighCardinality(TransformerMixin, BaseEstimator):
    '''A transformation for categorical variables
    with high cardinality based on the paper
    "A Preprocessing Scheme for High-Cardinality Categorical
    Attributes in Classification and Prediction Problems"
    by Daniele Micci-Barreca.
    
    So fat, y has to be a pandas series and X needs to be a pandas Dataframe.
    
    You should use assuption='mean' for
    binarylabel and median for cont labels to take care of outliers
    
    '''
    
    def __init__(self, cols=None, assumption='median',
                 f=1,  copy=True):
        self.copy = copy
        self.lookup_table = {}
        self.posterior = {}
        self.prior = None
        self.ni = {}
        self.f = f
        self.assumption = assumption
        self.values = {}
        self.cols = cols
    
    def fit(self, X, y):
        # join X and y
        df = X.join(pd.DataFrame(y))
        df.fillna('NaN', axis=1, inplace=True)
        
        if self.cols == None:
            self.cols = X.columns.tolist()
            
        for col in self.cols:
            # compute posterior and prior probabilities
            if self.assumption == 'median':
                self.posterior[col] = df.groupby(col).agg('median')
                self.prior = y.median()
            else:
                self.posterior[col] = df.groupby(col).agg('mean')
                self.prior = y.mean()
                        
            # compute counts per class i
            self.ni[col] = df.groupby(col)\
                             .agg('count')[y.to_frame().columns.tolist()[0]]
        
            # create a lookup table with posterior and weight column
            self.lookup_table[col] = pd.concat([self.posterior[col],
                                                self.ni[col]], axis=1)
                
            # rename columns of lookup table
            self.lookup_table[col].columns = [col+'_posterior', col+'_count']
            self.values[col] = self.posterior[col].index.tolist()
            
            # add dummy index row to lookup_table for values
            # which do not appear in test data
            # and if no np.nan appears in training
            if 'NaN' not in self.values[col]:
                lookup_NaN = pd.DataFrame({col+'_posterior': 0,
                                           col+'_count': 0}, index=['NaN'])
                self.lookup_table[col] = self.lookup_table[col]\
                                             .append(lookup_NaN)
        
            # add weight column to lookup_table
            self.lookup_table[col].loc[:, col+'_weight'] = self.weight_func(
                self.lookup_table[col][col+'_count'],
                self.ni[col].median(), self.f)
        
            # add transformation result S_i(X_i)
            # as a column to the lookup_table
            self.lookup_table[col].loc[:, col+'_S'] = self.S(
                self.lookup_table[col][col+'_weight'],
                self.lookup_table[col][col + '_posterior'], self.prior)
        
        # return the instance of the class including the fitted values
        return self
        
    def transform(self, X):
        # create a copy
        X_ = X if not self.copy else X.copy()
        # fill nan values
        
        X_.fillna('NaN', axis=1, inplace=True)
        # iterate over all different columns, here keys
        # lookup = None
        
        for key in self.lookup_table:
            # values which appear in test dataset but not in training
            missing_in_training = list(set(X_[key].unique().tolist())
                                       - set(self.values[key]))
            # fill these values with 'NaN'
            X_.loc[:, key] = X_.loc[:, key].apply(
                lambda x: 'NaN' if x in missing_in_training else x)
            # if no 'NaN' appears, drop the dummy NaN row in lookup Table
            if 'NaN' not in X_[key].unique().tolist():
                lookup = self.lookup_table[key].drop('NaN', axis=0)
            else:
                lookup = self.lookup_table[key]
            # join the S-column of the lookup table to the feature dataframe
            X_ = lookup.loc[:, key+'_S'].to_frame().merge(
                X_, left_index=True, right_on=key, how='inner')
            # drop the categorical column
            X_.drop(key, axis=1, inplace=True)
        # return the DF
        return X_
          
    def weight_func(self, n, k, f):
        '''weight function: small f leads to a hard threshold,
        large f to a soft threshold between
        the prior and posterior probability'''
        # set default f to 1 and k to mean of class, so ni
        # return 1
        return 1.0/(1 + np.exp(-(n-k)/f))
    
    def S(self, weight, posterior, prior):
        # complete transformation
        return weight * posterior + (1-weight) * prior

    
class DropTooManyNulls(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.1, copy=True):
        self.threshold = threshold
        self.copy = copy
        self.cols = None
        
    def fit(self, X, y=None):
        nullPercentage = X.isnull().sum(axis=0) / len(X)
        self.cols = nullPercentage.loc[nullPercentage > self.threshold]\
                                  .index.tolist()
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_.drop(self.cols, axis=1, inplace=True)
        return X_

    
class DropTooManyUnique(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.1, copy=True):
        self.threshold = threshold
        self.copy = copy
        self.cols = None

    def fit(self, X, y=None):
        nunique = X.nunique()
        self.cols = nunique.loc[nunique > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        X_.drop(self.cols, axis=1, inplace=True)
        return X_
