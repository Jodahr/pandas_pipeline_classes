from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import numpy as np
import chardet
from chardet.universaldetector import UniversalDetector
from bs4 import UnicodeDammit

def eval(modelList, X_test):
    '''Function which takes a List of ModelTuples ('name', model)
     and a test dataset and returns a Dictionary with predictions and probabilities where the Dict 
     Key is the name of the model.'''
    evalDict = {}
    for model in modelList:
        print("computing predictions for model " + str(model[0]) + "\n")
        estimator = model[1]
        y_pred, y_pred_proba = estimator.predict(X_test), estimator.predict_proba(X_test)
        evalDict[model[0]] = [y_pred, y_pred_proba]
    return evalDict


def precision_recall(evalDict, y_test):
    '''Function which takes an 'evalDict' and test labels y_test as input and returns 
    a Dict containing precision, recall and threshold. The Key will be the Model Name.'''
    precRecall_dict = {}
    for key, value in evalDict.iteritems():
        print("computing precision, recall and thresholds for model {}").format(key)
        # value[1][:,1] is probability for positive class with label (1)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test,value[1][:,1])
        precRecall_dict[key] = [precision, recall, thresholds]
    return precRecall_dict


def roc_curve(evalDict, y_test):
    '''Function which takes an 'evalDict' and test labels y_test as input and returns 
    a Dict containing fpr, tpr, thresholds and auc for the models. The Key will be the Model Name.'''
    fprTpr_dict = {}
    for key, value in evalDict.iteritems():
        print("computing fpr, tpr and thresholds for model {}").format(key)
        fpr, tpr, thresholds= sklearn.metrics.roc_curve(y_test,value[1][:,1])
        auc = sklearn.metrics.auc(fpr, tpr)
        fprTpr_dict[key] = [fpr, tpr, thresholds, auc]
    return fprTpr_dict


def precision_recall(evalDict, y_test):
    '''Function which takes an 'evalDict' and test labels y_test as input and returns 
    a Dict containing precision, recall and threshold. The Key will be the Model Name.'''
    precRecall_dict = {}
    for key, value in evalDict.iteritems():
        print("computing precision, recall and thresholds for model {}").format(key)
        # value[1][:,1] is probability for positive class with label (1)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test,value[1][:,1])
        precRecall_dict[key] = [precision, recall, thresholds]
    return precRecall_dict


def roc_curve(evalDict, y_test):
    '''Function which takes an 'evalDict' and test labels y_test as input and returns 
    a Dict containing fpr, tpr, thresholds and auc for the models. The Key will be the Model Name.'''
    fprTpr_dict = {}
    for key, value in evalDict.iteritems():
        print("computing fpr, tpr and thresholds for model {}").format(key)
        fpr, tpr, thresholds= sklearn.metrics.roc_curve(y_test,value[1][:,1])
        auc = sklearn.metrics.auc(fpr, tpr)
        fprTpr_dict[key] = [fpr, tpr, thresholds, auc]
    return fprTpr_dict


def get_featureImportance(features_list, importance_list):
    '''Function to get feature importance. The input is a list of columns which 
    you can get via 'DF.columns.tolist()' and a feature importance list which 
    you can get via the command model.feature_importances_.tolist(). 
    Features which have been 'OneHotEncoded' need to have a '=' suffix 
    and will be summed. The output is a DataFrame in descending order of the Feature Importance'''
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(features_list, importance_list):
        feats[feature] = importance #add the name/value pair
    importancesDF = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
    imp_list = importancesDF.index.tolist()
    imp_set = list(set([element.split('=')[0] for element in imp_list]))
    impDict = {}
    for element in imp_set:
        impDict[element] = importancesDF.filter(regex=element, axis=0).sum()
    return pd.DataFrame.from_dict(impDict, orient='index').sort_values(by='importance', axis=0, ascending=False)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,
                                      color="g", labels=('Precision', 'Recall')):
    '''Functions takes precisions, recalls and thresholds as arguments and generates a plot with two curves.
        Optionally you can define the color and the labels.'''
    plt.plot(thresholds, precisions[:-1], color + "--", label = labels[0])
    plt.plot(thresholds, recalls[:-1], color + "-", label = labels[1])
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])

def plot_precision_vs_recall(precisions, recalls, color="g", label='model'):
    plt.plot(recalls, precisions, color + "--", label = label)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="upper right")
    plt.ylim([0,1])

def plot_roc_curve(fpr, tpr, auc, color="g", label='model'):
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, color + "--", label = label + ", AUC = {}".format(auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Raite")
    plt.legend(loc="lower right")
    plt.ylim([0,1])

def plot_random():
    plt.plot([0,1],[0,1], label='Zufall')

def drop_featuersWithManyNulls(df, threshold):
    df_series = df.isnull().sum() / len(df)
    colList_toDrop = df_series[df_series > threshold].index.tolist()
    return df.drop(colList_toDrop, axis=1)

def getNumerics(df):
    return df.select_dtypes(exclude=[object])



def getCategorical(df):
    return df.select_dtypes(include=[object])


# credit goes to http://notmatthancock.github.io/2015/10/28/confusion-matrix.html
def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'Recall: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Precision: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()


def encodeDF(df, verbose=False):
    '''This function decodes and encodes the whole DataFrame.'''
    # extract columnNames
    allCols = df.columns.tolist()
    # decode/encode columnNames to Unicode
    # UnicodeDammit seems to have some probs
    # allCols_utf = [UnicodeDammit(col).unicode_markup for col in allCols]
    allCols_utf = [col.decode(chardet.detect(col)['encoding']).encode('utf8') for col in allCols]
    # set unicode ColNames
    df.columns = allCols_utf
    # select non_numeric columns
    cols = df.select_dtypes(include=[object]).columns.tolist()
    nCols = len(cols)
    counter = 1
    for col in cols:
        print('encode/decode col {} ({}/{})'.format(col, counter, nCols))
        try:
            df[col] = df[col].apply(lambda x: x.decode(chardet.detect(x)['encoding']).encode('utf8'))
        except:
            try:
                df[col].apply(lambda x: UnicodeDammit(x).unicode_markup)
            except:
                print("Decoding/Encoding of column {} not possible".format(col))
        counter +=1
    return df

    
