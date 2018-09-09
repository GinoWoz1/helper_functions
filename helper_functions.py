# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:15:28 2018

@author: jjonus
"""

from sklearn.model_selection import learning_curve,validation_curve,cross_val_score,GridSearchCV,train_test_split,RepeatedKFold
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from scipy.cluster import hierarchy as hc # used for dendrogram analysis
from fancyimpute import MICE # imputation
from sklearn.linear_model import Ridge
import itertools # create combo list
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns 
import pandas as pd
import numpy as np
from rfpimp import feature_corr_matrix
import math
import os
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve


"""""""""""""""""""""
 processing module

"""""""""""""""""""""

import missingno as msno

def missing(dataframe,graph=False):
    dataframe_na = (dataframe.isnull().sum()/len(dataframe)) * 100
    dataframe_na = dataframe_na.drop(dataframe_na[dataframe_na ==0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio':dataframe_na})
    print(missing_data.head(20))
    if graph == True:
        missing_data = dataframe.columns[dataframe.isnull().any()].tolist()
        msno.matrix(dataframe[missing_data])
        msno.heatmap(dataframe[missing_data],figsize=(20,20))

def drop_cols(dataframe,columns_remove):
    cols =list(dataframe.columns.values)
    for col in columns_remove:
        for c in cols:
            if c == col:
                cols.remove(c)
            elif c != col:
                continue
    dataframe = dataframe[cols]
    return dataframe

    
def normalize(dataframe):
    numeric_feats = dataframe.dtypes[dataframe.dtypes != "object"].index
    # Check the skew of all numerical features
    numeric_feats = dataframe[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :numeric_feats}).reset_index()
    skewness.rename(columns = {'index':'feature'},inplace=True)
    scaler = StandardScaler()
    scale_features = skewness.feature.values.tolist()
    dataframe[scale_features] = scaler.fit_transform(dataframe[scale_features])
    return dataframe

"""""""""""""""""""""
 analysis module

"""""""""""""""""""""


# TODO built function to analyze columns with 0s
"""
all_data_na = (X_train[X_train == 0] / len(X_train)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0]).index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
"""

def ind_corr(dataframe,independent_variable,upper_cutoff,lower_cutoff,plot=False):
    """
    Built to take the indepdendent variable (separate from data set), add it to dataframe. 
    conduct a correlation matwrix dependent on top values, and return dataframe with columns and their correlation to SalePrice
    """
    # build corrmat
    new_df = dataframe.copy()
    new_df['target'] = independent_variable
    corrmat = new_df.corr()
    """
    return dataframe with all the values above a certain cutoff (positive and negative correlated variables)
    """
    dataset = corrmat.iloc[-1].to_frame().reset_index()
    dataset.rename(columns = {'index':'correlated variable','target':'target_correlation_value'},inplace=True)
    cutoff_data = dataset[(dataset['target_correlation_value'] > upper_cutoff) | (dataset['target_correlation_value'] < lower_cutoff)] 
    cutoff_data = cutoff_data[cutoff_data['correlated variable'] != 'target']
    columns = cutoff_data['correlated variable'].tolist()
    if plot == True:
        cm = np.corrcoef(new_df[columns].values.T)
        hm = sns.heatmap(cm, cbar=True, square=True, fmt='.2f', yticklabels=columns, xticklabels=columns)
        plt.show()
    return cutoff_data,columns

def find_outliers(model, X, y, sigma=3,plot = False):

    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X,y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        
    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid)/std_resid    
    outliers = z[abs(z)>sigma].index
    
    # print and plot the results
    print('Score=',model.score(X,y))
    print('---------------------------------------')

    print('mean of residuals:',mean_resid)
    print('std of residuals:',std_resid)
    print('---------------------------------------')

    print(len(outliers),'outliers:')
    print(outliers.tolist())
    if plot == True:
        plt.figure(figsize=(15,5))
        ax_131 = plt.subplot(1,3,1)
        plt.plot(y,y_pred,'.')
        plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
        plt.legend(['Accepted','Outlier'])
        plt.xlabel('y')
        plt.ylabel('y_pred');
    
        ax_132=plt.subplot(1,3,2)
        plt.plot(y,y-y_pred,'.')
        plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],'ro')
        plt.legend(['Accepted','Outlier'])
        plt.xlabel('y')
        plt.ylabel('y - y_pred');
    
        ax_133=plt.subplot(1,3,3)
        z.plot.hist(bins=50,ax=ax_133)
        z.loc[outliers].plot.hist(color='r',bins=50,ax=ax_133)
        plt.legend(['Accepted','Outlier'])
        plt.xlabel('z')
        
        plt.savefig('outliers.png')
    
    return outliers

"""""""""""""""""""""
 data types
 
"""""""""""""""""""""
       
def df_datatypes(dataframe):
    print(dataframe.columns.to_series().groupby(dataframe.dtypes).groups)
    
def cardinality(dataframe,datatype):
    d = []
    columns = dataframe.dtypes[dataframe.dtypes == datatype].index
    #columns = dataframe.columns.tolist()
    for c in columns:
      data = dataframe[c].nunique()
      d.append({'Column':c,'UniqueValue':data})
    df = pd.DataFrame(d).sort_values(ascending=False,by = 'UniqueValue')
    print(df)
      
    
def define_vars(dataframe,card_thresh):
    dtypes = dataframe.dtypes
    cat_feats = dataframe.dtypes[dataframe.dtypes == "object"].index
    numeric_feats = dataframe.dtypes[dataframe.dtypes != "object"].index
    col_nunique = dict()
    for col in numeric_feats:
        col_nunique[col] = dataframe[col].nunique()
    col_nunique = pd.Series(col_nunique)
    cols_discrete = col_nunique[col_nunique<card_thresh].index.tolist()
    cols_continuous = col_nunique[col_nunique>=card_thresh].index.tolist()
    return cols_discrete,cols_continuous,cat_feats

"""""""""""""""""""""
analysis regression 
 
"""""""""""""""""""""

def PlotCatRegress(dataframe,target,cat_feats,fcols=3):
    fcols = 3
    frows = ceil(len(cat_feats)/fcols)
    plt.figure(figsize = (20,frows*4))
    
    for i,col in enumerate(cat_feats):
        plt.subplot(frows,fcols,i+1)
        sns.violinplot(dataframe[col],dataframe[target],inner="stick")

def PlotContRegress(dataframe,target,columns):
    fcols = 2
    frows = len(columns)
    plt.figure(figsize=(3*fcols,4*frows))
    i=0
    for col in columns:
        i+=1
        ax=plt.subplot(frows,fcols,i)
        sns.regplot(x=col,y=dataframe[target],data=dataframe,ax=ax)
    plt.xlabel(col)

def pear_corr(dataframe):
    df_corr_mat = feature_corr_matrix(dataframe)
    df_corr_mat = df_corr_mat.dropna(axis='columns',how='all')
    df_corr_mat = df_corr_mat.dropna()
    df_corr_mat = df_corr_mat.values
    corr_condensed = hc.distance.squareform(1-df_corr_mat)
    z = hc.linkage(corr_condensed,method='average')
    fig = plt.figure(figsize=(20,10))
    dendrogram = hc.dendrogram(z,labels=dataframe.columns,orientation = 'left',leaf_font_size = 8)
    
"""""""""""""""""""""
 analysis classifier smodule

"""""""""""""""""""""

def PlotCatClass(dataframe,target,cat_feats,fcols=3):
    fcols = fcols
    frows = ceil(len(cat_feats)/fcols)
    plt.figure(figsize=(20,frows*4))
    
    for i,col in enumerate(cat_feats):
        plt.subplot(frows,fcols,i+1)
        sns.countplot(dataframe[col],hue = dataframe[target])

def PlotContClass(dataframe,target,columns,bins=10,fcols=2,):
    fcols = fcols
    frows = ceil(len(columns)/fcols)
    plt.figure(figsize=(20,frows*4))
    
    for i,col in enumerate(columns):
        plt.subplot(frows,fcols,i+1)
        sns.countplot(dataframe[col],hue = dataframe[target])


"""""""""""""""""""""
 RF analysis importances

"""""""""""""""""""""

def dropcol_importances(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I

# combination function
    
def list_trans(features):
    list_o_lists = []
    for c in range(1,len(features)+1):
        print(c)
        data =  list(itertools.combinations(features,c))
        list_o_lists.append(data)
    new_list = [[]]
    for i in list(list_o_lists):
        for c in i:
            new_list.append(c)
    new_list = [x for x in new_list if x]
    return new_list


"""""""""""""""""""""
 metrics and training

"""""""""""""""""""""


def rmse(x,y): return math.sqrt(((x-y)**2).mean())


def TrainRegress(model,scorer,sigma,param_grid=[], X=[], y=[], splits=5, repeats=5):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    
    # perform a grid search if param_grid given
    if len(param_grid)>0:
        # setup grid search parameters
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring=scorer,
                               verbose=1, return_train_score=True)

        # search the grid
        gsearch.fit(X,y)

        # extract best model from the grid
        model = gsearch.best_estimator_        
        best_idx = gsearch.best_index_

        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)       
        cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
        cv_std = grid_results.loc[best_idx,'std_test_score']

    # no grid search, just cross-val score for given model    
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y, scoring=scorer, cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)
    
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)
    
    
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('----------------------')
    print('cross_val: mean=',cv_mean,', std=',cv_std)

    # residual plots
    y_pred = pd.Series(y_pred,index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid)/std_resid    
    n_outliers = sum(abs(z)>sigma)
    outliers = z[abs(z)>sigma].index
    print(len(outliers),'outliers:')
    print(outliers.tolist())
        
    
    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y_pred');
    plt.title('corr = {:.3f}'.format(np.corrcoef(y,y_pred)[0][1]))
    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred');
    plt.title('std resid = {:.3f}'.format(std_resid))
    
    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))
    df_outliers = X.copy()
    df_outliers['Target'] = y
    df_outliers['Prediction'] = y_pred
    df_outliers['Residual'] = resid
    df_outliers['Outlier'] = 0
    for index in outliers.tolist():
        df_outliers.loc[index,'Outlier'] = 1
    
    
    return model, cv_score, grid_results,df_outliers

"""
Class
"""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def TrainClass(model,scorer,splits =5,repeats = 5,normalize= False,graph= False,param_grid=[], X=[], y_true=[]):


    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    
    # perform a grid search if param_grid given
    if len(param_grid)>0:
        # setup grid search parameters
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring=scorer,
                               verbose=1, return_train_score=True)

        # search the grid
        gsearch.fit(X,y_true)

        # extract best model from the grid
        model = gsearch.best_estimator_        
        best_idx = gsearch.best_index_

        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)       
        cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
        cv_std = grid_results.loc[best_idx,'std_test_score']

    # no grid search, just cross-val score for given model    
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y_true, scoring=scorer, cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)
    
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)
    
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('----------------------')
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    
    # confusion matrix
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    cm = cnf_matrix
    np.set_printoptions(precision=2)
    print(cnf_matrix)
    """
    Graphing
    """
    
    classes = y_true.unique()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title(title='Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return model, cv_score, grid_results


def rmsle(y_true, y_pred): 
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_true))**2))


def learning_c(m,X_train,y_train,cv):
    train_sizes,train_scores,test_scores = learning_curve(estimator=m,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)
    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)
    
    plt.plot(train_sizes,train_mean,color = 'blue',marker='o',markersize=5,label='training accuracy')
    plt.fill_between(train_sizes,train_mean + train_std,train_mean - train_std,alpha=0.15,color='blue')
    plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
    plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15,color='blue')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.0,1.1])
    plt.show()
    
