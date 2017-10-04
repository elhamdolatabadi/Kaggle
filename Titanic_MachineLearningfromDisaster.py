# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:00:15 2017

@author: dolatae
"""

# data analysis and wrangling
from __future__ import division

import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, RandomizedSearchCV, GridSearchCV


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.metrics import classification_report, roc_curve ,auc
from scipy.stats import randint, expon

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.model_selection import StratifiedKFold        
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import xgboost as xgb                                    
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/                                              
"""
Functioncs Definitions

"""

pd.options.display.max_rows = 100

"""
Feature Engineering
"""
def status(feature):
    print 'Processing',feature,': ok'
    
def get_data():
    # reading train data
    train = pd.read_csv('train.csv')    
    # reading test data
    test = pd.read_csv('test.csv')

    # extracting and then removing the targets from the training data
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data 
    data = train.append(test)
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    
    return data



def process_ticket(data):
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    data['Ticket'] = data['Ticket'].map(cleanTicket) 
    mapping = pd.unique(data['Ticket'])
    t_mapping = dict(zip(mapping, range(len(mapping))))
    data['Ticket'] = data['Ticket'].map( t_mapping,na_action='ignore')
    return data

    
def recover_train_test_target(data):
    train0 = pd.read_csv('train.csv')
    target = train0.Survived
    train = data.head(891)
    test = data.iloc[891:]
    
    return train, test, target 
    
def prepare_data():
    combine = get_data()
    combine.drop('PassengerId', inplace=True, axis=1)
    combine['Sex'] = LabelEncoder().fit_transform(combine['Sex'])
    combine['Name'] = combine['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    titles = combine['Name'].unique()

    combine['Age'].fillna(-1, inplace=True)
    medians = dict()
    for title in titles:
        median = combine.Age[(combine["Age"] != -1) & (combine['Name'] == title)].median()
        medians[title] = median    
    for index, row in combine.iterrows():
        if row['Age'] == -1:
            combine.loc[index, 'Age'] = medians[row['Name']]
            #combine['Age'] = StandardScaler().fit_transform(data['Age'].values.reshape(-1, 1))

    combine['Fare'].fillna(-1, inplace=True)
    median = dict()
    for pclass in combine['Pclass'].unique():
        median = combine.Fare[(combine["Fare"] != -1) & (combine['Pclass'] == pclass)].median()
        medians[pclass] = median
    for index, row in combine.iterrows():
        if row['Fare'] == -1:
            combine.loc[index, 'Fare'] = medians[row['Pclass']]
            #combine['Fare'] = StandardScaler().fit_transform(combine['Fare'].values.reshape(-1, 1))
            #combine['Pclass'] = StandardScaler().fit_transform(data['Pclass'].values.reshape(-1, 1))

    replacement = {
        6: 0,
        4: 0,
        5: 1,
        0: 2,
        2: 3,
        1: 4,
        3: 5,
        9: 0,
    }
    combine['Parch'] = combine['Parch'].apply(lambda x: replacement.get(x))
    #combine['Parch'] = StandardScaler().fit_transform(combine['Parch'].values.reshape(-1, 1))
    
    replacement = {
        5: 0,
        8: 0,
        4: 1,
        3: 2,
        0: 3,
        2: 4,
        1: 5
    }
    combine['SibSp'] = combine['SibSp'].apply(lambda x: replacement.get(x))
    #combine['SibSp'] = StandardScaler().fit_transform(combine['SibSp'].values.reshape(-1, 1))

    combine = process_ticket(combine)
    #combine['SibSp'] = StandardScaler().fit_transform(combine['SibSp'].values.reshape(-1, 1))
    
    combine['Embarked'].fillna(-1, inplace=True)
    medians = dict()
    for title in titles:
        median = combine.Embarked[(combine["Embarked"] != -1) & (combine['Name'] == title)].value_counts().argmax()
        medians[title] = median
    for index, row in combine.iterrows():
        if row['Embarked'] == -1:
            combine.loc[index, 'Embarked'] = medians[row['Name']]
    replacement = {
        'S': 0,
        'Q': 1,
        'C': 2
    }
    combine['Embarked'] = combine['Embarked'].apply(lambda x: replacement.get(x))
    #data['Embarked'] = StandardScaler().fit_transform(data['Embarked'].values.reshape(-1, 1))
    
    pclass = combine['Pclass'].unique()
    combine['Cabin'].fillna('U', inplace=True)
    combine['Cabin'] = combine['Cabin'].apply(lambda x: x[0])
    #medians = dict()
    #for pclass_ in pclass:
    #    median = combine.Cabin[(combine["Cabin"] != -1) & (combine['Pclass'] == pclass_)].value_counts().argmax()
    #    medians[pclass_] = median
    #for index, row in combine.iterrows():
    #    if row['Cabin'] == -1:
    #        combine.loc[index, 'Cabin'] = medians[row['Pclass']]
    replacement = {
        'T': 0,
        'A': 1,
        'G': 2,
        'C': 3,
        'F': 4,
        'B': 5,
        'E': 6,
        'D': 7,
        'U': 8,
    }
    combine['Cabin'] = combine['Cabin'].apply(lambda x: replacement.get(x))
    #combine['Cabin'] = StandardScaler().fit_transform(combine['Cabin'].values.reshape(-1, 1))

#    replacement = {
#        'Don': 0,#'Royalty', #Spain, Portugal, Italy, Iberoamerica
#        'Dona': 0,#'Royalty',
#        'Master':0,#'Royalty', #an English honorific for boys and young men
#        'Lady':0,#'Royalty',    
#        'Sir': 0,#'Royalty',
#        'the Countess': 0,#'Royalty',#European countries
#        'Jonkheer': 0,#'Royalty', # Dutch
#        'Mr': 1,#'married',
#        'Mrs':1,#'married',
#        'Mme':1,# 'married',
#        'Ms': 1,# 'married',
#        'Miss':2,# 'single',
#        'Mlle':2, #'single',  
#        'Rev':3, #'staff',
#        'Capt': 3,#'staff',
#        'Dr': 3,#'staff',    
#        'Col': 4,#'military',
#        'Major':4# 'military',
#    }    
#    
    replacement = {
        'Don': 0,#'Royalty', #Spain, Portugal, Italy, Iberoamerica
        'Dona': 0,#'Royalty',
        'Master':0,#'Royalty', #an English honorific for boys and young men
        'Lady':0,#'Royalty',    
        'Sir': 0,#'Royalty',
        'the Countess': 0,#'Royalty',#European countries
        'Jonkheer': 0,#'Royalty', # Dutch
        'Mr': 1,#'married',
        'Mrs':1,#'married',
        'Mme':1,# 'married',
        'Ms': 1,# 'married',
        'Miss':2,# 'single',
        'Mlle':2, #'single',  
        'Rev':3, #'staff',
        'Capt': 3,#'staff',
        'Dr': 3,#'staff',    
        'Col': 4,#'military',
        'Major':4# 'military',
    }    
    combine['Name'] = combine['Name'].apply(lambda x: replacement.get(x))
    
    return combine
    
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return xval


def scorer(clf, X, cutoff):
    return (clf.predict_proba(X)[:,1]>cutoff).astype(int)
    
def custome_f1(clf, X,y,cutoff):
    def f1_cutoff(clf,X,y):
        ypred = scorer(clf,X,cutoff)
        return f1_score(y,ypred)
    return f1_cutoff

def cutoff_predict(X,cutoff):
    return (X[:,1]>cutoff).astype(int)

def get_classifier(model_name,cv,n_feat):
    _scoring = 'accuracy'
    FS_al = mutual_info_classif
    if model_name == 'lr':        
        pipe = Pipeline([('ss', StandardScaler()),('FS', SelectKBest(mutual_info_classif)), ('lr', LogisticRegression())])
        param_grid = {'lr__C': np.logspace(-5, 15, num=20,base=2)}
        clf = GridSearchCV(pipe, param_grid, cv=cv, scoring=_scoring, n_jobs=-1, verbose=2) 
        
    elif model_name == 'svm':
        print('svm fitting')
        pipe = Pipeline([('ss', StandardScaler()),('FS', SelectKBest(FS_al)),('svc', SVC(probability=True))])
        param_grid = {'svc__C': np.logspace(-5, 15, num=20,base=2), 
                      'svc__gamma': np.logspace(-15, 5, num=20,base=2),
                      'svc__class_weight' : ['balanced', None],
                      'FS__k':range(1,n_feat+1)}
                      #'FS__score_func': [mutual_info_classif,chi2,f_classif] }
        n_iter = 10
        clf = RandomizedSearchCV(pipe, param_grid, cv=cv, n_iter = n_iter, scoring=_scoring, n_jobs=-1, verbose=2)
    
    elif model_name == 'rf':  
        pipe = Pipeline([('ss', StandardScaler()),('FS', SelectKBest(FS_al)), ('rf', RandomForestClassifier())])
        param_grid = {'rf__max_depth': np.concatenate((range(1,20),[None]),axis=0),
                      'rf__max_features': ['sqrt', None , 'log2'],
                      'rf__min_samples_split': np.concatenate((np.arange(2,10,3),[17,50,75,100,300]),axis=0),
                      'rf__min_samples_leaf': np.concatenate((np.arange(2,10),[17,100,300]),axis=0),
                      'rf__n_estimators': [10, 50, 100,1000],
                      'rf__bootstrap': [True, False],
                      'rf__class_weight' : ['balanced', None],
                      'FS__k':range(1,n_feat+1)}
                      
        n_iter = 100
        clf = RandomizedSearchCV(pipe, param_grid, cv=cv, n_iter = n_iter, scoring=_scoring, n_jobs=-1, verbose=2)       
    
    elif model_name == 'gb':  
        pipe = Pipeline([('ss', StandardScaler()),('FS', SelectKBest(FS_al)), ('gb', GradientBoostingClassifier())])
        param_grid = {'gb__max_features': ['sqrt', None , 'log2'],
                      'gb__min_samples_split': np.concatenate((np.arange(2,10,3),[17,50,75,100,300]),axis=0),
                      'gb__min_samples_leaf': np.concatenate((np.arange(2,10),[17,100,300]),axis=0),
                      'gb__n_estimators': [10, 50,100, 1000],
                      'gb__warm_start': [True, False],
                      'gb__max_depth': np.concatenate((range(1,20),[None]),axis=0),
                      'FS__k':range(1,n_feat+1)}
                      
        n_iter = 100
        clf = RandomizedSearchCV(pipe, param_grid, cv=cv, n_iter = n_iter, scoring=_scoring, n_jobs=-1, verbose=2)
        
    elif model_name == 'dt':  
        pipe = Pipeline([('ss', StandardScaler()),('FS', SelectKBest(FS_al)), ('dt', DecisionTreeClassifier())])
        param_grid = {'dt__max_depth': np.concatenate((range(1,20),[None]),axis=0), 
                      'dt__max_features': ['sqrt', None , 'log2'],#randint(1, np.floor(np.sqrt(n_feat)))
                      'dt__min_samples_split': range(2,100),
                      'dt__min_samples_leaf': range(2,100),
                      'dt__class_weight' : ['balanced', None],
                      'FS__k':range(1,n_feat+1)}
        n_iter = 20
        clf = RandomizedSearchCV(pipe, param_grid, cv=cv, n_jobs=-1,n_iter = n_iter, scoring=_scoring, verbose=2)    
        
    elif model_name == 'xgb':
        print('xgb fitting')  
        ind_params = {'seed': 1,
                       'silent': True,
                       'objective':'binary:logistic'} 
        pipe = Pipeline([('ss', StandardScaler()),('FS', SelectKBest(mutual_info_classif)) , ('xgb', xgb.XGBClassifier(**ind_params))])
        param_grid = {'xgb__max_depth': [1,15,100,1000],
                      'xgb__min_child_weight': [1,15,100,1000],
                      'xgb__learning_rate': [0.01,0.1,1],
                      'xgb__gamma': np.arange(0,5,.5),
                      'xgb__n_estimators': [10,100,1000],
                      'FS__k':range(5,n_feat+1)}
        n_iter = 10
        clf = GridSearchCV(pipe, param_grid, cv=cv, scoring=_scoring, n_jobs=-1, verbose=2) 

    return clf
                 
if __name__ == "__main__":
    
    # load data
    data = prepare_data()
    #data = data.drop(['Ticket','Cabin'], axis=1)
    train, test, targets = recover_train_test_target(data)
    x1, x2, y1, y2 = train_test_split(train.as_matrix(), targets.as_matrix(), test_size=0.18, random_state=1)
    
    k=5
    skf = StratifiedKFold(n_splits=k)
    cross_validation = skf.split(x1,y1)  
    n_feat = x1.shape[1]
    clf = get_classifier('xgb',cross_validation,n_feat)
    clf.fit(x1,y1)
    y_true, y_pred = y2, clf.predict(x2)
    yscore = accuracy_score(y_true, y_pred)  
    print(yscore)

    
#    iteration_=10
#    output = np.empty(shape = [418,iteration_] )
#    c=0
#    for n in range(0,iteration_):
#       xtrain = train.as_matrix()
#       xtest = test.as_matrix()
#       ytrain = targets.as_matrix()
#       cross_validation = skf.split(xtrain,ytrain) 
#       model = get_classifier('xgb',cross_validation,n_feat)
#       model = model.fit(xtrain,ytrain)  
#       output[:,c] = model.predict(xtest)
#       c = c+1
#    count_mat = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=output.astype(int))
#    finaloutput = np.argmax(count_mat,axis=1)
#    df_output = pd.DataFrame()
#    aux = pd.read_csv('test.csv')
#    df_output['PassengerId'] = aux['PassengerId']
#    df_output['Survived'] = finaloutput
#    df_output[['PassengerId','Survived']].to_csv('xgb_output_gridsearch.csv',index=False)
##    
