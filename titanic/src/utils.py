# -*- coding: utf-8 -*-

import pandas as pd


def load_train_data(path):
    train = pd.read_csv(path)
    
    train['Sex'][train['Sex'] == 'male'] = 0
    train['Sex'][train['Sex'] == 'female'] = 1
    train['Sex'] = train['Sex'].astype(int)
    
    fill_train = train.iloc[:, [0,1,2,4,5,6,7,9]]
    med = fill_train['Age'].median()
    fill_train.loc[:, 'Age'] = fill_train['Age'].fillna(med)
    
    fill_train = fill_train.values
    ids = fill_train[:, 0]
    labels = fill_train[:, 1]
    data = fill_train[:, 2:]
    
    return ids, data, labels, med


def load_test_data(path, med):
    test = pd.read_csv(path)
    
    test['Sex'][test['Sex'] == 'male'] = 0
    test['Sex'][test['Sex'] == 'female'] = 1
    test['Sex'] = test['Sex'].astype(int)
    
    fill = test.iloc[:, [0,1,3,4,5,6,8]]
    fill.loc[:, 'Age'] = fill['Age'].fillna(med)
    
    fill = fill.values
    ids = fill[:, 0]
    data = fill[:, 1:]
    
    return ids, data


def scheduler(th_epoch, lr, gamma):
    def _scheduler(epoch):
        if epoch < th_epoch:
            return lr 
        else:
            return lr * gamma ** (epoch - th_epoch)

    return _scheduler
