# -*- coding: utf-8 -*-

import pandas as pd


CLASSES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


def convert_attr(data):
    for class_, value in [('male', 0), ('female', 1)]:
        data.loc[data.loc[:, 'Sex'] == class_, 'Sex'] = value
    data.loc[:, 'Sex'] = data.loc[:, 'Sex'].astype(int)
    
    for class_, value in [('C', 1), ('Q', 2), ('S', 3)]:
        data.loc[data.loc[:, 'Embarked'] == class_, 'Embarked'] = value
    data.loc[:, 'Embarked'] = data.loc[:, 'Embarked'].fillna(0)
    data.loc[:, 'Embarked'] = data.loc[:, 'Embarked'].astype(int)
    return data


def load_train_data(path):
    train = pd.read_csv(path)
    train = convert_attr(train)

    # 欠損値補完
    med = train['Age'].median()
    train.loc[:, 'Age'] = train['Age'].fillna(med)

    # 標準化
    data = train.loc[:, CLASSES]
    mean = data.mean()
    std = data.std()
    data = ((data - mean) / std).values

    ids = train.loc[:, 'PassengerId'].values
    labels = train.loc[:, 'Survived'].values
    
    return ids, data, labels, med, mean, std


def load_test_data(path, med, mean, std):
    test = pd.read_csv(path)
    test = convert_attr(test)

    test.loc[:, 'Age'] = test['Age'].fillna(med)

    data = test.loc[:, CLASSES]
    data = ((data - mean) / std).values

    ids = test.loc[:, 'PassengerId'].values
    
    return ids, data
