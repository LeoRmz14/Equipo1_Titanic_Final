#!/usr/bin/env python
# coding: utf-8

# # Random Forest

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[2]:


def Create_fold(data,fold,step):
    r1=fold*step
    if fold<4:
        r2=r1+step
    else:
        r2=len(data)
    Test=data.iloc[r1:r2]
    Train=data.drop(index=data.index[r1:r2])
    return Test, Train


# In[7]:


def RandomF(X_train, X_test, y_train, y_test,Test):
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    real_pred=random_forest.predict(Test)
    return accuracy, real_pred


# In[8]:


Pred=pd.DataFrame()
for i in range(1,5):
    doc="DataDummy"+str(i)+".csv"
    Data=pd.read_csv(doc)
    test_doc="TestDummy"+str(i)+".csv"
    Test=pd.read_csv(test_doc)
    Test=Test.drop('ID',axis=1)
    step=round(len(Data)/5)
    for fold in range(5):
        test,train=Create_fold(Data,fold,step)
        y_test=test['Survived'].values
        y_train=train['Survived'].values
        X_test=test.drop('Survived',axis=1).values
        X_train=train.drop('Survived',axis=1).values
        presicion, Predicciones=RandomF(X_train, X_test, y_train, y_test,Test)
        Pred[str('Dataset'+str(i)+ " fold "+str(fold+1))]=Predicciones
        print('DataSet {}, fold {}, precisión: {}'.format(i,fold+1,round(presicion,2)))
        print('')


# In[11]:


Test=pd.read_csv("TestDummy.csv")
Pred['PassengerId']=Test['ID']
Pred=Pred[['PassengerId','Dataset1 fold 1', 'Dataset1 fold 2', 'Dataset1 fold 3',
       'Dataset1 fold 4', 'Dataset1 fold 5', 'Dataset2 fold 1',
       'Dataset2 fold 2', 'Dataset2 fold 3', 'Dataset2 fold 4',
       'Dataset2 fold 5', 'Dataset3 fold 1', 'Dataset3 fold 2',
       'Dataset3 fold 3', 'Dataset3 fold 4', 'Dataset3 fold 5',
       'Dataset4 fold 1', 'Dataset4 fold 2', 'Dataset4 fold 3',
       'Dataset4 fold 4', 'Dataset4 fold 5']]
Pred.to_csv("Predicciones Random.csv",index=False)


# In[10]:


Pred.columns

