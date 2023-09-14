#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[29]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[33]:


#FUNCIÓN PARA CREAR FOLDS DE LOS DATOS
def Create_fold(data,fold,step):
    r1=fold*step
    if fold<4:
        r2=r1+step
    else:
        r2=len(data)
    Test=data.iloc[r1:r2]
    Train=data.drop(index=data.index[r1:r2])
    return Test, Train


# In[34]:


def RegresionL(X_train,X_test,y_train,y_test,Test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    real_pred=model.predict(Test)
    return accuracy, real_pred


# In[35]:


Test=pd.read_csv("TestDummy.csv")
Test.head()


# In[36]:


Pred=pd.DataFrame()
for i in range(1,5):
    doc="DataDummy"+str(i)+".csv"
    test_doc="TestDummy"+str(i)+".csv"
    Data=pd.read_csv(doc)
    Test=pd.read_csv(test_doc)
    Test=Test.drop('ID',axis=1)
    step=round(len(Data)/5)
    for fold in range(5):
        test,train=Create_fold(Data,fold,step)
        y_test=test['Survived'].values
        y_train=train['Survived'].values
        X_test=test.drop('Survived',axis=1).values
        X_train=train.drop('Survived',axis=1).values
        presicion, Predicciones = RegresionL(X_train,X_test,y_train,y_test,Test)
        Pred[str('Dataset'+str(i)+ " fold "+str(fold+1))]=Predicciones
        print('DataSet {}, fold {}, precisión: {}'.format(i,fold+1,round(presicion,2)))
        print('')


# In[38]:


Test=pd.read_csv("TestDummy.csv")
Pred['PassengerId']=Test['ID']
Pred=Pred[['PassengerId','Dataset1 fold 0', 'Dataset1 fold 1', 'Dataset1 fold 2',
       'Dataset1 fold 3', 'Dataset1 fold 4', 'Dataset2 fold 0',
       'Dataset2 fold 1', 'Dataset2 fold 2', 'Dataset2 fold 3',
       'Dataset2 fold 4', 'Dataset3 fold 0', 'Dataset3 fold 1',
       'Dataset3 fold 2', 'Dataset3 fold 3', 'Dataset3 fold 4',
       'Dataset4 fold 0', 'Dataset4 fold 1', 'Dataset4 fold 2',
       'Dataset4 fold 3', 'Dataset4 fold 4']]
Pred.to_csv("Predicciones Regresión Logistica.csv",index=False)

