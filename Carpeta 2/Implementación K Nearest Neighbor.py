#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbor 

# In[5]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#FUNCIÓN DE IMPLEMENTACIÓN DEL MODELO KNN
def knn_Model(X_train, X_test, y_train, y_test, k,Test):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    knn_classifier=KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train,y_train)
    y_pred=knn_classifier.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    real_pred=knn_classifier.predict(Test)
    return accuracy,real_pred


# In[6]:


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


# In[7]:


Pred=pd.DataFrame()
for i in range(1,5):
    doc="DataDummy"+str(i)+".csv"
    Data=pd.read_csv(doc)
    step=round(len(Data)/5)
    test_doc="TestDummy"+str(i)+".csv"
    Test=pd.read_csv(test_doc)
    Test=Test.drop('ID',axis=1)
    for fold in range(5):
        test,train=Create_fold(Data,fold,step)
        y_test=test['Survived'].values
        y_train=train['Survived'].values
        X_test=test.drop('Survived',axis=1).values
        X_train=train.drop('Survived',axis=1).values
        presicion, Predicciones=knn_Model(X_train, X_test, y_train, y_test, 3,Test)
        Pred[str('Dataset'+str(i)+ " fold "+str(fold+1))]=Predicciones
        print('DataSet {}, fold {}, precisión: {}'.format(i,fold+1,round(presicion,2)))
        print('')


# In[8]:


Test=pd.read_csv("TestDummy.csv")
Pred['PassengerId']=Test['ID']
Pred=Pred[['PassengerId','Dataset1 fold 1', 'Dataset1 fold 2', 'Dataset1 fold 3',
       'Dataset1 fold 4', 'Dataset1 fold 5', 'Dataset2 fold 1',
       'Dataset2 fold 2', 'Dataset2 fold 3', 'Dataset2 fold 4',
       'Dataset2 fold 5', 'Dataset3 fold 1', 'Dataset3 fold 2',
       'Dataset3 fold 3', 'Dataset3 fold 4', 'Dataset3 fold 5',
       'Dataset4 fold 1', 'Dataset4 fold 2', 'Dataset4 fold 3',
       'Dataset4 fold 4', 'Dataset4 fold 5']]
Pred.to_csv("Predicciones KNN.csv",index=False)

