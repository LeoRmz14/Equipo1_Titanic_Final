#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[106]:


df=pd.read_csv('TestLimpia.csv')
Data=pd.DataFrame()
df.head()
Data['ID']=df['Id']


# In[12]:


df['Survived'].value_counts()


# In[107]:


#SACAR VARIABLE DUMMY DEL SEXO DE LOS PASAJEROS
male=(df['Sex']=='male').astype(int)
female=(df['Sex']=='female').astype(int)
Data['Male']=male
Data['Female']=female


# In[108]:


#AGERGAR LA VARIABLE DE EDAD AL DATAFRAME Y CONVERTIR LA VARIABLE AGE CLASSIFICATION A DUMMYS 
Data['Age']=df['Age']
Adult=(df['Age Clasification']=='Adult').astype(int)
Child=(df['Age Clasification']=='Child').astype(int)
Old=(df['Age Clasification']=='Old').astype(int)
Data['Child']=Child
Data['Adult']=Adult
Data['Old']=Old


# In[109]:


#CREAR VARIABLES DUMMY DE LAS CLASES 
First=(df['Class']=='First').astype(int)
Second=(df['Class']=='Second').astype(int)
Third=(df['Class']=='Third').astype(int)
Data['First Class']=First
Data['Second Class']=Second
Data['Third Class']=Third


# In[110]:


#AGREGAR LAS VARIABLES DE COSTO DE BOLETO AL DATA FRAME Y CREAR VARIABLES DUMMYS PARA EL TIPO DE CABINA
Data['Cost']=df['Cost']
#NoCabin=(df['Cabin']=='-').astype(int)
CabinaA=(df['Cabin']=='A').astype(int)
CabinaB=(df['Cabin']=='B').astype(int)
CabinaC=(df['Cabin']=='C').astype(int)
CabinaD=(df['Cabin']=='D').astype(int)
CabinaE=(df['Cabin']=='E').astype(int)
CabinaF=(df['Cabin']=='F').astype(int)
CabinaG=(df['Cabin']=='G').astype(int)
#Data['No Cabin']=NoCabin
Data['Cabin A']=CabinaA
Data['Cabin B']=CabinaB
Data['Cabin C']=CabinaC
Data['Cabin D']=CabinaD
Data['Cabin E']=CabinaE
Data['Cabin F']=CabinaF
Data['Cabin G']=CabinaG


# In[111]:


#CREAR VARIABLES DUMMY PARA LAS CLASES SOCIALES
Data['Clase Baja-Baja']=(df['Social Classes']=='Baja-baja').astype(int)
Data['Clase Baja-Alta']=(df['Social Classes']=='Baja-alta').astype(int)
Data['Clase Media-Baja']=(df['Social Classes']=='Media-baja').astype(int)
Data['Clase Media-Alta']=(df['Social Classes']=='Media-alta').astype(int)
Data['Clase Alta-Baja']=(df['Social Classes']=='Alta-baja').astype(int)
Data['Clase Alta-Alta']=(df['Social Classes']=='Alta-alta').astype(int)


# In[112]:


#AGREGAR VARIABLE DE HERMANOS/PAREJA Y PADRES/HIJOS
Data['Siblings/Spouses']=df['Siblings/Spouses']
Data['Parents/Children']=df['Parents/Children']


# In[113]:


#AGREGAR VARIABLE DUMMY DE LOS PUERTOS DE EMBARQUE DEL DF ORIGINAL
Data['No Port']=(df['Embarked']=='-').astype(int)
Data['Southampton Port']=(df['Embarked']=='Southampton').astype(int)
Data['Cherbourg Port']=(df['Embarked']=='Cherbourg').astype(int)
Data['Queenstown Port']=(df['Embarked']=='Queenstown').astype(int)


# In[114]:


#AGREGAR VARIABLES DE LOS TITULOS DE LAS PERSONAS DENTRO DEL BARCO TOMANDO EN CUENTA UNICAMENTE LOS 4 TITULOS MAS IMPORTANTES 
#Y LOS DEMAS APILADOS COMO UNA SOLA VARIABLE
Data['Mr']=(df['Title']==' Mr').astype(int)
Data['Miss']=(df['Title']==' Miss').astype(int)
Data['Mrs']=(df['Title']==' Mrs').astype(int)
Data['Master']=(df['Title']==' Master').astype(int)
Other=((df['Title']==' Don').astype(int))+((df['Title']==' Rev').astype(int))+((df['Title']==' Dr').astype(int))+((df['Title']==' Mme').astype(int))+((df['Title']==' Ms').astype(int))+((df['Title']==' Major').astype(int))+((df['Title']==' Lady').astype(int))+((df['Title']==' Sir').astype(int))+((df['Title']==' Mlle').astype(int))+((df['Title']==' Col').astype(int))+((df['Title']==' Capt').astype(int))+((df['Title']==' the Countess').astype(int))+((df['Title']==' Jonkheer').astype(int))
Data['Other']=Other


# In[11]:


#AGREGAR LA COLUMNA DE SOBREVIVIENTES EN EL NUEVO DF
Data['Survived']=df['Survived']


# In[120]:


Data.columns


# In[121]:


Data.to_csv('TestDummy.csv',index=False)


# # DataSet 1

# In[122]:


Data1=Data[['ID','Male', 'Female', 'Age','First Class','Second Class', 'Third Class', 'Cost', 'Cabin A', 'Cabin B', 'Cabin C',
       'Cabin D', 'Cabin E', 'Cabin F', 'Cabin G','Siblings/Spouses','Parents/Children','Southampton Port', 'Cherbourg Port',
       'Queenstown Port']]


# In[123]:


Data1.to_csv('TestDummy1.csv',index=False)


# # DataSet 2

# In[124]:


Data2=Data[['ID','Male', 'Female', 'Age', 'Child', 'Adult', 'Old', 'First Class','Second Class', 'Third Class', 'Cost',
           'Clase Baja-Baja','Clase Baja-Alta', 'Clase Media-Baja', 'Clase Media-Alta','Clase Alta-Baja', 'Clase Alta-Alta',
            'Siblings/Spouses','Parents/Children']]


# In[125]:


Data2.to_csv('TestDummy2.csv',index=False)


# # DataSet 3

# In[126]:


Data3=Data[['ID','Mr', 'Miss', 'Mrs', 'Master', 'Other','Male', 'Female', 'Age', 'Child', 'Adult', 'Old', 'First Class',
       'Second Class', 'Third Class', 'Cost','Clase Baja-Baja','Clase Baja-Alta', 'Clase Media-Baja', 'Clase Media-Alta',
       'Clase Alta-Baja', 'Clase Alta-Alta','Southampton Port', 'Cherbourg Port','Queenstown Port']]


# In[127]:


Data3.to_csv('TestDummy3.csv',index=False)


# # DataSet 4

# In[128]:


Data4=Data[['ID','Male', 'Female', 'Age','Cost','Siblings/Spouses','Parents/Children']]


# In[129]:


Data4.to_csv('TestDummy4.csv',index=False)

