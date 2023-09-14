#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[111]:


#LEER LOS DATASETS, CAMBIAR LOS NOMBRES DE LAS COLUMNAS Y CAMBIAS EL TIPO DE DATO EN COLUMNAS DE CABINA Y CLASE
train2=pd.read_csv('train.csv')
train=pd.read_csv('test.csv')
gender=pd.read_csv('gender_submission.csv')


train.rename(columns={'SibSp':'Siblings/Spouses','Parch':'Parents/Children','Pclass':'Class','Fare':'Cost',
                      'PassengerId':'Id'},inplace=True)
train['Cabin']=train['Cabin'].astype(str)
train['Class']=train['Class'].astype(str)



train2.rename(columns={'SibSp':'Siblings/Spouses','Parch':'Parents/Children','Pclass':'Class','Fare':'Cost',
                      'PassengerId':'Id'},inplace=True)
train2['Cabin']=train2['Cabin'].astype(str)
train2['Class']=train2['Class'].astype(str)


# In[112]:


train.head()


# In[113]:


def CostNan(row):
    data=row["Cost"]
    if data!=data:
        row['Cost']=14.4542
        return row
    else: 
        return row
train=train.apply(CostNan,axis='columns')


# In[114]:


train.head()


# In[115]:


#CREAR FUNCIÓN PARA SEPARAR NOMBRE, APELLIDO Y TITULO
def Name_Last(row):
    data=row['Name']
    FullName=str.split(data,',')
    row['Last Name']=FullName[0]
    if '.' in FullName[-1]:
        Name=str.split(FullName[-1],'.')
        row['Name']=Name[-1]
        row['Title']=Name[0]
        return row
    else:
        row['Name']=FullName[-1]
        row['Title']=np.nan
        return row
train=train.apply(Name_Last,axis='columns')
train2=train2.apply(Name_Last,axis='columns')


# In[116]:


#CREAR UNA FUNCIÓN PARA SEPARAR LOS PARENTESIS DENTRO DE LOS NOMBRES Y SOLO DEJAR LOS NOMBRES 
def Parentesis(row):
    Data=row['Name']
    pattern='\((.*)\)'
    Parentesis=re.findall(pattern,Data)
    if len(Parentesis)>=1:
        row['Parentesis']=Parentesis[0]
        Name=str.split(row['Name'],'(')
        row['Name']=Name[0]
        return row
    else:
        row['Parentesis']=np.nan
        return row
train=train.apply(Parentesis,axis='columns')
train2=train2.apply(Parentesis,axis='columns')


# In[117]:


#ESCOGER Y ORDENAR POR RELEVANCIA LAS COLUMNAS NECESARIAS DEL DATASET
train2=train2[['Id','Title','Name','Last Name','Sex','Age','Class','Cost','Cabin','Siblings/Spouses',
             'Parents/Children','Embarked','Survived']]

train=train[['Id','Title','Name','Last Name','Sex','Age','Class','Cost','Cabin','Siblings/Spouses',
             'Parents/Children','Embarked']]


# In[118]:


#CREAR FUNCIÓN PARA DEVOLVER UNICAMENTE LA SECCIÓN DE LA CABINA Y REGRESAR '-' EN CASO DE NO TENER CABINA
def Cabin(row):
    pattern='[A-Z]'
    data=row['Cabin']
    if data=='nan':
        row['Cabin']='-'
        return row
    else:
        found=re.findall(pattern,data)
        row['Cabin']=found[0]
        return row
    
train=train.apply(Cabin,axis='columns')
train2=train2.apply(Cabin,axis='columns')


# In[119]:


train2.head()


# In[120]:


#CAMBIAR LOS DATOS DE LAS CLASES DE PASAJEROS POR UNA VARIABLE ORDINAL
def Class_Order(row):
    data=row['Class']
    if data=='1':
        row['Class']='First'
        return row
    elif data=='2':
        row['Class']='Second'
        return row
    elif data=='3':
        row['Class']='Third'
        return row
train=train.apply(Class_Order,axis='columns') 
train2=train2.apply(Class_Order,axis='columns') 


# In[121]:


#CAMBIAR LOS DATOS DEL PURTO DEL EMBARCAMIENTO POR SU NOMBRE COMPLETO
def Port_Name(row):
    data=row['Embarked']
    if data=='C':
        row['Embarked']='Cherbourg'
        return row
    elif data=='Q':
        row['Embarked']='Queenstown'
        return row
    elif data=='S':
        row['Embarked']='Southampton'
        return row
    elif data!=data:
        row['Embarked']='-'
        return row
train=train.apply(Port_Name,axis='columns')
train2=train2.apply(Port_Name,axis='columns')


# In[122]:


#ESTABLECER UNA JERARQUIA DE LAS CLASES DE PASAJEROS 
ordinal_Class=['Third','Second','First']
train['Class']=pd.Categorical(train['Class'],categories=ordinal_Class,ordered=True)
train2['Class']=pd.Categorical(train2['Class'],categories=ordinal_Class,ordered=True)


# In[123]:


#ESTABLECER UNA JERARQUIA DE LAS CLASES DE LAS CABINAS 
ordinal_Cabin=['-','G','F','E','D','C','B','A']
train['Cabin']=pd.Categorical(train['Cabin'],categories=ordinal_Cabin,ordered=True)
train2['Cabin']=pd.Categorical(train2['Cabin'],categories=ordinal_Cabin,ordered=True)


# In[48]:


#CREAR UN DATAFRAME CON LAS EDADES MAXIMAS Y MINIMAS DE ACUERDO A LOS TTULOS EXISTENTES DE LOS PASAJEROS
#Promedio_Edades=train.groupby('Title').agg({'Age':(np.min,np.max,np.mean)})
#Promedio_Edades=Promedio_Edades.droplevel(0,axis=1)


# In[124]:


train2.head()


# In[125]:


#CREAR UN DATAFRAME CON LAS EDADES MAXIMAS, MINIMAS, IQR, Q1,Q3, Lower bound y Upper bound
#DE ACUERDO A LOS TTULOS EXISTENTES DE LOS PASAJEROS
Min=[]
Max=[]
Mean=[]
IQR=[]
Last=[]
First=[]
Len=[]
Q1=[]
Q3=[]
for title in train2['Title'].unique().tolist():
    df=train2[train2['Title']==title].dropna(subset='Age')
    minimo=np.min(df['Age'])
    maximo=np.max(df['Age'])
    media=np.mean(df['Age'])
    q1=np.percentile(df['Age'],25)
    q3=np.percentile(df['Age'],75)
    iqr=q3-q1
    last=q3+1.5*iqr
    first=q1-1.5*iqr
    NumTitle=len(df)
    Min.append(minimo)
    Max.append(maximo)
    Mean.append(media)
    IQR.append(iqr)
    Last.append(last)
    First.append(first)
    Len.append(NumTitle)
    Q1.append(q1)
    Q3.append(q3)
Ages_df=pd.DataFrame({'Title':train2['Title'].unique().tolist(),'# Title':Len,
                     'Min':Min,'Max':Max,'Mean':Mean,'Q1':Q1,'Q3':Q3,'IQR':IQR,'Lower bound':First,'Upper bound':Last})
Ages_df.set_index('Title',inplace=True)


# In[126]:


Ages_df


# In[127]:


#RELLENAR LOS VALORES FALTANTES DE LAS EDADES CON UN NUMERO RANDON DE ACUERDO AL DATAFRAME DE LAS EDADES MAXIMAS Y MINIMAS
#POR TITULO DEL PASAJERO
def Edades_Faltantes(df,Promedio):
    Age=df.iloc[x]['Age']
    Title=df.iloc[x]['Title']
    if Age!=Age:
        
        if Promedio.loc[Title]['Min']>Promedio.loc[Title]['Lower bound']:
            minimo=Promedio.loc[Title]['Min']
        else:
            minimo=Promedio.loc[Title]['Lower bound']
            
        if Promedio.loc[Title]['Max']<Promedio.loc[Title]['Upper bound']:
            maximo=Promedio.loc[Title]['Max']
        else:
            maximo=Promedio.loc[Title]['Upper bound'] 
            
        Age=np.random.randint(minimo,maximo+1,1)[0]
        df.at[x,'Age']=Age
    return df
    
for x in range(0,len(train)):
    Edades_Faltantes(train,Ages_df)


# In[128]:


#DEFINIR UNA FUNCIÓN DE CLASIFICACIÓN DE ETAPA DE VIDA DEPENDIENDO DE LA EDAD 
#Y AGREGAR UNA NUEVA COLUMNA LLAMADA AGE CLASIFICATION
def Clasificacion_Edad(row):
    data=row['Age']
    if data<18:
        row['Age Clasification']='Child'
        return row
    elif data<60:
        row['Age Clasification']='Adult'
        return row
    else:
        row['Age Clasification']='Old'
        return row
train=train.apply(Clasificacion_Edad,axis='columns')


# In[129]:


#DEFINIR UNA FUNCIÓN DE CLASIFICACIÓN DE CLASE SOCIAL DEPENDIENDO DEL DINERO PAGADO POR BOLETO 
#Y AGREGAR UNA NUEVA COLUMNA LLAMADA SOCIAL CLASS
def Cost_Classification(df,variable):
    INFO=df[variable].describe()
    MIN=INFO.loc['min']
    q1=INFO.loc['25%']
    q3=INFO.loc['75%']
    IQR=q3-q1
    MAX=q3+1.5*IQR
    Clases=np.linspace(MIN,MAX,6)[1:]
    Social_clases=[]
    for x in range(0,len(df)):
        cost=df.iloc[x][variable]
        if cost<=Clases[0]:
            Social_clases.append('Baja-baja')
        elif cost<=Clases[1]:
            Social_clases.append('Baja-alta')
        elif cost<=Clases[2]:
            Social_clases.append('Media-baja')
        elif cost<=Clases[3]:
            Social_clases.append('Media-alta')
        elif cost<=Clases[4]:
            Social_clases.append('Alta-baja')
        else:
            Social_clases.append('Alta-alta')
    df['Social Classes']=Social_clases
    return df
train=Cost_Classification(train,'Cost')


# In[130]:


ordinal_SocialClass=['Baja-baja','Baja-alta','Media-baja','Media-alta','Alta-baja','Alta-alta']
train['Social Classes']=pd.Categorical(train['Social Classes'],categories=ordinal_SocialClass,ordered=True)


# In[54]:


Classes=train['Class'].value_counts()
sns.barplot(x=Classes.index,y=Classes.values)
plt.xlabel('Clases')
plt.ylabel('# de personas')
plt.title('Numero de personas por clase')
plt.show()


# In[55]:


sns.boxplot(data=train, x="Cost", y="Class",hue="Class", dodge=False)
plt.title('Boxplot del costo de boleto vs la clase')
plt.show()


# In[56]:


sns.boxplot(data=train, x="Cost", y="Cabin",hue="Cabin", dodge=False)
plt.title('Boxplot de dinero pagado por boleto vs Cabina')
plt.show()


# In[57]:


sns.boxplot(data=train,x='Age',y='Title',showfliers=True)
plt.title('Boxplot de las edades vs titulo')
plt.show()


# In[58]:


sns.histplot(data=train,x='Age',bins=10)
plt.xlabel('Edad')
plt.ylabel('# de personas')
plt.title('Histograma de las edades de las personas en el barco')
plt.show()


# In[59]:


#FUNCIÓN PARA REALIZAR GRAFICAS DE BARRAS APILADAS CONSIDERANDO VIVOS Y MUERTOS POR CADA VARIABLE
def GraphVariable(df,variable):
    lista=df[variable].unique().tolist()
    vivos=[]
    muertos=[]
    for x in lista:
        DF=df[df[variable]==x]
        vivos.append(len(DF[DF['Survived']==1]))
        muertos.append(len(DF[DF['Survived']==0]))
    fig, ax = plt.subplots()
    ax.bar(lista,vivos,label='Vivos')
    ax.bar(lista,muertos, bottom = vivos,label='Muertos')
    plt.xlabel(variable)
    plt.ylabel('# de personas')
    plt.title('Vivos y Muertos vs '+variable)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


# In[60]:


#GRAFICA DE PERSONAS VIVAS Y MUERTAS VS SU TITULO ASIGNADO
GraphVariable(train,'Title')


# In[61]:


#GRAFICA DE PERSONAS VIVAS Y MUERTAS VS SU CLASES ASIGNADA
GraphVariable(train,'Class')


# In[62]:


#GRAFICA DE PERSONAS VIVAS Y MUERTAS VS SU SEXO
GraphVariable(train,'Sex')


# In[63]:


#GRAFICA DE PERSONAS VIVAS Y MUERTAS VS SU EMBARCAMIENTO
GraphVariable(train,'Embarked')


# In[64]:


#GRAFICA DE PERSONAS VIVAS Y MUERTAS VS SU EMBARCAMIENTO
GraphVariable(train,'Age Clasification')


# In[65]:


GraphVariable(train,'Embarked')


# In[66]:


#GRAFICA DE NUMERO DE PERSONAS VIVAS Y MUERTAS VS EL # DE HERMANOS O ESPOSAS QUE TENIAN DENTRO DEL TITANIC
GraphVariable(train,'Siblings/Spouses')


# In[67]:


#GRAFICA DE NUMERO DE PERSONAS VIVAS Y MUERTAS VS EL # DE PADRES O HIJOS QUE TENIAN DENTRO DEL TITANIC
GraphVariable(train,'Parents/Children')


# In[68]:


#GRAFICA DEL DINERO PROMEDIOPOR BOLETO VS EL PUERTO DE EMBARCACIÓN
Embarcamiento=train['Embarked'].unique().tolist()
Dinero_X_Embarcamiento=[]
for x in Embarcamiento:
    data=train[train['Embarked']==x]['Cost']
    Dinero_X_Embarcamiento.append(np.mean(data))
fig, ax = plt.subplots()
ax.bar(Embarcamiento, Dinero_X_Embarcamiento)
plt.xlabel('Embarcamiento')
plt.ylabel('$ Promedio de Boleto')
plt.title('Dinero vs Puerto')
plt.show()


# In[69]:


GraphVariable(train,'Social Classes')


# In[131]:


train.head()


# In[133]:


train=train[['Id', 'Title', 'Name', 'Last Name', 'Sex', 'Age','Age Clasification', 'Class', 'Cost',
       'Cabin','Social Classes', 'Siblings/Spouses', 'Parents/Children', 'Embarked']]
train.to_csv('TestLimpia.csv',index=False)

