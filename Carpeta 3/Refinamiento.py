#!/usr/bin/env python
# coding: utf-8

# # Regresión Logistica

# In[19]:


import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[18]:


df = pd.read_csv("DataDummy1.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

TestDf = pd.read_csv("TestDummy1.csv")
Test = TestDf.drop("ID", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300, 400, 500],
    'fit_intercept': [True, False],
    'intercept_scaling': [1, 2, 3, 4, 5],
    'class_weight': [None, 'balanced'],
    'warm_start': [True, False],
    'multi_class': ['auto', 'ovr', 'multinomial'],
}

logistic_regression = LogisticRegression()

random_search = RandomizedSearchCV(logistic_regression, param_distributions=param_dist, n_iter=1000, cv=5, n_jobs=-1)

# Listas para almacenar métricas
accuracies = []
biases = []
variances = []

# Realiza la validación cruzada y registra métricas en cada iteración
for i in range(1, 11):  # Realizar 10 iteraciones
    random_search.fit(X_train, y_train)
    best_logistic_regression = random_search.best_estimator_
    
    # Calcula el accuracy en datos de prueba
    accuracy = best_logistic_regression.score(X_test, y_test)
    accuracies.append(accuracy)
    
    # Calcula el bias y la varianza utilizando validación cruzada
    scores = cross_val_score(best_logistic_regression, X_train, y_train, cv=5)
    bias = 1 - np.mean(scores)
    variance = np.std(scores)
    biases.append(bias)
    variances.append(variance)
    
    print(f"Iteración {i}: Accuracy = {accuracy:.2f}, Bias = {bias:.2f}, Variance = {variance:.2f}")

# Graficar las métricas en cada iteración


# In[20]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(range(1, 11), accuracies, marker='o')
plt.title('Accuracy por Iteración')
plt.xlabel('Iteración')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 2)
plt.plot(range(1, 11), biases, marker='o')
plt.title('Bias por Iteración')
plt.xlabel('Iteración')
plt.ylabel('Bias')

plt.subplot(1, 3, 3)
plt.plot(range(1, 11), variances, marker='o')
plt.title('Varianza por Iteración')
plt.xlabel('Iteración')
plt.ylabel('Varianza')

plt.tight_layout()
plt.show()

print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LogisticRegression(solver="liblinear",penalty="l2",max_iter=400,C=0.1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
TestPred=model.predict(Test)


# In[24]:


Prediccion=pd.DataFrame()
Prediccion['PassengerId']=TestDf['ID']
Prediccion['Survived']=TestPred
Prediccion.to_csv("RS LG.csv",index=False)


# # VSM

# In[ ]:


import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC

df = pd.read_csv("DataDummy1.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

TestDf = pd.read_csv("TestDummy1.csv")
Test = TestDf.drop("ID", axis=1)


param_dist = {
    'C': np.logspace(-3, 3, 7),            # Parámetro de regularización C
    'kernel': ['linear', 'poly', 'rbf'],   # Tipo de kernel
    'gamma': np.logspace(-3, 3, 7),        # Parámetro gamma (solo para kernel 'rbf' y 'poly')
    'degree': [2, 3, 4],                  # Grado del kernel polinómico (solo para kernel 'poly')
}
svm = SVC()

random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1)

random_search.fit(X, y)

best_params = random_search.best_params_
print("Mejores hiperparámetros:", best_params)

best_svm = SVC(**best_params)
best_svm.fit(X, y)

scores = cross_val_score(best_svm, X, y, cv=5)
print("Puntuaciones de validación cruzada:", scores)
print("Puntuación media de validación cruzada:", np.mean(scores))


# # Random Forest 

# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint


df = pd.read_csv("DataDummy1.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

TestDf = pd.read_csv("TestDummy1.csv")
Test = TestDf.drop("ID", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()

param_dist = {
    'n_estimators': randint(10, 200),            # Número de árboles en el bosque
    'max_features': ['auto', 'sqrt', 'log2'],    # Número máximo de características a considerar en cada división
    'max_depth': [None] + list(np.arange(1, 20)), # Profundidad máxima del árbol
    'min_samples_split': randint(2, 20),         # Mínimo número de muestras requeridas para dividir un nodo interno
    'min_samples_leaf': randint(1, 20),          # Mínimo número de muestras requeridas para ser una hoja
    'bootstrap': [True, False]                   # Si se deben utilizar muestras de arranque para construir árboles
}

random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Mejores hiperparámetros encontrados:", best_params)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


# In[5]:


TestPred=best_model.predict(Test)
Prediccion=pd.DataFrame()
Prediccion['PassengerId']=TestDf['ID']
Prediccion['Survived']=TestPred
Prediccion.to_csv("RS RF.csv",index=False)


# # Decision Tree

# In[6]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("DataDummy1.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

TestDf = pd.read_csv("TestDummy1.csv")
Test = TestDf.drop("ID", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'max_depth': [None] + list(np.arange(1, 20)),
    'max_features': ['auto', 'sqrt', 'log2'] + list(np.arange(0.5, 1.0, 0.1)),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'criterion': ['gini', 'entropy']
}

tree = DecisionTreeClassifier()

random_search = RandomizedSearchCV(tree, param_distributions=param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1, random_state=42)

random_search.fit(X, y)

print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

print("Mejor puntuación de validación cruzada:", random_search.best_score_)

best_model = random_search.best_estimator_


# In[7]:


TestPred=best_model.predict(Test)
Prediccion=pd.DataFrame()
Prediccion['PassengerId']=TestDf['ID']
Prediccion['Survived']=TestPred
Prediccion.to_csv("RS DT.csv",index=False)


# # KNN

# In[8]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("DataDummy1.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

TestDf = pd.read_csv("TestDummy1.csv")
Test = TestDf.drop("ID", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_dist = {
    'n_neighbors': list(range(1, 21)),  # Número de vecinos a considerar
    'weights': ['uniform', 'distance'],  # Tipo de peso ('uniform' o 'distance')
    'p': [1, 2]  # Parámetro de distancia (1 para distancia de Manhattan, 2 para Euclidiana)
}

knn = KNeighborsClassifier()

random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)

random_search.fit(X_train, y_train)

print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

best_knn = random_search.best_estimator_
accuracy = best_knn.score(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}")

TestPred=best_model.predict(Test)
Prediccion=pd.DataFrame()
Prediccion['PassengerId']=TestDf['ID']
Prediccion['Survived']=TestPred
Prediccion.to_csv("RS KNN.csv",index=False)

