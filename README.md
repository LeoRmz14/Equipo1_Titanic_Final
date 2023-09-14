# Aprendizaje Automatizado en el Titanic: Predicciones de Supervivencia con Clasificación Binaria

En el presente repositorio se encuentran los archivos .csv y .py que muestran la solución al reto del bloque para la materia: Inteligencia Artifical Avanzada para la Ciencia de Datos. 

El reto se basó en estructurar y limpiar la base de datos de tripalantes del Titanic. La base de datos fue obtenida de la plataforma Kaggle y se muestra en el siguiente enlace: https://www.kaggle.com/competitions/titanic/data?select=train.csv

# Equipo 1: 
Pablo Gabriel Galeana Benítez - A01735281
Leonardo Ramírez Ramírez - A01351715
Miguel Chávez Silva - A01661109
Alessandro Arguelles Elias - A00829536
Luis Eduardo Aguilar Moreno - A01367545
Agustín Tapia - A01367639


## Contenido General

- [Introducción](#Breve explicación del reto)
- [Dataset](#Dataset)
- [Modelos](#Modelos ML implementados)
- [Correcciones y Entregas Previas](#Correcciones y Entregas Previas)
- [Repositorio](#Sobre el repositorio)

## Breve explicación del reto

Brevemente el reto se basó en implementación de modelos de clasificación binaria en la predicción de supervivencia de tripulantes del Titanic pues, a pesar de existir un factor de suerte, presumiblemente existieron grupos con una mayor probabilidad de supervivencia que otros según ciertos atributos. La importancia del problema radica en la tarea de detectar variables significativas en la variable predictora: sobreviviente o no sobreviviente. Esto con fin de generar un último modelo generalizable con predicciones lo más acertadas posibles según métricas de evaluación como la precisión. 

## Dataset

Los dataset obtenidos de Kaggle (https://www.kaggle.com/competitions/titanic/data?select=train.csv) fueron dos: train.csv y test.csv. 
En ambos archivos se encuentran las variables: PassengerId, Survived, Pclass, Name, Age, SibSip, Parch, Sexo, Fare, Ticket, Cabin y Embarked.

Como pauta importante para implementar modelos de Machine Learning se debe dividir el dataset en training y testing para verificar underfitting, overfitting y el rendimiento general del modelo. Sin embargo, las bases de datos ya estaban previamente separadas. El archivo train.csv es la base de datos que se utilizó para entrenar los modelos de ML implementados, y el archivo test.csv se utilizó para hacer predicciones de los modelos y validar su rendimiento a través métricas de precisión ('accuracy'). 
Sin embargo, como paso previo al training y validación de los modelos, se estructuró la base de datos train.csv con estrategias de limpieza y corrección de datos faltantes. 

En general la metodología seguida para solucionar el reto fue:
- Definición del problema
- Limpieza y estrucuración de la base de datos
- Selección de variables significativas
- Aplicación de modelos de ML
- Refinaminamiento de los modelos de ML
- Resultados finales 

## Modelos ML implementados

Los modelos de Machine Learning implementados fueron:
1. Árboles de decisión
2. Regresión Logística
3. K-Nearest Neighbor
4. Feedforward Multilayer

Donde cada modelo se implementó utilizando una combinación de hiperparámetros óptimizada tras implementar herramientas de refinamiento. Estos resultados se encuentran en el pdf del reporte.

## Correcciones y Entregas Previas

Las entregas previas realizadas fueron:

Entrega 1 en Github: https://github.com/Mikeeee40/TitanicEquipo1

Entrega 2 en Github: https://github.com/LuisAguilar3456/Avance-Ciencia-de-Datos-1

Los comentarios y las correciones aplicadas para la entrega final fueron:

Ahora se incluye en el reporte final la razón por la que se utilizó un número aleatorio para asignar las edades faltantes a instancias que no tenian este atributo. Se incluyen las correciones específicadas sobre el pdf en la entrega 2, donde se agregaron referencias en introducción y en la explicación del funcionamiento de cada modelo de ML utilizado agregando más información en cada uno sobre su sustento matemático. Se corrigeron ligeros detalles de redacción y ortografía. Se mejoró la presentación de los resultados obtenidos en tablas estructuradas para garantizar una mejor visualización. Además, se completó el archivo final con el refinamiento de cada modelo, los resultados finales obtenidos y conclusiones.


## Sobre el repositorio

El repositorio contiene los archivos utilizados para los modelos y los resultados obtenidos de la siguiente manera:
 
En el repositorio: Se muestran las bases de datos utilizadas train.csv y test.csv

1ra carpeta "Limpieza y visualización de datos": Se incluyen los procesos ETL implementados así como el One Hot Encoding (variables dummie). Además, se incluyen los 10 csv para las diferentes predicciones de acuerdo al dataset (Se seleccionaron 5 datasets de entrenamiento y 5 de prueba)


2da carpeta "Implementación de modelos": Se incluyen los códigos de los modelos implementados sin refinamiento

3ra carpeta "Refinamiento del modelo": Se incluyen los con códigos donde se utiliza RandomSearchCV como estrategia de refinamiento

4ta carpeta "Resultados": Se incluyen los resultados finales que se subieron a la plataforma Kaggle
