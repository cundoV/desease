# IMPORTACION DE LIBRERIAS 
import functions as fn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report





#IMPORTACION DE LOS DF
df_trainging= pd.read_csv("data/Training.csv")
df_test = pd.read_csv("data/Testing.csv")

#PRIMERA VISTA DE LOS DF
df_trainging.head()
df_test.head()
df_trainging.dtypes
df_trainging.shape


#LIMPIEZA DE LOS DF

# Me fijo si hay alguna columna que tennga valores NULL y la dropeo
df_trainging.isnull().any()[df_trainging.isnull().any()]
df_test.isnull().any()[df_test.isnull().any()]

df_trainging = df_trainging.drop(columns=['Unnamed: 133'])

# Me fijo si hay filas repetidas y las dropeo
df_trainging.duplicated(keep=False).sum()
df_test.duplicated(keep=False).sum()

df_trainging = df_trainging.drop_duplicates(keep='first')

# Me fijo si tengo filas repetidas con distinto diganostico, no tengo ninguna
df_trainging.duplicated(subset=df_trainging.columns.difference(['prognosis']), keep=False).sum()
df_test.duplicated(subset=df_test.columns.difference(['prognosis']), keep=False).sum()

# Me fijo si tengo valores distintos a 0 y 1, no tengo ninguno
conteo_valores_distintos= df_trainging[df_trainging.columns.difference(['prognosis'])].isin([0, 1]).all()
conteo_valores_distintos[conteo_valores_distintos != True]

conteo_valores_distintos= df_test[df_test.columns.difference(['prognosis'])].isin([0, 1]).all()
conteo_valores_distintos[conteo_valores_distintos != True]


#SEGUNDA VISTA DE LOS DF

#lista de enfermedades y sintomas 
Enfermedades = df_trainging['prognosis'].unique().tolist()
Sintomas = df_trainging.columns.difference(['prognosis']).values.tolist()

# Filtrar las filas por la columna 'prognosis'
filtro_train = df_trainging.loc[df_trainging['prognosis'] == 'Fungal infection']
filtro_test = df_test.loc[df_test['prognosis'] == 'Fungal infection']

filtro_train = df_trainging.loc[df_trainging['prognosis'] == 'Common Cold']
filtro_test = df_test.loc[df_test['prognosis'] == 'Common Cold']

# Eliminar las columnas con todos sus valores iguales a 0
filtro_train.loc[:, (filtro_train != 0).any(axis=0)]
filtro_test.loc[:, (filtro_test != 0).any(axis=0)]



#este grafico me perimte enteder que hay un sesgo hacia la derecha, lo cual voy a tener valores muy lejos de la normal
#gracias a este grafico voy a graficar los sitomas con frecuencia mayor de 50 apariciones en los 303 diagnosticos
fn.gauss(df_trainging)

#este grafico me muestra los porsetnajes de aparicion del determinado sintoma eleji 10 pq estaba en el rango del grafico de arriba
fn.sintomasfrec(df_trainging)

#con respecto a los otros sintomas, este grafico me permite observar la media mediana y la desviacion estandar de los porsentajes de los 
#otros sintomas los cuales son muy pequenos  
fn.otrossintomas(df_trainging)

#mas o menos con estos graficos podemos decir cuales sintomas son mas frecuentes 
# una aplicacion de esto seria que la intrefaz de usuario pregunte primero con estos sitomas 

#est grafico se puede ver la cantidad de sintmoas de cada enfermedad/diagnostico
fn.diagcomplejos(df_trainging,Enfermedades)


#MODELO DE ML
#MODELO RANDOM FOREST

X_train = df_trainging.drop(columns=['prognosis'])
Y_train = df_trainging['prognosis']

X_test = df_test.drop(columns=['prognosis'])
Y_test = df_test['prognosis']


# Definir el modelo
rf = RandomForestClassifier()

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Configurar la búsqueda en grid
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Ejecutar la búsqueda
grid_search.fit(X_train, Y_train)

print(f"Mejores hiperparámetros: {grid_search.best_params_}")
print(f"Mejor score de validación cruzada: {grid_search.best_score_}")

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, Y_train)


y_pred = best_rf.predict(X_test)


# Evaluar el rendimiento
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

#Con esta precisión del 97.62% y una matriz de confusión perfecta, parece que el modelo de clasificación es altamente efectivo en la identificación de diversas enfermedades. 
# Sin embargo, vale la pena investigar más a fondo por qué la infección fúngica tiene un rendimiento ligeramente inferior. En general, los resultados indican una alta capacidad 
# del modelo para clasificar una amplia gama de enfermedades con precisión y confianza.