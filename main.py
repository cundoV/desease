# IMPORTACION DE LIBRERIAS 
import functions as fn
import pandas as pd
import numpy as np
import openpyxl 
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_curve, auc, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize




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
