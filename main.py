import pandas as pd
import numpy as np
import openpyxl 
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line


df = pd.read_csv("data/Training.csv")
df.head()
df.dtypes
df.shape

# Me fijo si hay alguna columna que tennga valores NULL y la dropeo
df.isnull().any()[df.isnull().any()]
df = df.drop(columns=['Unnamed: 133'])

# Me fijo si hay filas repetidas y las dropeo
df.duplicated(keep=False).sum()
df = df.drop_duplicates(keep='first')

# Me fijo si tengo filas repetidas con distinto diganostico, no tengo ninguna
df.duplicated(subset=df.columns.difference(['prognosis']), keep=False).sum()

# Me fijo si tengo valores distintos a 0 y 1, no tengo ninguno
conteo_valores_distintos= df[df.columns.difference(['prognosis'])].isin([0, 1]).all()
conteo_valores_distintos[conteo_valores_distintos != True]

Enfermedades = df['prognosis'].unique().tolist()
Sintomas = df.columns.difference(['prognosis']).values.tolist()


dfs_por_enfermedad = []  # Crear una lista vacía

# Iterar sobre cada enfermedad y crear un DataFrame para cada una
for enfermedad in Enfermedades:
    # Filtrar el DataFrame original por enfermedad
    df_filtrado = df[df['prognosis'] == enfermedad]
    # Agregar el DataFrame filtrado a la lista
    dfs_por_enfermedad.append(df_filtrado)
    
# Lista para almacenar los DataFrames con las columnas filtradas
dfs_con_unos = []

# Iterar sobre cada DataFrame en la lista
for df_enfermedad in dfs_por_enfermedad:
    # Filtrar las columnas que tienen valores de 1 para cada fila
    df_filtrado = df_enfermedad.loc[:, (df_enfermedad == 1).any()]
    # Agregar el DataFrame filtrado a la lista
    dfs_con_unos.append(df_filtrado)

for i in range(41):
    dfs_con_unos[i]['prognosis'] = Enfermedades[i]

dfs_con_unos[3]