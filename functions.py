#FUNICONES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
        
def gauss(df):

    suma_columnas = df.drop(columns=['prognosis']).sum().sort_values(ascending=False)
   
    # Calcular la media y la desviación estándar de la serie
    media = np.mean(suma_columnas)
    desviacion_estandar = np.std(suma_columnas)

    # Generar valores de x para la campana de Gauss
    x = np.linspace(min(suma_columnas), max(suma_columnas), 100)

    # Calcular la función de densidad de probabilidad de la distribución normal (campana de Gauss)
    pdf = 1 / (desviacion_estandar * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - media) / desviacion_estandar) ** 2)

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, label='Campana de Gauss', color='blue')  # Graficar la campana de Gauss
    plt.hist(suma_columnas, bins=5, density=True, alpha=0.5, color='green', label='Datos originales')  # Graficar el histograma de los datos originales
    plt.title('Gráfico de Campana de Gauss')
    plt.xlabel('Valores')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.grid(True)
    plt.show()


def sintomasfrec(df):
        
    suma_columnas = df.drop(columns=['prognosis']).sum().sort_values(ascending=False)

    cinco_mas_grandes = suma_columnas.nlargest(10).sort_values(ascending=True)
    
    # Graficar las n columnas con las sumas más grandes

    ax = cinco_mas_grandes.plot(kind='barh', figsize=(10, 6), color='skyblue', title='Prosetaje de aparicion de sintoma en 303 diagnosticos ', xlabel='Porsentaje [%]', ylabel='Sintomas')

    # Obtener los  valores de la serie
    valores = (cinco_mas_grandes.values)

    # Establecer los ticks en el eje x para que coincidan con los valores de la serie
    ax.set_xticks(valores)
    ax.set_xticklabels(np.round(valores*100/303, decimals=0).astype(int))

    valores = cinco_mas_grandes.values

    # Iterar sobre los valores y agregar líneas verticales y etiquetas
    for i, v in enumerate(valores):
        ax.plot([v, v], [i - (i+0.5), i + 0.2], color='grey')  # Agregar línea vertical al final de la barra

    # Mostrar el gráfico
    plt.show()
    
    
def otrossintomas(df):
    
    suma_columnas = df.drop(columns=['prognosis']).sum().sort_values(ascending=False)
    otros = suma_columnas.nsmallest(121).sort_values(ascending=True).mul(100).div(303)

    fig, ax = plt.subplots()
    ax.boxplot(otros, showmeans = True, meanline = True, showfliers = False)
        
    # Establece los límites del eje y
    ax.set_ylim(-2, 10)  # Ajusta los límites según tus datos

    # Personaliza los ticks del eje y
    ax.set_yticks(range(0, 10, 1))  # Establece los ticks cada 10 unidades
    ax.yaxis.set_tick_params(labelsize=8)  # Ajusta el tamaño de las etiquetas

    plt.title('Media, mediana y desviacion de los posentajes de apariciones de los 121 sintomas restantes en 303 diagnosticos ')
    plt.ylabel('Porsentaje [%]')
    plt.legend()
    
    
    # Muestra el gráfico
    plt.show()
    
def diagcomplejos(df,Enfermedades):
    
    valores = []

    for i in range(len(Enfermedades)):
        filtro_train = df.loc[df['prognosis'] == Enfermedades[i]]
        filtro_train = filtro_train.loc[:, (filtro_train != 0).any(axis=0)]
        valores.append(len(filtro_train.columns) - 1)

    serie = pd.Series(valores)
    serie_con_nuevos_indices = pd.Series(serie.values, index=Enfermedades).sort_values(ascending=True)

    ax = serie_con_nuevos_indices.plot(kind='barh', figsize=(10, 6), color='skyblue', title='Enfermedad/Diagnostico y cantidad sintomas', xlabel='Numero de sintomas', ylabel='Enfermedades')

    # Obtener los  valores de la serie
    valores = (serie_con_nuevos_indices.values)

    # Establecer los ticks en el eje x para que coincidan con los valores de la serie
    ax.set_xticks(valores)
    ax.set_xticklabels(np.round(valores*100/303, decimals=0).astype(int))

    valores = serie_con_nuevos_indices.values

    # Iterar sobre los valores y agregar líneas verticales y etiquetas
    for i, v in enumerate(valores):
        ax.plot([v, v], [i - (i+0.5), i + 0.2], color='grey')  # Agregar línea vertical al final de la barra

    # Mostrar el gráfico
    plt.show()
