import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para generar imágenes en servidores
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import mode
import base64
import io

def leer_datos_excel():
    archivo = pd.read_excel("datos.xlsx")
    archivo.columns = archivo.columns.str.strip()
    tabla_reorganizada = archivo.pivot(
        index='Unidad territorial',
        columns='Variable',
        values=['2018', '2019', '2020', '2021', '2022', '2023', '2024']
    ).reset_index()

#datos del dataset

    tabla_reorganizada.columns = ['Unidad territorial', 'Homicidios_2018', 'Tasa_2018',
                                  'Homicidios_2019', 'Tasa_2019', 'Homicidios_2020', 'Tasa_2020',
                                  'Homicidios_2021', 'Tasa_2021', 'Homicidios_2022', 'Tasa_2022',
                                  'Homicidios_2023', 'Tasa_2023', 'Homicidios_2024', 'Tasa_2024']
    return tabla_reorganizada

def generar_grafico(region_index=0, variable='Homicidios'):
    tabla = leer_datos_excel()
    region = tabla.iloc[region_index]
    
    # Rango de años
    años = list(range(2018, 2025))
    x = [[a] for a in años]

    # Valores Y (dependen de si es 'Homicidios' o 'Tasa')
    y = []
    for año in años:
        valor = region.get(f'{variable}_{año}', None)
        y.append(valor if pd.notnull(valor) else 0)  # Reemplaza NaN con 0

    # Modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(x, y)
    predicciones = modelo.predict(x)

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(años, y, color='blue', label='Datos reales', s=100)
    plt.plot(años, predicciones, color='red', linewidth=2, label='Regresión lineal')
    plt.title(f"{region['Unidad territorial']}: Evolución de {variable} (2018-2024)")
    plt.xlabel('Año')
    plt.ylabel("Cantidad" if variable == 'Homicidios' else "Tasa por 100 mil hab.")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Etiquetas en los puntos
    for i, año in enumerate(años):
        plt.annotate(f'{y[i]:.2f}', (año, y[i]), xytext=(5, 5), textcoords='offset points')

    # Convertir a imagen base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    imagen_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return imagen_base64

def obtener_regiones():
    tabla = leer_datos_excel()
    return tabla['Unidad territorial'].tolist()
