import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from .utils import leer_datos_excel  # Ajusta la ruta según tu estructura

def generar_grafico(region_index=0, variable='Homicidios', formato='base64'):
    tabla = leer_datos_excel()
    region = tabla.iloc[region_index]
    años = [[a] for a in range(2018, 2025)]
    y = [region[f'{variable}_{año}'] for año in range(2018, 2025)]

    modelo = LinearRegression()
    modelo.fit(años, y)
    predicciones = modelo.predict(años)

    plt.figure(figsize=(10,6))
    plt.scatter([a[0] for a in años], y, color='blue', label='Datos reales', s=100)
    plt.plot([a[0] for a in años], predicciones, color='red', label='Regresión lineal')
    plt.title(f"{region['Unidad territorial']}: Evolución de {variable} (2018-2024)")
    plt.xlabel('Año')
    plt.ylabel(variable)
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    if formato == 'base64':
        imagen_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return imagen_base64
    elif formato == 'buffer':
        return buf
    else:
        raise ValueError("Formato no válido: usa 'base64' o 'buffer'")
