from django.shortcuts import render
from .utils import generar_grafico, obtener_regiones, leer_datos_excel
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
from .plot_generator import generar_grafico  # importa tu función
from django.http import HttpResponse
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.colors as mcolors  # al inicio del archivo
import matplotlib.cm as cm





# Create your views here.

def grafico_view(request):
    tabla = leer_datos_excel()
    
    # Obtener parámetros GET
    region_index = int(request.GET.get('region_index', 0))
    variable = request.GET.get('variable', 'Homicidios')
    
    # Generar gráfico
    grafico_html = generar_grafico(region_index, variable, formato='base64')


    # Lista de regiones para el dropdown
    regiones = [(i, row['Unidad territorial']) for i, row in tabla.iterrows()]

    return render(request, 'analisis/grafico.html', {
        'grafico_html': grafico_html,
        'regiones': regiones,
        'region_seleccionada': region_index,
        'variable_seleccionada': variable,
    })

def generar_grafico(region_index=0, variable='Homicidios', formato='base64'):
    tabla = leer_datos_excel()
    region = tabla.iloc[region_index]

    # --- Datos base ---
    años = np.array(list(range(2018, 2025)))
    x = años.reshape(-1, 1)
    y = np.array([region.get(f'{variable}_{año}', 0) for año in años])

    # --- Modelo de regresión lineal ---
    modelo = LinearRegression()
    modelo.fit(x, y)
    predicciones = modelo.predict(x)

    # --- Pronóstico futuro ---
    años_futuros = np.arange(2018, 2031)
    x_futuros = años_futuros.reshape(-1, 1)
    pred_futuros = modelo.predict(x_futuros)

    # --- Intervalo de confianza simple ---
    residuales = y - modelo.predict(x)
    error_std = np.std(residuales)
    y_upper = pred_futuros + 1.96 * error_std
    y_lower = pred_futuros - 1.96 * error_std

    # --- Crear figura Plotly ---
    fig = go.Figure()

    # Datos reales
    fig.add_trace(go.Scatter(
        x=años,
        y=y,
        mode='markers+text',
        name='Datos reales',
        marker=dict(size=10, color='blue'),
        text=[f"{val:.2f}" for val in y],
        textposition='top center'
    ))

    # Línea de regresión
    color_reg = 'red'
    fig.add_trace(go.Scatter(
        x=años_futuros,
        y=pred_futuros,
        mode='lines',
        name='Regresión lineal + pronóstico',
        line=dict(color=color_reg, width=2, dash='dash')
    ))

    # --- Intervalo de confianza (área sombreada) ---
    rgb = mcolors.to_rgb(color_reg)
    rgba_fill = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.15)'

    fig.add_trace(go.Scatter(
        x=np.concatenate([años_futuros, años_futuros[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor=rgba_fill,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))

    # --- Configuración de diseño ---
    fig.update_layout(
        title=f"{region['Unidad territorial']}: Evolución y proyección de {variable} (2018–2030)",
        xaxis_title='Año',
        yaxis_title='Tasa por 100 mil hab.' if variable == 'Tasa' else 'Cantidad',
        legend=dict(x=0, y=1),
        template='plotly_white',
        hovermode='x unified'
    )

    # --- Salida según formato ---
    if formato in ['html', 'base64']:
        return pio.to_html(fig, full_html=False)
    elif formato == 'buffer':
        img_bytes = pio.to_image(fig, format='png')  # Kaleido requerido
        return io.BytesIO(img_bytes)
    else:
        raise ValueError("Formato no soportado: usa 'html', 'base64' o 'buffer'")


def generar_grafico_comparativo(region_indices=[0], variable='Homicidios', formato='base64'):
    """
    Genera un gráfico interactivo comparando una o varias regiones con su regresión lineal y pronóstico.

    Parámetros:
    - region_indices: int o lista de ints -> índices de las regiones
    - variable: str -> columna base (por ejemplo, 'Homicidios' o 'Tasa')
    - formato: 'html', 'base64' o 'buffer'
    """

    tabla = leer_datos_excel()

    # Aceptar un solo índice o lista
    if isinstance(region_indices, int):
        region_indices = [region_indices]

    # Años base y de pronóstico
    años = np.array(list(range(2018, 2025)))
    años_futuros = np.arange(2018, 2031)

    # Paleta de colores automática (distintos tonos)
    cmap = cm.get_cmap('tab10', len(region_indices))

    fig = go.Figure()

    for idx, i_region in enumerate(region_indices):
        region = tabla.iloc[i_region]
        nombre_region = region['Unidad territorial']

        # --- Datos base ---
        x = años.reshape(-1, 1)
        y = np.array([region.get(f'{variable}_{año}', 0) for año in años])

        # --- Modelo lineal ---
        modelo = LinearRegression()
        modelo.fit(x, y)

        # --- Predicciones ---
        x_futuros = años_futuros.reshape(-1, 1)
        pred_futuros = modelo.predict(x_futuros)

        # --- Intervalo de confianza ---
        residuales = y - modelo.predict(x)
        error_std = np.std(residuales)
        y_upper = pred_futuros + 1.96 * error_std
        y_lower = pred_futuros - 1.96 * error_std

        # --- Colores (línea y sombreado) ---
        rgb = cmap(idx)[:3]
        rgba_color = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.9)'
        rgba_fill = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.15)'

        # --- Datos reales ---
        fig.add_trace(go.Scatter(
            x=años,
            y=y,
            mode='markers+text',
            name=f'{nombre_region} - Datos reales',
            marker=dict(size=9, color=rgba_color),
            text=[f"{val:.2f}" for val in y],
            textposition='top center'
        ))

        # --- Regresión + proyección ---
        fig.add_trace(go.Scatter(
            x=años_futuros,
            y=pred_futuros,
            mode='lines',
            name=f'{nombre_region} - Regresión lineal',
            line=dict(color=rgba_color, width=2)
        ))

        # --- Intervalo de confianza (área sombreada) ---
        fig.add_trace(go.Scatter(
            x=np.concatenate([años_futuros, años_futuros[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=rgba_fill,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

    # --- Diseño general ---
    fig.update_layout(
        title=f"Comparación de {variable} por región (2018–2030)",
        xaxis_title='Año',
        yaxis_title='Tasa por 100 mil hab.' if variable == 'Tasa' else 'Cantidad',
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.6)'),
        template='plotly_white',
        hovermode='x unified'
    )

    # --- Salida según formato ---
    if formato in ['html', 'base64']:
        return pio.to_html(fig, full_html=False)
    elif formato == 'buffer':
        img_bytes = pio.to_image(fig, format='png')
        return io.BytesIO(img_bytes)
    else:
        raise ValueError("Formato no soportado: usa 'html', 'base64' o 'buffer'")
    
def grafico_comparativo_view(request):
    region_indices = request.GET.getlist('region_index', [0])
    region_indices = [int(i) for i in region_indices]
    variable = request.GET.get('variable', 'Homicidios')

    grafico_html = generar_grafico_comparativo(region_indices, variable)

    tabla = leer_datos_excel()
    regiones = [(i, row['Unidad territorial']) for i, row in tabla.iterrows()]

    return render(request, 'analisis/grafico_comparativo.html', {
        'grafico_html': grafico_html,
        'regiones': regiones,
        'regiones_seleccionadas': region_indices,
        'variable_seleccionada': variable,
    })
