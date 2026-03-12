from django.shortcuts import render
from .utils import generar_grafico, obtener_regiones, leer_datos_excel
from sklearn.linear_model import LinearRegression
import os
from django.conf import settings
import json
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
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
from sklearn.preprocessing import PolynomialFeatures
from django.template.loader import render_to_string
import matplotlib
matplotlib.use('Agg')  # Para evitar problemas con GUI
from datetime import datetime
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from xhtml2pdf import pisa






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

    return render(request, 'analisis/regresion_lineal/grafico.html', {
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
        title=f"Comparación de {variable} por región (2018–2030)", #la variable es homicidio
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

    tabla = leer_datos_excel()
    regiones = [(i, row['Unidad territorial']) for i, row in tabla.iterrows()]

    # 🔹 Generar un gráfico individual por región
    graficos_html = []
    for i in region_indices:
        grafico_html = generar_grafico_comparativo([i], variable) #hacer un callback a la funcion creada anteriormente
        graficos_html.append(grafico_html)

    return render(request, 'analisis/regresion_lineal/grafico_comparativo.html', {
        'graficos_html': graficos_html,
        'regiones': regiones,
        'regiones_seleccionadas': region_indices,
        'variable_seleccionada': variable,
    })


def graficos_dobles(request):
    region_indices = request.GET.getlist('region_index', [0])
    region_indices = [int(i) for i in region_indices]

    tabla = leer_datos_excel()
    regiones = [(i, row['Unidad territorial']) for i, row in tabla.iterrows()]

    graficos = []
    for i in region_indices:
        nombre_region = tabla.iloc[i]['Unidad territorial']
        grafico_homicidios = generar_grafico_comparativo([i], 'Homicidios')
        grafico_tasa = generar_grafico_comparativo([i], 'Tasa')
        graficos.append({
            'nombre': nombre_region,
            'homicidios': grafico_homicidios,
            'tasa': grafico_tasa
        })

    return render(request, 'analisis/regresion_lineal/graficos_dobles.html', {
        'graficos': graficos,
        'regiones': regiones,
        'regiones_seleccionadas': region_indices,
    })

def generar_grafico_polinomico(region_indices=[0], variable='Homicidios', grado=2, formato='base64'):
    """
    Genera un gráfico de regresión polinómica (grado variable) para una o varias regiones.
    """

    tabla = leer_datos_excel()

    if isinstance(region_indices, int):
        region_indices = [region_indices]

    años = np.array(list(range(2018, 2025)))
    años_futuros = np.arange(2018, 2031)

    cmap = cm.get_cmap('tab10', len(region_indices))
    fig = go.Figure()

    for idx, i_region in enumerate(region_indices):
        region = tabla.iloc[i_region]
        nombre_region = region['Unidad territorial']

        x = años.reshape(-1, 1)
        y = np.array([region.get(f'{variable}_{año}', 0) for año in años])

        # Modelo polinómico dinámico
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(x)

        modelo = LinearRegression()
        modelo.fit(X_poly, y)

        x_futuros_poly = poly.transform(años_futuros.reshape(-1, 1))
        pred_futuros = modelo.predict(x_futuros_poly)

        residuales = y - modelo.predict(X_poly)
        error_std = np.std(residuales)
        y_upper = pred_futuros + 1.96 * error_std
        y_lower = pred_futuros - 1.96 * error_std

        # Colores
        rgb = cmap(idx)[:3]
        rgba_color = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.9)'
        rgba_fill = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.15)'

        # Datos reales
        fig.add_trace(go.Scatter(
            x=años, y=y,
            mode='markers+text',
            name=f'{nombre_region} - Datos reales',
            marker=dict(size=9, color=rgba_color),
            text=[f"{val:.2f}" for val in y],
            textposition='top center'
        ))

        # Curva polinómica
        fig.add_trace(go.Scatter(
            x=años_futuros, y=pred_futuros,
            mode='lines',
            name=f'{nombre_region} - Regresión polinómica (grado {grado})',
            line=dict(color=rgba_color, width=2, dash='solid')
        ))

        # Área de confianza
        fig.add_trace(go.Scatter(
            x=np.concatenate([años_futuros, años_futuros[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=rgba_fill,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=f"Regresión Polinómica (grado {grado}) de {variable} por Región (2018–2030)",
        xaxis_title="Año",
        yaxis_title="Tasa por 100 mil hab." if variable == "Tasa" else "Cantidad",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.6)')
    )

    return pio.to_html(fig, full_html=False)

def grafico_polinomico(request):
    region_indices = request.GET.getlist('region_index', [0])
    region_indices = [int(i) for i in region_indices]
    variable = request.GET.get('variable', 'Homicidios')
    grado = int(request.GET.get('grado', 2))  # ← nuevo parámetro dinámico

    grafico_html = generar_grafico_polinomico(region_indices, variable, grado)

    tabla = leer_datos_excel()
    regiones = [(i, row['Unidad territorial']) for i, row in tabla.iterrows()]

    return render(request, 'analisis/regresion_polinomica/grafico_polinomico.html', {
        'grafico_html': grafico_html,
        'regiones': regiones,
        'regiones_seleccionadas': region_indices,
        'variable_seleccionada': variable,
        'grado': grado,
    })

    
#GENERAR PDF
def generar_pdf_xhtml(request):
    """
    Genera un PDF con gráficos de homicidios usando xhtml2pdf
    """
    try:
        # Obtener regiones seleccionadas del request
        region_indices = request.GET.getlist('region_index', [])
        if not region_indices:
            region_indices = ['0']  # Default a primera región si no hay selección
        
        region_indices = [int(i) for i in region_indices]
        
        # Leer datos (asumiendo que tienes esta función)
        tabla = leer_datos_excel()
        
        años = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        datos_pdf = []
        
        # Procesar cada región seleccionada
        for i in region_indices:
            if i >= len(tabla):
                continue
                
            nombre_region = tabla.iloc[i]['Unidad territorial']
            
            # Obtener datos numéricos
            homicidios = []
            tasas = []
            for año in años:
                homicidios.append(float(tabla.iloc[i][f'Homicidios_{año}']))
                tasas.append(float(tabla.iloc[i][f'Tasa_{año}']))
            
            # Generar gráficos en base64
            grafico_homicidios = crear_grafico_base64(
                años, homicidios, 
                f'Evolución de Homicidios - {nombre_region}',
                'Año', 'Homicidios', '#e74c3c'
            )
            
            grafico_tasa = crear_grafico_base64(
                años, tasas,
                f'Evolución de Tasa - {nombre_region}',
                'Año', 'Tasa', '#27ae60'
            )
            
            # Calcular estadísticas básicas
            stats = {
                'promedio_homicidios': sum(homicidios) / len(homicidios), #la suma de homicidios dividido el largo de los homicidios
                'promedio_tasa': sum(tasas) / len(tasas),
                'max_homicidios': max(homicidios),
                'min_homicidios': min(homicidios),
                'tendencia': 'Creciente' if homicidios[-1] > homicidios[0] else 'Decreciente'
            }
            
            datos_pdf.append({
                'nombre': nombre_region,
                'homicidios': grafico_homicidios,
                'tasa': grafico_tasa,
                'datos_homicidios': homicidios,
                'datos_tasa': tasas,
                'stats': stats,
                'años': años
            })
        
        # Estadísticas generales
        stats_generales = {
            'total_regiones': len(region_indices),
            'fecha_generacion': datetime.now().strftime("%d/%m/%Y %H:%M"),
            'total_homicidios_2024': sum(
                float(tabla.iloc[i][f'Homicidios_2024']) 
                for i in region_indices if i < len(tabla)
            )
        }
        
        # Contexto para el template
        context = {
            'datos': datos_pdf,
            'stats_generales': stats_generales,
            'años': años
        }
        
        # Renderizar HTML
        html_string = render_to_string('analisis/regresion_lineal/pdf_template.html', context)
        
        # Crear PDF en memoria
        result = BytesIO()
        pdf = pisa.CreatePDF(html_string, dest=result, encoding='UTF-8')
        
        # Verificar si hubo errores
        if pdf.err:
            return HttpResponse('Error al generar el PDF', status=500)
        
        # Preparar respuesta HTTP
        response = HttpResponse(result.getvalue(), content_type='application/pdf')
        filename = f"analisis_homicidios_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        # Log del error para debugging
        print(f"Error generando PDF: {str(e)}")
        return HttpResponse(f'Error: {str(e)}', status=500)

def crear_grafico_base64(x, y, titulo, xlabel, ylabel, color):
    """
    Crea un gráfico con matplotlib y lo retorna como string base64
    """
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, marker='o', color=color, linewidth=2, markersize=8)
        plt.title(titulo, fontsize=12, pad=20)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Agregar valores en los puntos
        for xi, yi in zip(x, y):
            plt.annotate(f'{yi:.1f}', (xi, yi), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar en buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creando gráfico: {e}")
        return ""
    
def generar_pdf__simple_xhtml(request):

    try:

        # Obtener regiones seleccionadas
        region_indices = request.GET.getlist('region_index') #llamar mediante el id del html

        if not region_indices:
            return HttpResponse("No se seleccionó ninguna región", status=400)

        region_indices = [int(i) for i in region_indices]

        tabla = leer_datos_excel()

        años = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

        datos_pdf = []

        for region_index in region_indices:

            if region_index >= len(tabla):
                continue

            nombre_region = tabla.iloc[region_index]['Unidad territorial']

            homicidios = []
            tasas = []

            for año in años:
                homicidios.append(float(tabla.iloc[region_index][f'Homicidios_{año}']))
                tasas.append(float(tabla.iloc[region_index][f'Tasa_{año}']))

            grafico_homicidios = crear_grafico_base64(
                años, homicidios,
                f'Evolución de Homicidios - {nombre_region}',
                'Año', 'Homicidios', '#e74c3c'
            )

            grafico_tasa = crear_grafico_base64(
                años, tasas,
                f'Evolución de Tasa - {nombre_region}',
                'Año', 'Tasa', '#27ae60'
            )

            stats = {
                'promedio_homicidios': sum(homicidios) / len(homicidios),
                'promedio_tasa': sum(tasas) / len(tasas),
                'max_homicidios': max(homicidios), #el maximo de homicidios
                'min_homicidios': min(homicidios), #el minimo de homicidios
                'tendencia': 'Creciente' if homicidios[-1] > homicidios[0] else 'Decreciente'
            }

            datos_pdf.append({
                'nombre': nombre_region,
                'homicidios': grafico_homicidios,
                'tasa': grafico_tasa,
                'datos_homicidios': homicidios,
                'datos_tasa': tasas,
                'stats': stats,
                'años': años
            })

        stats_generales = {
            'total_regiones': len(datos_pdf),
            'fecha_generacion': datetime.now().strftime("%d/%m/%Y %H:%M"),
            'total_homicidios_2024': sum(
                float(tabla.iloc[i]['Homicidios_2024'])
                for i in region_indices if i < len(tabla)
            )
        }

        context = {
            "datos": datos_pdf,
            "stats_generales": stats_generales,
            "años": años
        }

        html_string = render_to_string(
            "analisis/regresion_lineal/pdf_template_simple.html",
            context
        )

        result = BytesIO()
        pdf = pisa.CreatePDF(html_string, dest=result)

        if pdf.err:
            return HttpResponse("Error generando PDF", status=500)

        response = HttpResponse(result.getvalue(), content_type="application/pdf")
        response['Content-Disposition'] = 'attachment; filename="analisis_homicidios.pdf'

        return response

    except Exception as e:
        print(e)
        return HttpResponse("Error interno", status=500)

def grafico_heatmap_homicidios(regiones=None, años=None):

    tabla = leer_datos_excel()

    if años is None: #mostrar el año de los homicidios, el excel solo muestra hasta los datos del 2024
        años = list(range(2018, 2025))

    if regiones:
        tabla = tabla.iloc[regiones]

    data = []

    for i, row in tabla.iterrows():
        fila = []
        for año in años:
            fila.append(row[f'Homicidios_{año}'])
        data.append(fila)

    df = pd.DataFrame(
        data,
        columns=años,
        index=tabla['Unidad territorial']
    )

    fig = px.imshow(
        df,
        labels={
            "x": "Año",
            "y": "Región",
            "color": "Homicidios"
        },
        color_continuous_scale="Reds",
        aspect="auto",
        title="Heatmap de Homicidios por Región y Año"
    )

    fig.update_layout(
        height=650
    )

    return pio.to_html(fig, full_html=False)

def heatmap_homicidios(request): #mapa de calor de los homicidios

    regiones = request.GET.getlist("region_index")
    años = request.GET.getlist("años")

    if regiones:
        regiones = list(map(int, regiones))

    if años:
        años = list(map(int, años))

    grafico = grafico_heatmap_homicidios(regiones, años)

    tabla = leer_datos_excel()

    return render(request, "analisis/regresion_lineal/heatmap.html", {
        "grafico": grafico,
        "regiones": tabla["Unidad territorial"]
    })

import unicodedata

def limpiar_nombre_region(nombre): #al final no sirvio
    if not isinstance(nombre, str):
        return nombre

    # quitar "Región de"
    nombre = nombre.replace("Región De ", "")
    nombre = nombre.replace("Región del ", "")
    nombre = nombre.replace("Región Del ", "")
    nombre = nombre.replace("Región Metropolitana De ", "Metropolitana De ")

    # quitar tildes
    nombre = ''.join(
        c for c in unicodedata.normalize('NFD', nombre)
        if unicodedata.category(c) != 'Mn'
    )

    return nombre.strip()

def generar_mapa_homicidios(año):

    tabla = leer_datos_excel() #ver la tabla del excel
    tabla.columns = tabla.columns.astype(str).str.strip()

    columna = f"Homicidios_{año}"

    # arreglar el nombre de las regiones en excel igual que el json para que muestre el mapa
    mapa_regiones = {
        "Región De Arica Y Parinacota": "Arica y Parinacota",
        "Región De Tarapacá": "Tarapacá",
        "Región De Antofagasta": "Antofagasta",
        "Región De Atacama": "Atacama",
        "Región De Coquimbo": "Coquimbo",
        "Región De Valparaíso": "Valparaíso",
        "Región Metropolitana De Santiago": "Región Metropolitana de Santiago",
        "Región Del Libertador Gral. Bernardo O'higgins": "Libertador General Bernardo O'Higgins",
        "Región Del Maule": "Maule",
        "Región del Ñuble": "Ñuble",
        "Región Del Biobío": "Bío-Bío",
        "Región De La Araucanía": "La Araucanía",
        "Región De Los Ríos": "Los Ríos",
        "Región De Los Lagos": "Los Lagos",
        "Región De Aysén Del Gral. Carlos Ibáñez Del Campo": "Aisén del General Carlos Ibáñez del Campo",
        "Región De Magallanes Y De La Antártica Chilena": "Magallanes y Antártica Chilena"
    }

    tabla["region_geo"] = tabla["Unidad territorial"].map(mapa_regiones)

    ruta_geojson = os.path.join(settings.BASE_DIR, "static/mapas/cl.json")

    with open(ruta_geojson, encoding="utf-8") as f:
        geojson = json.load(f)

    fig = px.choropleth(
        tabla,
        geojson=geojson,
        locations="region_geo",
        featureidkey="properties.name",
        color=columna,
        color_continuous_scale="Reds",
        hover_name="Unidad territorial",
        title=f"Homicidios en Chile ({año})"
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False
    )

      # 🔹 HACER EL MAPA MÁS GRANDE
    fig.update_layout(
        width=1400,
        height=900,
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    return pio.to_html(fig, full_html=False)



def mapa_homicidios_view(request):

    año = request.GET.get("año", "2024")

    mapa_html = generar_mapa_homicidios(año) #callback para mostrar el mapa en la pagina

    return render(request, "analisis/regresion_lineal/mapachile.html", {
        "mapa_html": mapa_html,
        "año_seleccionado": año
    })