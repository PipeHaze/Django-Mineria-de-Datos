from django.urls import path
from . import views

app_name = 'graficos'

urlpatterns = [
    path('', views.grafico_view, name='grafico'),
    path('grafico_comparativo/', views.grafico_comparativo_view, name='grafico_comparativo'),
    path('graficos_dobles/', views.graficos_dobles, name='graficos_dobles'),
    path('grafico_polinomico/', views.grafico_polinomico, name='grafico_polinomico'),
    path('generar-pdf/', views.generar_pdf_xhtml, name='generar_pdf'),
    path('generar_pdf_simple/', views.generar_pdf__simple_xhtml, name='grafico_simple'),
    path("heatmap/", views.heatmap_homicidios, name="heatmap_homicidios"),
    path("mapachile/", views.mapa_homicidios_view, name="mapa_homicidios"),


]
