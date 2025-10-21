from django.urls import path
from .views import grafico_view, grafico_comparativo_view

urlpatterns = [
    path('', grafico_view, name='grafico'),
    path('grafico_comparativo/', grafico_comparativo_view, name='grafico_comparativo'),


]
