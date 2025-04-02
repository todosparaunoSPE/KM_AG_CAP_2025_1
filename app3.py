# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:59:59 2025

@author: jperezr
"""

import streamlit as st
import pandas as pd
import folium
from sklearn.cluster import KMeans
import numpy as np
import random
from deap import base, creator, tools, algorithms
from math import radians, sin, cos, sqrt, atan2
import requests

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Función para calcular la distancia en kilómetros usando la fórmula haversine
def calcular_distancia_km(lat1, lon1, lat2, lon2):
    # Radio de la Tierra en kilómetros
    R = 6371.0
    
    # Convertir grados a radianes
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    # Diferencia de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula haversine
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distancia en kilómetros
    distancia = R * c
    return distancia

# Función para cargar los datos
@st.cache_data
def load_data():
    # Cambia 'BASE_UTILIZADA_2025.csv' por 'BASE_UTILIZADA_2025.xlsx'
    data = pd.read_excel('BASE_UTILIZADA_2025.xlsx')  # Asegúrate de que el archivo XLSX esté en la misma carpeta
    return data

# Función para ejecutar el algoritmo genético
def ejecutar_algoritmo_genetico(df_filtrado):
    # Definir la función objetivo
    def evaluar_ubicacion(individual, df_filtrado):
        total_distancia = 0
        for idx, row in df_filtrado.iterrows():
            # Asegúrate de que individual sea una lista o tupla con dos elementos
            if isinstance(individual, (list, tuple)) and len(individual) == 2:
                distancia = np.sqrt((row['LATITUD'] - individual[0])**2 + (row['LONGITUD'] - individual[1])**2)
                total_distancia += distancia * row['PEA']
            else:
                raise ValueError("El individuo no tiene la estructura correcta: [latitud, longitud]")
        return total_distancia,

    # Configurar el algoritmo genético
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)  # Usar list en lugar de tuple

    # Rangos válidos para latitud y longitud en México
    lat_range = (14.0, 33.0)  # Ajusta según sea necesario
    lon_range = (-118.0, -86.0)  # Ajusta según sea necesario

    toolbox = base.Toolbox()
    toolbox.register("attr_lat", random.uniform, lat_range[0], lat_range[1])
    toolbox.register("attr_lon", random.uniform, lon_range[0], lon_range[1])
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_lat, toolbox.attr_lon), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluar_ubicacion, df_filtrado=df_filtrado)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # Función para limitar los valores de latitud y longitud después de la mutación
    def mutar_limitada(individual, indpb):
        # Convertir el individuo a lista para poder modificarlo
        individual = list(individual)
        # Aplicar mutación Gaussiana
        if random.random() < indpb:
            individual[0] += random.gauss(0, 1)  # Mutar latitud
            individual[1] += random.gauss(0, 1)  # Mutar longitud
        # Limitar los valores dentro de los rangos válidos
        individual[0] = np.clip(individual[0], lat_range[0], lat_range[1])
        individual[1] = np.clip(individual[1], lon_range[0], lon_range[1])
        # Convertir de nuevo a Individual
        return creator.Individual(individual),

    toolbox.register("mutate", mutar_limitada, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Ejecutar el algoritmo genético
    population = toolbox.population(n=50)
    NGEN = 40
    CXPB = 0.7
    MUTPB = 0.2

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    # Obtener la mejor solución
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

# Cargar los datos
df = load_data()

# Título de la aplicación
st.title("2o escenario-Modelos utilizados para la propuesta de apertura de nuevos CAP's o en su caso reubicación: K-Means y Algoritmos Genéticos")
# Subtítulo corregido
st.markdown("### Marzo del 2025")

# Sidebar con sección de ayuda y tu nombre
with st.sidebar:
    st.header("Ayuda")
    st.write("""
    Esta aplicación utiliza el modelo de K-Means y Algoritmos Genéticos para sugerir una ubicación óptima basada en la PEA (Población Económicamente Activa) de los municipios de un estado seleccionado.
    - **PIN Rojo**: Municipio PENSIONISSSTE.
    - **PIN Verde**: Municipios AZTECA, CITIBANAMEX, COPPEL, INBURSA, INVERCAP, PRINCIPAL, PROFUTURO, SURA, XXI-BANORTE.
    - **PIN Negro**: Otros municipios.
    - **PIN Azul**: Ubicación sugerida por K-Means.
    - **PIN Anaranjado**: Ubicación sugerida por Algoritmos Genéticos.
    """)
    st.write("Desarrollado por: **Javier Horacio Pérez Ricárdez**")

    # Botón para descargar el reporte PDF
    with open("K_means_CAP.pdf", "rb") as pdf_file:
        st.download_button(
            label="Descargar reporte (PDF)",
            data=pdf_file,
            file_name="K_means_CAP.pdf",
            mime="application/pdf"
        )

# Selectbox para filtrar por ESTADO
estado_seleccionado = st.selectbox('Selecciona un estado', df['ESTADO'].unique())

# Filtrar los datos según el estado seleccionado
df_filtrado = df[df['ESTADO'] == estado_seleccionado].copy()

# Mostrar los datos filtrados
st.write(f"Datos filtrados para el estado de {estado_seleccionado}:")
st.dataframe(df_filtrado)

# Aplicar K-Means para encontrar la ubicación sugerida
if not df_filtrado.empty:
    X = df_filtrado[['LATITUD', 'LONGITUD', 'PEA']]
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(X)
    centroide_kmeans = kmeans.cluster_centers_[0]
    latitude_kmeans = centroide_kmeans[0]
    longitude_kmeans = centroide_kmeans[1]

    # Ejecutar el algoritmo genético
    best_individual = ejecutar_algoritmo_genetico(df_filtrado)
    latitude_genetico = best_individual[0]
    longitude_genetico = best_individual[1]

    # Crear un DataFrame con las ubicaciones sugeridas
    ubicaciones_sugeridas_df = pd.DataFrame({
        'MODELO': ['K-Means', 'Algoritmo Genético'],
        'LATITUD': [latitude_kmeans, latitude_genetico],
        'LONGITUD': [longitude_kmeans, longitude_genetico]
    })
    st.write("Detalles de las ubicaciones sugeridas:")
    st.dataframe(ubicaciones_sugeridas_df)

    # Crear el mapa con Folium
    m = folium.Map(location=[latitude_kmeans, longitude_kmeans], zoom_start=6)

    # Diccionario de colores para los municipios
    colores_municipios = {
        'PENSIONISSSTE': 'red',
        'AZTECA': 'green',
        'CITIBANAMEX': 'green',
        'COPPEL': 'green',
        'INBURSA': 'green',
        'INVERCAP': 'green',
        'PRINCIPAL': 'green',
        'PROFUTURO': 'green',
        'SURA': 'green',
        'XXI-BANORTE': 'green'
    }

    # Agregar marcadores para los municipios
    for idx, row in df_filtrado.iterrows():
        municipio = row['MUNICIPIO'].strip().upper()
        color = colores_municipios.get(municipio, 'black')
        folium.Marker(
            location=[row['LATITUD'], row['LONGITUD']],
            popup=f"Municipio: {municipio}<br>PEA: {row['PEA']}",
            icon=folium.Icon(color=color)
        ).add_to(m)

    # Agregar el marcador para la ubicación sugerida por K-Means (azul)
    folium.Marker(
        location=[latitude_kmeans, longitude_kmeans],
        popup="Ubicación sugerida por K-Means",
        icon=folium.Icon(color='blue')
    ).add_to(m)

    # Agregar el marcador para la ubicación sugerida por Algoritmos Genéticos (gris)
    folium.Marker(
        location=[latitude_genetico, longitude_genetico],
        popup="Ubicación sugerida por Algoritmos Genéticos",
        icon=folium.Icon(color='orange')
    ).add_to(m)

    # Mostrar el mapa en Streamlit
    st.write(f"Mapa de los municipios en {estado_seleccionado} con las ubicaciones sugeridas:")
    st.components.v1.html(m._repr_html_(), height=500)

    # Guardar el mapa en un archivo HTML
    map_path = f'mapa_{estado_seleccionado}.html'
    m.save(map_path)

    # Botón para descargar el mapa
    with open(map_path, "rb") as file:
        btn = st.download_button(
            label=f"Descargar el mapa de {estado_seleccionado} con las ubicaciones sugeridas en HTML",
            data=file,
            file_name=map_path,
            mime="application/html"
        )

    # DataFrame con las ubicaciones sugeridas y los municipios con mayor PEA
    st.write("### Análisis de Distancias")
    st.write("Comparación de las ubicaciones sugeridas con los municipios de mayor PEA:")

    # Obtener los municipios con mayor PEA
    top_municipios = df_filtrado.nlargest(5, 'PEA')  # Top 5 municipios con mayor PEA

    # Crear un DataFrame para el análisis
    analisis_distancias = []
    for idx, row in top_municipios.iterrows():
        distancia_kmeans = calcular_distancia_km(row['LATITUD'], row['LONGITUD'], latitude_kmeans, longitude_kmeans)
        distancia_genetico = calcular_distancia_km(row['LATITUD'], row['LONGITUD'], latitude_genetico, longitude_genetico)
        analisis_distancias.append({
            'MUNICIPIO': row['MUNICIPIO'],
            'PEA': row['PEA'],
            'LATITUD': row['LATITUD'],
            'LONGITUD': row['LONGITUD'],
            'Distancia a K-Means (km)': distancia_kmeans,
            'Distancia a Algoritmo Genético (km)': distancia_genetico
        })

    # Convertir a DataFrame
    analisis_distancias_df = pd.DataFrame(analisis_distancias)

    # Crear una fila con la suma de las columnas
    suma_pea = analisis_distancias_df['PEA'].sum()
    suma_distancia_kmeans = analisis_distancias_df['Distancia a K-Means (km)'].sum()
    suma_distancia_genetico = analisis_distancias_df['Distancia a Algoritmo Genético (km)'].sum()

    # Crear un DataFrame con la fila de suma
    fila_suma = pd.DataFrame({
        'MUNICIPIO': ['SUMA'],
        'PEA': [suma_pea],
        'LATITUD': [''],
        'LONGITUD': [''],
        'Distancia a K-Means (km)': [suma_distancia_kmeans],
        'Distancia a Algoritmo Genético (km)': [suma_distancia_genetico]
    })

    # Concatenar la fila de suma al DataFrame original
    analisis_distancias_df = pd.concat([analisis_distancias_df, fila_suma], ignore_index=True)

    # Mostrar el DataFrame
    st.dataframe(analisis_distancias_df)

    # Incrustar el mapa desde la URL proporcionada
    st.write("### Mapa con los PIN de K-Means (azul), Algoritmo Genético (anaranjado) CAP's de PENSIONISSSTE (rojo)")
    #st.write("A continuación se muestra el mapa incrustado desde la URL proporcionada:")
    mapa_url = "https://todosparaunospe.github.io/mapa1_2025/"
    st.components.v1.iframe(mapa_url, height=500)

    # Descargar el mapa desde la URL
    st.write("### Descargar el Mapa de las ubicaciones de los modelos: K-Means, Algoritmo Genético y CAP's de PENSIONISSSTE")
    response = requests.get(mapa_url)
    if response.status_code == 200:
        with open("mapa_pensionissste.html", "wb") as file:
            file.write(response.content)
        with open("mapa_pensionissste.html", "rb") as file:
            st.download_button(
                label="Descargar Mapa de PENSIONISSSTE (HTML)",
                data=file,
                file_name="mapa_pensionissste.html",
                mime="application/html"
            )
    else:
        st.error("No se pudo descargar el mapa desde la URL proporcionada.")



else:
    
    
    st.warning(f"No hay datos disponibles para el estado de {estado_seleccionado}.")
