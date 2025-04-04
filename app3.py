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
import matplotlib.pyplot as plt
import time

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
    data = pd.read_excel('BASE_UTILIZADA_2025.xlsx')
    return data

# Función para ejecutar el algoritmo genético
def ejecutar_algoritmo_genetico(df_filtrado):
    # Definir la función objetivo
    def evaluar_ubicacion(individual, df_filtrado):
        total_distancia = 0
        for idx, row in df_filtrado.iterrows():
            if isinstance(individual, (list, tuple)) and len(individual) == 2:
                distancia = np.sqrt((row['LATITUD'] - individual[0])**2 + (row['LONGITUD'] - individual[1])**2)
                total_distancia += distancia * row['PEA']
            else:
                raise ValueError("El individuo no tiene la estructura correcta: [latitud, longitud]")
        return total_distancia,

    # Configurar el algoritmo genético
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Rangos válidos para latitud y longitud en México
    lat_range = (14.0, 33.0)
    lon_range = (-118.0, -86.0)

    toolbox = base.Toolbox()
    toolbox.register("attr_lat", random.uniform, lat_range[0], lat_range[1])
    toolbox.register("attr_lon", random.uniform, lon_range[0], lon_range[1])
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_lat, toolbox.attr_lon), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluar_ubicacion, df_filtrado=df_filtrado)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def mutar_limitada(individual, indpb):
        individual = list(individual)
        if random.random() < indpb:
            individual[0] += random.gauss(0, 1)
            individual[1] += random.gauss(0, 1)
        individual[0] = np.clip(individual[0], lat_range[0], lat_range[1])
        individual[1] = np.clip(individual[1], lon_range[0], lon_range[1])
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

    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

# Cargar los datos
df = load_data()

# Título de la aplicación
st.title("2o escenario-Modelos utilizados para la propuesta de apertura de nuevos CAP's o en su caso reubicación: K-Means y Algoritmos Genéticos")
st.markdown("### marzo del 2025")

# Sidebar con sección de ayuda
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

    # Agregar marcadores para las ubicaciones sugeridas
    folium.Marker(
        location=[latitude_kmeans, longitude_kmeans],
        popup="Ubicación sugerida por K-Means",
        icon=folium.Icon(color='blue')
    ).add_to(m)

    folium.Marker(
        location=[latitude_genetico, longitude_genetico],
        popup="Ubicación sugerida por Algoritmos Genéticos",
        icon=folium.Icon(color='orange')
    ).add_to(m)

    # Mostrar el mapa en Streamlit
    st.write(f"Mapa de los municipios en {estado_seleccionado} con las ubicaciones sugeridas:")
    st.components.v1.html(m._repr_html_(), height=500)

    # --------------------------------------------------
    # SECCIÓN DE SIMULACIÓN DE ALGORITMOS
    # --------------------------------------------------
    st.write("### Simulación de los algoritmos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**K-Means - Proceso de convergencia**")
        kmeans_points = df_filtrado[['LATITUD', 'LONGITUD']].values
        
        # Configurar figura
        fig_kmeans, ax_kmeans = plt.subplots(figsize=(8, 6))
        plt.close(fig_kmeans)  # Cerramos la figura para evitar doble renderizado
        
        if st.button('Iniciar simulación K-Means', key='kmeans_sim'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Configurar K-Means con inicialización manual y máximo de iteraciones
            kmeans_sim = KMeans(n_clusters=1, init=np.array([[np.mean(kmeans_points[:, 0]), 
                                                             np.mean(kmeans_points[:, 1])]]), 
                               max_iter=1, n_init=1)
            
            centroids_history = []
            
            for i in range(5):
                kmeans_sim.fit(kmeans_points)
                centroids_history.append(kmeans_sim.cluster_centers_[0])
                
                # Actualizar gráfico
                fig_kmeans, ax_kmeans = plt.subplots(figsize=(8, 6))
                ax_kmeans.scatter(kmeans_points[:, 0], kmeans_points[:, 1], c='blue', alpha=0.5, label='Municipios')
                ax_kmeans.scatter(centroids_history[-1][0], centroids_history[-1][1], 
                                c='red', marker='X', s=200, label='Centroide actual')
                
                if len(centroids_history) > 1:
                    ax_kmeans.plot([c[0] for c in centroids_history], [c[1] for c in centroids_history], 
                                  'r--', alpha=0.3, label='Trayectoria')
                
                ax_kmeans.set_xlabel('Latitud')
                ax_kmeans.set_ylabel('Longitud')
                ax_kmeans.set_title(f'Iteración {i+1} - K-Means')
                ax_kmeans.legend()
                
                col1.pyplot(fig_kmeans)
                plt.close(fig_kmeans)
                
                progress_bar.progress((i + 1) / 5)
                status_text.text(f'Iteración {i+1} de 5 completada')
                time.sleep(0.8)
            
            progress_bar.empty()
            status_text.success('¡Simulación K-Means completada!')
    
    with col2:
        st.write("**Algoritmo Genético - Evolución de soluciones**")
        genetic_points = df_filtrado[['LATITUD', 'LONGITUD', 'PEA']].values
        
        # Configurar figura
        fig_genetic, ax_genetic = plt.subplots(figsize=(8, 6))
        plt.close(fig_genetic)
        
        if st.button('Iniciar simulación Algoritmo Genético', key='genetic_sim'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Configuración simplificada del algoritmo genético para simulación
            toolbox = base.Toolbox()
            toolbox.register("attr_float", random.uniform, 14.0, 33.0)
            toolbox.register("individual", tools.initRepeat, creator.Individual, 
                           toolbox.attr_float, n=2)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            pop = toolbox.population(n=30)
            best_history = []
            
            for gen in range(8):
                # Evaluar fitness
                fits = [evaluar_ubicacion(ind, df_filtrado) for ind in pop]
                for fit, ind in zip(fits, pop):
                    ind.fitness.values = fit
                
                # Seleccionar el mejor
                best = tools.selBest(pop, k=1)[0]
                best_history.append(best)
                
                # Actualizar gráfico
                fig_genetic, ax_genetic = plt.subplots(figsize=(8, 6))
                ax_genetic.scatter(genetic_points[:, 0], genetic_points[:, 1], c='blue', alpha=0.5, label='Municipios')
                
                # Dibujar población
                for ind in pop:
                    ax_genetic.scatter(ind[0], ind[1], c='gray', alpha=0.1, s=30)
                
                # Dibujar mejores históricos
                for i, b in enumerate(best_history[:-1]):
                    ax_genetic.scatter(b[0], b[1], c='green', alpha=0.5, s=50)
                
                # Dibujar mejor actual
                ax_genetic.scatter(best_history[-1][0], best_history[-1][1], 
                                  c='red', marker='X', s=200, label='Mejor solución')
                
                ax_genetic.set_xlabel('Latitud')
                ax_genetic.set_ylabel('Longitud')
                ax_genetic.set_title(f'Generación {gen+1} - Algoritmo Genético')
                ax_genetic.legend()
                
                col2.pyplot(fig_genetic)
                plt.close(fig_genetic)
                
                progress_bar.progress((gen + 1) / 8)
                status_text.text(f'Generación {gen+1} de 8 completada')
                time.sleep(0.8)
                
                # Operadores genéticos
                offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
                pop = offspring
            
            progress_bar.empty()
            status_text.success('¡Simulación Algoritmo Genético completada!')

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

    # Análisis de distancias
    st.write("### Análisis de Distancias")
    st.write("Comparación de las ubicaciones sugeridas con los municipios de mayor PEA:")

    top_municipios = df_filtrado.nlargest(5, 'PEA')
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

    analisis_distancias_df = pd.DataFrame(analisis_distancias)
    suma_pea = analisis_distancias_df['PEA'].sum()
    suma_distancia_kmeans = analisis_distancias_df['Distancia a K-Means (km)'].sum()
    suma_distancia_genetico = analisis_distancias_df['Distancia a Algoritmo Genético (km)'].sum()

    fila_suma = pd.DataFrame({
        'MUNICIPIO': ['SUMA'],
        'PEA': [suma_pea],
        'LATITUD': [''],
        'LONGITUD': [''],
        'Distancia a K-Means (km)': [suma_distancia_kmeans],
        'Distancia a Algoritmo Genético (km)': [suma_distancia_genetico]
    })

    analisis_distancias_df = pd.concat([analisis_distancias_df, fila_suma], ignore_index=True)
    st.dataframe(analisis_distancias_df)

    # Mapa incrustado
    st.write("### Mapa con los PIN de K-Means (azul), Algoritmo Genético (anaranjado) CAP's de PENSIONISSSTE (rojo)")
    mapa_url = "https://todosparaunospe.github.io/mapa1_2025/"
    st.components.v1.iframe(mapa_url, height=500)

    # Descargar mapa PENSIONISSSTE
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
