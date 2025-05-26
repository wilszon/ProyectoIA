import streamlit as st
import pandas as pd
import joblib

# Cargar los modelos
scaler = joblib.load("scaler.pkl")
svc_model = joblib.load("xgboost_model.pkl")

# Título y subtítulo
st.title("Modelo de Predicción de Deserción Universitaria")
st.subheader("Realizado por Mateo Sandoval, Wilson Suarez, Cristian Cala")

# Introducción
st.write("Esta aplicación utiliza IA para predecir si un estudiante continuará o abandonará su carrera universitaria, basándose en factores académicos y personales.")

# Imagen
st.image("https://www.reporterosasociados.com.co/wp/wp-content/uploads/2023/06/Estudiante-universitaria-en-una-Aula-de-Clase.Foto-Ferran-Nadeu.jpg", use_container_width=True)

# Entrada de datos desde el sidebar
st.sidebar.header("Introduce los datos del estudiante")

# Inputs
study_time = st.sidebar.selectbox("Tiempo de Estudio", [1, 2, 3, 4], format_func=lambda x: ["< 1 hora", "2 horas", "3 horas", "4+ horas"][x - 1])
number_of_failures = st.sidebar.selectbox("Número de Inasistencias", list(range(0, 4)))
wants_higher_education = st.sidebar.selectbox("¿Desea Educación Superior?", ["Sí", "No"])
grade_1 = st.sidebar.slider("Nota 1", 0, 5, 3)
grade_2 = st.sidebar.slider("Nota 2", 0, 5, 3)

# Crear DataFrame
data = pd.DataFrame({
    "Study_Time": [study_time],
    "Number_of_Failures": [number_of_failures],
    "Wants_Higher_Education": [1 if wants_higher_education == "Sí" else 0],
    "Grade_1": [grade_1*4],
    "Grade_2": [grade_2*4]
})

# Normalizar
scaled_data = scaler.transform(data)

# Predicción
prediction = svc_model.predict(scaled_data)[0]

# Mostrar resultado
st.markdown("---")
if prediction:
    st.markdown("<h2 style='color: red; text-align: center;'>❌ Si vas a abandonar tu carrera ❌</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='color: blue; text-align: center;'>✅ Si vas a continuar con tu carrera ✅</h2>", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center;'>&copy; Unab2025</p>", unsafe_allow_html=True)
