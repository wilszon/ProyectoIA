import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelos
scaler = joblib.load("scaler.pkl")
nb_model = joblib.load("nb_model.pkl")


# Configuración de la aplicación
st.title("Modelo predicción de deserción universitaria con IA")
st.subheader("Realizado por Mateo Sandoval, Wilson Suarez, Cristian Cala")

# Introducción
st.write("""
Esta aplicación permite predecir si un estudiante universitario tiene riesgo de abandonar su carrera
o si continuará con éxito. Para ello, ingresa los datos en los campos correspondientes y presiona el botón de predicción.
""")

# Imagen
st.image("https://www.reporterosasociados.com.co/wp/wp-content/uploads/2023/06/Estudiante-universitaria-en-una-Aula-de-Clase.Foto-Ferran-Nadeu.jpg", use_container_width=True)

# Entradas del usuario
age = st.slider("Age", min_value=15, max_value=30, value=20)
address = st.selectbox("Address", ["U", "R"])
family_size = st.selectbox("Family_Size", ["GT3", "LE3"])
parental_status = st.selectbox("Parental_Status", ["A", "T"])
study_time = st.slider("Study_Time", min_value=1, max_value=4, value=2)
number_of_failures = st.slider("Number_of_Failures", min_value=0, max_value=3, value=0)
school_support = st.selectbox("School_Support", ["yes", "no"])
family_support = st.selectbox("Family_Support", ["yes", "no"])
extra_paid_class = st.selectbox("Extra_Paid_Class", ["yes", "no"])
extra_curricular = st.selectbox("Extra_Curricular_Activities", ["yes", "no"])
attended_nursery = st.selectbox("Attended_Nursery", ["yes", "no"])
wants_higher_education = st.selectbox("Wants_Higher_Education", ["yes", "no"])
internet_access = st.selectbox("Internet_Access", ["yes", "no"])
family_relationship = st.slider("Family_Relationship", min_value=1, max_value=5, value=3)
number_of_absences = st.slider("Number_of_Absences", min_value=0, max_value=93, value=5)
final_grade = st.slider("Final_Grade", min_value=0, max_value=5, value=3)

# Convertir las entradas a DataFrame
data = pd.DataFrame([[age, address, family_size, parental_status, study_time, number_of_failures, 
                      school_support, family_support, extra_paid_class, extra_curricular, attended_nursery, 
                      wants_higher_education, internet_access, family_relationship, number_of_absences, final_grade]],
                     columns=['Age', 'Address', 'Family_Size', 'Parental_Status', 'Study_Time',
                              'Number_of_Failures', 'School_Support', 'Family_Support', 'Extra_Paid_Class',
                              'Extra_Curricular_Activities', 'Attended_Nursery', 'Wants_Higher_Education',
                              'Internet_Access', 'Family_Relationship', 'Number_of_Absences', 'Final_Grade'])

# Normalizar los datos
data_scaled = scaler.transform(data)

# Botón de predicción
if st.button("Predecir Deserción"):
    prediction = svc_model.predict(data_scaled)[0]
    
    if prediction:
        st.error("⚠️ **Si vas a abandonar tu carrera**", icon="🚨")
    else:
        st.success("✅ **Si vas a continuar con tu carrera**", icon="🎓")

# Línea separadora
st.markdown("---")

# Copyright
st.markdown("<p style='text-align: center;'>&copy; Unab2025</p>", unsafe_allow_html=True)
