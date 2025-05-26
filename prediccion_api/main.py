from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelos
scaler = joblib.load("scaler.pkl")
model = joblib.load("xgboost_model.pkl")

app = FastAPI()

class Estudiante(BaseModel):
    study_time: int
    number_of_failures: int
    wants_higher_education: int
    grade_1: float
    grade_2: float

@app.post("/predict")
def predict(data: Estudiante):
    input_data = np.array([[data.study_time,
                            data.number_of_failures,
                            data.wants_higher_education,
                            data.grade_1 * 4,
                            data.grade_2 * 4]])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    return {"prediction": int(prediction)}