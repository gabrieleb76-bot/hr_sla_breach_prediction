"""
prediction.py

Funciones para cargar el modelo final y generar predicciones.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model(model_path: str):
    """
    Carga el modelo entrenado desde un archivo .pkl.

    Parameters
    ----------
    model_path : str
        Ruta del modelo guardado.

    Returns
    -------
    object
        Pipeline entrenado.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    return joblib.load(path)


def predict_sla_risk(model, input_data: pd.DataFrame) -> dict:
    """
    Genera predicción y probabilidad de incumplimiento SLA.

    Parameters
    ----------
    model
        Modelo entrenado.
    input_data : pd.DataFrame
        Datos del ticket.

    Returns
    -------
    dict
        Predicción, probabilidad y nivel de riesgo.
    """
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    breach_probability = probabilities[1]

    if breach_probability >= 0.70:
        risk_level = "Alto"
        recommendation = "Priorizar el ticket y revisar preventivamente."
    elif breach_probability >= 0.40:
        risk_level = "Medio"
        recommendation = "Monitorizar el ticket durante el ciclo de resolución."
    else:
        risk_level = "Bajo"
        recommendation = "Seguir flujo operativo estándar."

    return {
        "prediction": int(prediction),
        "breach_probability": round(breach_probability * 100, 2),
        "risk_level": risk_level,
        "recommendation": recommendation
    }
    