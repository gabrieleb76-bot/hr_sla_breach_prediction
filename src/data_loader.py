"""
data_loader.py

Funciones para cargar los datos procesados del proyecto SLA.
"""

from pathlib import Path
import pandas as pd
import streamlit as st


@st.cache_data
def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset final listo para modelado.

    Parameters
    ----------
    file_path : str
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    return pd.read_csv(path)