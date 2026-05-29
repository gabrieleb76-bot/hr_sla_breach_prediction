"""
feature_engineering.py

Funciones reutilizables para crear variables derivadas y construir
la variable objetivo del proyecto de incumplimientos de SLA.
"""

import numpy as np
import pandas as pd


def convert_to_datetime(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Convierte columnas especificadas a formato datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    columns : list
        Columnas que se convertirán a datetime.

    Returns
    -------
    pd.DataFrame
        Dataset con columnas convertidas.
    """
    df = df.copy()

    for column in columns:
        df[column] = pd.to_datetime(df[column])

    return df


def calculate_resolution_hours(
    df: pd.DataFrame,
    start_column: str = "First Response Time",
    end_column: str = "Time to Resolution",
    output_column: str = "horas_resolucion"
) -> pd.DataFrame:
    """
    Calcula la duración en horas entre dos columnas temporales.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    start_column : str
        Columna de inicio.
    end_column : str
        Columna de fin.
    output_column : str
        Nombre de la nueva columna calculada.

    Returns
    -------
    pd.DataFrame
        Dataset con la nueva columna de duración en horas.
    """
    df = df.copy()

    df[output_column] = (
        df[end_column] - df[start_column]
    ).dt.total_seconds() / 3600

    return df


def correct_negative_resolution_hours(
    df: pd.DataFrame,
    input_column: str = "horas_resolucion",
    output_column: str = "horas_resolucion_corregidas",
    correction_hours: int = 24
) -> pd.DataFrame:
    """
    Corrige valores negativos de duración sumando una ventana temporal.

    En este proyecto se utiliza una corrección de 24 horas porque los
    valores negativos se encontraban dentro de una ventana diaria.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    input_column : str
        Columna con duración original.
    output_column : str
        Nombre de la columna corregida.
    correction_hours : int
        Número de horas a sumar cuando la duración es negativa.

    Returns
    -------
    pd.DataFrame
        Dataset con duración corregida.
    """
    df = df.copy()

    df[output_column] = np.where(
        df[input_column] < 0,
        df[input_column] + correction_hours,
        df[input_column]
    )

    return df


def create_sla_expected_hours(
    df: pd.DataFrame,
    priority_column: str = "Ticket Priority",
    output_column: str = "sla_esperado_horas"
) -> pd.DataFrame:
    """
    Crea una columna de SLA esperado en horas según la prioridad del ticket.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    priority_column : str
        Columna que contiene la prioridad del ticket.
    output_column : str
        Nombre de la columna de SLA esperado.

    Returns
    -------
    pd.DataFrame
        Dataset con SLA esperado.
    """
    df = df.copy()

    sla_by_priority = {
        "Critical": 6,
        "High": 12,
        "Medium": 18,
        "Low": 24
    }

    df[output_column] = df[priority_column].map(sla_by_priority)

    return df


def create_sla_breach_target(
    df: pd.DataFrame,
    resolution_column: str = "horas_resolucion_corregidas",
    sla_column: str = "sla_esperado_horas",
    target_column: str = "incumplio_sla"
) -> pd.DataFrame:
    """
    Crea la variable objetivo de incumplimiento de SLA.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    resolution_column : str
        Columna con horas de resolución corregidas.
    sla_column : str
        Columna con SLA esperado.
    target_column : str
        Nombre de la variable objetivo.

    Returns
    -------
    pd.DataFrame
        Dataset con variable objetivo.
    """
    df = df.copy()

    df[target_column] = np.where(
        df[resolution_column] > df[sla_column],
        1,
        0
    )

    return df