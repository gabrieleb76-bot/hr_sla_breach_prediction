"""
data_preprocessing.py

Funciones reutilizables para cargar, filtrar y preparar datos iniciales
del proyecto de predicción de incumplimientos de SLA en HR Operations.
"""

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Carga un dataset CSV desde una ruta especificada.

    Parameters
    ----------
    file_path : str
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado como DataFrame.
    """
    return pd.read_csv(file_path)


def filter_closed_tickets(
    df: pd.DataFrame,
    resolution_column: str = "Time to Resolution"
) -> pd.DataFrame:
    """
    Filtra únicamente tickets que tienen tiempo de resolución disponible.

    Estos tickets se consideran válidos para entrenamiento supervisado
    porque conocemos el resultado final del caso.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.
    resolution_column : str
        Columna que indica el tiempo o fecha de resolución.

    Returns
    -------
    pd.DataFrame
        Dataset filtrado con tickets cerrados/resueltos.
    """
    return df[df[resolution_column].notna()].copy()


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Calcula la cantidad de valores nulos por columna.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a revisar.

    Returns
    -------
    pd.Series
        Valores nulos ordenados de mayor a menor.
    """
    return df.isna().sum().sort_values(ascending=False)


def check_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """
    Calcula el porcentaje de valores nulos por columna.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a revisar.

    Returns
    -------
    pd.Series
        Porcentaje de nulos ordenado de mayor a menor.
    """
    return (
        df.isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Cuenta filas duplicadas exactas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a revisar.

    Returns
    -------
    int
        Número de filas duplicadas.
    """
    return df.duplicated().sum()


def select_modeling_columns(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Selecciona las columnas finales utilizadas para modelado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    columns : list
        Lista de columnas a conservar.

    Returns
    -------
    pd.DataFrame
        Dataset con columnas seleccionadas.
    """
    return df[columns].copy()


def save_dataset(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Guarda un DataFrame como archivo CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a guardar.
    output_path : str
        Ruta de salida.
    """
    df.to_csv(output_path, index=False)