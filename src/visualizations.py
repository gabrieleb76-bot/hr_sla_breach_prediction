"""
visualizations.py

Funciones para crear visualizaciones de la app Streamlit.
"""

import pandas as pd
import plotly.express as px


def plot_target_distribution(df: pd.DataFrame):
    """
    Grafica la distribución de cumplimiento/incumplimiento SLA.
    """
    target_counts = (
        df["incumplio_sla"]
        .value_counts()
        .rename(index={0: "Cumplió SLA", 1: "Incumplió SLA"})
        .reset_index()
    )

    target_counts.columns = ["Resultado", "Cantidad"]

    fig = px.bar(
        target_counts,
        x="Resultado",
        y="Cantidad",
        text="Cantidad",
        title="Distribución de cumplimiento SLA"
    )

    fig.update_traces(
        texttemplate="%{text}",
        textposition="inside"
    )

    fig.update_layout(
        xaxis_title="Resultado",
        yaxis_title="Número de tickets",
        template="plotly_white"
    )

    return fig


def plot_breach_by_priority(df: pd.DataFrame):
    """
    Grafica la tasa de incumplimiento por prioridad.
    """
    grouped = (
        df.groupby("Ticket Priority")["incumplio_sla"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"incumplio_sla": "tasa_incumplimiento"})
    )

    grouped["etiqueta"] = grouped["tasa_incumplimiento"].apply(
        lambda value: f"{value:.2f}%"
    )

    fig = px.bar(
        grouped,
        x="Ticket Priority",
        y="tasa_incumplimiento",
        text="etiqueta",
        title="Tasa de incumplimiento por prioridad"
    )

    fig.update_traces(
        texttemplate="%{text}",
        textposition="inside"
    )

    fig.update_layout(
        xaxis_title="Prioridad",
        yaxis_title="% Incumplimiento",
        template="plotly_white"
    )

    return fig


def plot_breach_by_channel(df: pd.DataFrame):
    """
    Grafica la tasa de incumplimiento por canal.
    """
    grouped = (
        df.groupby("Ticket Channel")["incumplio_sla"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"incumplio_sla": "tasa_incumplimiento"})
    )

    grouped["etiqueta"] = grouped["tasa_incumplimiento"].apply(
        lambda value: f"{value:.2f}%"
    )

    fig = px.bar(
        grouped,
        x="Ticket Channel",
        y="tasa_incumplimiento",
        text="etiqueta",
        title="Tasa de incumplimiento por canal"
    )

    fig.update_traces(
        texttemplate="%{text}",
        textposition="inside"
    )

    fig.update_layout(
        xaxis_title="Canal",
        yaxis_title="% Incumplimiento",
        template="plotly_white"
    )

    return fig


def plot_model_metrics():
    """
    Grafica métricas finales del modelo optimizado.
    """
    metrics_df = pd.DataFrame({
        "Métrica": [
            "Accuracy",
            "Precision clase 1",
            "Recall clase 1",
            "F1-score clase 1"
        ],
        "Valor": [0.7292, 0.6115, 0.8019, 0.6939]
    })

    metrics_df["etiqueta"] = metrics_df["Valor"].apply(
        lambda value: f"{value:.4f}"
    )

    fig = px.bar(
        metrics_df,
        x="Métrica",
        y="Valor",
        text="etiqueta",
        title="Métricas del modelo final — Random Forest optimizado"
    )

    fig.update_traces(
        texttemplate="%{text}",
        textposition="inside"
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        template="plotly_white"
    )

    return fig


def plot_feature_importance():
    """
    Grafica importancia agregada de variables.
    """
    importance_df = pd.DataFrame({
        "Variable": [
            "Ticket Priority",
            "Product Purchased",
            "Ticket Subject",
            "Customer Age",
            "Ticket Type",
            "Ticket Channel"
        ],
        "Importancia": [0.7734, 0.1035, 0.0491, 0.0283, 0.0237, 0.0219]
    })

    importance_df["etiqueta"] = importance_df["Importancia"].apply(
        lambda value: f"{value:.4f}"
    )

    fig = px.bar(
        importance_df.sort_values("Importancia"),
        x="Importancia",
        y="Variable",
        orientation="h",
        text="etiqueta",
        title="Importancia agregada de variables"
    )

    fig.update_traces(
        texttemplate="%{text}",
        textposition="inside"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Importancia",
        yaxis_title="Variable"
    )

    return fig