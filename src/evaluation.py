"""
evaluation.py

Funciones reutilizables para evaluar modelos de clasificación
en el proyecto de predicción de incumplimientos de SLA.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    model_name: str
) -> dict:
    """
    Calcula métricas principales para clasificación binaria.

    Parameters
    ----------
    y_true
        Valores reales.
    y_pred
        Predicciones del modelo.
    model_name : str
        Nombre del modelo.

    Returns
    -------
    dict
        Diccionario con métricas calculadas.
    """
    return {
        "modelo": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_clase_1": precision_score(y_true, y_pred),
        "recall_clase_1": recall_score(y_true, y_pred),
        "f1_clase_1": f1_score(y_true, y_pred)
    }


def evaluate_multiple_models(
    models: dict,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
) -> pd.DataFrame:
    """
    Entrena y evalúa múltiples modelos usando el mismo preprocesamiento.

    Parameters
    ----------
    models : dict
        Diccionario con nombre y modelo.
    preprocessor
        Preprocesador de Scikit-learn.
    X_train
        Variables de entrenamiento.
    X_test
        Variables de test.
    y_train
        Target de entrenamiento.
    y_test
        Target de test.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa de métricas.
    """
    from sklearn.pipeline import Pipeline

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocesamiento", preprocessor),
                ("modelo", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name
        )

        results.append(metrics)

    return pd.DataFrame(results)


def print_classification_summary(
    y_true,
    y_pred
) -> None:
    """
    Imprime el classification report del modelo.

    Parameters
    ----------
    y_true
        Valores reales.
    y_pred
        Predicciones del modelo.
    """
    print(classification_report(y_true, y_pred))


def get_confusion_matrix(
    y_true,
    y_pred
):
    """
    Devuelve la matriz de confusión.

    Parameters
    ----------
    y_true
        Valores reales.
    y_pred
        Predicciones del modelo.

    Returns
    -------
    np.ndarray
        Matriz de confusión.
    """
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list = None,
    title: str = "Matriz de confusión"
) -> None:
    """
    Grafica una matriz de confusión.

    Parameters
    ----------
    y_true
        Valores reales.
    y_pred
        Predicciones del modelo.
    labels : list
        Etiquetas de las clases.
    title : str
        Título del gráfico.
    """
    if labels is None:
        labels = ["Cumplió SLA", "Incumplió SLA"]

    matrix = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    display.plot(ax=ax, values_format="d")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def sort_results_by_metric(
    results_df: pd.DataFrame,
    metric: str = "recall_clase_1",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Ordena la tabla de resultados por una métrica específica.

    Parameters
    ----------
    results_df : pd.DataFrame
        Tabla de resultados.
    metric : str
        Métrica utilizada para ordenar.
    ascending : bool
        Orden ascendente o descendente.

    Returns
    -------
    pd.DataFrame
        Tabla ordenada.
    """
    return results_df.sort_values(by=metric, ascending=ascending)