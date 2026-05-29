"""
modeling.py

Funciones reutilizables para preparar datos, crear pipelines
y entrenar modelos de Machine Learning.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "incumplio_sla"
):
    """
    Separa variables predictoras y variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de modelado.
    target_column : str
        Nombre de la variable objetivo.

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Variable objetivo.
    """
    X = df.drop(columns=target_column)
    y = df[target_column]

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Divide los datos en train y test manteniendo la proporción de clases.

    Parameters
    ----------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Variable objetivo.
    test_size : float
        Proporción del conjunto de test.
    random_state : int
        Semilla de reproducibilidad.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def create_preprocessor(
    numeric_features: list,
    categorical_features: list
) -> ColumnTransformer:
    """
    Crea un preprocesador para variables numéricas y categóricas.

    Parameters
    ----------
    numeric_features : list
        Columnas numéricas.
    categorical_features : list
        Columnas categóricas.

    Returns
    -------
    ColumnTransformer
        Preprocesador de Scikit-learn.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), numeric_features),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor


def create_model_pipeline(
    preprocessor: ColumnTransformer,
    model
) -> Pipeline:
    """
    Crea un pipeline con preprocesamiento y modelo.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Transformador de variables.
    model
        Modelo de Scikit-learn.

    Returns
    -------
    Pipeline
        Pipeline completo.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocessor),
            ("modelo", model)
        ]
    )

    return pipeline


def get_classification_models(random_state: int = 42) -> dict:
    """
    Devuelve un diccionario con modelos de clasificación supervisada.

    Parameters
    ----------
    random_state : int
        Semilla de reproducibilidad.

    Returns
    -------
    dict
        Diccionario de modelos.
    """
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Regresión Logística": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        )
    }

    return models


def create_final_random_forest_model(random_state: int = 42) -> RandomForestClassifier:
    """
    Crea el modelo final Random Forest optimizado.

    Parameters
    ----------
    random_state : int
        Semilla de reproducibilidad.

    Returns
    -------
    RandomForestClassifier
        Modelo Random Forest con hiperparámetros optimizados.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight="balanced",
        random_state=random_state
    )