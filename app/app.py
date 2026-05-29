"""
app.py

Aplicación Streamlit para el proyecto:
Modelo predictivo para anticipar incumplimientos de SLA en HR Operations.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to Python path so Streamlit can import modules from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_processed_data
from src.metrics import calculate_kpis
from src.visualizations import (
    plot_target_distribution,
    plot_breach_by_priority,
    plot_breach_by_channel,
    plot_model_metrics,
    plot_feature_importance
)
from src.prediction import load_model, predict_sla_risk


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tickets_hr_sla_model_ready.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_random_forest_sla.pkl"


st.set_page_config(
    page_title="SLA Breach Prediction | HR Operations",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Carga de datos y modelo
# -----------------------------
df = load_processed_data(DATA_PATH)
model = load_model(MODEL_PATH)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controles")

st.sidebar.markdown("### Filtros del dataset")

priority_filter = st.sidebar.multiselect(
    "Filtrar por prioridad",
    options=sorted(df["Ticket Priority"].unique()),
    default=sorted(df["Ticket Priority"].unique())
)

channel_filter = st.sidebar.multiselect(
    "Filtrar por canal",
    options=sorted(df["Ticket Channel"].unique()),
    default=sorted(df["Ticket Channel"].unique())
)

ticket_type_filter = st.sidebar.multiselect(
    "Filtrar por tipo de ticket",
    options=sorted(df["Ticket Type"].unique()),
    default=sorted(df["Ticket Type"].unique())
)

df_filtered = df[
    (df["Ticket Priority"].isin(priority_filter)) &
    (df["Ticket Channel"].isin(channel_filter)) &
    (df["Ticket Type"].isin(ticket_type_filter))
].copy()


# -----------------------------
# Header
# -----------------------------
st.title("Modelo predictivo para anticipar incumplimientos de SLA en HR Operations")
st.markdown(
    """
    Esta aplicación muestra un prototipo interactivo basado en Machine Learning
    para anticipar tickets con riesgo de incumplir su SLA.
    
    El objetivo es apoyar la priorización operativa y pasar de una gestión reactiva
    a una gestión preventiva basada en datos.
    """
)


# -----------------------------
# Tabs principales
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen",
    "Datos",
    "Análisis",
    "Modelo",
    "Simulador",
    "Conclusiones"
])


# -----------------------------
# TAB 1: Resumen
# -----------------------------
with tab1:
    st.header("Resumen ejecutivo")

    kpis = calculate_kpis(df_filtered)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tickets analizados", f"{kpis['total_tickets']:,}")
    col2.metric("Tasa incumplimiento SLA", f"{kpis['sla_breach_rate']}%")
    col3.metric("Tasa cumplimiento SLA", f"{kpis['sla_compliance_rate']}%")
    col4.metric("Tickets críticos", f"{kpis['critical_rate']}%")

    st.markdown("---")

    st.subheader("Problema de negocio")
    st.write(
        """
        Los equipos de HR Operations gestionan tickets con diferentes niveles de prioridad.
        Cuando un ticket incumple su SLA, puede generar retrasos, escalaciones y pérdida de eficiencia operativa.
        
        Este proyecto utiliza Machine Learning para anticipar qué tickets tienen mayor riesgo de incumplimiento.
        """
    )

    st.subheader("Resultado del modelo")
    st.success(
        """
        El modelo final seleccionado fue Random Forest optimizado.
        Alcanzó un recall de 0.8019 para la clase 1, lo que significa que detecta aproximadamente
        8 de cada 10 tickets que realmente incumplen SLA.
        """
    )


# -----------------------------
# TAB 2: Datos
# -----------------------------
with tab2:
    st.header("Vista general del dataset")

    st.write("Dataset filtrado según los controles de la barra lateral.")

    st.dataframe(df_filtered, use_container_width=True)

    st.markdown("### Dimensiones del dataset filtrado")
    st.write(f"Filas: {df_filtered.shape[0]}")
    st.write(f"Columnas: {df_filtered.shape[1]}")

    with st.expander("Ver columnas del dataset"):
        st.write(df_filtered.columns.tolist())


# -----------------------------
# TAB 3: Análisis
# -----------------------------
with tab3:
    st.header("Análisis exploratorio")

    if df_filtered.empty:
        st.warning("No hay datos disponibles con los filtros seleccionados.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_target_distribution(df_filtered),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                plot_breach_by_priority(df_filtered),
                use_container_width=True
            )

        st.plotly_chart(
            plot_breach_by_channel(df_filtered),
            use_container_width=True
        )

    st.info(
        """
        Estos gráficos permiten entender cómo se distribuyen los incumplimientos
        según prioridad y canal. No buscan decorar la app, sino apoyar decisiones
        de priorización operativa.
        """
    )


# -----------------------------
# TAB 4: Modelo
# -----------------------------
with tab4:
    st.header("Modelo predictivo y evaluación")

    st.subheader("Modelo final seleccionado")
    st.write("Random Forest optimizado")
    st.info(
    """
    El modelo fue seleccionado porque ofreció el mejor equilibrio entre detección de incumplimientos
    y control de falsas alarmas. En este proyecto se prioriza el recall de la clase 1,
    ya que el objetivo operativo es detectar la mayor cantidad posible de tickets que incumplen SLA.
    """
)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "0.7292")
    col2.metric("Precision clase 1", "0.6115")
    col3.metric("Recall clase 1", "0.8019")
    col4.metric("F1-score clase 1", "0.6939")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.plotly_chart(
            plot_model_metrics(),
            use_container_width=True
        )

    with col_b:
        st.plotly_chart(
            plot_feature_importance(),
            use_container_width=True
        )

    st.warning(
        """
        Nota metodológica: la prioridad del ticket aparece como variable muy importante
        porque el SLA esperado fue definido en función de la prioridad.
        Esto es coherente operativamente, pero debe interpretarse como una limitación del proyecto.
        """
    )


# -----------------------------
# TAB 5: Simulador
# -----------------------------
with tab5:
    st.header("Simulador de predicción de riesgo SLA")

    st.write(
        """
        Introduce las características de un ticket para estimar su riesgo de incumplimiento SLA.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        customer_age = st.slider(
            "Edad del cliente / empleado",
            min_value=int(df["Customer Age"].min()),
            max_value=int(df["Customer Age"].max()),
            value=int(df["Customer Age"].median())
        )

        product = st.selectbox(
            "Producto / servicio asociado",
            options=sorted(df["Product Purchased"].unique())
        )

        ticket_type = st.selectbox(
            "Tipo de ticket",
            options=sorted(df["Ticket Type"].unique())
        )

    with col2:
        ticket_subject = st.selectbox(
            "Asunto del ticket",
            options=sorted(df["Ticket Subject"].unique())
        )

        ticket_priority = st.selectbox(
            "Prioridad del ticket",
            options=sorted(df["Ticket Priority"].unique())
        )

        ticket_channel = st.selectbox(
            "Canal del ticket",
            options=sorted(df["Ticket Channel"].unique())
        )

    input_ticket = pd.DataFrame([{
        "Customer Age": customer_age,
        "Product Purchased": product,
        "Ticket Type": ticket_type,
        "Ticket Subject": ticket_subject,
        "Ticket Priority": ticket_priority,
        "Ticket Channel": ticket_channel
    }])

    st.markdown("### Ticket simulado")
    st.dataframe(input_ticket, use_container_width=True)

    if st.button("Predecir riesgo de incumplimiento SLA"):
        result = predict_sla_risk(model, input_ticket)

        st.markdown("---")
        st.subheader("Resultado de la predicción")

        col_pred1, col_pred2, col_pred3 = st.columns(3)

        prediction_label = "Incumple SLA" if result["prediction"] == 1 else "Cumple SLA"

        col_pred1.metric(
            "Predicción del modelo",
            prediction_label
        )

        col_pred2.metric(
            "Probabilidad de incumplimiento",
            f"{result['breach_probability']}%"
        )

        col_pred3.metric(
            "Nivel de riesgo operativo",
            result["risk_level"]
        )

        if result["risk_level"] == "Alto":
            st.error(f"🚨 Recomendación operativa: {result['recommendation']}")
            st.markdown(
                """
                **Acción sugerida:** asignar prioridad inmediata, revisar capacidad del equipo
                y considerar escalación preventiva.
                """
            )

        elif result["risk_level"] == "Medio":
            st.warning(f"⚠️ Recomendación operativa: {result['recommendation']}")
            st.markdown(
                """
                **Acción sugerida:** mantener seguimiento activo del ticket y revisar su evolución
                antes de que se acerque al umbral de SLA.
                """
            )

        else:
            st.success(f"✅ Recomendación operativa: {result['recommendation']}")
            st.markdown(
                """
                **Acción sugerida:** mantener el flujo operativo estándar sin escalación preventiva.
                """
            )

        with st.expander("Interpretación del resultado"):
            st.write(
                """
                El modelo estima el riesgo de incumplimiento SLA a partir de las variables del ticket.
                Este resultado no debe interpretarse como una decisión automática, sino como una señal
                de apoyo para priorización operativa.
                """
            )


# =========================
# PESTAÑA: CONCLUSIONES
# =========================
with tab6:
    st.markdown("## Conclusiones ejecutivas")
    st.caption(
        "Síntesis final del valor del modelo, sus limitaciones metodológicas y los próximos pasos recomendados."
    )

    # KPIs ejecutivos
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
    label="Modelo final seleccionado",
    value="Random Forest optimizado",
    delta="Modelo final"
)

    with col2:
        st.metric(
            label="Capacidad de detección",
            value="80.19%",
            delta="Recall clase 1"
        )

    with col3:
        st.metric(
            label="Objetivo de negocio",
            value="Priorización preventiva",
            delta="Riesgo de incumplimiento SLA"
        )

    st.markdown("---")

    # Conclusión principal
    st.markdown("### Conclusión principal")
    st.success(
        "El modelo Random Forest optimizado permite anticipar una parte relevante de los tickets con riesgo de incumplir SLA. "
        "Con un recall de 0.8019 para la clase de incumplimiento, detecta aproximadamente 8 de cada 10 tickets que realmente incumplen SLA."
    )

    # Dos columnas: valor de negocio y limitaciones
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### Valor para negocio")
        st.markdown(
            """
            - Permite pasar de una **gestión reactiva** a una **gestión preventiva**.
            - Ayuda a **priorizar tickets de mayor riesgo** antes de que se produzca el retraso.
            - Reduce el riesgo de **escalaciones, cuellos de botella y pérdida de eficiencia operativa**.
            - Sirve como base para una futura **herramienta de soporte a la decisión** en HR Operations.
            """
        )

        st.markdown("### Recomendación ejecutiva")
        st.info(
            "Usar este modelo como un prototipo de priorización operativa: no para reemplazar el criterio humano, "
            "sino para ayudar al equipo a identificar de forma temprana los casos con mayor riesgo."
        )

    with col_right:
        st.markdown("### Limitaciones metodológicas")
        st.warning(
            "El dataset utilizado fue adaptado desde tickets de soporte y todavía no representa directamente datos reales de HR Operations."
        )
        st.warning(
            "La variable objetivo `incumplio_sla` fue construida mediante una regla de negocio basada en prioridad. "
            "Por ello, la prioridad aparece como variable muy influyente y debe interpretarse con cautela."
        )

    st.markdown("### Próximos pasos priorizados")

    next_col1, next_col2 = st.columns(2)

    with next_col1:
        st.markdown(
            """
            **Prioridad alta**
            
            1. Incorporar datos reales de HR Operations.  
            2. Añadir variables operativas como carga de trabajo, agente asignado, región o reasignaciones.  
            3. Validar el modelo con nuevos periodos o nuevos equipos.
            """
        )

    with next_col2:
        st.markdown(
            """
            **Prioridad media**
            
            4. Monitorizar el rendimiento del modelo con datos nuevos.  
            5. Reentrenar el modelo de forma periódica.  
            6. Desplegar la app en Streamlit Community Cloud como activo de portfolio.
            """
        )

    st.markdown("---")
    st.caption("Proyecto de Machine Learning | HR Operations SLA Prediction | Gabriel Bohórquez")
