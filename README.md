# Modelo predictivo para anticipar incumplimientos de SLA en HR Operations  
## (Predictive Model to Anticipate SLA Breaches in HR Operations)

---

## 1. Resumen ejecutivo

Este proyecto desarrolla un modelo de Machine Learning para anticipar qué tickets operativos tienen mayor probabilidad de incumplir su SLA.

El objetivo no es únicamente predecir retrasos, sino ayudar a los equipos de HR Operations a priorizar casos críticos antes de que se produzca el incumplimiento.

El proyecto utiliza un dataset público de tickets de soporte como base operativa y lo adapta a un contexto de HR Operations, donde los procesos suelen gestionarse mediante tickets, prioridades, canales, tipos de caso y tiempos de resolución.

El modelo final seleccionado fue un **Random Forest optimizado**, que alcanzó un **recall de aproximadamente 80% para la clase de incumplimiento SLA**.

Esto significa que el modelo fue capaz de detectar cerca de 8 de cada 10 tickets que realmente incumplieron su SLA.

---

## 2. Problema de negocio

Los equipos de HR Operations gestionan un alto volumen de casos relacionados con procesos como onboarding, documentación, payroll, beneficios, cambios de datos, accesos internos y consultas de empleados.

Muchos de estos casos están sujetos a SLA, es decir, tiempos máximos esperados de resolución.

Cuando un caso incumple su SLA, puede generar:

- retrasos operativos,
- escalaciones,
- pérdida de confianza en el servicio,
- presión adicional sobre el equipo,
- peor experiencia del empleado,
- incumplimiento de KPIs internos.

El problema principal es que muchas veces los equipos detectan el riesgo demasiado tarde.

Por eso, la pregunta de negocio del proyecto es:

> ¿Podemos anticipar qué tickets tienen mayor probabilidad de incumplir su SLA antes de que ocurra?

---

## 3. Objetivo del proyecto

Construir un modelo predictivo capaz de clasificar tickets según su riesgo de incumplimiento SLA.

La variable objetivo del proyecto es:

| Variable | Valor | Significado |
|---|---:|---|
| `incumplio_sla` | 0 | El ticket cumplió el SLA |
| `incumplio_sla` | 1 | El ticket incumplió el SLA |

El objetivo operativo es pasar de una gestión reactiva a una gestión preventiva basada en riesgo.

---

## 4. Enfoque analítico

El proyecto sigue un flujo de trabajo orientado a negocio:

```text
Problema de negocio
↓
Entendimiento de datos
↓
Creación de variable objetivo
↓
Preparación del dataset final
↓
Entrenamiento de modelos
↓
Comparación de modelos
↓
Optimización
↓
Interpretación
↓
Recomendaciones de negocio


## Aplicación Streamlit

El proyecto incluye una aplicación interactiva desarrollada con Streamlit.

La app permite:

- Explorar el dataset final.
- Consultar KPIs operativos.
- Filtrar tickets por prioridad, canal y tipo.
- Visualizar patrones de incumplimiento SLA.
- Revisar métricas del modelo final.
- Simular la predicción de riesgo de incumplimiento SLA para un nuevo ticket.
- Obtener una recomendación operativa automática.
- Consultar conclusiones, limitaciones y próximos pasos.

### Ejecutar localmente

Desde la raíz del proyecto:

```bash
streamlit run app/app.py

La aplicación se abrirá en:

http://localhost:8501

Enlace publico: https://canva.link/mgiyjhi7zm3opox


