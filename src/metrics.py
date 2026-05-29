"""
metrics.py

Funciones para calcular KPIs operativos del proyecto SLA.
"""

import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Calcula KPIs principales del dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset filtrado o completo.

    Returns
    -------
    dict
        Diccionario de KPIs.
    """
    total_tickets = len(df)

    if total_tickets == 0:
        return {
            "total_tickets": 0,
            "sla_breach_rate": 0,
            "sla_compliance_rate": 0,
            "critical_rate": 0
        }

    sla_breach_rate = df["incumplio_sla"].mean() * 100
    sla_compliance_rate = 100 - sla_breach_rate

    if "Ticket Priority" in df.columns:
        critical_rate = (df["Ticket Priority"].eq("Critical").mean()) * 100
    else:
        critical_rate = 0

    return {
        "total_tickets": total_tickets,
        "sla_breach_rate": round(sla_breach_rate, 2),
        "sla_compliance_rate": round(sla_compliance_rate, 2),
        "critical_rate": round(critical_rate, 2)
    }
    