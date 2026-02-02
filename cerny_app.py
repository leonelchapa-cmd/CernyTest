import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp

st.set_page_config(page_title="Modelo Černý - 8 Activos", layout="wide")

st.title("🚀 Modelo de Černý (2019) con Matriz de Correlación")
st.markdown("""
Este modelo calcula la **Matriz de Covarianza** automáticamente y determina cuánto invertir en activos con riesgo y cuánto mantener en **Cash**.
""")

# --- SIDEBAR: PARÁMETROS ---
st.sidebar.header("⚙️ Configuración Global")
rf = st.sidebar.number_input("Tasa Libre de Riesgo (rf)", value=0.01, format="%.4f", step=0.001)
gamma_val = st.sidebar.slider("Gamma (Aversión al Riesgo)", 0.1, 30.0, 15.0, 0.1)
alpha = st.sidebar.slider("Alpha (Proporción Máxima)", 0.0, 1.0, 1.0, 0.01)

activos = ['AAPL', 'IAU', 'MBB', 'SLV', 'SPY', 'UNG', 'VCIT', 'WMT']

# --- SIDEBAR: RETORNOS ---
st.sidebar.subheader("📈 Retornos Esperados")
mu_vals = {
    'AAPL': 0.2513, 'IAU': 0.0932, 'MBB': 0.0197, 'SLV': 0.1025,
    'SPY': 0.1389, 'UNG': -0.1289, 'VCIT': 0.0408, 'WMT': 0.1668
}
mu_input = [st.sidebar.number_input(f"Retorno {a}", value=mu_vals[a], format="%.4f") for a in activos]
mu = np.array(mu_input)

# --- SIDEBAR: VOLATILIDADES ---
st.sidebar.subheader("📉 Volatilidades")
sigma_vals = {
    'AAPL': 0.2670, 'IAU': 0.1611, 'MBB': 0.0439, 'SLV': 0.3118,
    'SPY': 0.1404, 'UNG': 0.4861, 'VCIT': 0.0611, 'WMT': 0.1789
}
sigma_input = [st.sidebar.number_input(f"Volatilidad {a}", value=sigma_vals[a], format="%.4f") for a in activos]
sigma = np.array(sigma_input)

# --- MATRIZ DE CORRELACIÓN ---
corr_matrix = np.array([
    [1, 0.120146116, 0.251909112, 0.225280126, 0.594099, 0.036377312, 0.394173541, 0.249878949],
    [0.120146116, 1, 0.280243412, 0.771645042, 0.098369012, -0.053217378, 0.374305838, 0.110369145],
    [0.251909112, 0.280243412, 1, 0.193700541, 0.308116899, -0.008117399, 0.7971215, 0.125847523],
    [0.225280126, 0.771645042, 0.193700541, 1, 0.281330563, 0.018332692, 0.365295774, 0.079684812],
    [0.594099, 0.098369012, 0.308116899, 0.281330563, 1, 0.082000895, 0.509621512, 0.377308192],
    [0.036377312, -0.053217378, -0.008117399, 0.018332692, 0.082000895, 1, -0.026349723, 0.179822569],
    [0.394173541, 0.374305838, 0.7971215, 0.365295774, 0.509621512, -0.026349723, 1, 0.170894586],
    [0.249878949, 0.110369145, 0.125847523, 0.079684812, 0.377308192, 0.179822569, 0.170894586, 1]
])

# --- CÁLCULO DE COVARIANZA ---
D = np.diag(sigma)
Sigma = D @ corr_matrix @ D

# --- OPTIMIZACIÓN ---
excess_risky = mu - rf
pi_risky = cp.Variable(len(mu), nonneg=True)
objective = cp.Maximize(excess_risky @ pi_risky - (gamma_val / 2) * cp.quad_form(pi_risky, Sigma))
constraints = [cp.sum(pi_risky) <= alpha]
problem = cp.Problem(objective, constraints)
problem.solve()

# Resultados
weights = pi_risky.value
sum_risky = weights.sum()
cash_weight = alpha - sum_risky  # Lo que sobra se va a Cash
utilidad = excess_risky @ weights - (gamma_val / 2) * (weights.T @ Sigma @ weights)

# --- VISUALIZACIÓN ---
col1, col2, col3 = st.columns([3, 3, 2])

with col1:
    st.write("### ⚖️ Pesos Óptimos (π_risky)")
    # Crear DataFrame incluyendo Cash
    activos_con_cash = activos + ['CASH (Libre de Riesgo)']
    pesos_con_cash = np.append(weights, cash_weight)
    
    weights_df = pd.DataFrame({'Activo': activos_con_cash, 'Peso': pesos_con_cash})
    st.dataframe(weights_df.style.format({"Peso": "{:.2%}"}))
    
    st.metric("Inversión en Riesgo", f"{sum_risky:.2%}")
    st.metric("Inversión en Cash", f"{cash_weight:.2%}")

with col2:
    st.write("### 📊 Pesos Relativos (π_hoy)")
    # Normalizar pesos respecto a alpha
    pi_hoy = weights / alpha if alpha > 0 else np.zeros_like(weights)
    cash_hoy = cash_weight / alpha if alpha > 0 else 0
    
    pi_hoy_df = pd.DataFrame({
        'Activo': activos_con_cash, 
        'Peso Hoy': np.append(pi_hoy, cash_hoy)
    })
    st.dataframe(pi_hoy_df.style.format({"Peso Hoy": "{:.2%}"}))
    st.metric("Suma Total (Alpha)", f"{(pi_hoy.sum() + cash_hoy):.0%}")

with col3:
    st.write("### 🎯 Función de Utilidad")
    st.metric("Utilidad Máxima", f"{utilidad:.6f}")
    st.info(f"El modelo sugiere dejar un **{cash_weight:.2%}** en efectivo para optimizar el riesgo.")

st.success("✅ Modelo actualizado. El efectivo (Cash) se calcula automáticamente restando la inversión en riesgo de Alpha.")