import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

st.set_page_config(
    page_title="Dashboard Ambiental",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { font-family: "Inter", sans-serif; }
    .card {
        background: linear-gradient(135deg, #ffffffcc, #f0f6ffcc);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    .metric-number { font-size: 30px; font-weight: 700; }
    .header {
        display:flex; align-items:center; gap:12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DB_CONFIG = {
    "host": "DB_HOST",
    "port": "DB_´PRT",
    "database": "DB_DATABASE",
    "user": "DB_USER",
    "password": "DB_PASSWORD"
}

@st.cache_data(ttl=1800)
def load_data():
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(
            "SELECT temp, iluminacao, data FROM estudos.projeto_integrador2 ORDER BY data ASC",
            conn
        )
    df["data"] = pd.to_datetime(df["data"])
    df["dia"] = df["data"].dt.date
    df["temp"] = df["temp"].astype(float)
    df["iluminacao"] = df["iluminacao"].astype(float)
    return df

df = load_data()

st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Escolha a página:",
    ["Dashboard Geral", "Comparação de Intervalos"]
)

st.sidebar.markdown("### Alertas automáticos")
th_temp = st.sidebar.number_input("Limiar da temperatura (°C)", value=30.0)
th_ilum = st.sidebar.number_input("Limiar da iluminação (lux)", value=300.0)

if page == "Dashboard Geral":

    st.markdown(
        '<div class="header"><img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="40">'
        '<h1 style="margin:0;">Dashboard Ambiental</h1></div>',
        unsafe_allow_html=True
    )

    st.sidebar.header("Filtros")
    min_day, max_day = df["dia"].min(), df["dia"].max()
    date_start = st.sidebar.date_input("Data inicial", min_day)
    date_end = st.sidebar.date_input("Data final", max_day)

    if date_start > date_end:
        st.error("Data inicial deve ser menor ou igual à data final.")
        st.stop()

    dfp = df[(df["dia"] >= date_start) & (df["dia"] <= date_end)]
    if dfp.empty:
        st.warning("Sem dados no intervalo selecionado.")
        st.stop()

    agg = dfp.groupby("dia").agg(
        temp_mean=("temp","mean"),
        temp_median=("temp","median"),
        temp_min=("temp","min"),
        temp_max=("temp","max"),
        temp_std=("temp","std"),
        ilum_mean=("iluminacao","mean"),
        ilum_median=("iluminacao","median"),
        ilum_min=("iluminacao","min"),
        ilum_max=("iluminacao","max"),
        ilum_std=("iluminacao","std"),
    ).reset_index()

    st.subheader("Estatísticas principais (Temperatura)")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f'<div class="card"><div class="metric-number">{agg["temp_mean"].mean():.2f}</div>Média</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="metric-number">{agg["temp_median"].median():.2f}</div>Mediana</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="metric-number">{agg["temp_min"].min():.2f}</div>Mínimo</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><div class="metric-number">{agg["temp_max"].max():.2f}</div>Máximo</div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="card"><div class="metric-number">{agg["temp_std"].mean():.2f}</div>Desvio Padrão</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Estatísticas principais (Iluminação)")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f'<div class="card"><div class="metric-number">{agg["ilum_mean"].mean():.2f}</div>Média</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="metric-number">{agg["ilum_median"].median():.2f}</div>Mediana</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="metric-number">{agg["ilum_min"].min():.2f}</div>Mínimo</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><div class="metric-number">{agg["ilum_max"].max():.2f}</div>Máximo</div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="card"><div class="metric-number">{agg["ilum_std"].mean():.2f}</div>Desvio Padrão</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Temperatura (Média Diária)", 
        "Iluminação (Média Diária)", 
        "Correlação", 
        "Previsão"
    ])

    with tab1:
        st.subheader("Temperatura — média diária")
        fig = px.line(agg, x="dia", y="temp_mean", markers=True)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Iluminação — média diária")
        fig2 = px.line(agg, x="dia", y="ilum_mean", markers=True)
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Correlação entre temperatura e iluminação")
        fig_scatter = px.scatter(
            agg,
            x="ilum_mean",
            y="temp_mean",
            trendline="ols",
            labels={"ilum_mean": "Iluminação (lux)", "temp_mean": "Temperatura (°C)"},
            title="Relação entre temperatura e iluminação"
        )
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.subheader("Previsão (Regressão Linear)")
        horizon = st.slider("Dias para prever", 1, 60, 7)
        y = agg["temp_mean"].values
        X = np.arange(len(y)).reshape(-1,1)
        model = LinearRegression()
        model.fit(X, y)

        future_idx = np.arange(len(y), len(y)+horizon).reshape(-1,1)
        preds = model.predict(future_idx)
        future_dates = pd.date_range(agg["dia"].max() + timedelta(days=1), periods=horizon)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=agg["dia"], y=y, mode="lines+markers", name="Histórico"))
        fig4.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name="Previsão"))
        fig4.update_layout(template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)

        forecast_df = pd.DataFrame({
            "data": future_dates,
            "previsao_temp": preds
        })

        st.dataframe(
            forecast_df.style.format({
                "previsao_temp": "{:.2f}"
            })
        )
    
    st.markdown("---")
    
    st.subheader("Alertas automáticos")

    alerts = []
    temp_alerts = agg[agg["temp_mean"] > th_temp]
    ilum_alerts = agg[agg["ilum_mean"] > th_ilum]

    if not temp_alerts.empty:
        alerts.append("Temperatura acima do limite")
        st.warning("Temperatura acima do limiar")
        st.dataframe(temp_alerts)

    if not ilum_alerts.empty:
        alerts.append("Iluminação acima do limite")
        st.warning("Iluminação acima do limiar")
        st.dataframe(ilum_alerts)

    if not alerts:
        st.success("Nenhum alerta encontrado.")

elif page == "Comparação de Intervalos":

    st.title("Comparação entre dois intervalos de datas")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Intervalo A")
        A_start = st.date_input("A — data início", df["dia"].min(), key="a1")
        A_end = st.date_input("A — data fim", df["dia"].max(), key="a2")

    with col2:
        st.subheader("Intervalo B")
        B_start = st.date_input("B — data início", df["dia"].min(), key="b1")
        B_end = st.date_input("B — data fim", df["dia"].max(), key="b2")

    if A_start > A_end or B_start > B_end:
        st.error("Datas inválidas.")
        st.stop()

    def interval_stats(start, end):
        d = df[(df["dia"] >= start) & (df["dia"] <= end)]
        if d.empty:
            return None
        agg = d.groupby("dia").agg(temp_mean=("temp","mean"), ilum_mean=("iluminacao","mean")).reset_index()
        return agg

    A = interval_stats(A_start, A_end)
    B = interval_stats(B_start, B_end)

    if A is None or B is None:
        st.error("Um dos intervalos não contém dados.")
        st.stop()

    st.subheader("Temperatura — comparação")
    figA = go.Figure()
    figA.add_trace(go.Scatter(x=A["dia"], y=A["temp_mean"], name="A"))
    figA.add_trace(go.Scatter(x=B["dia"], y=B["temp_mean"], name="B"))
    figA.update_layout(template="plotly_white")
    st.plotly_chart(figA, use_container_width=True)

    st.subheader("Iluminação — comparação")
    figB = go.Figure()
    figB.add_trace(go.Scatter(x=A["dia"], y=A["ilum_mean"], name="A"))
    figB.add_trace(go.Scatter(x=B["dia"], y=B["ilum_mean"], name="B"))
    figB.update_layout(template="plotly_white")
    st.plotly_chart(figB, use_container_width=True)

    format_cols = {
        "temp_mean": "{:.2f}",
        "ilum_mean": "{:.2f}",
    }

    st.subheader("Resumo numérico")
    colA, colB = st.columns(2)
    colA.write("Intervalo A")
    colA.dataframe(A.style.format(format_cols))
    colB.write("Intervalo B")
    colB.dataframe(B.style.format(format_cols))
    

