import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="❄️ Monitoramento Polar",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');

    .stApp { 
        background: linear-gradient(180deg, #E3F2FD 0%, #F5FBFF 100%);
        font-family: "Inter", sans-serif;
    }

    .card {
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        color: #01579B;
        transition: transform 0.2s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15);
    }

    .metric-number { 
        font-size: 36px; 
        font-weight: 800; 
        color: #0288D1;
        text-shadow: 0px 0px 10px rgba(2, 136, 209, 0.2);
    }

    .metric-label {
        font-size: 14px;
        color: #546E7A;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    h1, h2, h3, h4, h5 {
        color: #01579B !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #F1F8FF;
        border-right: 1px solid #D1E9FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DB_CONFIG = {
    "host": "dataiesb.iesbtech.com.br",
    "port": "5432",
    "database": "2212120006_Marco",
    "user": "2212120006_Marco",
    "password": "2212120006_Marco"
}

@st.cache_data(ttl=1800)
def load_data():
    try:
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
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

ICE_COLOR_SCALE = ["#00BCD4", "#0288D1", "#01579B", "#81D4FA", "#4DD0E1"]
ICE_LINE_COLOR = "#0288D1"
ICE_LINE_COLOR_2 = "#00BCD4"

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2324/2324147.png", width=50)
st.sidebar.title("Controle Geral")
page = st.sidebar.radio(
    "Navegação:",
    ["❄️ Dashboard Geral", "🧊 Comparação de Intervalos"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Alertas")
th_temp = st.sidebar.number_input("Temp. Máxima (°C)", value=30.0)
th_ilum = st.sidebar.number_input("Ilum. Máxima (lux)", value=300.0)

if page == "❄️ Dashboard Geral":

    st.markdown('<h1 style="margin-bottom: 20px;">❄️ Dashboard Ambiental <span style="font-size:20px; color:#4FC3F7">| Monitoramento em Tempo Real</span></h1>', unsafe_allow_html=True)

    st.sidebar.header("📅 Filtro Temporal")
    min_day, max_day = df["dia"].min(), df["dia"].max()
    date_start = st.sidebar.date_input("Início", min_day)
    date_end = st.sidebar.date_input("Fim", max_day)

    if date_start > date_end:
        st.error("Data inicial deve ser menor ou igual à data final.")
        st.stop()

    dfp = df[(df["dia"] >= date_start) & (df["dia"] <= date_end)]
    if dfp.empty:
        st.warning("🥶 Sem dados neste período congelado.")
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

    def card_metric(label, value, suffix=""):
        return f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-number">{value}{suffix}</div>
        </div>
        """

    st.subheader("🌡️ Temperatura (°C)")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(card_metric("Média", f"{agg['temp_mean'].mean():.1f}"), unsafe_allow_html=True)
    c2.markdown(card_metric("Mediana", f"{agg['temp_median'].median():.1f}"), unsafe_allow_html=True)
    c3.markdown(card_metric("Mínima", f"{agg['temp_min'].min():.1f}"), unsafe_allow_html=True)
    c4.markdown(card_metric("Máxima", f"{agg['temp_max'].max():.1f}"), unsafe_allow_html=True)
    c5.markdown(card_metric("Desvio Padrão", f"{agg['temp_std'].mean():.2f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("💡 Iluminação (Lux)")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(card_metric("Média", f"{agg['ilum_mean'].mean():.0f}"), unsafe_allow_html=True)
    c2.markdown(card_metric("Mediana", f"{agg['ilum_median'].median():.0f}"), unsafe_allow_html=True)
    c3.markdown(card_metric("Mínima", f"{agg['ilum_min'].min():.0f}"), unsafe_allow_html=True)
    c4.markdown(card_metric("Máxima", f"{agg['ilum_max'].max():.0f}"), unsafe_allow_html=True)
    c5.markdown(card_metric("Desvio Padrão", f"{agg['ilum_std'].mean():.2f}"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Temperatura (Evolução)", 
        "🔦 Iluminação (Evolução)", 
        "🔗 Correlação", 
        "🔮 Previsão Futura"
    ])

    def update_ice_layout(fig):
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#01579B"),
            hovermode="x unified"
        )
        return fig

    with tab1:
        st.subheader("Evolução da Temperatura")
        fig = px.line(agg, x="dia", y="temp_mean", markers=True)
        fig.update_traces(line_color=ICE_LINE_COLOR, line_width=3, marker_size=8)
        fig = update_ice_layout(fig)
        fig.update_traces(fill='tozeroy', fillcolor='rgba(2, 136, 209, 0.1)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Evolução da Iluminação")
        fig2 = px.line(agg, x="dia", y="ilum_mean", markers=True)
        fig2.update_traces(line_color=ICE_LINE_COLOR_2, line_width=3, marker_size=8)
        fig2 = update_ice_layout(fig2)
        fig2.update_traces(fill='tozeroy', fillcolor='rgba(0, 188, 212, 0.1)')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Correlação: Calor vs Luz")
        fig_scatter = px.scatter(
            agg,
            x="ilum_mean",
            y="temp_mean",
            trendline="ols",
            labels={"ilum_mean": "Iluminação (lux)", "temp_mean": "Temperatura (°C)"},
            color_discrete_sequence=[ICE_LINE_COLOR]
        )
        fig_scatter = update_ice_layout(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.subheader("Previsão Linear (Tendência)")
        horizon = st.slider("Dias para prever", 1, 60, 7)
        y = agg["temp_mean"].values
        X = np.arange(len(y)).reshape(-1,1)
        model = LinearRegression()
        model.fit(X, y)

        future_idx = np.arange(len(y), len(y)+horizon).reshape(-1,1)
        preds = model.predict(future_idx)
        future_dates = pd.date_range(agg["dia"].max() + timedelta(days=1), periods=horizon)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=agg["dia"], y=y, mode="lines+markers", name="Histórico", line=dict(color=ICE_LINE_COLOR)))
        fig4.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name="Previsão", line=dict(color=ICE_LINE_COLOR_2, dash='dot')))
        fig4 = update_ice_layout(fig4)
        st.plotly_chart(fig4, use_container_width=True)

        forecast_df = pd.DataFrame({"Data Prevista": future_dates, "Temp. Esperada (°C)": preds})
        st.dataframe(forecast_df.style.format({"Temp. Esperada (°C)": "{:.2f}"}), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔔 Alertas Automáticos")

    col_alert1, col_alert2 = st.columns(2)
    
    temp_alerts = agg[agg["temp_mean"] > th_temp]
    ilum_alerts = agg[agg["ilum_mean"] > th_ilum]

    if not temp_alerts.empty:
        col_alert1.warning(f"🔥 **{len(temp_alerts)} dias** com temperatura acima de {th_temp}°C")
        col_alert1.dataframe(temp_alerts[["dia", "temp_mean"]], hide_index=True)
    else:
        col_alert1.success("❄️ Temperatura dentro dos limites seguros.")

    if not ilum_alerts.empty:
        col_alert2.warning(f"☀️ **{len(ilum_alerts)} dias** com iluminação acima de {th_ilum} lux")
        col_alert2.dataframe(ilum_alerts[["dia", "ilum_mean"]], hide_index=True)
    else:
        col_alert2.success("🌑 Iluminação dentro dos limites seguros.")

elif page == "🧊 Comparação de Intervalos":

    st.title("🧊 Comparação de Períodos")

    col1, col2 = st.columns(2)
    with col1:
        st.info("**Período A (Base)**")
        A_start = st.date_input("Início A", df["dia"].min(), key="a1")
        A_end = st.date_input("Fim A", df["dia"].max(), key="a2")
    with col2:
        st.info("**Período B (Comparativo)**")
        B_start = st.date_input("Início B", df["dia"].min(), key="b1")
        B_end = st.date_input("Fim B", df["dia"].max(), key="b2")

    if A_start > A_end or B_start > B_end:
        st.error("Datas inválidas nos seletores.")
        st.stop()

    def interval_stats(start, end):
        d = df[(df["dia"] >= start) & (df["dia"] <= end)]
        if d.empty: return None
        agg = d.groupby("dia").agg(temp_mean=("temp","mean"), ilum_mean=("iluminacao","mean")).reset_index()
        return agg

    A = interval_stats(A_start, A_end)
    B = interval_stats(B_start, B_end)

    if A is None or B is None:
        st.error("Falta de dados em um dos intervalos para comparação.")
        st.stop()

    st.subheader("🌡️ Comparativo: Temperatura")
    figA = go.Figure()
    figA.add_trace(go.Scatter(x=A["dia"], y=A["temp_mean"], name="Período A", line=dict(color=ICE_LINE_COLOR)))
    figA.add_trace(go.Scatter(x=B["dia"], y=B["temp_mean"], name="Período B", line=dict(color=ICE_LINE_COLOR_2)))
    figA.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(figA, use_container_width=True)

    st.subheader("💡 Comparativo: Iluminação")
    figB = go.Figure()
    figB.add_trace(go.Scatter(x=A["dia"], y=A["ilum_mean"], name="Período A", line=dict(color=ICE_LINE_COLOR)))
    figB.add_trace(go.Scatter(x=B["dia"], y=B["ilum_mean"], name="Período B", line=dict(color=ICE_LINE_COLOR_2)))
    figB.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(figB, use_container_width=True)
    
    st.markdown("### 📊 Dados Brutos")
    format_dict = {
        "temp_mean": "{:.2f}",
        "ilum_mean": "{:.2f}"
    }
    c_a, c_b = st.columns(2)
    c_a.write("Dados do Período A")
    c_a.dataframe(A.style.format(format_dict), use_container_width=True)
    c_b.write("Dados do Período B")
    c_b.dataframe(B.style.format(format_dict), use_container_width=True)