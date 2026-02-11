import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# carregar dados
#df = pd.read_csv(r"C:\Users\gabri\Desktop\FIAP\Fase_5\Tech_Challenge\data\base_final.csv")  # salve seu df como csv antes
df = pd.read_parquet(r"C:\Users\gabri\Desktop\FIAP\Fase_5\Tech_Challenge\data\base.parquet")
cols = ["IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE"]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

st.set_page_config(layout="wide")

st.title("📊 Painel Preditivo Educacional")

# ----------------------------
# SIDEBAR
# ----------------------------

aluno = st.sidebar.selectbox("Selecionar aluno", df["RA"].unique())
df_aluno = df[df["RA"] == aluno]

# ----------------------------
# 1️⃣ Visão geral
# ----------------------------

st.header("Visão Geral do Programa")

media = df.groupby("ANO")[["IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE"]].mean().reset_index()

fig = px.line(media, x="ANO", y="INDE", markers=True, title="Evolução média do INDE")
fig.update_xaxes(type="category")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 2️⃣ Radar do aluno
# ----------------------------

st.header("Perfil do Aluno")

ultimo = df_aluno.sort_values("ANO").iloc[-1]

indicadores = ["IAN","IDA","IEG","IAA","IPS","IPP","IPV"]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=ultimo[indicadores].values,
    theta=indicadores,
    fill='toself',
    name="Aluno"
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,10])),
    showlegend=False
)

st.plotly_chart(fig_radar, use_container_width=True)

# ----------------------------
# 3️⃣ Alunos em risco
# ----------------------------

st.header("🚨 Alunos em Risco")

risco = df[df["prob_risco"] > 0.7]
st.dataframe(risco.sort_values("prob_risco", ascending=False))

# ----------------------------
# 4️⃣ Correlação
# ----------------------------

st.header("Correlação entre indicadores")

corr = df[["IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE"]].corr()

fig_corr = px.imshow(corr, text_auto=True)
st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------
# 5️⃣ Evolução individual
# ----------------------------

st.header("Evolução do aluno")

fig_ind = px.line(df_aluno, x="ANO", y="INDE", markers=True)
fig_ind.update_xaxes(type="category")
st.plotly_chart(fig_ind, use_container_width=True)

