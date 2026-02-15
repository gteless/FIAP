import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Painel Preditivo Educacional", layout="wide")
st.title("📊 Painel Preditivo Educacional (Ranking de Risco)")

# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip()
    return df

def pick_score_col(df: pd.DataFrame):
    for c in ["prob", "prob_risco", "score", "PROB", "PROB_RISCO"]:
        if c in df.columns:
            return c
    return None

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols_exist = [c for c in cols if c in df.columns]
    if cols_exist:
        df[cols_exist] = df[cols_exist].apply(pd.to_numeric, errors="coerce")
    return df

def precision_at_k(df: pd.DataFrame, score_col: str, y_col: str, k: int) -> float:
    if df.empty or k <= 0 or score_col not in df.columns or y_col not in df.columns:
        return np.nan
    top = df.sort_values(score_col, ascending=False).head(k)
    y = pd.to_numeric(top[y_col], errors="coerce")
    if y.notna().sum() == 0:
        return np.nan
    return float(y.fillna(0).sum() / k)

def lift_at_k(df: pd.DataFrame, score_col: str, y_col: str, k: int) -> float:
    if df.empty or y_col not in df.columns:
        return np.nan
    y = pd.to_numeric(df[y_col], errors="coerce")
    base_rate = float(y.mean()) if y.notna().any() else np.nan
    p_at_k = precision_at_k(df, score_col, y_col, k)
    if np.isnan(base_rate) or base_rate == 0 or np.isnan(p_at_k):
        return np.nan
    return p_at_k / base_rate

def recalcular_risco_por_corte_validacao(df: pd.DataFrame, corte: float) -> pd.DataFrame:
    d = df.copy()
    if not {"RA", "ANO", "INDE"}.issubset(d.columns):
        return d

    d["RA"] = d["RA"].astype(str).str.strip()
    d["ANO"] = pd.to_numeric(d["ANO"], errors="coerce").astype("Int64")
    d["INDE"] = pd.to_numeric(d["INDE"], errors="coerce")

    d = d.sort_values(["RA", "ANO"])
    d["delta_INDE"] = d.groupby("RA")["INDE"].diff()

    d["risco_calc"] = (d["delta_INDE"] < corte).astype("Int64")
    d["risco_futuro_calc"] = d.groupby("RA")["risco_calc"].shift(-1).astype("Int64")
    return d

def recalcular_risco_por_corte_2024(df_2024: pd.DataFrame, df_valid: pd.DataFrame, corte: float) -> pd.DataFrame:
    d24 = df_2024.copy()
    if not {"RA", "INDE"}.issubset(d24.columns):
        return d24
    if not {"RA", "ANO", "INDE"}.issubset(df_valid.columns):
        return d24

    d24["RA"] = d24["RA"].astype(str).str.strip()
    d24["INDE"] = pd.to_numeric(d24["INDE"], errors="coerce")

    dv = df_valid.copy()
    dv["RA"] = dv["RA"].astype(str).str.strip()
    dv["ANO"] = pd.to_numeric(dv["ANO"], errors="coerce")
    dv["INDE"] = pd.to_numeric(dv["INDE"], errors="coerce")

    inde_prev = (
        dv[dv["ANO"] < 2024]
        .dropna(subset=["RA", "ANO", "INDE"])
        .sort_values(["RA", "ANO"])
        .groupby("RA", as_index=False)
        .tail(1)[["RA", "INDE"]]
        .rename(columns={"INDE": "INDE_prev"})
    )

    d24 = d24.merge(inde_prev, on="RA", how="left")
    d24["delta_INDE"] = d24["INDE"] - d24["INDE_prev"]
    d24["risco_calc"] = (d24["delta_INDE"] < corte).astype("Int64")
    return d24

def make_curve_df(dfv: pd.DataFrame, score_col: str, y_col: str, ks: list[int]) -> pd.DataFrame:
    if dfv.empty or score_col not in dfv.columns or y_col not in dfv.columns:
        return pd.DataFrame(columns=["K", "Precision@K", "Lift@K"])
    ks_eff = [kk for kk in ks if kk <= len(dfv)]
    rows = []
    for kk in ks_eff:
        rows.append({
            "K": kk,
            "Precision@K": precision_at_k(dfv, score_col, y_col, kk),
            "Lift@K": lift_at_k(dfv, score_col, y_col, kk),
        })
    return pd.DataFrame(rows, columns=["K", "Precision@K", "Lift@K"])

def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

# =========================
# PATHS (fixos)
# =========================
path_valid = "Fase_5/Tech_Challenge/data/base_validacao.parquet"
path_2024  = "Fase_5/Tech_Challenge/data/base_2024_pred.parquet"

# =========================
# LOAD DATA
# =========================
try:
    df_valid = load_parquet(path_valid)
except Exception as e:
    st.error(f"Erro ao carregar validação: {e}")
    st.stop()

try:
    df_2024 = load_parquet(path_2024)
except Exception as e:
    st.error(f"Erro ao carregar previsões 2024: {e}")
    st.stop()

# padronizações básicas
for d in (df_valid, df_2024):
    if "RA" in d.columns:
        d["RA"] = d["RA"].astype(str).str.strip()
    if "ANO" in d.columns:
        d["ANO"] = pd.to_numeric(d["ANO"], errors="coerce").astype("Int64")

indicadores = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE"]
df_valid = safe_numeric(df_valid, indicadores)
df_2024 = safe_numeric(df_2024, indicadores)

score_valid = pick_score_col(df_valid)
score_2024 = pick_score_col(df_2024)

if score_valid is None:
    st.warning("Não encontrei coluna de score na validação (esperado: prob/prob_risco).")
if score_2024 is None:
    st.warning("Não encontrei coluna de score na base 2024 (esperado: prob/prob_risco).")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Filtros")

max_k = int(max(len(df_valid), len(df_2024))) if max(len(df_valid), len(df_2024)) > 0 else 10
max_k = max(10, max_k)
default_k = 30 if max_k >= 30 else max_k

k = st.sidebar.slider("Top K alunos para priorizar", min_value=10, max_value=max_k, value=default_k, step=10)
limiar = st.sidebar.slider("Probabilidade (aba Ranking 2024)", min_value=0.0, max_value=1.0, value=0.70, step=0.05)

corte_delta = st.sidebar.slider(
    "Definir grau de severidade",
    min_value=-5.0, max_value=0.0, value=-1.0, step=0.1
)

# recalcula risco conforme slider
df_valid_calc = recalcular_risco_por_corte_validacao(df_valid, corte_delta)
df_2024_calc = recalcular_risco_por_corte_2024(df_2024, df_valid_calc, corte_delta)

ras_2024 = sorted(df_2024_calc["RA"].dropna().unique()) if "RA" in df_2024_calc.columns else []
aluno = st.sidebar.selectbox("Selecionar aluno (2024)", ras_2024) if ras_2024 else None
df_aluno_2024 = df_2024_calc[df_2024_calc["RA"] == aluno].copy() if aluno else pd.DataFrame()

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📈 Validação (Ranking)", "🚨 Ranking 2024", "👤 Perfil do Aluno"])
mapa_risco = {0: "🟢 Sem risco", 1: "🔴 Entrou em risco"}

# =========================
# TAB 1: VALIDATION
# =========================
with tab1:
    st.subheader("Validação temporal (ex.: treino 2022 → teste 2023)")
    st.caption(f"Definição dinâmica (apenas análise): **risco_calc = (delta_INDE < {corte_delta:.2f})**")

    if not score_valid or "risco_futuro" not in df_valid_calc.columns:
        st.info("Para validação, preciso de colunas: score ('prob'/'prob_risco') e 'risco_futuro' na base_validacao.parquet.")
    else:
        dfv = df_valid_calc.copy()

        # garante tipos
        dfv[score_valid] = pd.to_numeric(dfv[score_valid], errors="coerce")
        dfv["risco_futuro"] = pd.to_numeric(dfv["risco_futuro"], errors="coerce").astype("Int64")

        # remove linhas inválidas
        dfv = dfv.dropna(subset=[score_valid, "risco_futuro"]).copy()

        # opcional: só onde delta existe (não é o 1º ano do RA)
        if "delta_INDE" in dfv.columns:
            dfv = dfv.dropna(subset=[score_valid, "risco_futuro"]).copy()

        st.write("Linhas válidas para métricas:", len(dfv))

        if len(dfv) == 0:
            st.warning("Após filtrar score/risco_futuro (e delta), não sobrou nenhuma linha. Verifique se o parquet tem score e risco_futuro preenchidos.")
        else:
            base_rate = float(pd.to_numeric(dfv["risco_futuro"], errors="coerce").mean())
            st.metric("Taxa base de risco (risco_futuro real)", f"{base_rate:.2%}")

            # AUC
            try:
                auc = roc_auc_score(dfv["risco_futuro"].astype(int), dfv[score_valid].clip(0, 1))
                st.metric("ROC AUC (validação)", f"{auc:.3f}")
            except Exception as e:
                st.info(f"Não foi possível calcular ROC AUC: {e}")

            p_k = precision_at_k(dfv, score_valid, "risco_futuro", min(k, len(dfv)))
            l_k = lift_at_k(dfv, score_valid, "risco_futuro", min(k, len(dfv)))

            colA, colB, colC = st.columns(3)
            colA.metric(f"Precision@{min(k, len(dfv))}", f"{p_k:.2%}" if not np.isnan(p_k) else "—")
            colB.metric(f"Lift@{min(k, len(dfv))}", f"{l_k:.2f}x" if not np.isnan(l_k) else "—")

            # concordância real vs calc (se existir)
            if "risco_futuro_calc" in dfv.columns:
                dfv["risco_futuro_calc"] = pd.to_numeric(dfv["risco_futuro_calc"], errors="coerce").astype("Int64")
                m = dfv.dropna(subset=["risco_futuro_calc"]).copy()
                if len(m) > 0:
                    concord = float((m["risco_futuro"].astype(int) == m["risco_futuro_calc"].astype(int)).mean())
                    colC.metric("Concordância (real vs calc)", f"{concord:.2%}")
                else:
                    colC.metric("Concordância (real vs calc)", "—")
            else:
                colC.metric("Concordância (real vs calc)", "—")

            # curvas
            ks = [10, 20, 30, 50, 100, 150, 200]
            df_curve = make_curve_df(dfv, score_valid, "risco_futuro", ks)

            if df_curve.empty or df_curve["K"].isna().all():
                st.warning("Não consegui montar a curva Precision@K/Lift@K (amostra pequena).")
            else:
                fig1 = px.line(df_curve, x="K", y="Precision@K", markers=True, title="Precision@K (Validação)")
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = px.line(df_curve, x="K", y="Lift@K", markers=True, title="Lift@K (Validação)")
                st.plotly_chart(fig2, use_container_width=True)

            # tabela topK
            k_eff = min(k, len(dfv))
            st.caption(f"Top {k_eff} da validação (para inspecionar)")
            top_valid = dfv.sort_values(score_valid, ascending=False).head(k_eff).copy()

            top_valid["Risco Real"] = top_valid["risco_futuro"].map(mapa_risco)
            if "risco_futuro_calc" in top_valid.columns:
                top_valid["Risco Futuro (calc)"] = top_valid["risco_futuro_calc"].map(mapa_risco)
            else:
                top_valid["Risco Futuro (calc)"] = "—"

            cols_show = [c for c in ["RA", "NOME", "ANO", score_valid, "INDE", "Risco Real"] if c in top_valid.columns]
            st.dataframe(top_valid[cols_show], use_container_width=True, hide_index=True)

            st.error(f"{int(pd.to_numeric(top_valid['risco_futuro'], errors='coerce').fillna(0).sum())} alunos do Top {k_eff} realmente entraram em risco (real).")

# =========================
# TAB 2: RANKING 2024
# =========================
with tab2:
    st.subheader("Ranking de priorização (2024)")
    st.caption(f"Análise dinâmica: **risco_calc = (delta_INDE < {corte_delta:.2f})**")

    if not score_2024:
        st.info("Não achei coluna de score na base_2024_pred.parquet (esperado: prob/prob_risco).")
    else:
        dfp = df_2024_calc.copy()
        dfp[score_2024] = pd.to_numeric(dfp[score_2024], errors="coerce").clip(0, 1)
        dfp = dfp.dropna(subset=[score_2024]).copy()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total alunos (2024)", f"{len(dfp):,}".replace(",", "."))
        col2.metric(f"Média {score_2024}", f"{float(dfp[score_2024].mean()):.2%}")
        if "risco_calc" in dfp.columns:
            col3.metric("Taxa risco (calc)", f"{float(pd.to_numeric(dfp['risco_calc'], errors='coerce').mean()):.2%}")
        else:
            col3.metric("Taxa risco (calc)", "—")

        k_eff = min(k, len(dfp))

        st.caption(f"Top {k_eff} alunos com maior score")
        top_2024 = dfp.sort_values(score_2024, ascending=False).head(k_eff).copy()
        if "risco_calc" in top_2024.columns:
            top_2024["Risco (calc)"] = top_2024["risco_calc"].map(mapa_risco)

        cols_show = [c for c in ["RA", "NOME", "ANO", score_2024, "INDE", "delta_INDE", "Risco (calc)"] if c in top_2024.columns]
        st.dataframe(top_2024[cols_show], use_container_width=True, hide_index=True)

        if "risco_calc" in top_2024.columns:
            st.error(f"{int(pd.to_numeric(top_2024['risco_calc'], errors='coerce').fillna(0).sum())} alunos do Top {k_eff} estão em risco (calc).")

        st.caption(f"Lista por probabilidade: {score_2024} ≥ {limiar:.2f}")
        acima = dfp[dfp[score_2024] >= limiar].sort_values(score_2024, ascending=False).copy()
        cols_show2 = [c for c in cols_show if c in acima.columns]
        st.dataframe(acima[cols_show2], use_container_width=True, hide_index=True)

        fig = px.histogram(dfp, x=score_2024, nbins=30, title="Distribuição do score (2024)")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: STUDENT PROFILE
# =========================
with tab3:
    st.subheader("Perfil do aluno (2024)")
    if aluno is None or df_aluno_2024.empty:
        st.info("Selecione um aluno no menu lateral.")
    else:
        row = df_aluno_2024.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RA", str(row.get("RA", "")))
        c2.metric("Nome", str(row.get("NOME", ""))[:40])

        if score_2024:
            c3.metric("Probabilidade (2024)", f"{safe_float(row.get(score_2024, 0)):.2%}")
        else:
            c3.metric("Probabilidade (2024)", "—")

        c4.metric("delta_INDE", f"{safe_float(row.get('delta_INDE', np.nan), default=np.nan):.3f}" if not pd.isna(row.get("delta_INDE", np.nan)) else "—")

        if "risco_calc" in df_aluno_2024.columns:
            risco_val = row.get("risco_calc", 0)
            risco_val = int(0 if pd.isna(risco_val) else risco_val)
            if risco_val == 1:
                st.error(f"🔴 Risco (calc) pelo corte {corte_delta:.2f}")
            else:
                st.success(f"🟢 Sem risco (calc) pelo corte {corte_delta:.2f}")

        ind_exist = [c for c in ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"] if c in df_aluno_2024.columns]
        if len(ind_exist) >= 3:
            vals = pd.to_numeric(df_aluno_2024[ind_exist].iloc[0], errors="coerce").fillna(0).values
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=ind_exist, fill="toself"))
            fig_radar.update_layout(
                title="Radar de indicadores (2024)",
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Indicadores insuficientes para radar.")

        st.caption("Registro completo (2024)")
        st.dataframe(df_aluno_2024, use_container_width=True, hide_index=True)
