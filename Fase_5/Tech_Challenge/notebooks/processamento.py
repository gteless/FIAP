# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pipeline de Predição de Risco Escolar
# Notebook gerado automaticamente.

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# %%
def limpar_colunas(df):
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def padronizar(df, ano):
    df = limpar_colunas(df)

    col_inde = [
        c for c in df.columns
        if "INDE" in c and (str(ano)[-2:] in c or str(ano) in c)
    ]
    df["INDE"] = df[col_inde[0]] if col_inde else None

    mapa = {
        "Nome": "NOME",
        "Nome Anonimizado": "NOME",
        "Gênero": "SEXO",
        "Idade 22": "IDADE",
        "Idade": "IDADE",
        "Rec Psicologia": "REC_PSICO",
        "Rec Psicologia ": "REC_PSICO",
        "Rec Psicologia.": "REC_PSICO", 
        "IAN": "IAN",
        "IDA": "IDA",
        "IEG": "IEG",
        "IAA": "IAA",
        "IPS": "IPS",
        "IPP": "IPP",
        "IPV": "IPV",
        "Defas": "DEFASAGEM",
        "Defasagem": "DEFASAGEM",
        "Destaque IEG": "DESTAQUE_IEG",
        "Destaque IDA": "DESTAQUE_IDA",
    }

    df = df.rename(columns=mapa)

    COLS = [
        "RA", "NOME", "SEXO", "IDADE", "DEFASAGEM", "REC_PSICO",
        "IAN", "IDA", "IEG", "IAA",
        "IPS", "IPP", "IPV", "INDE",
        "DESTAQUE_IEG", "DESTAQUE_IDA"
    ]

    for col in COLS:
        if col not in df.columns:
            df[col] = None

    df = df[COLS].copy()
    df["ANO"] = ano
    return df


# %%
arquivo = r"C:\Users\gabri\Desktop\FIAP\Fase_5\Tech_Challenge\data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

df_2022 = padronizar(pd.read_excel(arquivo, sheet_name="PEDE2022"), 2022)
df_2023 = padronizar(pd.read_excel(arquivo, sheet_name="PEDE2023"), 2023)
df_2024 = padronizar(pd.read_excel(arquivo, sheet_name="PEDE2024"), 2024)

df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
df.head()

# %%
cols = ['IDADE','DEFASAGEM','IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE', 'ANO']

# verifica se as colunas possuem algum valor que não sejam números
for c in cols:
    erro = pd.to_numeric(df[c], errors='coerce').isna() & df[c].notna()
    print(c, "-> valores inválidos:", erro.sum(), "-> valor identificado: ", df.loc[erro, c].unique())

# %%
# Converte para NaN os valores incorretos da coluna INDE
df['INDE'] = pd.to_numeric(df['INDE'], errors='coerce')
# Converte para NaN os valores incorretos da coluna IDADE
df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

# %%
df["SEXO"].unique()

# %%
#padroniza a coluna sexo
mapa_sexo = {
    "Menina": "F",
    "Feminino": "F",
    "Menino": "M",
    "Masculino": "M"
}

df["SEXO"] = df["SEXO"].map(mapa_sexo)


# %%
df["SEXO"].value_counts()

# %%
#converte para números para usar a coluna sexo como feature no modelo
df["SEXO"] = df["SEXO"].map({"F": 0, "M": 1})

# ✅ Features do texto (simples e fortes)
texto_cols = ["DESTAQUE_IEG", "DESTAQUE_IDA"]

# flag: existe observação?
df["tem_alerta_IEG"] = df["DESTAQUE_IEG"].notna().astype(int)
df["tem_alerta_IDA"] = df["DESTAQUE_IDA"].notna().astype(int)

# flag: contém "melhor" (ajuste palavras se quiser)
df["melhorar_IEG"] = df["DESTAQUE_IEG"].astype(str).str.contains("melhor", case=False, na=False).astype(int)
df["melhorar_IDA"] = df["DESTAQUE_IDA"].astype(str).str.contains("melhor", case=False, na=False).astype(int)


cols_num = ["IDADE","DEFASAGEM","IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE"]

df[cols_num] = df[cols_num].apply(pd.to_numeric, errors="coerce")
df[cols_num] = df[cols_num].fillna(df[cols_num].median(numeric_only=True))

df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype("Int64")
# %%
df = df.sort_values(["RA", "ANO"])

df["delta_IDA"] = df.groupby("RA")["IDA"].diff()
df["delta_INDE"] = df.groupby("RA")["INDE"].diff()

df["trend_IDA"] = (
    df.groupby("RA")["delta_IDA"]
      .transform(lambda x: x.rolling(2).mean())
)

df["trend_INDE"] = (
    df.groupby("RA")["delta_INDE"]
      .transform(lambda x: x.rolling(2).mean())
)

df["media_IDA"] = (
    df.groupby("RA")["IDA"]
      .transform(lambda x: x.expanding().mean())
)

df["media_INDE"] = (
    df.groupby("RA")["INDE"]
      .transform(lambda x: x.expanding().mean())
)

# %%
# novas features fortes
df["quedas_consecutivas"] = (
    df.groupby("RA")["delta_INDE"]
      .transform(lambda x: (x < 0).rolling(2).sum())
)

df["volatilidade"] = df.groupby("RA")["INDE"].transform("std")

df["risco"] = (df["delta_INDE"] < -1).astype(int)

df_shift = df.sort_values(["RA", "ANO"]).copy()
df_shift["risco_futuro"] = df_shift.groupby("RA")["risco"].shift(-1)

df_model = df_shift.dropna(subset=["risco_futuro"]).copy()
df_model["risco_futuro"] = df_model["risco_futuro"].astype(int)

# ✅ Features de Psicologia (2022 tem, outros anos podem não ter)
df_model["REC_PSICO"] = df_model["REC_PSICO"].fillna("Não avaliado").astype(str).str.strip()

mapa_psico_nivel = {
    "Sem limitações": 0,
    "Não indicado": 0,
    "Não avaliado": 0,
    "Requer avaliação": 1,
    "Não atendido": 2,
}

df_model["psico_nivel"] = df_model["REC_PSICO"].map(mapa_psico_nivel).fillna(0).astype(int)

# flags opcionais (bem úteis)
df_model["psico_requer_avaliacao"] = (df_model["REC_PSICO"] == "Requer avaliação").astype(int)
df_model["psico_nao_atendido"] = (df_model["REC_PSICO"] == "Não atendido").astype(int)
# %%
features = [
    "SEXO", "IDADE", "DEFASAGEM",
    "IAN", "IDA", "IEG", "IAA",
    "IPS", "IPP", "IPV",
    "delta_IDA",
    "trend_IDA",
    "media_IDA",
    "quedas_consecutivas",
    "volatilidade",
    "tem_alerta_IEG", "tem_alerta_IDA",
    "melhorar_IEG", "melhorar_IDA", "psico_nivel",
    "psico_requer_avaliacao",
    "psico_nao_atendido",
]

train_anos = [2022]
test_ano = 2023

df_model = df_model.groupby("RA").filter(lambda x: len(x) >= 2)


X_train = df_model[df_model["ANO"].isin(train_anos)][features].fillna(0)
y_train = df_model[df_model["ANO"].isin(train_anos)]["risco_futuro"]

X_test = df_model[df_model["ANO"] == test_ano][features].fillna(0)
y_test = df_model[df_model["ANO"] == test_ano]["risco_futuro"]


# %%
def avaliar_modelo(model, X_train, y_train, X_test, y_test, limiar=0.25):
    model.fit(X_train, y_train)

    if len(getattr(model, "classes_", [])) < 2:
        print("⚠️ Treino ficou com 1 classe só:", model.classes_)
        print("Distribuição y_train:\n", y_train.value_counts())
        return None

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > limiar).astype(int)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\nROC AUC:", roc_auc_score(y_test, y_prob))
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    return y_prob



# %%
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42
)

y_prob_rf = avaliar_modelo(rf, X_train, y_train, X_test, y_test, limiar=0.25)


# %%
xgb = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    scale_pos_weight=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

y_prob_xgb = avaliar_modelo(xgb, X_train, y_train, X_test, y_test, limiar=0.25)


# %%
def testar_limiares(y_test, y_prob):
    for t in [0.5, 0.3, 0.25, 0.2, 0.15]:
        print(f"\nLimiar: {t}")
        y_pred = (y_prob > t).astype(int)
        print(classification_report(y_test, y_pred))



# %%
testar_limiares(y_test, y_prob_xgb)


# %%
print(df_model["ANO"].value_counts())


# %%
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)

plt.plot(thresholds, tpr, label="Recall risco")
plt.plot(thresholds, 1-fpr, label="Especificidade")
plt.xlabel("Limiar")
plt.legend()
plt.show()

# %%
df_test = df_model[df_model["ANO"] == test_ano].copy()

X_test_alinhado = df_test[features].fillna(0)

df_test["prob"] = xgb.predict_proba(X_test_alinhado)[:, 1]

top = df_test.sort_values("prob", ascending=False).head(30)

print("Quantos riscos reais no TOP 30?")
print(top["risco_futuro"].sum())


# %%
if y_prob_xgb is not None:
    df_test = df_model[df_model["ANO"] == test_ano].copy()
    X_test_alinhado = df_test[features].fillna(0)

    df_test["prob"] = xgb.predict_proba(X_test_alinhado)[:, 1]

    top30 = df_test.sort_values("prob", ascending=False).head(30)
    print("Quantos riscos reais no TOP 30?", int(top30["risco_futuro"].sum()))

    for k in [10, 20, 30, 50, 100]:
        topk = df_test.sort_values("prob", ascending=False).head(k)
        acertos = int(topk["risco_futuro"].sum())
        print(f"Top {k}: {acertos} acertos → {(acertos/k):.2%}")


# %%
df_2024 = df[df["ANO"] == 2024].copy()

# garante numéricos no 2024 também
df_2024[cols_num] = df_2024[cols_num].apply(pd.to_numeric, errors="coerce")
med_train = df_model[df_model["ANO"].isin(train_anos)][cols_num].median(numeric_only=True)
df_2024[cols_num] = df_2024[cols_num].fillna(med_train)

# garante que todas as features existam no 2024
for f in features:
    if f not in df_2024.columns:
        df_2024[f] = 0

X_2024 = df_2024[features].fillna(0)
df_2024["prob"] = xgb.predict_proba(X_2024)[:, 1]
# %%
df_test.to_parquet(r"C:\Users\gabri\Desktop\FIAP\Fase_5\Tech_Challenge\data\base_validacao.parquet")
df_2024.to_parquet(r"C:\Users\gabri\Desktop\FIAP\Fase_5\Tech_Challenge\data\base_2024_pred.parquet")

