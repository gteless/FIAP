# Importação de bibliotecas

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_theme(style="whitegrid", palette="pastel")

# Carregamento do pipeline usando decorador

NOME_PIPELINE = 'modelo_obesidade_pipeline_COMPLETO.joblib'

@st.cache_data
def carrega_pipeline():
    """Carrega o Pipeline de ML salvo com joblib."""
    try:
        pipeline = joblib.load(NOME_PIPELINE)
        return pipeline
    except FileNotFoundError:
        st.error(f"Erro: O arquivo do pipeline '{NOME_PIPELINE}' não foi encontrado.")
        return None


def carrega_dados():
    try:
        df = pd.read_csv("Obesity_Final.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Arquivo de dados não encontrado.")
        return None

df_painel = carrega_dados()
pipeline_modelo = carrega_pipeline()


# Só executa as conversões se df não for None
if df_painel is not None:
    # 1) Calcular IMC se tiver peso e altura (arredondado)
    if 'peso' in df_painel.columns and 'altura' in df_painel.columns:
        # evita divisão por zero
        df_painel['imc'] = (df_painel['peso'] / (df_painel['altura'].replace(0, np.nan) ** 2)).round(2)
    else:
        # se não tem peso/altura, cria coluna imc vazia para evitar KeyError mais adiante
        df_painel['imc'] = np.nan

    # 2) Mapear binários armazenados como 1/0 para rótulos (apenas para exibição nos gráficos)
    map_binario = {1: 'Sim', 0: 'Não'}
    if 'fuma' in df_painel.columns and df_painel['fuma'].dropna().dtype != object:
        df_painel['fuma_label'] = df_painel['fuma'].map(map_binario)
    elif 'fuma' in df_painel.columns:
        # se já for texto, padroniza rótulo
        df_painel['fuma_label'] = df_painel['fuma'].replace({'yes': 'Sim', 'no': 'Não', 'Yes': 'Sim', 'No': 'Não'}).fillna(df_painel['fuma'])

    if 'come_alimentos_caloricos' in df_painel.columns and df_painel['come_alimentos_caloricos'].dropna().dtype != object:
        df_painel['come_alimentos_caloricos_label'] = df_painel['come_alimentos_caloricos'].map(map_binario)
    elif 'come_alimentos_caloricos' in df_painel.columns:
        df_painel['come_alimentos_caloricos_label'] = df_painel['come_alimentos_caloricos']

    if 'monitora_calorias_consumidas' in df_painel.columns and df_painel['monitora_calorias_consumidas'].dropna().dtype != object:
        df_painel['monitora_calorias_consumidas_label'] = df_painel['monitora_calorias_consumidas'].map(map_binario)
    elif 'monitora_calorias_consumidas' in df_painel.columns:
        df_painel['monitora_calorias_consumidas_label'] = df_painel['monitora_calorias_consumidas']

    # 3) Padronizar consome_alcool: garantir numérico 0-3 e criar coluna de rótulos para exibição
    mapa_alcool = {
        'Não': 0, 'no': 0, 'No': 0, 'NO': 0,
        'Algumas vezes': 1, 'Sometimes': 1, 'sometimes': 1,
        'Frequentemente': 2, 'Frequently': 2, 'frequently': 2,
        'Sempre': 3, 'Always': 3, 'always': 3
    }
    if 'consome_alcool' in df_painel.columns:
        # tenta mapear strings para números; se já for numérico, mantém
        if df_painel['consome_alcool'].dtype == object:
            df_painel['consome_alcool'] = df_painel['consome_alcool'].map(mapa_alcool)
        # agora garante rótulo legível para gráficos
        mapa_rotulos_alcool = {0: 'Não', 1: 'Algumas vezes', 2: 'Frequentemente', 3: 'Sempre'}
        df_painel['consome_alcool_label'] = df_painel['consome_alcool'].map(mapa_rotulos_alcool)
    



# Funções para a formatação dos dados categóricos obtidos via Streamlit para o formato esperado pelo pipeline

def converte_binarios(df):
    # Converte Sim/Não em 1/0 para colunas binárias
    binarios = ['historico_familiar','come_alimentos_caloricos','fuma','monitora_calorias_consumidas']
    for coluna in binarios:
        df[coluna] = df[coluna].apply(lambda x: 1 if x == 'Sim' else 0)
    #print(f'Campos {binarios} convertidos para binário. Sim = 1, Não = 0')
    return df

def converte_genero(df):
    # Mapeamento do Gênero: Mulher=1, Homem=0
    df['genero'] = df['genero'].apply(lambda x: 1 if x == 'Mulher' else 0)
    #print('Campo de genero convertido para binário. Mulher = 1, Homem = 0')
    return df

def converte_categoricos (df):
    # Mapeamento: Não=0, Algumas vezes=1, Frequentemente=2, Sempre=3
    categoricos = ['come_entre_refeicoes', 'consome_alcool']
    dict = {    'Não' : 0,
                'Algumas vezes' : 1,
                'Frequentemente' : 2,
                'Sempre' : 3
}
    for col in categoricos:
        df[col] = df[col].map(dict)
    #print(f'Campos categóricos {categoricos} convertidos.')
    return df

def converte_costuma_comer_vegetais (df):
    # Mapeamento: Raramente=0, Às vezes=1, Sempre=2
    categoricos = ['costuma_comer_vegetais']
    dict = {    'Raramente' : 0,
                'Às vezes' : 1,
                'Sempre' : 2
}
    for col in categoricos:
        df[col] = df[col].map(dict)
    #print(f'Campos categóricos (vegetais) {categoricos} convertidos.')
    return df


# ==============================
# 🎨 Função para aplicar o tema do Streamlit aos gráficos
# ==============================
def aplica_tema_streamlit():
    """Aplica as cores do tema atual do Streamlit a Seaborn e Matplotlib, com fallback seguro."""
    try:
        cores = st.get_option("theme")
    except Exception:
        cores = None

    if not cores:
        # Fallback padrão caso o tema não esteja disponível
        cores = {
            "primaryColor": "#1f77b4",
            "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#F0F2F6",
            "textColor": "#31333F"
        }

    sns.set_palette([
        cores["primaryColor"],
        cores["secondaryBackgroundColor"],
        cores["textColor"],
    ])

    plt.rcParams.update({
        "axes.facecolor": cores["backgroundColor"],
        "figure.facecolor": cores["backgroundColor"],
        "text.color": cores["textColor"],
        "axes.labelcolor": cores["textColor"],
        "xtick.color": cores["textColor"],
        "ytick.color": cores["textColor"]
    })

    sns.set_style("whitegrid", {
        'axes.edgecolor': cores["secondaryBackgroundColor"],
        'grid.color': cores["secondaryBackgroundColor"]
    })



# Configuração da Página
st.set_page_config(
    page_title="Preditor de Obesidade", # Título que aparece na aba do navegador
    page_icon="🏥",                      # Ícone na aba do navegador (pode ser um emoji ou caminho para um arquivo)
    layout="wide",                       # Define o layout para ocupar toda a largura da tela
    initial_sidebar_state="auto"         # Define o estado inicial da barra lateral
)

# Aplica o tema visual do Streamlit aos gráficos
aplica_tema_streamlit()

# Título Principal
st.title("🏥 Ferramenta de Auxílio ao Diagnóstico de Obesidade")
st.markdown("---") 


############ PAINEL ANALITICO ##############

# =========================
# PAINEL ANALÍTICO
# =========================
if df_painel is not None:
    # Seção 1 - Perfil da Amostra
    st.header("📊 Perfil da Amostra")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Número total de participantes", len(df_painel))
        if 'idade' in df_painel.columns:
            st.metric("Idade média", f"{df_painel['idade'].mean():.1f} anos")

    with col2:
        if 'genero' in df_painel.columns:
            df_plot = df_painel.copy()
            df_plot['genero_label'] = df_plot['genero'].map({0: 'Homem', 1: 'Mulher'}).fillna('Desconhecido')
            genero_count = df_plot['genero_label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(4.5, 4.5))  # proporção quadrada
            ax.pie(
                genero_count, 
                labels=genero_count.index, 
                autopct='%1.1f%%', 
                startangle=90
            )
            ax.set_title("Distribuição por Gênero")
            ax.axis('equal')  # mantém o gráfico circular
            st.pyplot(fig, use_container_width=True)  # ocupa toda a largura da coluna

    with col3:
        if 'imc' in df_painel.columns:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))  # mesmo tamanho do de cima
            sns.histplot(df_painel['imc'].dropna(), bins=15, kde=True, ax=ax)
            ax.set_title("Distribuição do IMC")
            st.pyplot(fig, use_container_width=True)


    st.markdown("---")

    # Seção 2 - Hábitos Alimentares e Estilo de Vida
    st.header("🥗 Hábitos Alimentares e Estilo de Vida")

    col1, col2 = st.columns(2)
    with col1:
        if 'come_alimentos_caloricos_label' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='come_alimentos_caloricos_label', order=['Não', 'Sim'], ax=ax)
            ax.set_title("Consumo de alimentos calóricos")
            ax.set_xlabel("")
            st.pyplot(fig)
        elif 'come_alimentos_caloricos' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='come_alimentos_caloricos', ax=ax)
            ax.set_title("Consumo de alimentos calóricos")
            st.pyplot(fig)

    with col2:
        if 'consome_alcool_label' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='consome_alcool_label', order=['Não', 'Algumas vezes', 'Frequentemente', 'Sempre'], ax=ax)
            ax.set_title("Consumo de álcool")
            ax.set_xlabel("")
            st.pyplot(fig)
        elif 'consome_alcool' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='consome_alcool', ax=ax)
            ax.set_title("Consumo de álcool")
            st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        if 'fuma_label' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='fuma_label', order=['Não', 'Sim'], ax=ax)
            ax.set_title("Fumantes na amostra")
            ax.set_xlabel("")
            st.pyplot(fig)
        elif 'fuma' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='fuma', ax=ax)
            ax.set_title("Fumantes na amostra")
            st.pyplot(fig)

    with col4:
        if 'monitora_calorias_consumidas_label' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='monitora_calorias_consumidas_label', order=['Não', 'Sim'], ax=ax)
            ax.set_title("Monitora calorias?")
            ax.set_xlabel("")
            st.pyplot(fig)
        elif 'monitora_calorias_consumidas' in df_painel.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df_painel, x='monitora_calorias_consumidas', ax=ax)
            ax.set_title("Monitora calorias?")
            st.pyplot(fig)

    st.markdown("---")

    # Seção 3 - Desempenho do Modelo
    st.header("🤖 Desempenho do Modelo")

    if pipeline_modelo is not None and 'nivel_obesidade' in df_painel.columns:
        # Protege caso o pipeline exija colunas específicas: remove apenas a coluna target
        try:
            X = df_painel.drop(columns=['nivel_obesidade'])
            y = df_painel['nivel_obesidade']
            y_pred = pipeline_modelo.predict(X)

            acc = accuracy_score(y, y_pred)
            st.metric("Acurácia do Modelo", f"{acc*100:.2f}%")

            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Matriz de Confusão")
            st.pyplot(fig)

            st.subheader("Relatório de Classificação")
            st.text(classification_report(y, y_pred))
        except Exception as e:
            st.error(f"Erro ao avaliar o modelo: {e}")
    else:
        st.warning("⚠️ Não foi possível avaliar o desempenho do modelo. Verifique se o dataset contém 'nivel_obesidade' e se o pipeline está disponível.")

else:
    st.warning("⚠️ Carregue o dataset e o modelo para exibir o painel.")

st.markdown("---")
############ PAINEL ANALITICO ##############

st.markdown("---") 

# Descrição/Subtítulo (Ajustado com o contexto do desafio)
st.markdown(
    """
    #### **Contexto Médico:**
    Desenvolvido para auxiliar a equipe médica, este sistema utiliza um modelo de Machine Learning 
    para prever o nível de obesidade de um indivíduo. A obesidade é uma condição multifatorial 
    que prejudica a saúde, e este modelo integra dados antropométricos, genéticos e comportamentais 
    para um **pré-diagnóstico rápido**.

    **Instruções:** Preencha os campos abaixo com as informações do paciente para obter o diagnóstico preditivo.
    """
)

st.divider() # Linha final do cabeçalho

# Captura das Features

col4, col5, col6 = st.columns(3)

# Coluna 1

with col4:

    st.subheader("Dados Básicos (paciente)")

    idade = st.number_input("Idade (anos)", min_value=18, max_value=120, value=25, step=1)

    altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")

    peso = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, format="%.1f") 

    genero = st.radio("Gênero", options=['Mulher', 'Homem'], horizontal=True, index=0)

    historico_familiar = st.radio("Histórico Familiar de Sobrepeso/Obesidade", options=['Sim', 'Não'], horizontal=True, index=1)

# Coluna 2
with col5:
    st.subheader("Hábitos Alimentares")

    numero_refeicoes_diarias = st.number_input("Número de Refeições Diárias", min_value=1, max_value=10, value=3, step=1)

    consumo_diario_agua = st.number_input("Consumo Diário de Água (litros)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.1f")

    come_alimentos_caloricos = st.radio("Costuma Comer Alimentos Calóricos", options=['Sim', 'Não'], horizontal=True, index=1)  

    costuma_comer_vegetais = st.radio("Costuma Comer Vegetais", options=['Raramente', 'Às vezes', 'Sempre'], horizontal=True, index=0)

    come_entre_refeicoes = st.radio("Costuma Comer Entre as Refeições", options=['Não', 'Às vezes', 'Frequentemente', 'Sempre'], horizontal=True, index=1)

    monitora_calorias_consumidas = st.radio("Monitora as Calorias Consumidas", options=['Sim', 'Não'], horizontal=True, index=1)


# Coluna 3
with col6:
    st.subheader("Outros Dados")

    frequencia_atividade_fisica = st.number_input("Frequência de Atividade Física (dias por semana)", min_value=0, max_value=7, value=3, step=1)

    tempo_dispositivos_tecnologicos = st.number_input("Tempo em Dispositivos Tecnológicos (horas por dia)", min_value=0, max_value=24, value=4, step=1)

    meio_de_transporte = st.selectbox("Meio de Transporte Principal", options=['Transporte_Publico', 'Caminhando', 'Automovel', 'Motocicleta', 'Bicicleta'], index=0)

    col3_1, col3_2 = st.columns(2)
    with col3_1:
        consome_alcool = st.radio("Consome Álcool", options=['Não', 'Às vezes', 'Frequentemente', 'Sempre'], horizontal=True, index=1)
    with col3_2:
        fuma = st.radio("Fuma", options=['Sim', 'Não'], horizontal=True, index=1)



st.divider() # Linha separadora antes do botão

# Botão de Previsão
if st.button("DIAGNÓSTICO PREDITIVO (Prever Obesidade)", type="primary"):
    if pipeline_modelo is not None:
        # Criação do DataFrame com os dados de entrada
        dados_entrada = pd.DataFrame({
            'genero': [genero],
            'idade': [idade],
            'altura': [altura],
            'peso': [peso],
            'historico_familiar': [historico_familiar],
            'come_alimentos_caloricos': [come_alimentos_caloricos],
            'costuma_comer_vegetais': [costuma_comer_vegetais],
            'numero_refeicoes_diarias': [numero_refeicoes_diarias],
            'come_entre_refeicoes': [come_entre_refeicoes],
            'fuma': [fuma],
            'consumo_diario_agua': [consumo_diario_agua],
            'monitora_calorias_consumidas': [monitora_calorias_consumidas],
            'frequencia_atividade_fisica': [frequencia_atividade_fisica],
            'tempo_dispositivos_tecnologicos': [tempo_dispositivos_tecnologicos],
            'consome_alcool': [consome_alcool],
            'meio_de_transporte': [meio_de_transporte]
        })

        # Formatação dos dados categóricos
        dados_entrada = converte_binarios(dados_entrada)
        dados_entrada = converte_genero(dados_entrada)
        dados_entrada = converte_categoricos(dados_entrada)
        dados_entrada = converte_costuma_comer_vegetais(dados_entrada)

        # Realiza a previsão usando o pipeline carregado
        # O pipeline deve cuidar do OHE/Label Encoding restante e Scaling
        previsao = pipeline_modelo.predict(dados_entrada)

         # Mapeamento da previsão para rótulos legíveis
        mapa_obesidade = {
            0: "Abaixo do Peso (Risco Mínimo)",
            1: "Peso Normal (Saudável)",
            2: "Sobrepeso I (Atenção)",
            3: "Sobrepeso II (Risco Moderado)",
            4: "Obesidade Grau I (Risco Alto)",
            5: "Obesidade Grau II (Risco Crítico)",
            6: "Obesidade Grau III (Risco Máximo)"
        }
        
        # Definições resumidas dos graus de obesidade
        mapa_definicoes = {
            0: "A pessoa pode estar com peso abaixo do ideal. Embora o risco de doenças crônicas relacionadas à obesidade seja baixo, é crucial investigar se há problemas nutricionais ou outras condições médicas subjacentes que causem o baixo peso. Uma avaliação nutricional completa é recomendada.",
            1: "O peso do indivíduo é classificado como normal (saudável). Isso indica um risco reduzido de complicações de saúde associadas ao excesso de peso. A manutenção de um estilo de vida equilibrado e a monitoração periódica são a melhor conduta.",
            2: "A pessoa está classificada com Sobrepeso Grau I. Esta é a primeira categoria de excesso de peso. O acúmulo de gordura corporal, embora ainda não seja considerado obesidade, exige atenção e ajustes no estilo de vida para evitar a progressão para graus mais severos e o aumento do risco de comorbidades.",
            3: "O indivíduo está com Sobrepeso Grau II, indicando um risco moderado de desenvolver condições de saúde associadas ao excesso de peso. A intervenção médica, nutricional e a incentivo à atividade física são fortemente recomendados neste estágio para a reversão do quadro.",
            4: "A classificação aponta para Obesidade Grau I. Este grau representa um risco alto para a saúde, aumentando a probabilidade de doenças cardiovasculares, diabetes tipo 2 e outras comorbidades. É fundamental iniciar um plano de tratamento e acompanhamento médico e multiprofissional (nutricionista, educador físico).",
            5: "O resultado indica Obesidade Grau II. Esta condição é considerada de alto risco e clinicamente significativa. Requer atenção imediata e um plano de tratamento intensivo e monitorado, com foco na perda de peso sustentável para reduzir o risco de complicações graves à saúde.",
            6: "A classificação de Obesidade Grau III (Obesidade Mórbida) representa o maior risco à saúde, com severas implicações para a qualidade de vida e longevidade. O tratamento é urgente e pode envolver intervenções médicas, cirúrgicas e acompanhamento contínuo de uma equipe de saúde especializada."
        }

        nivel_obesidade = mapa_obesidade.get(previsao[0], "Desconhecido")
        texto_definicao = mapa_definicoes.get(previsao[0], "Não foi possível obter uma definição detalhada para este resultado.")

        st.markdown("---")
        
        # Exibe o resultado da previsão com a definição (substituindo o sucesso/warning/error)
        
        if previsao[0] <= 1:
            st.success(f"### Previsão: **{nivel_obesidade}**")
            st.markdown(f"**Análise:** {texto_definicao}")
        elif previsao[0] <= 3:
            st.warning(f"### Previsão: **{nivel_obesidade}** (Requer Acompanhamento)")
            st.markdown(f"**Análise:** {texto_definicao}")
        else:
            st.error(f"### Previsão: **{nivel_obesidade}** (URGENTE: Risco à Saúde)")
            st.markdown(f"**Análise:** {texto_definicao}")
            
    else:
        st.error("O modelo não está disponível no momento. Tente novamente mais tarde.")

