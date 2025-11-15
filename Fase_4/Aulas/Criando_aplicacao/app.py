import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames
from sklearn.pipeline import Pipeline
import joblib
from joblib import load


dados = pd.read_csv('https://raw.githubusercontent.com/FIAP/Pos_Tech_DTAT/refs/heads/Deploy-de-Aplica%C3%A7%C3%B5es-Machine-Learning/df_clean.csv', 
                    sep=',')

st.write('# Simulador de avaliação de crédito')

st.write('### Idade')
input_idade = float(st.slider('Selecione a sua idade', 18, 100))

st.write('### Nível de escolaridade')
input_grau_escolaridade = st.selectbox('Qual é o seu grau de escolaridade?', dados['Grau_escolaridade'].unique())

st.write('### Estado civil')
input_estado = st.selectbox('Qual é o seu estado civil?', dados['Estado_civil'].unique())

st.write('### Qual é o tamanho da sua família?')
membros_familia = float(st.slider('Selecione a quantidade de integrantes da sua família', 1, 20))

st.write('### Carro próprio')
input_carro_proprio = st.radio('Você possui um automóvel?', ['Sim', 'Não'])
input_carro_proprio_dict = {'Sim':1,'Não':0}
input_carro_proprio = input_carro_proprio_dict.get(input_carro_proprio)

st.write('### Casa próprio')
input_casa_proprio = st.radio('Você possui um imóvel?', ['Sim', 'Não'])
input_casa_proprio_dict = {'Sim':1,'Não':0}
input_casa_proprio = input_casa_proprio_dict.get(input_casa_proprio)

st.write('### Tipo de residência')
input_tipo_moradia = st.selectbox('Qual é o tipo do seu imóvel?', dados['Moradia'].unique())

st.write('### Categoria de renda')
input_categoria_renda = st.selectbox('Qual é sua categoria de renda?', dados['Categoria_de_renda'].unique())

st.write('### Ocupação')
input_ocupacao = st.selectbox('Qual é a sua ocupação?', dados['Ocupacao'].unique())

st.write('### Experiência')
input_tempo_experiencia = float(st.slider('Qual é o seu tempo de experiência?', 0, 50))

st.write('### Rendimentos')
input_rendimentos = float(st.number_input('Digite o seu rendimento anual (em R$), pressione ENTER para confirmar', 0))

st.write('### Telefone corporativo')
input_telefone_trabalho = st.radio('Você possui um telefone corporativo?', ['Sim', 'Não'])
input_telefone_trabalho_dict = {'Sim':1,'Não':0}
input_telefone_trabalho = input_telefone_trabalho_dict.get(input_telefone_trabalho)

st.write('### Telefone fixo')
input_telefone = st.radio('Você possui um telefone fixo?', ['Sim', 'Não'])
input_telefone_dict = {'Sim':1,'Não':0}
input_telefone = input_telefone_dict.get(input_telefone)

st.write('### Email fixo')
input_email = st.radio('Você possui um email?', ['Sim', 'Não'])
input_email_dict = {'Sim':1,'Não':0}
input_email = input_email_dict.get(input_email)

novo_cliente = [0,
                input_carro_proprio,
                input_casa_proprio,
                input_telefone_trabalho,
                input_telefone,
                input_email,
                membros_familia,
                input_rendimentos,
                input_idade,
                input_tempo_experiencia,
                input_categoria_renda,
                input_grau_escolaridade,
                input_estado,
                input_tipo_moradia,
                input_ocupacao,
                0
                ]


def data_split (df, test_size):
  SEED=1561651
  treino_df, teste_df = train_test_split(df,test_size=test_size, random_state=SEED)
  return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(dados, 0.2)

cliente_predict_df = pd.DataFrame([novo_cliente], columns=teste_df.columns)

teste_novo_cliente = pd.concat([teste_df, cliente_predict_df], ignore_index=True)

#Pipeline
def pipeline_teste(df):

  pipeline = Pipeline([
    ('feature_dropper', DropFeatures()),
    ('OneHotEncoding', OneHotEncodingNames()),
    ('ordinal_feature', OrdinalFeature()),
    ('min_max_scaler', MinMaxWithFeatNames()),
  ])
  df_pipeline = pipeline.fit_transform(df)
  return df_pipeline

teste_novo_cliente = pipeline_teste(teste_novo_cliente)

cliente_pred = teste_novo_cliente.drop(['Mau'], axis=1)

if st.button('Enviar'):
  model = joblib.load('modelo/xgb.joblib')
  final_pred = model.predict(cliente_pred)

  if final_pred[-1]==0:
    st.success('### Parabéns! Você teve o cartão de crédito aprovado')
    st.balloons()
  else:
    st.error('### Infelizmente, não será possível liberar crédito para você nesse momento.')
