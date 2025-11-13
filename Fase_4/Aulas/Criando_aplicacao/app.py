import pandas as pd
import streamlit as st

dados = pd.read_csv('https://raw.githubusercontent.com/FIAP/Pos_Tech_DTAT/refs/heads/Deploy-de-Aplica%C3%A7%C3%B5es-Machine-Learning/df_clean.csv', 
                    sep=',')

st.write('# Simulador de avaliação de crédito')

st.write('### Idade')
input_idade = float(st.slider('Selecione a sua idade'), 18, 100)

st.write('### Nível de escolaridade')
input_grau_escolaridade = st.selectbox('Qual é o seu grau de escolaridade?', dados['Grau_escolaridade'].unique)

st.write('### Estado civil')
input_estado = st.selectbox('Qual é o seu estado civil?', dados['Estado_civil'].unique)

st.write('### Qual é o tamanho da sua família?')
input_idade = float(st.slider('Selecione a sua idade'), 1, 20)
