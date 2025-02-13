import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

# 1. Configuração do Streamlit
st.title("Análise de Dados e Predições")
st.sidebar.header("Opções")

# Criando abas para separar as seções
aba1, aba2 = st.tabs(["Análise de Dados", "Predição"])

# 2. Leitura dos arquivos reduzidos
def carregar_dados_reduzidos():
    file_path = "dados_reduzidos.csv"
    try:
        return pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        st.error(f"Erro ao carregar {file_path}: {e}")
        return pd.DataFrame()

data_reduzida = carregar_dados_reduzidos()

# 3. Processamento dos dados
def preprocessar_dados(data):
    data['dataNotificacao'] = pd.to_datetime(data['dataNotificacao'], errors='coerce')
    data = data.dropna(subset=['dataNotificacao'])
    data = data[(data['dataNotificacao'] >= '2022-01-01') & (data['dataNotificacao'] <= '2024-12-31')]
    data['mes'] = data['dataNotificacao'].dt.to_period('M')
    data['ano_mes'] = data['dataNotificacao'].dt.to_period('M').astype(str)
    return data

data_reduzida = preprocessar_dados(data_reduzida)

# 4. Função para gerar gráficos
def plotar_grafico(titulo, dados, xlabel, ylabel, cor):
    fig, ax = plt.subplots(figsize=(10, 6))
    dados.head(10).plot(kind='bar', color=cor, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 5. Análises
def gerar_analises(data, titulo):
    st.subheader(titulo)
    sintomas_counts = data['sintomas'].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    municipios_counts = data['municipioNotificacao'].value_counts()
    raca_counts = data['racaCor'].value_counts()
    vacina_counts = data['codigoDosesVacina'].value_counts()
    mes_counts = data['ano_mes'].value_counts().sort_index()
    
    st.write("### Top 10 Sintomas")
    plotar_grafico("Top 10 Sintomas Registrados", sintomas_counts, "Sintomas", "Número de Ocorrências", 'skyblue')
    
    st.write("### Top 10 Municípios com Mais Incidências")
    plotar_grafico("Top 10 Municípios com Mais Incidências", municipios_counts, "Município", "Número de Ocorrências", 'salmon')
    
    st.write("### Incidências por Raça/Cor")
    plotar_grafico("Incidências por Raça/Cor", raca_counts, "Raça/Cor", "Número de Ocorrências", 'green')
    
    st.write("### Incidências por Mês")
    plotar_grafico("Incidências por Mês", mes_counts, "Mês", "Número de Ocorrências", 'purple')
    
    st.write("### Doses de Vacina Administradas")
    plotar_grafico("Doses de Vacina Administradas", vacina_counts, "Doses", "Número de Ocorrências", 'orange')

# Aba 1 - Análise de Dados
with aba1:
    st.header("Análise da Base Reduzida")
    gerar_analises(data_reduzida, "Dados da Base Reduzida")

# 6. Predição do próximo mês (Base reduzida e balanceada)
def prever_proximo_mes(notificacoes):
    notificacoes = notificacoes.copy()
    notificacoes['ano_mes'] = pd.to_datetime(notificacoes['ano_mes'].astype(str))
    notificacoes['timestamp'] = (notificacoes['ano_mes'] - notificacoes['ano_mes'].min()).dt.days
    
    X = notificacoes[['timestamp']]
    y = notificacoes['quantidade']
    
    modelo = make_pipeline(PolynomialFeatures(2), LinearRegression())
    modelo.fit(X, y)
    
    proximo_mes = pd.DataFrame({'timestamp': [notificacoes['timestamp'].max() + 30]})
    previsao = modelo.predict(proximo_mes)
    previsao = max(0, previsao[0])
    
    return previsao

with aba2:
    st.header("Predição de Casos")
    st.write("Previsão baseada na tendência dos últimos meses")
    
    # Predição para MG (Base reduzida)
    notificacoes_geral = data_reduzida.groupby('ano_mes').size().reset_index(name='quantidade')
    if st.button("Executar Previsão para MG (Base Reduzida)"):
        previsao_geral = prever_proximo_mes(notificacoes_geral)
        st.success(f"Previsão para o próximo mês em MG: {previsao_geral:.2f} casos")
    
    # Predição para Top 10 Municípios (Base reduzida)
    top10_cidades = data_reduzida['municipioNotificacao'].value_counts().nlargest(10).index
    data_top10 = data_reduzida[data_reduzida['municipioNotificacao'].isin(top10_cidades)]
    notificacoes_top10 = data_top10.groupby(['municipioNotificacao', 'ano_mes']).size().reset_index(name='quantidade')
    
    for municipio in top10_cidades:
        df_mun = notificacoes_top10[notificacoes_top10['municipioNotificacao'] == municipio]
        if st.button(f"Executar Previsão para {municipio} (Base Reduzida)"):
            previsao_municipio = prever_proximo_mes(df_mun)
            st.success(f"Previsão para o próximo mês em {municipio}: {previsao_municipio:.2f} casos")
    
    # Seção para Base Balanceada
    st.subheader("Predição com Base Balanceada")
    try:
        file_path_bal = "municipios_populacao.csv"
        populacao = pd.read_csv(file_path_bal, sep=',', encoding='utf-8')
        data_balanceada = data_reduzida.merge(populacao, left_on='municipioNotificacao', right_on='municipio', how='left')
        data_balanceada['taxa_10k'] = (data_balanceada.groupby(['municipioNotificacao', 'ano_mes']).transform('count')['dataNotificacao'] / data_balanceada['populacao']) * 1000000
        notificacoes_bal = data_balanceada.groupby('ano_mes')['taxa_10k'].sum().reset_index()
        if st.button("Executar Previsão para MG (Base Balanceada)"):
            previsao_bal = prever_proximo_mes(notificacoes_bal)
            st.success(f"Previsão para o próximo mês em MG (Balanceada): {previsao_bal:.2f} casos")
    except Exception as e:
        st.error(f"Erro ao processar base balanceada: {e}")
