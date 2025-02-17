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

# ----------------------------------------------------------------
# 2. Função genérica para carregar CSV (com cache)
@st.cache_data
def carregar_csv(file_path, **kwargs):
    """
    Tenta carregar um CSV e retorna um DataFrame.
    Em caso de erro, mostra st.error e retorna DataFrame vazio.
    """
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        st.error(f"Erro ao carregar '{file_path}': {e}")
        return pd.DataFrame()

# Leitura do arquivo reduzido (cache garantido)
data_reduzida = carregar_csv(
    "dados_reduzidos_sampled.csv",
    sep=';',
    encoding='utf-8',
    on_bad_lines='skip',
    low_memory=False
)

# ----------------------------------------------------------------
# 3. Processamento dos dados (cache)
@st.cache_data
def preprocessar_dados(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converte 'dataNotificacao' para datetime, filtra por intervalo e cria a coluna 'ano_mes'.
    """
    if data.empty:
        return data
    
    if 'dataNotificacao' not in data.columns:
        st.warning("Coluna 'dataNotificacao' não encontrada. Retornando DataFrame vazio.")
        return pd.DataFrame()
    
    data['dataNotificacao'] = pd.to_datetime(data['dataNotificacao'], errors='coerce')
    data = data.dropna(subset=['dataNotificacao'])
    data = data[(data['dataNotificacao'] >= '2022-01-01') & (data['dataNotificacao'] <= '2024-12-31')]
    data['ano_mes'] = data['dataNotificacao'].dt.to_period('M').astype(str)
    return data

data_reduzida = preprocessar_dados(data_reduzida)

# ----------------------------------------------------------------
# 4. Função para gerar gráficos (não necessita de cache)
def plotar_grafico(titulo, dados, xlabel, ylabel, cor):
    """
    Plota um gráfico de barras com os 10 primeiros itens de 'dados'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    dados.head(10).plot(kind='bar', color=cor, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ----------------------------------------------------------------
# 5. Função de análise dos dados (apenas exibe os gráficos)
def gerar_analises(data: pd.DataFrame, titulo: str):
    """
    Gera análises (gráficos) sobre sintomas, municípios, raça/cor, etc.
    """
    st.subheader(titulo)
    
    if data.empty:
        st.warning("Nenhum dado encontrado para análise.")
        return
    
    colunas_necessarias = ["sintomas", "municipioNotificacao", "racaCor", "codigoDosesVacina", "ano_mes"]
    for col in colunas_necessarias:
        if col not in data.columns:
            st.warning(f"Coluna '{col}' não encontrada no dataset.")
            return
    
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

# ----------------------------------------------------------------
# 6. Predição do próximo mês (cache para a função de predição)
@st.cache_data
def prever_proximo_mes(notificacoes: pd.DataFrame, col_y='quantidade') -> float:
    """
    Recebe um DataFrame com 'ano_mes' e a coluna alvo (ex.: 'quantidade'),
    realiza regressão polinomial e prevê o valor para o próximo mês (~+30 dias).
    """
    if notificacoes.empty:
        st.warning("Sem dados para previsão.")
        return 0
    
    if 'ano_mes' not in notificacoes.columns:
        st.warning("Coluna 'ano_mes' não encontrada para previsão.")
        return 0
    if col_y not in notificacoes.columns:
        st.warning(f"Coluna '{col_y}' não encontrada para previsão.")
        return 0
    
    notificacoes['ano_mes'] = pd.to_datetime(notificacoes['ano_mes'].astype(str), errors='coerce')
    notificacoes.dropna(subset=['ano_mes'], inplace=True)
    
    if notificacoes.empty:
        st.warning("Dados de 'ano_mes' inválidos para previsão.")
        return 0
    
    notificacoes['timestamp'] = (notificacoes['ano_mes'] - notificacoes['ano_mes'].min()).dt.days
    
    if len(notificacoes) < 2:
        st.warning("Dados insuficientes para uma previsão confiável.")
        return 0
    
    X = notificacoes[['timestamp']]
    y = notificacoes[col_y]
    
    modelo = make_pipeline(PolynomialFeatures(2), LinearRegression())
    modelo.fit(X, y)
    
    proximo_ts = notificacoes['timestamp'].max() + 30
    df_proximo = pd.DataFrame({'timestamp': [proximo_ts]})
    previsao = modelo.predict(df_proximo)
    
    return max(0, previsao[0])

# Aba 2 - Predição
with aba2:
    st.header("Predição de Casos")
    st.write("Previsão baseada na tendência dos últimos meses")
    
    # Predição para MG (Base reduzida)
    if not data_reduzida.empty:
        notificacoes_geral = data_reduzida.groupby('ano_mes').size().reset_index(name='quantidade')
    else:
        notificacoes_geral = pd.DataFrame()
    
    if st.button("Executar Previsão para MG (Base Reduzida)"):
        previsao_geral = prever_proximo_mes(notificacoes_geral, col_y='quantidade')
        st.success(f"Previsão para o próximo mês em MG: {previsao_geral:.2f} casos")
    
    # Predição para Top 10 Municípios (Base reduzida)
    if not data_reduzida.empty and 'municipioNotificacao' in data_reduzida.columns:
        top10_cidades = data_reduzida['municipioNotificacao'].value_counts().nlargest(10).index
        data_top10 = data_reduzida[data_reduzida['municipioNotificacao'].isin(top10_cidades)]
        notificacoes_top10 = data_top10.groupby(['municipioNotificacao', 'ano_mes']).size().reset_index(name='quantidade')
        
        for municipio in top10_cidades:
            df_mun = notificacoes_top10[notificacoes_top10['municipioNotificacao'] == municipio]
            if st.button(f"Executar Previsão para {municipio} (Base Reduzida)"):
                previsao_municipio = prever_proximo_mes(df_mun, col_y='quantidade')
                st.success(f"Previsão para o próximo mês em {municipio}: {previsao_municipio:.2f} casos")
    
    # Seção para Base Balanceada
    st.subheader("Predição com Base Balanceada")
    try:
        file_path_bal = "municipios_populacao.csv"
        populacao = carregar_csv(file_path_bal, sep=',', encoding='utf-8')
        
        if not populacao.empty and not data_reduzida.empty:
            data_balanceada = data_reduzida.merge(
                populacao, left_on='municipioNotificacao', right_on='municipio', how='left'
            )
            
            if 'populacao' not in data_balanceada.columns:
                st.warning("Coluna 'populacao' não encontrada na base de municípios.")
            else:
                count_notif = data_balanceada.groupby(['municipioNotificacao', 'ano_mes'])['dataNotificacao'].transform('count')
                data_balanceada['taxa_10k'] = (count_notif / data_balanceada['populacao']) * 1000000
                notificacoes_bal = data_balanceada.groupby('ano_mes')['taxa_10k'].sum().reset_index(name='quantidade')
                
                if st.button("Executar Previsão para MG (Base Balanceada)"):
                    previsao_bal = prever_proximo_mes(notificacoes_bal, col_y='quantidade')
                    st.success(f"Previsão para o próximo mês em MG (Balanceada): {previsao_bal:.2f} casos")
        else:
            st.warning("Não foi possível processar a base balanceada (arquivo vazio ou dados reduzidos vazios).")
    
    except Exception as e:
        st.error(f"Erro ao processar base balanceada: {e}")
