import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import folium
from streamlit_folium import folium_static
import json

# 1. Configuração do Streamlit
st.title("Análise de Dados e Predições")
st.sidebar.header("Opções")

# Criando três abas para separar as seções: Análise de Dados, Predição e Mapa
aba1, aba2, aba3 = st.tabs(["Análise de Dados", "Predição", "Mapa"])

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
    Plota um gráfico de barras com base em uma Series 'dados'.
    O índice da Series será o eixo X e os valores serão o eixo Y.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    dados.plot(kind='bar', color=cor, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ----------------------------------------------------------------
# 5. Função de análises interativas
def gerar_analises_interativas(data: pd.DataFrame):
    st.subheader("Análise Interativa da Base Reduzida")
    
    if data.empty:
        st.warning("Nenhum dado encontrado para análise.")
        return
    
    # ----------------- SINTOMAS -----------------
    st.write("### Sintomas")
    df_sintomas = data['sintomas'].str.get_dummies(sep=',')
    sintomas_unicos = sorted(df_sintomas.columns)
    sintomas_selecionados = st.multiselect(
        "Selecione os Sintomas a exibir:",
        options=sintomas_unicos,
        default=sintomas_unicos[:5]
    )
    if sintomas_selecionados:
        sintomas_counts = df_sintomas[sintomas_selecionados].sum().sort_values(ascending=False)
        plotar_grafico("Sintomas Selecionados", sintomas_counts, "Sintoma", "Número de Ocorrências", "skyblue")
    else:
        st.warning("Nenhum sintoma selecionado.")
    
    # ----------------- TOP 10 MUNICÍPIOS (GRÁFICO ESTÁTICO) -----------------
    st.write("### Top 10 Municípios com Mais Incidências")
    top10_municipios = data['municipioNotificacao'].value_counts().head(10)
    plotar_grafico("Top 10 Municípios com Mais Incidências", top10_municipios, "Município", "Número de Ocorrências", "red")
    
    # ----------------- MUNICÍPIOS (INTERATIVO) -----------------
    st.write("### Municípios (Seleção Manual)")
    municipios_unicos = data['municipioNotificacao'].astype(str).unique()
    municipios_selecionados = st.multiselect(
        "Selecione os Municípios a exibir:",
        options=sorted(municipios_unicos),
        default=sorted(municipios_unicos)[:5]
    )
    if municipios_selecionados:
        data_mun_filtrada = data[data['municipioNotificacao'].astype(str).isin(municipios_selecionados)]
        municipios_counts = data_mun_filtrada['municipioNotificacao'].value_counts()
        plotar_grafico("Municípios Selecionados", municipios_counts, "Município", "Número de Ocorrências", "salmon")
    else:
        st.warning("Nenhum município selecionado.")
    
    # ----------------- RAÇA/COR -----------------
    st.write("### Raça/Cor")
    racas_unicas = data['racaCor'].astype(str).unique()
    racas_selecionadas = st.multiselect(
        "Selecione as Raças/Cor a exibir:",
        options=sorted(racas_unicas),
        default=sorted(racas_unicas)[:5]
    )
    if racas_selecionadas:
        data_raca_filtrada = data[data['racaCor'].astype(str).isin(racas_selecionadas)]
        raca_counts = data_raca_filtrada['racaCor'].value_counts()
        plotar_grafico("Raça/Cor Selecionadas", raca_counts, "Raça/Cor", "Número de Ocorrências", "green")
    else:
        st.warning("Nenhuma raça/cor selecionada.")
    
    # ----------------- INCIDÊNCIAS POR MÊS/ANO -----------------
    st.write("### Incidências por Mês/Ano")
    ano_mes_unicos = sorted(data['ano_mes'].unique())
    ano_mes_selecionados = st.multiselect(
        "Selecione os Meses/Ano a exibir:",
        options=ano_mes_unicos,
        default=ano_mes_unicos
    )
    if ano_mes_selecionados:
        data_mes_filtrada = data[data['ano_mes'].isin(ano_mes_selecionados)]
        mes_counts = data_mes_filtrada['ano_mes'].value_counts().sort_index()
        plotar_grafico("Incidências por Mês/Ano (Selecionados)", mes_counts, "Mês/Ano", "Número de Ocorrências", "purple")
    else:
        st.warning("Nenhum mês/ano selecionado.")
    
    # ----------------- DOSES DE VACINA -----------------
    st.write("### Doses de Vacina Administradas")
    vacinas_unicas = data['codigoDosesVacina'].astype(str).unique()
    vacinas_selecionadas = st.multiselect(
        "Selecione as Vacinas a exibir:",
        options=sorted(vacinas_unicas),
        default=sorted(vacinas_unicas)[:5]
    )
    if vacinas_selecionadas:
        data_vac_filtrada = data[data['codigoDosesVacina'].astype(str).isin(vacinas_selecionadas)]
        vacina_counts = data_vac_filtrada['codigoDosesVacina'].value_counts()
        plotar_grafico("Doses de Vacina Administradas (Selecionadas)", vacina_counts, "Doses", "Número de Ocorrências", "orange")
    else:
        st.warning("Nenhuma vacina selecionada.")

# Aba 1 - Análise de Dados Interativa
with aba1:
    st.header("Análise da Base Reduzida")
    gerar_analises_interativas(data_reduzida)

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
    
    # Predição geral para MG (Base Reduzida)
    if not data_reduzida.empty:
        notificacoes_geral = data_reduzida.groupby('ano_mes').size().reset_index(name='quantidade')
    else:
        notificacoes_geral = pd.DataFrame()
    
    if st.button("Executar Previsão para MG (Base Reduzida)"):
        previsao_geral = prever_proximo_mes(notificacoes_geral, col_y='quantidade')
        st.success(f"Previsão para o próximo mês em MG: {previsao_geral:.2f} casos")
    
    # Predição para municípios específicos com filtro
    if not data_reduzida.empty and 'municipioNotificacao' in data_reduzida.columns:
        opcoes_municipios = sorted(data_reduzida['municipioNotificacao'].astype(str).unique())
        municipios_pred = st.multiselect(
            "Selecione Municípios para Previsão",
            options=opcoes_municipios,
            default=opcoes_municipios[:3]
        )
        if municipios_pred:
            dados_municipios = data_reduzida[data_reduzida['municipioNotificacao'].astype(str).isin(municipios_pred)]
            notificacoes_municipios = dados_municipios.groupby(['municipioNotificacao', 'ano_mes']).size().reset_index(name='quantidade')
            for municipio in municipios_pred:
                df_mun = notificacoes_municipios[notificacoes_municipios['municipioNotificacao'].astype(str) == municipio]
                if st.button(f"Executar Previsão para {municipio} (Base Reduzida)"):
                    previsao_municipio = prever_proximo_mes(df_mun, col_y='quantidade')
                    st.success(f"Previsão para o próximo mês em {municipio}: {previsao_municipio:.2f} casos")
    
    # Seção para Base Balanceada (sem filtros interativos avançados)
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

# ----------------------------------------------------------------

# Aba 3 - Mapa de Infecções em Minas Gerais
with aba3:
    st.header("Mapa de Infecções em Minas Gerais")
    
    # Agrupa os dados por município
    if not data_reduzida.empty and 'municipioNotificacao' in data_reduzida.columns:
        municipio_counts = data_reduzida['municipioNotificacao'].value_counts().reset_index()
        municipio_counts.columns = ['municipio', 'casos']
    else:
        municipio_counts = pd.DataFrame(columns=['municipio', 'casos'])
    
    # Carrega o arquivo GeoJSON com as fronteiras dos municípios de MG
    try:
        with open("geojs-31-mun.json", "r", encoding="utf-8") as f:
            geo_data = json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo GeoJSON: {e}")
        geo_data = None
    
    if geo_data is not None and not municipio_counts.empty:
        # Cria o mapa centrado em MG
        m = folium.Map(location=[-19.9167, -43.9345], zoom_start=7)

        # Passo 1: "Clipar" valores muito altos
        max_cutoff = 30000  # Ajuste conforme necessidade
        municipio_counts["casos_clipped"] = municipio_counts["casos"].clip(upper=max_cutoff)
        
        # Passo 2: Definir bins manuais
        # (Valores de exemplo, ajuste conforme a realidade)
        bins = [0, 10, 50, 200, 1000, 5000, 10000, 20000, 30000]

        folium.Choropleth(
            geo_data=geo_data,
            data=municipio_counts,
            columns=["municipio", "casos_clipped"],  # Usamos a coluna "casos_clipped"
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Casos de Infecção (com corte em 30k)",
            bins=bins,
            nan_fill_color="white",
            nan_fill_opacity=0.5
        ).add_to(m)

        folium_static(m)
    else:
        st.warning("Dados insuficientes ou erro ao carregar o GeoJSON para exibir o mapa.")

