import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calcular_taxa_notificacoes(arquivo_dados, arquivo_populacao):
    """Carrega e balanceia os dados de notificações por população."""
    df = pd.read_csv(arquivo_dados, dtype={"municipioNotificacao": "category"}, low_memory=False)
    df.columns = df.columns.str.strip()
    
    # Verificar e carregar corretamente o arquivo de população
    df_populacao = pd.read_csv(arquivo_populacao, encoding='ISO-8859-1', sep=None, engine='python')
    
    # Exibir as colunas detectadas para depuração
    print("Colunas detectadas:", df_populacao.columns.tolist())
    
    # Garantir que os nomes das colunas estão corretos
    if df_populacao.shape[1] == 1:
        raise ValueError("Erro ao carregar o arquivo: verifique se o delimitador está correto.")
    
    df_populacao.columns = ['municipio', 'populacao']
    
    # Remover espaços extras no nome do município
    df_populacao['municipio'] = df_populacao['municipio'].str.strip()
    
    # Converter a população para inteiro
    df_populacao['populacao'] = pd.to_numeric(df_populacao['populacao'], errors='coerce')
    df_populacao.dropna(subset=['populacao'], inplace=True)
    
    df['dataNotificacao'] = pd.to_datetime(df['dataNotificacao'], errors='coerce')
    df = df[(df['dataNotificacao'] >= '2022-01-01') & (df['dataNotificacao'] <= '2024-12-31')]
    
    df['ano_mes'] = df['dataNotificacao'].dt.to_period('M').astype(str)
    df.dropna(subset=['ano_mes', 'municipioNotificacao'], inplace=True)
    
    # Contagem de notificações antes do balanceamento
    top_municipios_antes = df['municipioNotificacao'].value_counts().nlargest(10)
    print("Top 10 municípios antes do balanceamento:")
    print(top_municipios_antes)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_municipios_antes.index, y=top_municipios_antes.values, palette='Blues_r')
    plt.xlabel("Município")
    plt.ylabel("Número de Notificações")
    plt.title("Top 10 Municípios com Mais Notificações Antes do Balanceamento")
    plt.xticks(rotation=45)
    plt.show()
    
    notificacoes_por_municipio = df.groupby(['municipioNotificacao', 'ano_mes']).size().reset_index(name='quantidade')
    
    notificacoes_por_municipio['municipioNotificacao'] = notificacoes_por_municipio['municipioNotificacao'].astype(str).str.strip()
    notificacoes_por_municipio = notificacoes_por_municipio.merge(df_populacao, left_on='municipioNotificacao', right_on='municipio', how='left')
    
    notificacoes_por_municipio.dropna(subset=['populacao'], inplace=True)
    notificacoes_por_municipio['taxa_100k'] = (notificacoes_por_municipio['quantidade'] / notificacoes_por_municipio['populacao']) * 100000
    notificacoes_por_municipio.drop(columns=['municipio'], inplace=True)
    
    # Contagem de notificações depois do balanceamento
    top_municipios_depois = notificacoes_por_municipio.groupby("municipioNotificacao")['taxa_100k'].sum().nlargest(10)
    print("Top 10 municípios depois do balanceamento:")
    print(top_municipios_depois)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_municipios_depois.index, y=top_municipios_depois.values, palette='Reds_r')
    plt.xlabel("Município")
    plt.ylabel("Taxa de Notificações por 100k Habitantes")
    plt.title("Top 10 Municípios com Mais Notificações Depois do Balanceamento")
    plt.xticks(rotation=45)
    plt.show()
    
    return notificacoes_por_municipio

def plotar_top10_municipios(df):
    """Plota gráfico empilhado das notificações mensais dos 10 municípios com mais casos."""
    top10 = df.groupby('municipioNotificacao')['taxa_100k'].sum().nlargest(10).index
    df_top10 = df[df['municipioNotificacao'].isin(top10)]
    
    df_pivot = df_top10.pivot(index='ano_mes', columns='municipioNotificacao', values='taxa_100k').fillna(0)
    df_pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='plasma')
    
    plt.xlabel('Mês')
    plt.ylabel('Notificações por 100 mil habitantes')
    plt.title('Notificações por Mês - Top 10 Municípios (Ajustado por População)')
    plt.xticks(rotation=45)
    plt.legend(title='Município')
    plt.show()

def treinar_modelo_predicao(df):
    """Treina um modelo de Random Forest para previsão de notificações."""
    df['ano_mes_num'] = df['ano_mes'].astype('category').cat.codes
    df = df.dropna()
    
    X = df[['ano_mes_num']]
    y = df['taxa_100k']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_rf = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo_rf.fit(X_train, y_train)
    
    y_pred_rf = modelo_rf.predict(X_test)
    print("Random Forest - MAE:", mean_absolute_error(y_test, y_pred_rf))
    
    return modelo_rf

def prever_proximo_mes(df, modelo):
    """Prevê o número de notificações para o próximo mês."""
    proximo_mes = df['ano_mes'].astype('category').cat.codes.max() + 1
    predicao = modelo.predict([[proximo_mes]])
    print(f"Previsão de notificações para o próximo mês: {int(predicao[0])}")

def prever_top10_municipios(df, modelo):
    """Prevê notificações para os 10 municípios com mais casos."""
    top10 = df.groupby('municipioNotificacao')['taxa_100k'].sum().nlargest(10).index
    df_top10 = df[df['municipioNotificacao'].isin(top10)]
    df_top10['ano_mes_num'] = df_top10['ano_mes'].astype('category').cat.codes
    
    predicoes = {}
    for municipio in top10:
        df_mun = df_top10[df_top10['municipioNotificacao'] == municipio]
        if not df_mun.empty:
            X_pred = [[df_mun['ano_mes_num'].max() + 1]]
            predicoes[municipio] = int(modelo.predict(X_pred)[0])
    
    print("Previsão de notificações para o próximo mês nos 10 municípios com mais casos:")
    for mun, pred in predicoes.items():
        print(f"{mun}: {pred} notificações")

# Chamada das funções principais
if __name__ == "__main__":
    df_resultado = calcular_taxa_notificacoes('dados_reduzidos.csv', 'municipios_populacao.csv')
    plotar_top10_municipios(df_resultado)
    modelo = treinar_modelo_predicao(df_resultado)
    prever_proximo_mes(df_resultado, modelo)
    prever_top10_municipios(df_resultado, modelo)


#municipios_populacao
