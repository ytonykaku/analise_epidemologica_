import pandas as pd

# 1. Leitura dos arquivos
file_path = "dados_reduzidos.csv"
populacao_path = "municipios_populacao.csv"
out_file = "dados_consolidados.csv"

try:
    data = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)
    print(f"{file_path}: carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar {file_path}: {e}")
    data = pd.DataFrame()

try:
    populacao = pd.read_csv(populacao_path, sep=',', encoding='utf-8')  # Ajustar separador
    print(f"{populacao_path}: carregado com sucesso!")
    
    # Padronizar nomes das colunas
    populacao.columns = populacao.columns.str.strip().str.lower()
    populacao.rename(columns={'município': 'municipio', 'população': 'populacao'}, inplace=True)
    
    if 'municipio' not in populacao.columns:
        raise KeyError("A coluna 'municipio' não foi encontrada no arquivo de população. Verifique o nome correto.")
except Exception as e:
    print(f"Erro ao carregar {populacao_path}: {e}")
    populacao = pd.DataFrame()

# 2. Processamento dos dados
data['dataNotificacao'] = pd.to_datetime(data['dataNotificacao'], errors='coerce')
data = data.dropna(subset=['dataNotificacao'])
data = data[data['dataNotificacao'].dt.year > 2021]
data['ano_mes'] = data['dataNotificacao'].dt.to_period('M').astype(str)

# 3. Balanceamento por população
data['municipioNotificacao'] = data['municipioNotificacao'].str.strip()
populacao['municipio'] = populacao['municipio'].str.strip()
data = data.merge(populacao, left_on='municipioNotificacao', right_on='municipio', how='left')

# Remover a coluna duplicada
if 'municipio' in data.columns:
    data.drop(columns=['municipio'], inplace=True)

# Calcular taxa de notificações por 10.000 habitantes
data['taxa_10k'] = (data.groupby(['municipioNotificacao', 'ano_mes'])['dataNotificacao'].transform('count') / data['populacao']) * 10000

data_consolidado = data[['municipioNotificacao', 'ano_mes', 'dataNotificacao', 'taxa_10k', 'populacao']]

data_consolidado.to_csv(out_file, index=False)
print(f"Arquivo {out_file} gerado com sucesso!")
