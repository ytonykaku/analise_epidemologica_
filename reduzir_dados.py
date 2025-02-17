import pandas as pd
import matplotlib.pyplot as plt
import glob

# 1. Leitura e união dos arquivos CSV com tratamento de linhas problemáticas
file_paths = glob.glob("*.csv")  # Certifique-se de executar no diretório correto

dataframes = []
for file in file_paths:
    try:
        df = pd.read_csv(file, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)
        dataframes.append(df)
        print(f"{file}: carregado com sucesso!")
        if file == "2024-1.csv":
            break  # Para a leitura ao encontrar o arquivo "2024-1.csv"
    except Exception as e:
        print(f"Erro ao carregar {file}: {e}")

# Concatenar todos os dataframes
full_data = pd.concat(dataframes, ignore_index=True)

# Reduzir a base de dados para 20% dos registros (removendo 80% aleatoriamente)
reduced_data = full_data.sample(frac=0.2, random_state=42)

# Salvar a base reduzida
reduced_data.to_csv("dados_reduzidos.csv", index=False, sep=';')

print("Arquivo 'dados_reduzidos.csv' gerado com 20% dos dados originais!")