import pandas as pd
import glob

# 1. Criar arquivos vazios para armazenar os dados processados
dados_completos_csv = "dados_completos.csv"
dados_balanceados_csv = "dados_balanceados.csv"

# Criar estrutura de DataFrame vazia para o primeiro arquivo
df_vazio = True

# 2. Carregar e processar arquivos CSV um por vez
file_paths = glob.glob("*.csv")  # Certifique-se de executar no diretório correto

for file in file_paths:
    try:
        for chunk in pd.read_csv(file, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False, chunksize=5000):
            chunk = chunk.drop_duplicates()
            
            # Limitar para 50.000 registros por arquivo
            chunk = chunk.sample(n=min(10000, len(chunk)), random_state=42)
            
            # Salvar em CSV aos poucos para evitar consumo excessivo de RAM
            chunk.to_csv(dados_completos_csv, mode='a', header=df_vazio, index=False)
            df_vazio = False
            
        print(f"{file}: processado e salvo parcialmente!")
    except Exception as e:
        print(f"Erro ao carregar {file}: {e}")

# 3. Recarregar dados já processados
full_data = pd.read_csv(dados_completos_csv, sep=',', encoding='utf-8', low_memory=False)

# 4. Criar DataFrame balanceado com até 1000 registros por município
balanced_data = full_data.groupby("municipioNotificacao").apply(lambda x: x.sample(n=min(1000, len(x)), random_state=42))
balanced_data = balanced_data.reset_index(drop=True)

# 5. Salvar dados balanceados
balanced_data.to_csv(dados_balanceados_csv, index=False)

print("Processamento concluído. Arquivos salvos: dados_completos.csv e dados_balanceados.csv")
