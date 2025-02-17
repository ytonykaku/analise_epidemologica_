import pandas as pd

# Caminho do arquivo CSV gerado pelo script anterior
populacao_path = "municipios_populacao.csv"

# Tentar carregar o arquivo e exibir informações
try:
    # Testar diferentes separadores caso necessário
    populacao = pd.read_csv(populacao_path, sep=',', encoding='utf-8')  
    print(f"{populacao_path}: carregado com sucesso!\n")

    # Exibir os nomes exatos das colunas
    print("Colunas disponíveis no arquivo de população:", populacao.columns.tolist())

    # Exibir os 5 municípios mais populosos
    populacao_sorted = populacao.sort_values(by=populacao.columns[1], ascending=False).head(5)
    
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Top 5 Municípios Mais Populosos", dataframe=populacao_sorted)

except Exception as e:
    print(f"Erro ao carregar {populacao_path}: {e}")
