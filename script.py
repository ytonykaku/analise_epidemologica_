import pandas as pd

def reduzir_csv(input_file, output_file, frac=0.2, sep=";", encoding="utf-8"):
    """
    Lê o CSV de entrada, seleciona uma amostra de tamanho frac (padrão 20% dos dados)
    e salva em um novo arquivo CSV.
    """
    # Carrega o CSV original
    try:
        df = pd.read_csv(input_file, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Erro ao carregar o arquivo {input_file}: {e}")
        return

    # Seleciona uma amostra aleatória com 20% dos dados
    df_reduzido = df.sample(frac=frac, random_state=42)  # random_state para reprodutibilidade

    # Salva o novo arquivo CSV
    try:
        df_reduzido.to_csv(output_file, sep=sep, encoding=encoding, index=False)
        print(f"Arquivo reduzido salvo em '{output_file}'. Total de linhas: {len(df_reduzido)}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo {output_file}: {e}")

if __name__ == '__main__':
    input_file = "dados_reduzidos.csv"         # arquivo original
    output_file = "dados_reduzidos_sampled.csv"  # arquivo reduzido
    reduzir_csv(input_file, output_file, frac=0.2)
