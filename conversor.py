import csv
import re

# Nome do arquivo de entrada (TXT) e saída (CSV)
arquivo_txt = "municipios.txt"
arquivo_csv = "municipios_populacao.csv"

# Lista para armazenar os dados
dados = []

# Ler o arquivo TXT e processar os dados
with open(arquivo_txt, "r", encoding="utf-8") as file:
    for linha in file:
        linha = linha.strip()
        if linha:  # Ignorar linhas vazias
            # Separar município e população usando regex para capturar números corretamente
            match = re.match(r"(.+?):?\s([\d.,]+)", linha)
            if match:
                municipio = match.group(1).strip()
                populacao = match.group(2).replace(".", "").replace(",", "").strip()  # Remover pontos e vírgulas
                dados.append([municipio, populacao])

# Escrever os dados no CSV
with open(arquivo_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Município", "População"])  # Cabeçalhos
    writer.writerows(dados)

print(f"Arquivo '{arquivo_csv}' criado com sucesso!")
