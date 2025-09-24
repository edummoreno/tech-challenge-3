import json
import argparse
import os


def processa_arquivo(caminho_entrada, caminho_saida):

    print("INFO: Lendo e validando o arquivo de entrada...")
    try:
        with open(caminho_entrada, 'r', encoding='utf-8') as infile, \
            open(caminho_saida, 'w', encoding='utf-8') as outfile:
    
            linhas_mantidas = 0
            linhas_descartadas = 0

            ###so mantem linhas que title e content tem conteudo.
            for line in infile:
                data = json.loads(line)
                if not data.get('title') or not data.get('content'):

                    linhas_descartadas += 1
                else:
                    outfile.write(line)
                    linhas_mantidas += 1

        print("Concluido")
        print(f"{linhas_mantidas} linhas mantidas e {linhas_descartadas} linhas descartadas")

    except FileNotFoundError:
        print(f"{arquivo_entrada} não encontrado")
    except Exception as e:
        print(f"Ocorreu um erro {e}")


if __name__ == "__main__":
    exemplo_uso = """
Exemplo de uso:
  python corrige_linhas.py meu_arquivo.json -o meu_arquivo_2.json
"""
    parser = argparse.ArgumentParser(
        description="Corrige e valida erros de formatação em arquivos JSON Lines.",
        epilog=exemplo_uso,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arquivo_entrada", help="Caminho para o arquivo de entrada a ser corrigido.")
    parser.add_argument("-o", "--output", help="(Opcional) Caminho para o arquivo de saída. Se não for fornecido, será criado com o sufixo '_2'.")
    args = parser.parse_args()

    arquivo_entrada = args.arquivo_entrada
    if args.output:
        arquivo_saida = args.output
    else:
        nome_base, extensao = os.path.splitext(arquivo_entrada)
        if nome_base.endswith('_1'):
            nome_base = nome_base[:-2] + '_2'
        else:
            nome_base = nome_base + '_2'

        arquivo_saida = f"{nome_base}{extensao}"

    processa_arquivo(arquivo_entrada, arquivo_saida)