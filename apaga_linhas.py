import json
import argparse
import os


def processa_arquivo(caminho_entrada, caminho_saida):

    print(f"INFO: Lendo, filtrando e formatando o arquivo: {caminho_entrada}...")
    try:
        with open(caminho_entrada, 'r', encoding='utf-8') as infile, \
             open(caminho_saida, 'w', encoding='utf-8') as outfile:
    
            CONTEUDO_INVALIDO = ['\\', '"']
            linhas_mantidas = 0
            linhas_descartadas = 0

            for line in infile:
                data = json.loads(line)

                title_limpo = (data.get('title') or "").strip()
                content_limpo = (data.get('content') or "").strip()

                if (not title_limpo or
                    not content_limpo or
                    title_limpo in CONTEUDO_INVALIDO or
                    content_limpo in CONTEUDO_INVALIDO):
                    
                    linhas_descartadas += 1
                else:
                    # --- ETAPA FINAL DE TRANSFORMAÇÃO ---
                    
                    # 1. Criamos um novo dicionário apenas com as chaves desejadas.
                    novo_objeto = {
                        "title": title_limpo,
                        "content": content_limpo
                    }
                    
                    # 2. Convertemos este novo dicionário Python de volta para uma string JSON.
                    #    O ensure_ascii=False é MUITO importante para manter acentos e caracteres especiais.
                    nova_linha_json = json.dumps(novo_objeto, ensure_ascii=False)
                    
                    # 3. Escrevemos a nova linha no arquivo de saída, adicionando a quebra de linha '\n'
                    #    para manter o formato JSON Lines.
                    outfile.write(nova_linha_json + '\n')
                    
                    linhas_mantidas += 1

            print("\nConcluído!")
            print(f"{linhas_mantidas} linhas mantidas e formatadas. {linhas_descartadas} linhas descartadas.")
            print(f"Arquivo final para fine-tuning salvo em: {caminho_saida}")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_entrada}' não foi encontrado")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# O if __name__ == "__main__": continua o mesmo...
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