import shutil
import re
import argparse
import os
import json # Importamos a biblioteca JSON para fazer a validação

"""
corrige_dados.py

Ferramenta de linha de comando para corrigir e validar arquivos JSON Lines (JSONL).

Aplica as seguintes correções em sequência:
1. Detecta e remove a última linha se estiver truncada/inválida.
2. Remove vírgulas sobrando antes de ']' e '}'.
3. Corrige aspas de abertura faltantes em valores de string (ex: ": , " -> ": " ").
"""

# --- Funções de Correção (Trabalham com uma única linha) ---

def corrige_virgulas(linha):
    """Recebe uma string (linha) e remove vírgulas sobrando."""
    linha_corrigida = re.sub(r',\s*]', ']', linha)
    linha_corrigida = re.sub(r',\s*}', '}', linha_corrigida)
    return linha_corrigida

def corrige_aspas(linha):
    """Recebe uma string (linha) e corrige o padrão de aspas faltantes."""
    linha_corrigida = linha.replace(': , ', ': "')
    return linha_corrigida


# --- Funções Principais (Orquestradores) ---

def faz_backup(caminho_entrada, caminho_backup):
    """Cria uma cópia de segurança do arquivo de entrada."""
    try:
        shutil.copy(caminho_entrada, caminho_backup)
        print(f"✅ Backup do arquivo original criado em: '{caminho_backup}'")
        return True
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo de entrada '{caminho_entrada}' não foi encontrado.")
        return False
    except Exception as e:
        print(f"❌ Erro ao criar o backup: {e}")
        return False

def processa_arquivo(caminho_entrada, caminho_saida):
    """
    Abre os arquivos, valida a última linha, lê linha por linha e aplica todas as correções.
    """
    # --- ETAPA DE VALIDAÇÃO ---
    print("INFO: Lendo e validando o arquivo de entrada...")
    try:
        with open(caminho_entrada, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo '{caminho_entrada}': {e}")
        return

    if not linhas:
        print("AVISO: Arquivo de entrada está vazio. Nada a fazer.")
        # Cria um arquivo de saída vazio para consistência
        open(caminho_saida, 'w').close()
        return
    
    # Verifica se a última linha é um JSON válido
    ultima_linha = linhas[-1].strip()
    if ultima_linha: # Só verifica se a linha não estiver em branco
        try:
            json.loads(ultima_linha)
        except json.JSONDecodeError:
            print(f"AVISO: A última linha (Nº {len(linhas)}) parece estar truncada ou inválida. Ela será removida do processamento.")
            linhas.pop() # Remove a última linha da lista

    # --- ETAPA DE CORREÇÃO E ESCRITA ---
    print(f"⚙️  Processando {len(linhas)} linhas válidas...")
    correcoes_feitas = 0
    
    with open(caminho_saida, 'w', encoding='utf-8') as arquivo_saida:
        for num, linha_original in enumerate(linhas, 1):
            linha_processada = linha_original

            # Encadeamento das Correções
            linha_processada = corrige_virgulas(linha_processada)
            linha_processada = corrige_aspas(linha_processada)
            
            if linha_original != linha_processada:
                correcoes_feitas += 1

            arquivo_saida.write(linha_processada)
            
    print("👍 Processamento concluído!")
    print(f"   - {len(linhas)} linhas processadas.")
    print(f"   - {correcoes_feitas} linha(s) tiveram correções aplicadas.")
    print(f"✅ Arquivo final salvo em: '{caminho_saida}'")


# --- Ponto de Entrada do Script (Lida com a Interface de Comando) ---

if __name__ == "__main__":
    # (O código aqui permanece o mesmo da versão anterior)
    exemplo_uso = """
Exemplo de uso:
  python corrige_dados.py meu_arquivo.json -o meu_arquivo_corrigido.json
"""
    parser = argparse.ArgumentParser(
        description="Corrige e valida erros de formatação em arquivos JSON Lines.",
        epilog=exemplo_uso,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arquivo_entrada", help="Caminho para o arquivo de entrada a ser corrigido.")
    parser.add_argument("-o", "--output", help="(Opcional) Caminho para o arquivo de saída. Se não for fornecido, será criado com o sufixo '_corrigido'.")
    args = parser.parse_args()

    arquivo_entrada = args.arquivo_entrada
    if args.output:
        arquivo_saida = args.output
    else:
        nome_base, extensao = os.path.splitext(arquivo_entrada)
        arquivo_saida = f"{nome_base}_corrigido{extensao}"
    
    arquivo_backup = f"{arquivo_entrada}.bak"

    if faz_backup(arquivo_entrada, arquivo_backup):
        processa_arquivo(arquivo_entrada, arquivo_saida)