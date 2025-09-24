import shutil
import re
import argparse
import os
import json
import time
"""
corrige_dados.py

Ferramenta de linha de comando para corrigir e validar arquivos JSON Lines (JSONL).

Aplica as seguintes correções em sequência:
1. Detecta e remove a última linha se estiver truncada/inválida.
2. Remove vírgulas sobrando antes de ']' e '}'.
3. Corrige aspas de abertura faltantes em valores de string (ex: ": , " -> ": " ").
"""



def escapa_aspas_internas_de_forma_segura(linha):
    def sub_funcao_de_correcao(match):
        parte_chave_e_separador = match.group(1)
        conteudo_valor = match.group(3)
        conteudo_corrigido = conteudo_valor.replace('""', '\\"')
        return f'{parte_chave_e_separador}"{conteudo_corrigido}"'

    # AQUI ESTÁ A MUDANÇA: adicionamos ""| na regex do conteúdo
    padrao = r'("([^"]+)":\s*)"((?:\\"|""|[^"])*)"'
    
    return re.sub(padrao, sub_funcao_de_correcao, linha)

def corrige_listas_como_string(linha):
    """
    Usa regex para corrigir listas que foram salvas inteiramente como strings.
    Ex: conserta ' ": "[1, 2]" ' para ' ": [1, 2] '
    """
    # O padrão encontra:
    # (:\s*)     - Grupo 1: Os dois-pontos e espaços
    # "(\[.*\])" - Grupo 2: Uma string que começa com '[' e termina com ']'
    # A substituição mantém o grupo 1 e o conteúdo da string (grupo 2) sem as aspas.
    linha_corrigida = re.sub(r'(:\s*)"(\[.*\])"', r'\1\2', linha)
    return linha_corrigida

# --- Funções de Correção (Trabalham com uma única linha) ---
def corrige_listas_iniciadas_como_string(linha):
    """
    Usa regex para corrigir listas cujo primeiro elemento é uma string
    que contém o colchete de abertura.
    Ex: conserta ' "[12", 13 ' para ' [12, 13 '
    """
    # O padrão procura por:
    # \"\[([0-9\.]+)  - Uma aspas, um colchete, e captura um ou mais dígitos/pontos (o número).
    # \"             - Seguido por uma aspas de fechamento.
    # A substituição coloca o colchete fora, e mantém apenas o número capturado.
    try:
        linha_corrigida = re.sub(r'\"\[([0-9\.]+)\"', r'[\1', linha)
        return linha_corrigida
    except Exception:
        return linha
    
def corrige_virgulas(linha):
    """
    Recebe uma string (linha) e remove vírgulas sobrando antes de ']' e '}'
    usando regex, que é mais robusto que .replace().
    """
    # Esta regra encontra uma vírgula, seguida por espaços (ou não), 
    # seguida por '}' ou ']' e remove a vírgula e os espaços.
    linha_corrigida = re.sub(r',\s*(}|])', r'\1', linha)
    return linha_corrigida

def retira_barras(linha):
    ###Recebe uma strin (linha) e retira barras
    linha_corrigida = linha.replace('\\\\', ' ')
    return linha_corrigida

def corrige_strings_sem_aspas(linha):
    """
    Usa regex para encontrar e corrigir valores de texto que não estão entre aspas.
    Exemplo: conserta ' "title": Mansell ' para ' "title": "Mansell" '.
    """
    # O padrão procura por:
    # (:\s*)      - Grupo 1: Os dois-pontos e qualquer espaço depois dele.
    # ([^\s"}.][^,}]*) - Grupo 2: O valor em si. Começa com algo que não é espaço, aspas ou '}',
    #              e continua até encontrar uma vírgula ou '}'.
    # Nós usamos re.sub para substituir o padrão encontrado por:
    # Grupo 1 + aspas + Grupo 2 + aspas
    try:
        linha_corrigida = re.sub(r'(:\s*)([^\s"}.][^,}]+)', r'\1"\2"', linha)
        return linha_corrigida
    except Exception:
        # Se o regex falhar por algum motivo, retorna a linha original para evitar quebrar
        return linha

# --- Funções Principais (Orquestradores) ---

"""
def faz_backup(caminho_entrada, caminho_backup):
    ###Cria uma cópia de segurança do arquivo de entrada.
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
"""
def processa_arquivo(caminho_entrada, caminho_saida):
    
    ###Abre os arquivos, valida a última linha, lê linha por linha e aplica todas as correções.
    
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
            
            # --- Dentro da função processa_arquivo ---

            # ETAPA 1: Correções estruturais de "larga escala".
            linha_processada = corrige_listas_como_string(linha_processada) # NOVO!
            linha_processada = corrige_listas_iniciadas_como_string(linha_processada)
            linha_processada = corrige_strings_sem_aspas(linha_processada)

            # ETAPA 2: Correções de conteúdo e sintaxe fina.
            linha_processada = escapa_aspas_internas_de_forma_segura(linha_processada) # ATUALIZADO!
            linha_processada = corrige_virgulas(linha_processada) # Use a versão com regex!

            # ETAPA 3: Transformações finais de conteúdo.
            linha_processada = retira_barras(linha_processada)
                
            if linha_original != linha_processada:
                correcoes_feitas += 1

            try: 
                json.loads(linha_processada)
                arquivo_saida.write(linha_processada)
            except json.JSONDecodeError:
                print(f"erro na linha de numero {num}")
                print(linha_processada)

                time.sleep(600)
                continue

            
    print("👍 Processamento concluído!")
    print(f"   - {len(linhas)} linhas processadas.")
    print(f"   - {correcoes_feitas} linha(s) tiveram correções aplicadas.")
    print(f"✅ Arquivo final salvo em: '{caminho_saida}'")


# --- Ponto de Entrada do Script (Lida com a Interface de Comando) ---

if __name__ == "__main__":
    exemplo_uso = """
Exemplo de uso:
  python corrige_dados.py meu_arquivo.json -o meu_arquivo_1.json
"""
    parser = argparse.ArgumentParser(
        description="Corrige e valida erros de formatação em arquivos JSON Lines.",
        epilog=exemplo_uso,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arquivo_entrada", help="Caminho para o arquivo de entrada a ser corrigido.")
    parser.add_argument("-o", "--output", help="(Opcional) Caminho para o arquivo de saída. Se não for fornecido, será criado com o sufixo '_1'.")
    args = parser.parse_args()

    arquivo_entrada = args.arquivo_entrada
    if args.output:
        arquivo_saida = args.output
    else:
        nome_base, extensao = os.path.splitext(arquivo_entrada)
        arquivo_saida = f"{nome_base}_1{extensao}"
    
    #arquivo_backup = f"{arquivo_entrada}.bak"

    #if faz_backup(arquivo_entrada, arquivo_backup):
    processa_arquivo(arquivo_entrada, arquivo_saida)