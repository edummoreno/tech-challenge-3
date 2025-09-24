import json
import argparse
import os
import re

def tenta_resgatar_linha(linha):
    """
    Aplica uma série de correções robustas, na ordem mais segura possível,
    para tentar recuperar uma linha de JSON mal formatada.
    """
    linha_corrigida = linha.strip()
    if not linha_corrigida:
        return "" # Retorna vazio se a linha estiver em branco

    # --- ETAPA 1: Lidar com aspas e escapes, a fonte mais comum de erros ---

    # Correção 1 (Segura): Padrão "" -> \"
    # Só mexe em aspas duplas, que é um padrão comum de exportação errada.
    linha_corrigida = linha_corrigida.replace('""', '\\"')

    # Correção 2 (Segura): Listas salvas como strings -> : "[...]" para : [...]
    # Usa regex para remover as aspas que englobam uma lista inteira.
    linha_corrigida = re.sub(r'(:\s*)"(\[.*\])"', r'\1\2', linha_corrigida)

    # --- ETAPA 2: Tentar uma primeira passagem de validação ---
    try:
        json.loads(linha_corrigida)
        return linha_corrigida # Se funcionou, a linha está boa! Retorna.
    except json.JSONDecodeError:
        # Se falhou, a linha tem erros mais profundos. Vamos para as correções "agressivas".
        pass

    # --- ETAPA 3: Correções agressivas (só rodam se a Etapa 2 falhar) ---

    # Correção 3 (Agressiva): Tenta colocar aspas em valores "soltos"
    # Ex: : Mansell, -> : "Mansell",
    try:
        linha_corrigida = re.sub(r'(:\s*)([^\s"\[{].*?[^\s])(\s*(?:,|}))', r'\1"\2"\3', linha_corrigida)
    except Exception:
        pass

    # Correção 4 (Agressiva): Tenta consertar o início de lista corrompido
    # Ex: "[12", -> [12,
    # Esta regra é mais específica e segura que a anterior.
    linha_corrigida = re.sub(r'\"\[([0-9\.]+)\",', r'[\1,', linha_corrigida)

    return linha_corrigida


def processa_arquivo(caminho_entrada, caminho_saida):
    print(f"INFO: Iniciando RESGATE de dados de {caminho_entrada}...")
    
    linhas_mantidas = 0
    linhas_descartadas_vazias = 0
    linhas_com_erro_final = 0

    with open(caminho_entrada, 'r', encoding='utf-8') as infile, \
         open(caminho_saida, 'w', encoding='utf-8') as outfile:

        for i, linha_original in enumerate(infile, 1):
            
            linha_resgatada = tenta_resgatar_linha(linha_original)
            if not linha_resgatada:
                continue

            try:
                data = json.loads(linha_resgatada)

                if data.get('title') and data.get('content'):
                    # Escreve a linha resgatada e formatada.
                    # json.dumps garante que a saída seja um JSONL perfeito.
                    outfile.write(json.dumps(data) + '\n')
                    linhas_mantidas += 1
                else:
                    linhas_descartadas_vazias += 1

            except json.JSONDecodeError:
                # Se mesmo após todas as tentativas a linha falhar, nós a descartamos.
                linhas_com_erro_final += 1
                if i < 20: # Mostra as 20 primeiras linhas com erro para diagnóstico
                    print(f"  - Erro final na linha {i}: {linha_original.strip()[:100]}...")


    print("\n✅ Processo de Resgate Concluído!")
    print(f"   - Linhas resgatadas e salvas: {linhas_mantidas}")
    print(f"   - Linhas descartadas (title/content vazio): {linhas_descartadas_vazias}")
    print(f"   - Linhas com formato irrecuperável: {linhas_com_erro_final}")
    print(f"✅ Arquivo final salvo em: '{caminho_saida}'")

# O if __name__ == "__main__" continua o mesmo.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ferramenta de resgate e filtragem para arquivos JSON Lines corrompidos.")
    parser.add_argument("arquivo_entrada", help="Caminho do arquivo de entrada a ser processado.")
    parser.add_argument("-o", "--output", help="Caminho do arquivo de saída. (Opcional)")
    args = parser.parse_args()
    # ... (resto do código para definir arquivo_saida)
    arquivo_entrada = args.arquivo_entrada
    if args.output:
        arquivo_saida = args.output
    else:
        nome_base, extensao = os.path.splitext(arquivo_entrada)
        arquivo_saida = f"{nome_base}_resgatado{extensao}"
    processa_arquivo(arquivo_entrada, arquivo_saida)