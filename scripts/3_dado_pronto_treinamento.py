import pandas as pd

# 1. CARREGAMENTO DOS SEUS DADOS JÁ LIMPOS
# Altere o caminho se necessário.
# Este script assume que 'trn_2.json' está na pasta 'data'.
json_path = 'data/trn_2.json'

try:
    df = pd.read_json(json_path, lines=True)
    print(f"Arquivo '{json_path}' carregado com sucesso! Contém {len(df)} linhas.")
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em '{json_path}'")
    exit()

# Garante que as colunas esperadas existem
if 'title' not in df.columns or 'content' not in df.columns:
    print("Erro: O JSON deve conter as colunas 'title' e 'content'.")
    exit()

# 2. FORMATAÇÃO DO PROMPT
# O template que ensina o modelo a seguir uma instrução.
# <bos>/<eos> são tokens especiais que marcam o início e fim de um exemplo.
prompt_template = """<bos><start_of_turn>user
Gere uma descrição para o seguinte produto:
### Título:
{}<end_of_turn>
<start_of_turn>model
{}<eos>"""

print("Formatando os dados no padrão de prompt...")
# Cria a nova coluna 'formatted_prompt' aplicando o template
df['text'] = df.apply(
    lambda row: prompt_template.format(row['title'], row['content']),
    axis=1
)

# 3. SALVAR O DATASET FINAL PARA TREINAMENTO
# Vamos salvar em formato JSON, que é muito usado com a biblioteca 'datasets'.
# Usamos 'orient="records"' para salvar no formato JSON Lines.
output_path = 'data/dataset_formatado_para_treino.json'
df[['text']].to_json(output_path, orient='records', lines=True, force_ascii=False)


# 4. VERIFICAÇÃO
print("\n--- Exemplo de um prompt formatado ---")
# Imprime o primeiro exemplo formatado para você ver como ficou
print(df['text'].iloc[0])

print(f"\nProcesso concluído! O dataset para treinamento foi salvo em:")
print(f"'{output_path}'")