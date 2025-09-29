import os, json
from llama_cpp import Llama

script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

gguf_path = os.path.join(project_root, 'LLM', 'gemma-2b-it.gguf')  # ajuste o nome exato se preciso
data_path = os.path.join(project_root, 'data', 'trn_2.json')

assert os.path.isfile(gguf_path), f"GGUF não encontrado: {gguf_path}"
assert os.path.getsize(gguf_path) > 10_000_000, "GGUF parece incompleto (OneDrive?). Marque 'Sempre manter neste dispositivo'."

print(f"Usando modelo: {gguf_path}")
print(f"Carregando um único exemplo de: {data_path}")

sample_product = None
line_to_get = 250
with open(data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == line_to_get:
            sample_product = json.loads(line)
            break
if sample_product is None:
    raise RuntimeError(f"Não foi possível ler a linha {line_to_get} de {data_path}")

print("Carregando Modelo GGUF (llama-cpp-python)...")
llm = Llama(
    model_path=gguf_path,
    n_ctx=4096,
    n_gpu_layers=0,                 # CPU puro
    n_threads=os.cpu_count() or 4   # use todos os núcleos disponíveis
)
print("Modelo Carregado!")

instruction = f"Qual é a descrição para o produto com o título: '{sample_product['title']}'?"
prompt = (
    "<bos><start_of_turn>user\n"
    f"{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

print("\n--- PROMPT ENVIADO ---")
print(prompt)

print("\n--- RESPOSTA DO MODELO ---")
out = llm.create_completion(prompt, max_tokens=256, temperature=0.7)
print(out["choices"][0]["text"])
