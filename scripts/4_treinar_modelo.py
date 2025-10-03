# Importações necessárias
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


# ETAPA 1: CONFIGURAÇÕES INICIAIS
# ---------------------------------

# O nome do modelo que vamos usar. Unsloth tem versões otimizadas.
model_name = "unsloth/gemma-2b-it" 
# Caminho para o nosso dataset formatado
dataset_path = "../data/dataset_formatado_para_treino.json"
# Onde vamos salvar nosso modelo treinado (os adaptadores LoRA)
output_dir = "LLM/gemma-2b-amazon-titles"

# ETAPA 2: CARREGAR O MODELO E TOKENIZER
print("Carregando o modelo...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
print("Modelo carregado com sucesso!")

# <<< patch para GPUs antigas (GTX 1050) >>>
try:
    # Algumas builds expõem aqui (GitHub)
    from cut_cross_entropy.transformers import cce_patch
except ImportError:
    # PyPI expõe no top-level
    from cut_cross_entropy import cce_patch

cce_patch("gemma", impl="torch_compile")
print("CCE patch aplicado (torch_compile).")

# ETAPA 3: CONFIGURAR O MODELO PARA TREINAMENTO (LoRA)
print("Configurando o modelo para o treinamento com LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)


# ETAPA 4: CARREGAR O DATASET
# -----------------------------
print(f"Carregando o dataset de '{dataset_path}'...")
dataset = load_dataset("json", data_files=dataset_path, split="train")
print("Dataset carregado!")


# ETAPA 5: CONFIGURAR OS ARGUMENTOS DE TREINAMENTO
# ------------------------------------------------
# Estes são os hiperparâmetros do nosso treinamento.
# Foram escolhidos para MAXIMIZAR a chance de caber em 4GB VRAM.
training_args = TrainingArguments(
    per_device_train_batch_size = 1, # !! MUITO IMPORTANTE para VRAM baixa.
    gradient_accumulation_steps = 8, # Simula um batch size maior (2*4=8) sem usar mais VRAM.
    warmup_steps = 5, # Número de passos para aquecer o learning rate.
    max_steps = 60, # Número máximo de passos de treinamento. Comece com pouco para testar. Aumente depois.
    learning_rate = 2e-4, # Taxa de aprendizado.
    fp16 = not torch.cuda.is_bf16_supported(), # Usa 16-bit precision (economiza memória).
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1, # A cada quantos passos ele mostra o log (loss).
    optim = "adamw_8bit", # Otimizador que também economiza memória.
    weight_decay = 0.01,
    lr_scheduler_type = "linear", # Tipo de agendador da taxa de aprendizado.
    seed = 3407,
    output_dir = output_dir, # Pasta para salvar os checkpoints.
)

# ETAPA 6: CRIAR E INICIAR O TREINADOR
# --------------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # A coluna do nosso dataset que tem o prompt formatado.
    max_seq_length = 1048,
    dataset_num_proc = 0,
    packing = False, # Manter False para datasets de instrução.
    args = training_args,
)

print("Iniciando o treinamento...")
trainer.train()
print("Treinamento concluído!")


# ETAPA 7: SALVAR O MODELO
# --------------------------
print(f"Salvando o modelo treinado em '{output_dir}'...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Modelo salvo com sucesso!")