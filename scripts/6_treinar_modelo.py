# scripts/5_treinar_modelo.py
from pathlib import Path
from shutil import disk_usage
import os, sys, traceback, torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Evita qualquer tentativa de usar bitsandbytes
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

def print_header():
    print("=== Boot ===")
    print("Python:", sys.executable)
    print("Torch:", torch.__version__)
    print("CUDA disponível?", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
            print("VRAM total (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
        except Exception as e:
            print("GPU info indisponível:", e)

def print_trainable_params(model):
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100 * trainable / total if total else 0
    print(f"Parâmetros treináveis: {trainable:,} de {total:,} ({pct:.2f}%)")

# Trainer que NÃO move o modelo (evita 'Cannot copy out of meta tensor')
class NoMoveSFTTrainer(SFTTrainer):
    def _move_model_to_device(self, model, device):
        return model  # não chama model.to(device)

if __name__ == "__main__":
    print_header()

    # --- paths / modelo base (sem Instruct) ---
    ROOT = Path(__file__).resolve().parents[1]
    DATASET = ROOT / "data" / "dataset_formatado_para_treino.json"
    OUT_DIR = ROOT / "LLM" / "ft-qwen2-0.5b-amazon-titles"

    # Troque aqui se quiser outra base:
    BASE_MODEL_ID = "Qwen/Qwen2-0.5B"           # <-- base, NÃO Instruct
    # Exemplos alternativos:
    # BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B" # base
    # BASE_MODEL_ID = "microsoft/phi-1_5"        # base (ver nota de LoRA abaixo)

    # Cache local fora de pastas sincronizadas
    CACHE_DIR = ROOT / ".hf_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total, used, free = disk_usage(str(ROOT))
    print(f"Disco livre (GB): {free/1e9:.2f}")
    print("Base:", BASE_MODEL_ID)
    print("Dataset em:", DATASET)

    assert DATASET.exists(), f"[ERRO] Dataset não encontrado: {DATASET}"

    # Evita fragmentação e paralelismo de tokenizer
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"

    # --- TOKENIZER ---
    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer OK.")

    # --- MODELO ---
    print("Carregando modelo (FP16)…")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",      # offload CPU+GPU se precisar
        )
        if hasattr(model, "config"):
            model.config.use_cache = False  # economiza VRAM no treino
    except RuntimeError as e:
        print("[ERRO] Falha ao carregar o modelo (possível OOM).")
        print(e)
        sys.exit(1)
    print("Modelo OK.")

    # --- LoRA (leve p/ 1050) ---
    print("Aplicando LoRA…")
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    # Para Phi-1_5, troque assim:
    # target_modules = ["Wqkv","fc1","fc2","out_proj"]

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    print("LoRA OK.")
    print_trainable_params(model)

    # (Pistas para o Trainer respeitar device map)
    if hasattr(model, "is_parallelizable"):
        model.is_parallelizable = True
    if hasattr(model, "model_parallel"):
        model.model_parallel = True
    setattr(model, "hf_device_map", getattr(model, "hf_device_map", {"": "auto"}))

    # --- DATASET ---
    print("Carregando dataset…")
    dataset = load_dataset("json", data_files=str(DATASET), split="train")
    print("Dataset OK. Linhas:", len(dataset))
    print("Colunas:", dataset.column_names)
    assert "text" in dataset.column_names, f"[ERRO] dataset não tem coluna 'text'. Colunas: {dataset.column_names}"
    if len(dataset) > 0:
        exemplo = dataset[0]["text"].replace("\n", " ")
        print("Exemplo:", (exemplo[:120] + "…") if len(exemplo) > 120 else exemplo)

    # --- (opcional) teste rápido ---
    QUICK_TEST = False       # True p/ validar ciclo e salvar mais rápido
    if QUICK_TEST:
        limit = min(5000, len(dataset))
        dataset = dataset.select(range(limit))
        print(f"[QUICK_TEST] usando subset com {len(dataset)} exemplos.")

    # --- TREINO ---
    print("Preparando TrainingArguments…")
    MAX_SEQ_LEN = 64 if QUICK_TEST else 96   # curto p/ caber na 1050
    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=5,
        max_steps=60 if QUICK_TEST else 300,  # aumente depois
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        lr_scheduler_type="linear",
        output_dir=str(OUT_DIR),
        save_total_limit=1,
        dataloader_num_workers=0,
        report_to=[],
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        skip_memory_metrics=True,
        # (sem place_model_on_device na sua versão)
    )

    try:
        print("Criando SFTTrainer…")
        trainer = NoMoveSFTTrainer(
            model=model,
            train_dataset=dataset,
            args=args,
            tokenizer=tokenizer,
            # IMPORTANTE: não passe data_collator custom — deixe o SFT cuidar
            max_seq_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            dataset_num_proc=1,   # sem multiprocess no Windows
        )

        print("Iniciando treino…")
        out = trainer.train()
        print("Treino finalizado. state:", out)

        print("Salvando…")
        trainer.save_model(OUT_DIR)   # salva adaptadores (LoRA)
        tokenizer.save_pretrained(OUT_DIR)
        print("Concluído. OUT_DIR:", OUT_DIR)

    except Exception:
        print("[ERRO] Exceção capturada durante preparo/treino:")
        traceback.print_exc()
        sys.exit(1)
