# scripts/5_treinar_modelo.py
from pathlib import Path
from shutil import disk_usage
import os, sys, traceback, torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
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

    # --- paths ---
    ROOT = Path(__file__).resolve().parents[1]
    MODEL_DIR = ROOT / "LLM" / "gemma-2-2b-it"
    DATASET   = ROOT / "data" / "dataset_formatado_para_treino.json"
    OUT_DIR   = ROOT / "LLM" / "gemma-2b-it-amazon-titles"

    # Cache local (fora de OneDrive)
    CACHE_DIR = ROOT / ".hf_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)

    # Garante que OUT_DIR exista desde o início
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total, used, free = disk_usage(str(ROOT))
    print(f"Disco livre (GB): {free/1e9:.2f}")

    print("Modelo em:", MODEL_DIR)
    print("Dataset em:", DATASET)
    assert MODEL_DIR.is_dir(), f"[ERRO] Diretório do modelo não existe: {MODEL_DIR}"
    assert DATASET.exists(),   f"[ERRO] Dataset não encontrado: {DATASET}"

    # Evita fragmentação e paralelismo de tokenizer
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"

    # --- TOKENIZER ---
    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer OK.")

    # --- DATA COLLATOR ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- MODELO ---
    print("Carregando modelo (FP16)…")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.float16,
            device_map="auto",          # offload automático CPU+GPU
            local_files_only=True,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False  # economiza VRAM no treino
    except RuntimeError as e:
        print("[ERRO] Falha ao carregar o modelo (possível OOM).")
        print(e)
        sys.exit(1)
    print("Modelo OK.")

    # --- LoRA ---
    print("Aplicando LoRA…")
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    print("LoRA OK.")
    print_trainable_params(model)

    # (Pistas para versões antigas do Trainer respeitarem device map)
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
    QUICK_TEST = False  # True para validar o pipeline rapidamente
    if QUICK_TEST:
        limit = min(5000, len(dataset))
        dataset = dataset.select(range(limit))
        print(f"[QUICK_TEST] usando subset com {len(dataset)} exemplos.")

    # --- TREINO ---
    print("Preparando TrainingArguments…")
    MAX_SEQ_LEN = 96 if QUICK_TEST else 128
    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=5,
        max_steps=30 if QUICK_TEST else 300,  # aumente depois
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
        # sem place_model_on_device (não existe na sua versão)
    )

    try:
        print("Criando SFTTrainer…")
        trainer = NoMoveSFTTrainer(  # <— não move o modelo
            model=model,
            train_dataset=dataset,
            args=args,
            tokenizer=tokenizer,
            data_collator=data_collator,
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
