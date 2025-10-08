# Tech Challenge — Fine‑tuning TinyLlama com LoRA (QLoRA)

Repositório do **Tech Challenge** para treinar o modelo [`TinyLlama/TinyLlama_v1.1`](https://huggingface.co/TinyLlama/TinyLlama_v1.1) a gerar **descrições de produtos** a partir de títulos. O pipeline roda em **Google Colab**, com **QLoRA (4‑bit)**, **pré‑tokenização**, **constant‑length packing** e **checkpoints versionados** por execução.

> **Demo**: vídeo de apresentação (**adicione o link aqui**).

---

## Visão geral

- **Base**: TinyLlama 1.1 (1.1B) carregado em 4‑bit via `bitsandbytes`.
- **Adaptação**: LoRA nas projeções de atenção e camadas MLP.
- **Dados**: `dataset_formatado_para_treino.json` (texto já no formato *instruct* com tags como `<start_of_turn>user` / `<end_of_turn>`).
- **Treino**: 1 época, `MAX_LEN=128`, *batch* efetivo 64 (8 × 8), *gradient checkpointing*, TF32.
- **Logs/artefatos**: CSV, JSON de histórico, *timing* e curva `loss x step`.

---

## Estrutura do repositório

```
.
├─ data/
│  └─ dataset_formatado_para_treino.json     # (não subir o arquivo grande)
├─ LLM/
│  ├─ ft-output/                             # saídas versionadas por execução (RUN_NAME)
│  │  └─ run-AAAAmmdd-HHMMSS-<tag>/
│  │     ├─ lora_checkpoints/                # checkpoints parciais do adapter
│  │     ├─ train_log.csv
│  │     ├─ training_history.json
│  │     ├─ timing.json
│  │     └─ train_curve.png
│  └─ tinyllama-lora/
│     └─ lora-final-AAAAmmdd-HHMMSS/         # adapter final + tokenizer
│         ├─ adapter_model.safetensors
│         ├─ adapter_config.json
│         ├─ tokenizer.json / tokenizer.model / tokenizer_config.json
│         └─ README.md
├─ tech_challenge_3.ipynb                     # notebook principal (Colab)
├─ tech_challenge_3.py                        # (opcional) export do notebook
└─ README.md
```

> Dica: mantenha `data/` fora do git ou use `.gitignore`. Se quiser preservar a estrutura vazia, crie arquivos `.gitkeep` em `LLM/ft-output/` e `LLM/tinyllama-lora/`.

---

## Reproduzindo no Colab (passo a passo)

1. **Abra o notebook** `tech_challenge_3.ipynb` no Colab com GPU ativada.
2. Execute as células **na ordem**:
   - **Célula 2–4**: montagem do Drive e *load* do dataset (com cópia para `/tmp` para I/O mais rápido).
   - **Célula 7**: define `MODEL_ID` e `MAX_LEN`, e utilitário `get_tokenizer()`.
   - **Célula 9**: configura *cache* Hugging Face (`HF_HOME`, etc.).
   - **Célula 10**: **treino** com QLoRA + packing + callbacks.
   - **Célula 11**: **inferência comparativa** (baseline x LoRA) com prompts compatíveis.
   - **Célula 12**: estatísticas e **gráfico loss x step**.
3. (Opcional) Antes da Célula 10, defina uma *tag* para versionar a saída:
   ```bash
   %env RUN_TAG=video
   ```
   O diretório do run será `LLM/ft-output/run-AAAAmmdd-HHMMSS-video/`.

### Dependências principais (instaladas no Colab)
- `torch 2.8.0+cu128`, `transformers`, `trl 0.8.0`, `peft 0.9.0`, `bitsandbytes 0.48.1`, `datasets`, `pandas`, `matplotlib`.

---

## Como funciona o treino (Célula 10)

- **Quantização 4‑bit (QLoRA)**: `BitsAndBytesConfig(load_in_4bit=True, quant_type="nf4")`.
- **LoRA**: `r=16`, `lora_alpha=16`, `lora_dropout=0.05`, *targets* `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`.
- **Pré‑tokenização**: converte `text → input_ids/attention_mask/labels`, depois **packing** concatena exemplos com `EOS` e fatia em blocos fixos (`MAX_LEN`).
- **Desempenho**: *DataLoader* com `persistent_workers=True`, `prefetch_factor` configurável e `num_workers` baseado em CPU.
- **Checkpoints & logs**: salvos automaticamente em `LLM/ft-output/run-.../` (diretório único por execução). O adapter final é salvo em `LLM/tinyllama-lora/lora-final-.../` junto com o **tokenizer** para garantir compatibilidade.

---

## Inferência (Célula 11)

A célula carrega:
- **Baseline** (TinyLlama original), com um prompt **limpo** para português.
- **Modelo afinado (LoRA)**, com o prompt no **formato das tags** usadas no treino (`<start_of_turn>user/.../<start_of_turn>model`).

Parâmetros de geração sugeridos: `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.1`, e `eos_token_id` incluindo `</s>` e `<end_of_turn>` para **parar no fim do turno**.

---

## Resultados do run de referência

- **Dados de treino (packed)**: ~102.6k exemplos  
- **Steps/época**: ~1603 — **tempo total** ≈ 90 min, **≈ 3.37 s/step**  
- **Training loss final** ≈ **1.57**  
- Curva disponível em: `LLM/ft-output/<run>/train_curve.png`

> Observação: a melhoria qualitativa é **sutil** com 1 época e `MAX_LEN=128`. Ganhos adicionais tendem a vir com **mais épocas**, **contexto maior** e prompt de inferência mais guiado.

---

## Dicas / Próximos passos

- Aumentar `MAX_LEN` para 256/512 se houver VRAM e dados longos.
- Experimentar **2–3 épocas** e *early stopping*.
- Usar **SFTTrainer + DataCollatorForCompletionOnlyLM** se quiser treinar **apenas a parte de “resposta”** sem o *prefixo* do prompt.
- Explorar *temperature* menor (0.3–0.5) para saídas mais factuais.

---

## Licenças e créditos

- Modelo **TinyLlama**: ver licença no repositório do Hugging Face.
- Este projeto usa **Hugging Face Transformers**, **TRL**, **PEFT** e **datasets**.

---

## Contato

- Autor: *Eduardo Moreno*  
- Repositório: https://github.com/edummoreno/tech-challenge-3  
- Vídeo: *(adicione o link assim que publicado)*
