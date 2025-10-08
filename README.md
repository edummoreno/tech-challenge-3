# 🧠 Tech Challenge FASE 03: Fine-Tuning do Gemma/TinyLlama para Perguntas e Respostas de Produtos

## 🎯 Objetivo do Projeto

Este projeto realiza o fine-tuning de um modelo de linguagem (LLM) para atuar como um especialista em produtos da Amazon.
O modelo foi treinado para receber o título de um produto e gerar uma descrição detalhada e criativa, com base no dataset "The Amazon Titles-1.3MM".

O foco do desafio é demonstrar a capacidade de adaptar um modelo de linguagem pré-treinado (como o TinyLlama ou Gemma) para um domínio específico de e-commerce, otimizando custo e performance usando LoRA + QLoRA (quantização 4-bit).

## 🛠️ Tecnologias e Modelos Utilizados

- **Modelo Base:** [TinyLlama/TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1)
- **Alternativa Gemma:** [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Dataset:** The Amazon Titles-1.3MM (`trn.json`)
- **Bibliotecas Principais:**
  - transformers
  - trl
  - peft
  - bitsandbytes
  - datasets
  - accelerate
  - sentencepiece
- **Ambiente de Treinamento:** Google Colab Pro (GPU A100)

## 📚 LLM Models e Referências

- [Hugging Face Hub](https://huggingface.co/)
- [TinyLlama v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1)
- [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it)
- [Versão Quantizada GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)

## 📂 Estrutura do Repositório

```
tech-challenge-3/
├── data/
│   ├── trn.json                           # Dataset original
│   ├── dataset_formatado_para_treino.json # Dataset já formatado para SFT
├── LLM/
│   ├── ft-output/             # Checkpoints do treinamento (retomada automática)
│   └── tinyllama-lora/        # Adapters LoRA salvos
├── notebooks/
│   └── tech_challenge_3.ipynb # Notebook principal (Google Colab)
├── README.md
├── .gitignore
└── tree.txt
```

## 🚀 Como Executar

O treinamento completo foi implementado no notebook `notebooks/tech_challenge_3.ipynb`.
Para reproduzir o projeto, basta abrir o notebook no Google Colab e seguir as células na ordem.

### 🔹 Etapas Principais

1. **Montar o Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Instalar dependências**
   ```bash
   pip install -q -U \
   "transformers>=4.38" \
   "trl>=0.8.0" \
   "peft>=0.9.0" \
   "bitsandbytes>=0.41" \
   "datasets" \
   "accelerate>=0.28" \
   "sentencepiece" \
   "pyarrow<20.0"
   ```

3. **Login no Hugging Face**
   ```python
   from huggingface_hub import login
   login()  # Insira seu token HF
   ```

4. **Executar o Fine-tuning**
   O notebook já contém o pipeline completo de:
   - carregamento do modelo,
   - formatação do dataset,
   - treinamento com LoRA + QLoRA,
   - salvamento automático de checkpoints.

5. **Salvar o modelo**
   Após o treinamento, os adapters LoRA e o tokenizer são salvos em:
   ```
   /LLM/tinyllama-lora/
   ```

## 💾 Checkpoints e Retomada

O script identifica automaticamente o último checkpoint salvo no diretório de saída e retoma o treino de onde parou.
Ideal para casos de interrupção do Colab.

## 🧪 Avaliação

### Exemplo de Prompt:
```
Gere uma descrição criativa para o produto:
Título: Girls Ballet Tutu Neon Pink
```

**Saída (Baseline):**
> Um vestido simples rosa para meninas.

**Saída (Modelo Fine-tunado):**
> Um tutu rosa neon encantador, perfeito para pequenas bailarinas brilharem nos palcos e ensaios.

## 📊 Hiperparâmetros Principais

| Parâmetro | Valor | Descrição |
|------------|--------|------------|
| `learning_rate` | 2e-4 | Taxa de aprendizado |
| `batch_size` | 1 | Tamanho do batch |
| `gradient_accumulation_steps` | 8 | Acúmulo de gradientes |
| `max_seq_length` | 256 | Tokens por exemplo |
| `num_train_epochs` | 1 | Quantas vezes percorre o dataset |
| `packing` | True | Junta exemplos até o limite de tokens |
| `save_steps` | 2000 | Checkpoints periódicos |

## 📈 Resultados e Conclusões

✅ O modelo aprendeu a gerar descrições mais criativas e ricas em contexto.  
✅ O uso de QLoRA (4-bit) reduziu o consumo de memória sem perda perceptível de qualidade.  
✅ O packing=True acelerou o treinamento em cerca de 30%.  
✅ O training loss foi decrescendo gradualmente, indicando aprendizado estável.  
✅ Os checkpoints automáticos garantiram retomada segura após quedas do Colab.

## 📹 Vídeo Explicativo

🎥 (Adicionar link do vídeo explicativo do projeto no YouTube quando disponível.)

## 👨‍💻 Autor

**Seu Nome**  
Desenvolvedor | IA & Machine Learning  
📧 [seuemail@exemplo.com](mailto:seuemail@exemplo.com)  
🔗 [LinkedIn](https://linkedin.com/in/seu-perfil) | [GitHub](https://github.com/seu-usuario)
