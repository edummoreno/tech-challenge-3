# tech-challenge-3




- utilizamos o gemma-3-4b-it-Q8_0.gguf

## Datasets

### Correção e Limpeza de Dados.

#### pre limpeza
- normaliza_dado.py

#### apaga linhas desnecessarias.
- apaga_linhas.py

# Tech Challenge FASE 03: Fine-Tuning do Gemma para Perguntas e Respostas de Produtos

## 🎯 Objetivo do Projeto

Este projeto realiza o fine-tuning de um modelo de linguagem (LLM) para atuar como um especialista em produtos. O modelo foi treinado para receber o título de um produto da Amazon e retornar sua descrição detalhada, baseado no dataset "The Amazon Titles-1.3MM".

## 🛠️ Tecnologias e Modelos Utilizados

- **Modelo Base:** Google Gemma (versão quantizada GGUF `gemma-3-4b-it-Q8_0.gguf`)
- **Dataset:** The Amazon Titles-1.3MM (`trn.json`)
- **Bibliotecas Principais:** Pandas, Hugging Face Transformers, ctransformers, PyTorch
- **Ambiente de Treinamento:** Google Cloud

## 📚 LLM models

- [hunggingface](https://huggingface.co/)
- [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- [modulo gguf quantizado](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)

## 📂 Estrutura do Repositório

- `data/`: Contém o dataset original e os dados tratados.
- `scripts/`:
    - `normaliza_dado.py`: Script para limpeza e normalização inicial dos dados.
    - `apaga_linhas.py`: Script para filtrar e remover dados desnecessários.
- `notebooks/`:
    - `Fine_Tuning_Gemma.ipynb`: Notebook principal com todo o processo de fine-tuning.
- `README.md`: Este documento.

## 🚀 Como Executar

1.  Clone o repositório: `git clone ...`
2.  Instale as dependências: `pip install -r requirements.txt`
3.  Execute o notebook `notebooks/Fine_Tuning_Gemma.ipynb` para ver o processo completo.