# MedAssist – Tech Challenge Fase 3

Assistente médico com LLM ajustado (fine-tuning), LangChain e LangGraph.
Todos os dados do repositório são **sintéticos e didáticos**; não use em atendimento real.

## Status
- [x] Etapa 1 – Dados: geração sintética, anonimização, curadoria e divisão
- [x] Etapa 2 – Fine-tuning (notebook Colab pronto, aguardando execução)
- [ ] Etapa 3 – Base de prontuários (SQLite) e RAG com citação de fonte
- [ ] Etapa 4 – Pipeline LangChain
- [ ] Etapa 5 – Fluxos LangGraph com validação humana
- [ ] Etapa 6 – Guardrails e logging de auditoria
- [ ] Etapa 7 – Relatório técnico, diagrama e roteiro do vídeo

## Etapa 1 – Como executar (somente biblioteca padrão, Python 3.10+)
```bash
python -m src.preprocessing.synthetic_data      # gera data/raw/hospital_sintetico.jsonl
python -m src.preprocessing.build_dataset       # anonimiza, cura e divide -> data/processed/
python -m unittest discover -s tests -t .       # testes
```
Saída: `data/processed/{train,val,test}.jsonl` e `relatorio_preparo.json`.

## Etapa 2 – Fine-tuning (rodar no Google Colab, com GPU)
Notebook: `notebooks/MedAssist_Finetuning_Etapa2.ipynb`.

1. Abra no Colab e ative uma GPU (Ambiente de execução → Alterar tipo → GPU).
2. Na Seção 2, troque `REPO_URL` pela URL do seu repositório Git (ou faça upload manual de `train.jsonl`/`val.jsonl`).
3. Execute as células em ordem.
4. Ao final, baixe `medassist-lora-adapter.zip` (pesos do adaptador) e `relatorio_avaliacao.json` (métricas).

**Modelo base:** `Qwen/Qwen2.5-1.5B-Instruct`, ajustado com QLoRA (4 bits + LoRA) — cabe numa GPU T4 gratuita.
**Avaliação (`src/finetuning/evaluate.py`):** perplexidade no teste, taxa de recusa correta em pedidos de prescrição/alta direta (segurança) e taxa de citação da fonte do protocolo (explainability).

## Estrutura
```
src/preprocessing/   anonymizer.py, synthetic_data.py, build_dataset.py
src/finetuning/      prompt_format.py (template único treino/inferência), evaluate.py
notebooks/           MedAssist_Finetuning_Etapa2.ipynb
tests/               testes unitários
data/raw|processed/  dados brutos sintéticos e dataset final
docs/                relatório técnico (etapa 7)
```
