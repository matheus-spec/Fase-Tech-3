# MedAssist – Tech Challenge Fase 3

Assistente médico com LLM ajustado (fine-tuning), LangChain e LangGraph.
Todos os dados do repositório são **sintéticos e didáticos**; não use em atendimento real.

## Status
- [x] Etapa 1 – Dados: geração sintética, anonimização, curadoria e divisão
- [ ] Etapa 2 – Fine-tuning (LoRA/QLoRA, Colab)
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

## Estrutura
```
src/preprocessing/   anonymizer.py, synthetic_data.py, build_dataset.py
tests/               testes unitários
data/raw|processed/  dados brutos sintéticos e dataset final
docs/                relatório técnico (etapa 7)
```
