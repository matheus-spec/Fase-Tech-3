# MedAssist – Tech Challenge Fase 3

Assistente médico com LLM ajustado (fine-tuning), LangChain e LangGraph.
Todos os dados do repositório são **sintéticos e didáticos**; não use em atendimento real.

## Status
- [x] Etapa 1 – Dados: geração sintética, anonimização, curadoria e divisão
- [x] Etapa 2 – Fine-tuning (notebook Colab pronto, aguardando execução)
- [x] Etapa 3 – Base de prontuários (SQLite) e RAG com citação de fonte
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
**Avaliação (`src/finetuning/evaluate.py`):** perplexidade só nos tokens da resposta (modelo ajustado vs. base), recusa EXPLÍCITA em pedidos de prescrição/dose/assinatura/alta/dados inexistentes com prompts fora do treino, ausência de fonte inventada e correção da fonte citada.

## Etapa 3 – Prontuários e busca com citação garantida

Resolve o problema achado na avaliação da Etapa 2: o LLM ajustado citava o *nome* certo do
protocolo, mas errava o *código* em 71% dos casos (decorar IDs alfanuméricos não é algo que
fine-tuning com poucos exemplos generalize bem). A solução não é treinar mais — é não depender
do modelo para esse dado.

- **`src/domain/protocolos.py`**: fonte única de verdade do catálogo de protocolos (usada também
  pela Etapa 1, para o dataset de fine-tuning nunca divergir do catálogo real).
- **`src/db/prontuarios.py`**: base SQLite sintética de pacientes, leitos e exames. Nunca grava
  nome, CPF ou outro identificador direto — só um `codigo_leito` (ex.: `leito-4`).
- **`src/rag/retriever.py`**: busca por palavra-chave (sem dependências externas) nos protocolos.
  A citação (`Fonte: HSF-PROT-00X vY – nome`) vem sempre do registro de dados que deu match, nunca
  de texto gerado — por construção, não tem como o código vir trocado.
- **`src/rag/contexto.py`**: une busca de protocolo + dados do paciente (se um leito for passado)
  numa única resposta, e registra cada consulta em `eventos_auditoria`. Se a pergunta não citar a
  doença mas o leito for conhecido, usa o protocolo já associado ao paciente no prontuário.

```bash
python -m src.db.prontuarios --db data/processed/prontuarios.db --pacientes 20
python -m unittest discover -s tests -t .
```

## Estrutura
```
src/domain/          protocolos.py (catálogo único, usado pela Etapa 1 e pela Etapa 3)
src/preprocessing/   anonymizer.py, synthetic_data.py, build_dataset.py
src/finetuning/      prompt_format.py (template único treino/inferência), evaluate.py
src/db/              prontuarios.py (SQLite sintético: pacientes, exames, auditoria)
src/rag/             retriever.py (busca com citação garantida), contexto.py
notebooks/           MedAssist_Finetuning_Etapa2.ipynb
tests/               testes unitários (32, cobrindo as 3 etapas)
data/raw|processed/  dados brutos sintéticos e dataset final
docs/                relatório técnico (etapa 7) e avaliação da etapa 2
```
