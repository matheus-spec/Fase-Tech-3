# MedAssist – Tech Challenge Fase 3

Assistente médico com LLM ajustado (fine-tuning), LangChain e LangGraph.
Todos os dados do repositório são **sintéticos e didáticos**; não use em atendimento real.

## Status
- [x] Etapa 1 – Dados: geração sintética, anonimização, curadoria e divisão
- [x] Etapa 2 – Fine-tuning (notebook Colab pronto, aguardando execução)
- [x] Etapa 3 – Base de prontuários (SQLite) e RAG com citação de fonte
- [x] Etapa 4 – Pipeline LangChain (testado localmente com stub; falta ligar ao modelo real no Colab)
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

## Etapa 4 – Pipeline (LangChain)

Une as três etapas anteriores numa única classe, `AssistenteMedico` (`src/chains/pipeline.py`):

1. **Guardrail** (`src/guardrails/seguranca.py`) — bloqueio por regra, não por IA. Se a
   pergunta pedir prescrição, dose, assinatura, alta ou "ignore o protocolo", a resposta
   sai daqui e **o LLM nunca é chamado**.
2. **Leito** (se informado) — busca o paciente no prontuário (Etapa 3). Leito inexistente
   também encerra aqui, sem chamar o LLM.
3. **Busca do protocolo** (Etapa 3) — por palavra-chave, com fallback para o protocolo do
   paciente se a pergunta não citar a doença.
4. **LLM** (Etapa 2) — só compõe o texto a partir do contexto já verificado.
5. **Correção de citação** — depois do LLM responder, a linha `Fonte: ...` é
   **substituída** pela fonte real da busca, nunca pela que o LLM escreveu. É assim que o
   problema da Etapa 2 (71% de código de protocolo errado) fica resolvido na arquitetura,
   em vez de depender do fine-tuning acertar sozinho.
6. **Auditoria** — toda consulta é registrada (`eventos_auditoria`).

```bash
pip install -r requirements.txt       # só langchain-core; nada de GPU aqui
python -m src.chains.demo_cli --leito leito-1   # demonstração com um stub no lugar do LLM
python -m unittest discover -s tests -t .
```

`src/chains/llm_cliente.py` define a interface `GeradorResposta` (só precisa de um método
`.gerar(prompt) -> str`), com duas implementações: `ModeloEco` (stub determinístico, usado
nos testes e no `demo_cli`, sem precisar de GPU) e `HuggingFaceGerador` (usa o `model` e o
`tokenizer` já carregados no notebook da Etapa 2, dentro do Colab). O pipeline não muda
nada ao trocar uma pela outra.

O arquivo usa `langchain_core` quando instalado (`pip install -r requirements.txt`) e,
caso contrário, uma implementação mínima com a mesma interface (`PromptTemplate`,
`RunnableLambda`, operador `|`) — isso permite testar a lógica do pipeline sem depender de
instalar a biblioteca, sem mudar o comportamento quando ela está instalada de verdade.

## Estrutura
```
src/domain/          protocolos.py (catálogo único, usado pelas Etapas 1, 3 e 4)
src/preprocessing/   anonymizer.py, synthetic_data.py, build_dataset.py
src/finetuning/      prompt_format.py (template único treino/inferência), evaluate.py
src/db/              prontuarios.py (SQLite sintético: pacientes, exames, auditoria)
src/rag/             retriever.py (busca com citação garantida), contexto.py
src/guardrails/      seguranca.py (bloqueio por regra, antes do LLM)
src/chains/          pipeline.py (AssistenteMedico), llm_cliente.py, demo_cli.py
notebooks/           MedAssist_Finetuning_Etapa2.ipynb
tests/               testes unitários (45, cobrindo as 4 etapas)
data/raw|processed/  dados brutos sintéticos e dataset final
docs/                relatório técnico (etapa 7) e avaliação da etapa 2
requirements.txt     dependências locais (langchain-core) e as do Colab (comentadas)
```
