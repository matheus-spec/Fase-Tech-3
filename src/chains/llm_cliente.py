"""Interface comum para "gerar uma resposta a partir de um prompt".

O pipeline (src/chains/pipeline.py) não sabe nem precisa saber se está falando
com o modelo ajustado de verdade ou com um stub — ele só chama `.gerar(prompt)`.
Isso permite montar e testar o pipeline inteiro numa máquina sem GPU
(com ModeloEco) e, no Colab, trocar por HuggingFaceGerador sem mudar
nenhuma linha do pipeline.
"""
from __future__ import annotations

from typing import Protocol


class GeradorResposta(Protocol):
    def gerar(self, prompt: str) -> str: ...


class ModeloEco(GeradorResposta):
    """Stub determinístico para rodar e testar o pipeline sem GPU/modelo.

    Não "inventa" nada: devolve o próprio contexto recebido no prompt,
    então serve para testar a orquestração (guardrail → busca → prompt →
    pós-processamento de citação), não a qualidade da escrita — essa parte
    só pode ser avaliada com o modelo real (Colab), como na Etapa 2.
    """

    MARCADOR_CONTEXTO_INICIO = "### CONTEXTO ###"
    MARCADOR_CONTEXTO_FIM = "### FIM DO CONTEXTO ###"

    def gerar(self, prompt: str) -> str:
        ini = prompt.find(self.MARCADOR_CONTEXTO_INICIO)
        fim = prompt.find(self.MARCADOR_CONTEXTO_FIM)
        if ini == -1 or fim == -1:
            return "Não encontrei um protocolo interno que corresponda a essa pergunta."
        return prompt[ini + len(self.MARCADOR_CONTEXTO_INICIO):fim].strip()


class HuggingFaceGerador(GeradorResposta):
    """Usa o modelo ajustado na Etapa 2 (model + tokenizer já carregados no Colab).

    Não importa torch/transformers no topo do arquivo de propósito: essa classe
    só é instanciada dentro do notebook do Colab, onde essas bibliotecas já
    estão carregadas; numa máquina sem GPU, o resto do pipeline funciona sem
    precisar instalá-las.

    Aplica o MESMO chat template (sistema + usuário) usado no treino da Etapa 2
    (src/finetuning/prompt_format.py) antes de gerar. Sem isso, o texto recebido
    do pipeline (que é só o "conteúdo", sem a estrutura de conversa) chega ao
    modelo num formato diferente do que ele foi ajustado para reconhecer, e a
    qualidade da resposta cai — foi o que se observou ao testar no Colab.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 300):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def gerar(self, prompt: str) -> str:
        from src.finetuning.prompt_format import SYSTEM_PROMPT

        mensagens = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        texto_formatado = self.tokenizer.apply_chat_template(
            mensagens, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer(texto_formatado, return_tensors="pt").to(self.model.device)
        saida = self.model.generate(
            **ids, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True,
        )
        texto = self.tokenizer.decode(saida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return texto.strip()
