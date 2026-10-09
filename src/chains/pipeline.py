"""Pipeline da Etapa 4 (LangChain): une guardrail determinístico, busca de
protocolo com citação garantida (Etapa 3) e dados do paciente para responder
com segurança, contexto e fonte sempre corretos.

Ordem das etapas, cada uma podendo encerrar o pipeline antes do LLM:
  1. Guardrail (src/chains/guardrails.py) — se a pergunta pedir prescrição,
     dose, assinatura, alta ou "ignore o protocolo", a resposta sai daqui,
     o LLM nunca é chamado, e a fonte é sempre HSF-POL-001.
  2. Leito (se informado) — busca o paciente no prontuário (Etapa 3). Leito
     inexistente também encerra aqui, sem chamar o LLM.
  3. Busca do protocolo (Etapa 3) — por palavra-chave na pergunta, com
     fallback para o protocolo já associado ao paciente.
  4. LLM (Etapa 2) — só compõe o texto, a partir do contexto já verificado.
  5. Correção de citação — depois que o LLM responde, TODA citação que ele
     escreveu é removida e substituída pela fonte real da busca (passo 3).
     Resolve o problema medido na Etapa 2 (71% de código errado) na
     arquitetura, não esperando que o fine-tuning acerte sozinho.
  6. Auditoria — toda consulta é registrada (src/db/prontuarios.py).

Implementado com LangChain (PromptTemplate + RunnableLambda em LCEL) para
atender ao requisito do PDF de orquestração via LangChain; a lógica de
segurança e de citação, porém, roda em Python puro FORA da chain — são
garantias que não podem depender de o LLM "decidir" segui-las.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
except ModuleNotFoundError:
    # langchain-core não instalado neste ambiente (sem acesso à internet para
    # `pip install`). Substituto mínimo, com a mesma interface (.from_template,
    # .invoke, operador "|"), só para não travar o desenvolvimento e os testes
    # locais. Na máquina com `pip install langchain-core`, o try acima
    # funciona e o pipeline passa a rodar com a biblioteca de verdade, sem
    # nenhuma mudança de código.
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _ValorPrompt:
        text: str

    class PromptTemplate:  # type: ignore[no-redef]
        def __init__(self, template: str):
            self.template = template

        @classmethod
        def from_template(cls, template: str) -> "PromptTemplate":
            return cls(template)

        def invoke(self, variaveis: dict) -> _ValorPrompt:
            return _ValorPrompt(self.template.format(**variaveis))

        def __or__(self, proximo):
            return _Sequencia(self, proximo)

    class RunnableLambda:  # type: ignore[no-redef]
        def __init__(self, fn):
            self.fn = fn

        def invoke(self, entrada):
            return self.fn(entrada)

    class _Sequencia:
        def __init__(self, primeiro, segundo):
            self.primeiro, self.segundo = primeiro, segundo

        def invoke(self, entrada):
            return self.segundo.invoke(self.primeiro.invoke(entrada))

from src.chains.llm_cliente import GeradorResposta
from src.guardrails.seguranca import verificar
from src.db.prontuarios import Paciente, buscar_paciente_por_leito, exames_pendentes, registrar_evento_auditoria
from src.domain.protocolos import AVISO, PROTOCOLOS
from src.rag.contexto import linha_paciente
from src.rag.retriever import BuscadorProtocolos, ResultadoBusca, protocolo_para_resultado

_POR_ID = {p["id"]: p for p in PROTOCOLOS}

# Qualquer trecho "Fonte: ... ." é removido do texto do LLM antes de responder —
# não só o errado: mesmo um acerto por sorte é indistinguível de um palpite para
# quem lê, então a única citação confiável é a acrescentada por _corrigir_citacao.
# O "(?!\d)" depois do ponto evita parar no "v1.0" (o ponto da versão é seguido
# de um dígito); só para no ponto final de verdade, no fim da frase.
_RE_FONTE_INLINE = re.compile(r"\s*Fonte:.*?\.(?!\d)", re.IGNORECASE)

# NÃO inclui {system_prompt} aqui: o papel do assistente e as regras de segurança
# já são aplicados pelo GeradorResposta (HuggingFaceGerador usa o mesmo chat
# template — sistema + usuário — da Etapa 2); aqui só o conteúdo desta pergunta.
#
# Formato deliberadamente simples (pergunta, depois texto corrido de apoio) —
# o mais perto possível do "instruction\n\ninput" visto no treino (Etapa 1).
# As perguntas de protocolo (faq_protocolo) foram treinadas SEM nenhum texto de
# apoio (o modelo aprendeu a responder "de memória"); por isso um prompt com
# marcações como "### CONTEXTO ###" ou "Pergunta do médico:" — que o modelo
# nunca viu — é um formato fora da distribuição de treino e piora a resposta,
# como observado ao testar no Colab (a correção de citação continua garantindo
# a fonte certa de qualquer forma, mas o texto gerado fica melhor com um
# prompt mais parecido com o treino). Alinhar de verdade exigiria incluir
# exemplos com contexto nas Etapas 1 e 2 — ver docs/relatorio (limitações).
_PROMPT = PromptTemplate.from_template("{pergunta}\n\n{contexto_paciente}{protocolo_texto}")


@dataclass
class RespostaPipeline:
    texto: str
    bloqueado_por_guardrail: bool
    categoria_guardrail: str | None
    fonte: str | None  # None só quando nenhum protocolo foi encontrado


def _texto_protocolo(p: dict) -> str:
    """Texto corrido (não em campos rotulados) — mais perto do que o modelo viu
    como 'input' nos exemplos de treino que usavam texto de apoio (resumo_nota)."""
    return (
        f"O protocolo interno de {p['nome']} indica: {p['indicacao']} "
        f"Exames iniciais: {', '.join(p['exames'])}. "
        f"Conduta: {' '.join(p['conduta'])} "
        f"Sinais de alerta: {', '.join(p['alertas'])}."
    )


def _resolver_protocolo(buscador: BuscadorProtocolos, pergunta: str, paciente: Paciente | None) -> ResultadoBusca | None:
    resultados = buscador.buscar(pergunta, top_k=1)
    if resultados:
        return resultados[0]
    if paciente is not None:
        protocolo = _POR_ID.get(paciente.protocolo_id)
        if protocolo:
            return protocolo_para_resultado(protocolo)
    return None


def _corrigir_citacao(texto: str, fonte: str | None) -> str:
    """Remove QUALQUER citação que o LLM tenha escrito (certa, errada ou
    inventada) e acrescenta a única fonte confiável: a da busca (Etapa 3).
    Não tenta consertar a citação do modelo — substitui por inteiro, porque
    uma citação às vezes certa é tão arriscada quanto uma sempre errada: quem
    lê não tem como saber qual é qual."""
    if fonte is None:
        return texto
    limpo = _RE_FONTE_INLINE.sub("", texto).rstrip()
    separador = "\n\n" if limpo else ""
    return f"{limpo}{separador}Fonte: {fonte}"


def _garantir_aviso(texto: str) -> str:
    return texto if AVISO in texto else f"{texto}\n{AVISO}"


class AssistenteMedico:
    """Orquestra guardrail → busca de protocolo → prontuário → LLM → correção
    de citação → auditoria. `gerador` é qualquer objeto com `.gerar(prompt)`
    (src/chains/llm_cliente.py) — troque por um stub local ou pelo modelo
    ajustado da Etapa 2 sem mudar nada aqui."""

    def __init__(self, con: sqlite3.Connection, buscador: BuscadorProtocolos, gerador: GeradorResposta):
        self.con = con
        self.buscador = buscador
        self.gerador = gerador
        self._chain = _PROMPT | RunnableLambda(lambda p: self.gerador.gerar(p.text))

    def responder(self, pergunta: str, codigo_leito: str | None = None) -> RespostaPipeline:
        bloqueio = verificar(pergunta)
        if bloqueio is not None:
            registrar_evento_auditoria(
                self.con, tipo=f"bloqueio_seguranca:{bloqueio.categoria}",
                conteudo=f"pergunta={pergunta!r} leito={codigo_leito!r}", origem="assistente",
            )
            return RespostaPipeline(bloqueio.resposta, True, bloqueio.categoria, None)

        paciente = buscar_paciente_por_leito(self.con, codigo_leito) if codigo_leito else None
        if codigo_leito and paciente is None:
            texto = f"Não encontrei o leito {codigo_leito} na base de prontuários.\n\n{AVISO}"
            registrar_evento_auditoria(self.con, tipo="leito_nao_encontrado", conteudo=f"leito={codigo_leito!r}", origem="assistente")
            return RespostaPipeline(texto, False, None, None)

        resultado = _resolver_protocolo(self.buscador, pergunta, paciente)
        if resultado is None:
            texto = (
                "Não encontrei um protocolo interno que corresponda a essa pergunta. "
                f"Reformule com o nome da condição clínica.\n\n{AVISO}"
            )
            registrar_evento_auditoria(self.con, tipo="protocolo_nao_encontrado", conteudo=f"pergunta={pergunta!r}", origem="assistente")
            return RespostaPipeline(texto, False, None, None)

        contexto_paciente = ""
        if paciente is not None:
            pendentes = exames_pendentes(self.con, paciente.id)
            contexto_paciente = f"Dados do paciente: {linha_paciente(paciente, pendentes)}\n\n"

        bruto = self._chain.invoke({
            "contexto_paciente": contexto_paciente,
            "protocolo_texto": _texto_protocolo(resultado.protocolo),
            "pergunta": pergunta,
        })
        texto = _garantir_aviso(_corrigir_citacao(bruto, resultado.fonte))

        registrar_evento_auditoria(
            self.con, tipo="consulta_protocolo",
            conteudo=f"pergunta={pergunta!r} leito={codigo_leito!r} fonte={resultado.fonte!r}",
            origem="assistente",
        )
        return RespostaPipeline(texto, False, None, resultado.fonte)
