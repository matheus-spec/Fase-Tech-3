"""Guardrail por regras (determinístico), aplicado ANTES de chamar o LLM.

Por que não confiar só no fine-tuning: a avaliação da Etapa 2 mostrou 100% de
recusa explícita nos 12 casos de teste, mas isso ainda é comportamento
*aprendido* — nada garante que ele se mantenha diante de uma frase que o
modelo nunca viu, ou após uma mudança de prompt, re-treino ou até um ataque
de "jailbreak" bem elaborado.

Este módulo é a segunda camada de defesa: um classificador por palavras-chave
que barra a pergunta e devolve uma recusa fixa, sem nunca passar pelo LLM.
Ele é mais simples que o modelo, mas também mais previsível — exatamente o
que se quer da camada de segurança de um sistema clínico (defesa em
profundidade: duas camadas independentes, cada uma cobrindo a fraqueza da
outra). Fica deliberadamente severo (prefere falso positivo a falso
negativo): é um MVP, refinado na Etapa 6 com o log de auditoria completo.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

FONTE_POLITICA = "HSF-POL-001 v1.0 – Política de uso do assistente virtual."
AVISO = "Apoio à decisão clínica: a conduta final deve ser validada pelo médico responsável."

# (categoria, resposta padrão, padrões regex que disparam o bloqueio)
# Regras combinam uma AÇÃO perigosa (verbo) com um ALVO (substantivo), para
# reduzir falsos positivos em perguntas legítimas ("qual o protocolo de
# antibiótico?" não deve ser bloqueada; "prescreva um antibiótico" deve).
# Ordem importa: categorias mais específicas (assinatura) vêm antes das mais
# genéricas (prescrição), já que "assine a receita" também contém "receita".
_REGRAS: list[tuple[str, str, list[str]]] = [
    (
        "assinatura",
        "Não posso assinar receitas, laudos ou solicitações. A assinatura e o CRM são do médico responsável.",
        [r"\bassin\w*\b.*\b(?:receita|laudo|atestado|solicita[cç][aã]o|crm)\b",
         r"\b(?:crm|carimb\w*)\b.*\bassin\w*\b"],
    ),
    (
        "prescricao",
        "Não posso prescrever medicamentos. A prescrição é decisão do médico responsável. "
        "Posso resumir o protocolo interno aplicável, com exames e sinais de alerta.",
        [r"\bprescrev\w*\b", r"\breceite\w*\b", r"\bqual\s+(?:remedio|medicamento)\s+.*(?:prescrev|escolh)",
         r"\b(?:passar|passe|fazer|faca|emitir|emita|escrever|escreva)\w*\b.*\bprescri\w*\b"],
    ),
    (
        "dose",
        "Não posso definir doses, vias ou intervalos. Isso é decisão do médico responsável. "
        "Posso indicar o protocolo interno aplicável ao quadro.",
        [r"\bdosagem\b.*\b(?:aplic|administr|tom)\w*\b", r"\bquant\w*\s+(?:mg|ml|comprimido|gota)",
         r"\bquant\w*\s+de\s+\w+.*\b(?:aplic|administr|tom)\w*\b",
         r"\b(?:aplic|administr)\w*\s+(?:agora|ja)\b.*\bdos[ea]"],
    ),
    (
        "alta",
        "Não posso dar alta nem liberar pacientes. A alta é decisão do médico responsável, após avaliação.",
        [r"\b(?:libera|autoriz|registr)\w*\b.*\balta\b",
         r"\balta\b.*\b(?:agora|ja\b|por\s+favor|sem\s+(?:outra\s+)?avalia)",
         r"\b(?:de|da|dar)\s+alta\b.*\b(?:paciente|leito)\b"],
    ),
    (
        "ignorar_protocolo",
        "Não posso ignorar os protocolos internos nem decidir no lugar do médico responsável. "
        "Posso indicar o protocolo aplicável ao quadro descrito.",
        [r"\b(?:ignor|esque[cç]|desconsider|pul[ae])\w*\b.*\bprotocolo",
         r"\bn[aã]o\s+(?:precisa|quero)\s+.*protocolo",
         r"\bdecid[ae]\w*\s+(?:voc[eê]|voce)\s+(?:mesmo\s+)?(?:o\s+tratamento|por\s+mim)"],
    ),
]


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento)


@dataclass
class Bloqueio:
    categoria: str
    resposta: str


def verificar(pergunta: str) -> Bloqueio | None:
    """Retorna o Bloqueio se a pergunta cair em alguma regra perigosa, senão None."""
    texto = _normalizar(pergunta)
    for categoria, resposta_base, padroes in _REGRAS:
        if any(re.search(p, texto) for p in padroes):
            return Bloqueio(categoria, f"{resposta_base}\n\nFonte: {FONTE_POLITICA}\n{AVISO}")
    return None
