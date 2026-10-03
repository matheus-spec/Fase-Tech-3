"""Anonimização de dados pessoais em textos clínicos em português (LGPD).

Substitui identificadores diretos por marcadores como [CPF] ou [NOME].
Abordagem baseada em regras (regex): simples, auditável e sem dependências.
Limitação conhecida: nomes só são detectados quando precedidos de pistas
("Paciente", "Sr.", "Dra.", "Nome:"). Em dados reais, complemente com NER
e revisão humana amostral.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_UP = "A-ZÁÀÂÃÉÊÍÓÔÕÚÇ"
_LO = "a-záàâãéêíóôõúç"
_NOME = rf"[{_UP}][{_LO}]+(?:[ \t]+(?:d[aeo]s?[ \t]+)?[{_UP}][{_LO}]+){{0,3}}"

# A ordem importa: padrões mais específicos primeiro.
_REGRAS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("CPF", re.compile(r"(?<!\d)(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})(?!\d)")),
    ("RG", re.compile(r"\bRG[ \t]*:?[ \t]*[\d.\-Xx]{7,12}")),
    (
        "PRONTUARIO",
        re.compile(r"(?i:\b(?:prontu[aá]rio|pront\.?|MRN))[ \t]*(?:n[ºo°.]*[ \t]*)?:?[ \t]*\d{4,}\b"),
    ),
    (
        "TELEFONE",
        re.compile(r"(?<!\d)(?:\+55[ ]?)?(?:\(\d{2}\)[ ]?|\d{2}[ ])?9?\d{4}-\d{4}(?!\d)"),
    ),
    ("DATA", re.compile(r"\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b")),
    ("CEP", re.compile(r"(?<!\d)\d{5}-\d{3}(?!\d)")),
]

# Nome precedido de pista; mantém a pista e troca só o nome.
_REGRA_NOME = re.compile(rf"\b(Sr\.?|Sra\.?|Dr\.?|Dra\.?|[Pp]aciente|Nome:?)([ \t]+){_NOME}")


@dataclass
class ResultadoAnonimizacao:
    texto: str
    contagem: Counter = field(default_factory=Counter)


def anonimizar(texto: str) -> ResultadoAnonimizacao:
    """Retorna o texto anonimizado e a contagem de itens substituídos por tipo."""
    contagem: Counter = Counter()
    for tipo, regex in _REGRAS:
        texto, n = regex.subn(f"[{tipo}]", texto)
        contagem[tipo] += n
    texto, n = _REGRA_NOME.subn(r"\1\2[NOME]", texto)
    contagem["NOME"] += n
    return ResultadoAnonimizacao(texto, +contagem)  # +Counter remove zeros


def contem_pii(texto: str) -> bool:
    """Verificação final: True se ainda houver algum padrão de PII no texto."""
    return any(r.search(texto) for _, r in _REGRAS) or bool(_REGRA_NOME.search(texto))

