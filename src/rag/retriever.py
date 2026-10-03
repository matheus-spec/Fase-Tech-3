"""Busca de protocolos internos com citação de fonte GARANTIDA.

Diferença central em relação a pedir a fonte ao LLM: aqui a citação (id, versão,
nome) vem direto do registro do protocolo que bateu na busca — o modelo nunca
"lembra" o código de cabeça, então não existe como ele inventar ou trocar um
código por outro (o problema medido na Etapa 2, com 28,6% de acerto).

Pontuação: sobreposição de palavras-chave entre a consulta e os campos do
protocolo (nome, indicação, exames, condutas, alertas), sem bibliotecas
externas. Simples e auditável — suficiente para um catálogo de poucas dezenas
de protocolos; um catálogo real usaria embeddings, mas o contrato de citação
(vinda dos dados, não do texto do modelo) seria o mesmo.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.domain.protocolos import AVISO, PROTOCOLOS, VERSAO

_PARADAS = {
    "a", "o", "os", "as", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "um", "uma", "uns", "umas", "e", "ou", "que", "qual",
    "quais", "devo", "deve", "é", "ao", "aos", "se", "como", "quando", "protocolo",
    "interno", "inicial", "paciente",
}


def _normalizar(texto: str) -> list[str]:
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", sem_acento)
    return [t for t in tokens if t not in _PARADAS and len(t) > 1]


def _texto_indexavel(p: dict) -> str:
    partes = [p["nome"], p["indicacao"], *p["exames"], *p["conduta"], *p["alertas"]]
    return " ".join(partes)


@dataclass
class ResultadoBusca:
    protocolo: dict
    pontuacao: float
    fonte: str  # citação pronta, construída a partir dos dados — não editável pelo modelo

    @property
    def id(self) -> str:
        return self.protocolo["id"]


class BuscadorProtocolos:
    def __init__(self, protocolos: list[dict] = PROTOCOLOS):
        self._protocolos = protocolos
        self._index = [(p, set(_normalizar(_texto_indexavel(p)))) for p in protocolos]

    def buscar(self, consulta: str, top_k: int = 1, minimo: float = 0.08) -> list[ResultadoBusca]:
        termos = set(_normalizar(consulta))
        if not termos:
            return []
        resultados = []
        for protocolo, termos_doc in self._index:
            intersecao = termos & termos_doc
            if not intersecao:
                continue
            pontuacao = len(intersecao) / len(termos)  # fração da pergunta coberta pelo protocolo
            if pontuacao >= minimo:
                resultados.append(ResultadoBusca(protocolo, pontuacao, _fonte(protocolo)))
        resultados.sort(key=lambda r: r.pontuacao, reverse=True)
        return resultados[:top_k]


def _fonte(p: dict) -> str:
    return f"{p['id']} {VERSAO} – Protocolo interno de {p['nome']}."


def protocolo_para_resultado(protocolo: dict) -> ResultadoBusca:
    """Converte um protocolo já conhecido (ex.: o do paciente no prontuário) em
    ResultadoBusca, sem passar pela busca por palavra-chave. Usado quando a
    pergunta não cita a doença, mas já sabemos o protocolo pelo contexto do
    leito — a citação continua vindo dos dados, nunca do texto do modelo."""
    return ResultadoBusca(protocolo, pontuacao=1.0, fonte=_fonte(protocolo))


def montar_resposta(consulta: str, resultado: ResultadoBusca | None) -> str:
    """Resposta padrão, com a fonte sempre vinda do registro (não do modelo)."""
    if resultado is None:
        return (
            "Não encontrei um protocolo interno que corresponda a essa pergunta. "
            "Reformule com o nome da condição clínica, ou consulte diretamente a base de protocolos.\n\n"
            f"{AVISO}"
        )
    p = resultado.protocolo
    corpo = (
        f"Indicação: {p['indicacao']}\n"
        f"Exames iniciais: {', '.join(p['exames'])}.\n"
        f"Conduta: {' '.join(p['conduta'])}\n"
        f"Sinais de alerta para escalonamento: {', '.join(p['alertas'])}."
    )
    return f"{corpo}\n\nFonte: {resultado.fonte}\n{AVISO}"
