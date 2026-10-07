"""Une a busca de protocolo (src/rag/retriever.py) com os dados do paciente
(src/db/prontuarios.py) numa única resposta. É o pedaço do PDF que pede para
'consultar bases estruturadas e contextualizar com dados atualizados do
paciente' — aqui, de forma simples, antes da orquestração completa em
LangChain (Etapa 4).

Dados do paciente (idade, sexo, exames pendentes) só aparecem quando o código
do leito é informado; nunca aparece nome, CPF ou outro identificador direto,
porque o banco (Etapa 3) não os armazena.
"""
from __future__ import annotations

import sqlite3

from src.db.prontuarios import Paciente, buscar_paciente_por_leito, exames_pendentes, registrar_evento_auditoria
from src.domain.protocolos import AVISO, PROTOCOLOS
from src.rag.retriever import BuscadorProtocolos, montar_resposta, protocolo_para_resultado

_POR_ID = {p["id"]: p for p in PROTOCOLOS}


def linha_paciente(p: Paciente, pendentes: list[sqlite3.Row]) -> str:
    nomes_pendentes = [e["nome"] for e in pendentes]
    protocolo = _POR_ID.get(p.protocolo_id, {}).get("nome", p.protocolo_id)
    pendencia = f"Exames pendentes: {', '.join(nomes_pendentes)}." if nomes_pendentes else "Sem exames pendentes registrados."
    return f"Paciente do leito {p.codigo_leito.replace('leito-', '')}, {p.idade} anos, em acompanhamento pelo protocolo de {protocolo}. {pendencia}"


def responder_com_contexto(
    con: sqlite3.Connection,
    buscador: BuscadorProtocolos,
    pergunta: str,
    codigo_leito: str | None = None,
) -> str:
    """Resposta final: contexto do paciente (se houver leito) + protocolo buscado + fonte.

    Se a pergunta não citar a doença (ex.: "qual a conduta inicial?") mas o leito
    for informado, usa o protocolo já associado ao paciente no prontuário, em vez
    de depender só da busca por palavra-chave na pergunta."""
    paciente = buscar_paciente_por_leito(con, codigo_leito) if codigo_leito else None

    if codigo_leito and paciente is None:
        texto = f"Não encontrei o leito {codigo_leito} na base de prontuários.\n\n{AVISO}"
        registrar_evento_auditoria(con, tipo="consulta_protocolo", conteudo=f"pergunta={pergunta!r} leito={codigo_leito!r}", origem="assistente")
        return texto

    resultados = buscador.buscar(pergunta, top_k=1)
    if resultados:
        resultado = resultados[0]
    elif paciente is not None:
        protocolo = _POR_ID.get(paciente.protocolo_id)
        resultado = protocolo_para_resultado(protocolo) if protocolo else None
    else:
        resultado = None

    resposta_protocolo = montar_resposta(pergunta, resultado)
    if paciente is None:
        texto = resposta_protocolo
    else:
        pendentes = exames_pendentes(con, paciente.id)
        contexto = linha_paciente(paciente, pendentes)
        texto = f"{contexto}\n\n{resposta_protocolo}"

    registrar_evento_auditoria(con, tipo="consulta_protocolo", conteudo=f"pergunta={pergunta!r} leito={codigo_leito!r}", origem="assistente")
    return texto
