"""Etapa 5 (LangGraph): fluxos de decisão automatizados.

Dois fluxos distintos, pedidos no PDF ("verificar exames pendentes, sugerir
tratamentos, emitir alertas para a equipe"):

  1. `verificar_exames_pendentes_geral` — automático, roda para todos os
     pacientes, sem aprovação: é um lembrete operacional (quais exames
     faltam), não uma decisão clínica.

  2. `construir_grafo_sugestao` + `decidir_sugestao` — usa o pipeline da
     Etapa 4 (guardrail + busca com citação garantida + LLM) para sugerir uma
     conduta por leito, mas o grafo PARA no estado "aguardando_aprovacao":
     nenhum alerta é enviado à equipe até um humano decidir. É aqui que a
     exigência do PDF ("nunca sem validação humana") vira um passo real do
     fluxo, registrado e auditável — não só uma instrução de prompt.
"""
from __future__ import annotations

from typing import Callable, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:
    # langgraph não instalado neste ambiente (sem acesso à internet para
    # `pip install`). Mesmo princípio da Etapa 4 (src/chains/pipeline.py):
    # substituto mínimo com a mesma interface (StateGraph, add_node, add_edge,
    # add_conditional_edges, set_entry_point, compile().invoke) — permite
    # montar e testar o fluxo aqui, e roda sem nenhuma mudança de código na
    # máquina com `pip install langgraph`.
    END = "__end__"

    class StateGraph:  # type: ignore[no-redef]
        def __init__(self, esquema_estado):
            self._nos: dict[str, Callable[[dict], dict]] = {}
            self._arestas: dict[str, str] = {}
            self._condicionais: dict[str, tuple[Callable[[dict], str], dict[str, str]]] = {}
            self._entrada: str | None = None

        def add_node(self, nome: str, fn: Callable[[dict], dict]) -> None:
            self._nos[nome] = fn

        def set_entry_point(self, nome: str) -> None:
            self._entrada = nome

        def add_edge(self, origem: str, destino: str) -> None:
            self._arestas[origem] = destino

        def add_conditional_edges(self, origem: str, condicao: Callable[[dict], str], mapa: dict[str, str]) -> None:
            self._condicionais[origem] = (condicao, mapa)

        def compile(self) -> "_GrafoCompilado":
            return _GrafoCompilado(self._nos, self._arestas, self._condicionais, self._entrada)

    class _GrafoCompilado:
        def __init__(self, nos, arestas, condicionais, entrada):
            self.nos, self.arestas, self.condicionais, self.entrada = nos, arestas, condicionais, entrada

        def invoke(self, estado: dict) -> dict:
            atual = self.entrada
            while atual != END:
                atualizacao = self.nos[atual](estado) or {}
                estado = {**estado, **atualizacao}
                if atual in self.condicionais:
                    condicao, mapa = self.condicionais[atual]
                    atual = mapa[condicao(estado)]
                else:
                    atual = self.arestas.get(atual, END)
            return estado

from src.chains.pipeline import AssistenteMedico
from src.db.prontuarios import buscar_paciente_por_leito, exames_pendentes, listar_pacientes, registrar_evento_auditoria


class EstadoSugestao(TypedDict, total=False):
    codigo_leito: str
    encontrado: bool
    paciente: dict | None
    exames_pendentes: list[str]
    sugestao: str | None
    fonte: str | None
    bloqueado_por_guardrail: bool
    status: str  # leito_nao_encontrado | aguardando_aprovacao | sem_sugestao | bloqueado_por_guardrail | aprovado | rejeitado
    alerta_equipe: str | None


def _no_buscar_paciente(con):
    def _executar(estado: EstadoSugestao) -> dict:
        paciente = buscar_paciente_por_leito(con, estado["codigo_leito"])
        if paciente is None:
            return {"encontrado": False, "status": "leito_nao_encontrado"}
        pendentes = [e["nome"] for e in exames_pendentes(con, paciente.id)]
        return {
            "encontrado": True,
            "paciente": {
                "id": paciente.id, "codigo_leito": paciente.codigo_leito,
                "idade": paciente.idade, "protocolo_id": paciente.protocolo_id,
            },
            "exames_pendentes": pendentes,
        }
    return _executar


def _no_sugerir_conduta(assistente: AssistenteMedico):
    def _executar(estado: EstadoSugestao) -> dict:
        resposta = assistente.responder("Qual a conduta inicial?", codigo_leito=estado["codigo_leito"])
        if resposta.bloqueado_por_guardrail:
            # Não deveria acontecer com uma pergunta fixa e inofensiva como essa,
            # mas se acontecer, o fluxo para aqui — nunca chega à aprovação.
            return {"status": "bloqueado_por_guardrail", "sugestao": resposta.texto, "bloqueado_por_guardrail": True}
        if resposta.fonte is None:
            return {"status": "sem_sugestao", "sugestao": resposta.texto}
        return {"status": "aguardando_aprovacao", "sugestao": resposta.texto, "fonte": resposta.fonte}
    return _executar


def _roteamento_apos_busca(estado: EstadoSugestao) -> str:
    return "encontrado" if estado.get("encontrado") else "nao_encontrado"


def construir_grafo_sugestao(con, assistente: AssistenteMedico):
    """Compila o grafo: buscar_paciente → (se encontrado) sugerir_conduta.
    Termina em "aguardando_aprovacao" — use `decidir_sugestao` para concluir."""
    grafo = StateGraph(EstadoSugestao)
    grafo.add_node("buscar_paciente", _no_buscar_paciente(con))
    grafo.add_node("sugerir_conduta", _no_sugerir_conduta(assistente))
    grafo.set_entry_point("buscar_paciente")
    grafo.add_conditional_edges(
        "buscar_paciente", _roteamento_apos_busca,
        {"encontrado": "sugerir_conduta", "nao_encontrado": END},
    )
    grafo.add_edge("sugerir_conduta", END)
    return grafo.compile()


def decidir_sugestao(con, estado: EstadoSugestao, aprovado: bool, aprovador: str) -> EstadoSugestao:
    """Continuação do fluxo acima, depois da decisão humana. O alerta para a
    equipe só é gerado se `aprovado` for True — nunca automaticamente — e a
    decisão (de qualquer jeito) fica registrada na auditoria."""
    if estado.get("status") != "aguardando_aprovacao":
        raise ValueError(
            f"Nada para decidir: status atual é {estado.get('status')!r}, esperado 'aguardando_aprovacao'."
        )
    leito = estado["codigo_leito"]
    if aprovado:
        alerta = f"[Alerta à equipe] Conduta aprovada por {aprovador} para o {leito}: {estado['sugestao']}"
        novo_status = "aprovado"
    else:
        alerta = None
        novo_status = "rejeitado"
    registrar_evento_auditoria(
        con, tipo=f"decisao_sugestao:{novo_status}",
        conteudo=f"leito={leito!r} aprovador={aprovador!r} fonte={estado.get('fonte')!r}",
        origem="medico",  # a coluna só aceita 'assistente' ou 'medico'; o nome de quem decidiu vai no conteudo
    )
    return {**estado, "status": novo_status, "alerta_equipe": alerta}


def verificar_exames_pendentes_geral(con) -> list[dict]:
    """Fluxo automático (sem aprovação humana): para cada paciente com algum
    exame pendente, gera um lembrete operacional. Não é uma decisão clínica —
    por isso não passa pela etapa de aprovação, diferente da sugestão de
    conduta acima."""
    alertas = []
    for paciente in listar_pacientes(con):
        pendentes = [e["nome"] for e in exames_pendentes(con, paciente.id)]
        if not pendentes:
            continue
        alertas.append({
            "codigo_leito": paciente.codigo_leito,
            "exames_pendentes": pendentes,
            "mensagem": f"[Lembrete] {paciente.codigo_leito}: exames pendentes — {', '.join(pendentes)}.",
        })
        registrar_evento_auditoria(
            con, tipo="lembrete_exame_pendente",
            conteudo=f"leito={paciente.codigo_leito!r} exames={pendentes!r}",
            origem="assistente",
        )
    return alertas
