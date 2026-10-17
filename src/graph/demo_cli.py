"""Demonstração manual dos fluxos da Etapa 5, sem GPU (usa ModeloEco).

Uso:
    python -m src.graph.demo_cli
"""
from __future__ import annotations

from src.chains.llm_cliente import ModeloEco
from src.chains.pipeline import AssistenteMedico
from src.db.prontuarios import conectar, criar_esquema, popular_sinteticamente
from src.graph.fluxo import construir_grafo_sugestao, decidir_sugestao, verificar_exames_pendentes_geral
from src.rag.retriever import BuscadorProtocolos


def main() -> None:
    with conectar(":memory:") as con:
        criar_esquema(con)
        popular_sinteticamente(con, n_pacientes=8, seed=3)

        print("=== Fluxo 1: verificação geral de exames pendentes (automático) ===")
        for alerta in verificar_exames_pendentes_geral(con):
            print(alerta["mensagem"])

        print()
        print("=== Fluxo 2: sugestão de conduta com aprovação humana (leito-1) ===")
        assistente = AssistenteMedico(con, BuscadorProtocolos(), ModeloEco())
        grafo = construir_grafo_sugestao(con, assistente)

        estado = grafo.invoke({"codigo_leito": "leito-1"})
        print(f"Status após o grafo: {estado['status']}  (nenhum alerta ainda)")
        print(f"Sugestão: {estado.get('sugestao')}")

        print("\n-- Um médico aprova --")
        final = decidir_sugestao(con, estado, aprovado=True, aprovador="Dra. Souza (CRM 123)")
        print(final["alerta_equipe"])


if __name__ == "__main__":
    main()
