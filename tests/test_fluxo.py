import sqlite3
import unittest

from src.chains.llm_cliente import ModeloEco
from src.chains.pipeline import AssistenteMedico
from src.db.prontuarios import criar_esquema, popular_sinteticamente
from src.graph.fluxo import construir_grafo_sugestao, decidir_sugestao, verificar_exames_pendentes_geral
from src.rag.retriever import BuscadorProtocolos


def _db_em_memoria(n=8, seed=3) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    criar_esquema(con)
    popular_sinteticamente(con, n_pacientes=n, seed=seed)
    return con


class TestVerificacaoExamesPendentes(unittest.TestCase):
    def test_lista_so_pacientes_com_pendencia_e_registra_auditoria(self):
        con = _db_em_memoria()
        alertas = verificar_exames_pendentes_geral(con)
        self.assertTrue(alertas)
        for a in alertas:
            self.assertTrue(a["exames_pendentes"])
            self.assertIn(a["codigo_leito"], a["mensagem"])
        n_eventos = con.execute(
            "SELECT COUNT(*) FROM eventos_auditoria WHERE tipo = 'lembrete_exame_pendente'"
        ).fetchone()[0]
        self.assertEqual(n_eventos, len(alertas))

    def test_nao_requer_aprovacao_humana(self):
        # Diferença de design: isto é lembrete operacional, não decisão clínica.
        con = _db_em_memoria()
        alertas = verificar_exames_pendentes_geral(con)
        for a in alertas:
            self.assertNotIn("aguardando_aprovacao", a)


class TestFluxoSugestao(unittest.TestCase):
    def setUp(self):
        self.con = _db_em_memoria()
        self.assistente = AssistenteMedico(self.con, BuscadorProtocolos(), ModeloEco())
        self.grafo = construir_grafo_sugestao(self.con, self.assistente)

    def test_leito_valido_para_em_aguardando_aprovacao(self):
        estado = self.grafo.invoke({"codigo_leito": "leito-1"})
        self.assertEqual(estado["status"], "aguardando_aprovacao")
        self.assertIsNotNone(estado["fonte"])
        self.assertTrue(estado["encontrado"])

    def test_nenhum_alerta_e_emitido_antes_da_aprovacao(self):
        estado = self.grafo.invoke({"codigo_leito": "leito-1"})
        self.assertNotIn("alerta_equipe", estado)

    def test_leito_inexistente_nao_chega_a_sugerir_nada(self):
        estado = self.grafo.invoke({"codigo_leito": "leito-999"})
        self.assertEqual(estado["status"], "leito_nao_encontrado")
        self.assertNotIn("sugestao", estado)

    def test_aprovacao_gera_alerta_e_grava_auditoria(self):
        estado = self.grafo.invoke({"codigo_leito": "leito-1"})
        antes = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        final = decidir_sugestao(self.con, estado, aprovado=True, aprovador="Dra. Souza")
        depois = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        self.assertEqual(final["status"], "aprovado")
        self.assertIsNotNone(final["alerta_equipe"])
        self.assertIn("Dra. Souza", final["alerta_equipe"])
        self.assertEqual(depois, antes + 1)

    def test_rejeicao_nao_gera_alerta_mas_grava_auditoria(self):
        estado = self.grafo.invoke({"codigo_leito": "leito-1"})
        antes = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        final = decidir_sugestao(self.con, estado, aprovado=False, aprovador="Dr. Lima")
        depois = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        self.assertEqual(final["status"], "rejeitado")
        self.assertIsNone(final["alerta_equipe"])
        self.assertEqual(depois, antes + 1)

    def test_nao_deixa_decidir_duas_vezes_o_mesmo_estado(self):
        estado = self.grafo.invoke({"codigo_leito": "leito-1"})
        final = decidir_sugestao(self.con, estado, aprovado=True, aprovador="Dra. Souza")
        with self.assertRaises(ValueError):
            decidir_sugestao(self.con, final, aprovado=True, aprovador="outra pessoa")

    def test_nao_deixa_decidir_estado_que_nunca_rodou(self):
        with self.assertRaises(ValueError):
            decidir_sugestao(self.con, {"codigo_leito": "leito-1", "status": "sem_sugestao"}, True, "x")


if __name__ == "__main__":
    unittest.main()
