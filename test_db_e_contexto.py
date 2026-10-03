import sqlite3
import unittest

from src.db.prontuarios import (
    buscar_paciente_por_leito, criar_esquema, exames_pendentes,
    popular_sinteticamente, registrar_evento_auditoria,
)
from src.rag.contexto import responder_com_contexto
from src.rag.retriever import BuscadorProtocolos


def _db_em_memoria() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    criar_esquema(con)
    return con


class TestProntuarios(unittest.TestCase):
    def setUp(self):
        self.con = _db_em_memoria()
        popular_sinteticamente(self.con, n_pacientes=15, seed=1)

    def tearDown(self):
        self.con.close()

    def test_popular_cria_pacientes_sem_dado_pessoal_direto(self):
        colunas = {d[0] for d in self.con.execute("SELECT * FROM pacientes LIMIT 1").description}
        for proibido in ("nome", "cpf", "telefone", "email"):
            self.assertNotIn(proibido, colunas)

    def test_busca_por_leito(self):
        p = buscar_paciente_por_leito(self.con, "leito-1")
        self.assertIsNotNone(p)
        self.assertEqual(p.codigo_leito, "leito-1")

    def test_leito_inexistente_retorna_none(self):
        self.assertIsNone(buscar_paciente_por_leito(self.con, "leito-999"))

    def test_exames_pendentes_filtra_status(self):
        p = buscar_paciente_por_leito(self.con, "leito-1")
        pendentes = exames_pendentes(self.con, p.id)
        todos = self.con.execute("SELECT status FROM exames WHERE paciente_id = ?", (p.id,)).fetchall()
        self.assertEqual(len(pendentes), sum(1 for r in todos if r["status"] == "pendente"))

    def test_auditoria_registra_evento(self):
        registrar_evento_auditoria(self.con, "teste", "conteudo x", "assistente")
        n = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        self.assertEqual(n, 1)

    def test_populacao_e_idempotente_reprodutivel(self):
        con2 = _db_em_memoria()
        popular_sinteticamente(con2, n_pacientes=15, seed=1)
        a = [dict(r) for r in self.con.execute("SELECT codigo_leito, idade, protocolo_id FROM pacientes")]
        b = [dict(r) for r in con2.execute("SELECT codigo_leito, idade, protocolo_id FROM pacientes")]
        self.assertEqual(a, b)
        con2.close()


class TestRespostaContextualizada(unittest.TestCase):
    def setUp(self):
        self.con = _db_em_memoria()
        popular_sinteticamente(self.con, n_pacientes=10, seed=2)
        self.buscador = BuscadorProtocolos()

    def tearDown(self):
        self.con.close()

    def test_sem_leito_nao_vaza_dado_de_paciente(self):
        texto = responder_com_contexto(self.con, self.buscador, "Qual a conduta inicial em sepse?")
        self.assertNotIn("Paciente do leito", texto)
        self.assertIn("Fonte: HSF-PROT-001", texto)

    def test_com_leito_valido_inclui_contexto_e_fonte(self):
        texto = responder_com_contexto(self.con, self.buscador, "Qual a conduta inicial?", codigo_leito="leito-1")
        self.assertIn("Paciente do leito 1", texto)
        self.assertIn("Fonte:", texto)

    def test_pergunta_vaga_com_leito_usa_protocolo_do_paciente(self):
        paciente = buscar_paciente_por_leito(self.con, "leito-1")
        texto = responder_com_contexto(self.con, self.buscador, "Qual a conduta inicial?", codigo_leito="leito-1")
        self.assertIn(f"Fonte: {paciente.protocolo_id}", texto)

    def test_leito_inexistente_nao_quebra(self):
        texto = responder_com_contexto(self.con, self.buscador, "Qual a conduta?", codigo_leito="leito-999")
        self.assertIn("Não encontrei o leito leito-999", texto)

    def test_nenhuma_resposta_vaza_cpf_ou_nome(self):
        for leito in ("leito-1", "leito-2", None):
            texto = responder_com_contexto(self.con, self.buscador, "conduta inicial em pneumonia", codigo_leito=leito)
            self.assertNotIn("CPF", texto)

    def test_toda_consulta_gera_evento_de_auditoria(self):
        antes = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        responder_com_contexto(self.con, self.buscador, "Qual a conduta inicial em sepse?")
        depois = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        self.assertEqual(depois, antes + 1)


if __name__ == "__main__":
    unittest.main()
