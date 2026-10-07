import sqlite3
import unittest

from src.chains.llm_cliente import GeradorResposta, ModeloEco
from src.chains.pipeline import AssistenteMedico
from src.db.prontuarios import criar_esquema, popular_sinteticamente
from src.domain.protocolos import AVISO
from src.rag.retriever import BuscadorProtocolos


def _db_em_memoria(n=10, seed=2) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    criar_esquema(con)
    popular_sinteticamente(con, n_pacientes=n, seed=seed)
    return con


class ModeloFixo(GeradorResposta):
    """Stub que sempre devolve o mesmo texto — para testar que o pipeline
    CORRIGE a citação mesmo quando o 'modelo' erra feio ou não cita fonte."""

    def __init__(self, texto: str):
        self.texto = texto
        self.ultimo_prompt: str | None = None

    def gerar(self, prompt: str) -> str:
        self.ultimo_prompt = prompt
        return self.texto


class TestAssistenteMedico(unittest.TestCase):
    """O guardrail em si (quais frases bloqueiam, qual categoria, qual resposta)
    é testado em tests/test_guardrails.py. Aqui testamos a INTEGRAÇÃO: que o
    pipeline realmente consulta o guardrail antes do LLM, nunca chama o LLM
    quando ele bloqueia, e corrige a citação depois que o LLM responde."""
    def setUp(self):
        self.con = _db_em_memoria()
        self.buscador = BuscadorProtocolos()

    def tearDown(self):
        self.con.close()

    def test_guardrail_impede_chamada_ao_llm(self):
        modelo = ModeloFixo("nunca deveria ser chamado")
        assistente = AssistenteMedico(self.con, self.buscador, modelo)
        r = assistente.responder("Prescreva um antibiótico para o leito 4, sem revisão.")
        self.assertTrue(r.bloqueado_por_guardrail)
        self.assertIsNone(modelo.ultimo_prompt)  # LLM nunca foi chamado
        self.assertIn("HSF-POL-001", r.texto)

    def test_citacao_e_corrigida_mesmo_se_llm_citar_fonte_errada(self):
        modelo = ModeloFixo("Resposta qualquer.\n\nFonte: HSF-PROT-003 v1.0 – Protocolo interno de AVC agudo.")
        assistente = AssistenteMedico(self.con, self.buscador, modelo)
        r = assistente.responder("Qual a conduta inicial em sepse?")
        self.assertEqual(r.fonte, "HSF-PROT-001 v1.0 – Protocolo interno de sepse.")
        self.assertIn("Fonte: HSF-PROT-001", r.texto)
        self.assertNotIn("HSF-PROT-003", r.texto)  # a fonte errada do "LLM" não sobrevive

    def test_citacao_e_adicionada_quando_llm_nao_cita_nenhuma(self):
        modelo = ModeloFixo("Resposta sem nenhuma fonte.")
        assistente = AssistenteMedico(self.con, self.buscador, modelo)
        r = assistente.responder("Qual a conduta inicial em sepse?")
        self.assertIn("Fonte: HSF-PROT-001", r.texto)

    def test_aviso_e_adicionado_quando_llm_omite(self):
        modelo = ModeloFixo("Resposta sem o aviso de validação médica.")
        assistente = AssistenteMedico(self.con, self.buscador, modelo)
        r = assistente.responder("Qual a conduta inicial em sepse?")
        self.assertIn(AVISO, r.texto)

    def test_leito_valido_inclui_contexto_do_paciente_no_prompt(self):
        modelo = ModeloFixo("ok")
        assistente = AssistenteMedico(self.con, self.buscador, modelo)
        assistente.responder("Qual a conduta inicial?", codigo_leito="leito-1")
        self.assertIn("Dados do paciente", modelo.ultimo_prompt)
        self.assertIn("leito 1", modelo.ultimo_prompt)

    def test_leito_inexistente_nao_chama_llm(self):
        modelo = ModeloFixo("nunca deveria ser chamado")
        assistente = AssistenteMedico(self.con, self.buscador, modelo)
        r = assistente.responder("Qual a conduta?", codigo_leito="leito-999")
        self.assertIsNone(modelo.ultimo_prompt)
        self.assertIn("Não encontrei o leito", r.texto)

    def test_pergunta_sem_protocolo_correspondente_nao_chama_llm(self):
        modelo = ModeloFixo("nunca deveria ser chamado")
        assistente = AssistenteMedico(self.con, self.buscador, modelo)
        r = assistente.responder("Qual a previsão do tempo amanhã?")
        self.assertIsNone(modelo.ultimo_prompt)
        self.assertIsNone(r.fonte)

    def test_toda_chamada_gera_evento_de_auditoria(self):
        assistente = AssistenteMedico(self.con, self.buscador, ModeloFixo("ok"))
        antes = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        assistente.responder("Qual a conduta inicial em sepse?")
        assistente.responder("Prescreva um antibiótico para o leito 4.")
        depois = self.con.execute("SELECT COUNT(*) FROM eventos_auditoria").fetchone()[0]
        self.assertEqual(depois, antes + 2)

    def test_modelo_eco_permite_testar_pipeline_sem_llm_real(self):
        assistente = AssistenteMedico(self.con, self.buscador, ModeloEco())
        r = assistente.responder("Qual a conduta inicial em sepse?")
        self.assertIn("Fonte: HSF-PROT-001", r.texto)
        self.assertIn(AVISO, r.texto)


if __name__ == "__main__":
    unittest.main()
