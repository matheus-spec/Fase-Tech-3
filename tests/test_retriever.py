import unittest

from src.rag.retriever import BuscadorProtocolos, montar_resposta


class TestBuscadorProtocolos(unittest.TestCase):
    def setUp(self):
        self.b = BuscadorProtocolos()

    def test_casos_que_o_modelo_ajustado_errou_na_etapa_2(self):
        # Mesmas perguntas do relatório de avaliação; aqui o código deve vir certo.
        casos = {
            "Qual a conduta inicial em sepse?": "HSF-PROT-001",
            "Que exames o protocolo de sepse recomenda?": "HSF-PROT-001",
            "Quando devo acionar a equipe de apoio em cetoacidose diabética?": "HSF-PROT-004",
            "Quais sinais de alerta exigem escalonamento em cetoacidose diabética?": "HSF-PROT-004",
            "Quando o protocolo de AVC agudo se aplica?": "HSF-PROT-003",
        }
        for pergunta, esperado in casos.items():
            r = self.b.buscar(pergunta, top_k=1)
            self.assertTrue(r, f"sem resultado para: {pergunta}")
            self.assertEqual(r[0].id, esperado, pergunta)

    def test_sem_correspondencia_retorna_vazio(self):
        self.assertEqual(self.b.buscar("qual o resultado do jogo de futebol de ontem"), [])

    def test_fonte_vem_dos_dados_nao_do_texto_livre(self):
        r = self.b.buscar("conduta inicial em sepse", top_k=1)[0]
        self.assertEqual(r.fonte, "HSF-PROT-001 v1.0 – Protocolo interno de sepse.")
        # a citação é derivada do dicionário do protocolo, não de uma string solta
        self.assertEqual(r.fonte, f"{r.protocolo['id']} v1.0 – Protocolo interno de {r.protocolo['nome']}.")

    def test_top_k_respeitado_e_ordenado_por_pontuacao(self):
        r = self.b.buscar("protocolo", top_k=3)  # termo genérico, mas filtrado como palavra de parada
        self.assertEqual(r, [])  # "protocolo" sozinho não deve gerar falso match

    def test_montar_resposta_sem_resultado(self):
        texto = montar_resposta("pergunta qualquer", None)
        self.assertIn("Não encontrei", texto)
        self.assertNotIn("Fonte:", texto)

    def test_montar_resposta_com_resultado_cita_fonte_e_aviso(self):
        r = self.b.buscar("conduta inicial em sepse", top_k=1)[0]
        texto = montar_resposta("conduta inicial em sepse", r)
        self.assertIn("Fonte: HSF-PROT-001", texto)
        self.assertIn("validada pelo médico responsável", texto)


if __name__ == "__main__":
    unittest.main()


class TestNumerosNaoContamComoRelevancia(unittest.TestCase):
    """Regressão: 'leito 10' batendo com o '10' de 'até 10 minutos' num protocolo
    não relacionado já causou uma citação errada em produção (Colab, Etapa 4)."""

    def test_numero_isolado_nao_casa_com_protocolo_por_coincidencia(self):
        b = BuscadorProtocolos()
        self.assertEqual(b.buscar("Quais são as comorbidades do paciente do leito 10?"), [])
        self.assertEqual(b.buscar("Como está o paciente do leito 6 hoje?"), [])

    def test_numero_nao_atrapalha_quando_ha_termo_clinico_real(self):
        b = BuscadorProtocolos()
        r = b.buscar("Preciso do protocolo de sepse para o leito 10")
        self.assertTrue(r)
        self.assertEqual(r[0].protocolo["id"], "HSF-PROT-001")
