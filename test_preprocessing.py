import random
import unittest

from src.preprocessing.anonymizer import anonimizar, contem_pii
from src.preprocessing.build_dataset import curar, dividir
from src.preprocessing.synthetic_data import gerar_faq_protocolos, gerar_notas_clinicas


class TestAnonimizador(unittest.TestCase):
    def test_identificadores_diretos(self):
        t = ("Paciente Maria Souza Lima, CPF 123.456.789-09, prontuário nº 456789, "
             "nasc. 05/03/1980, tel. (11) 98765-4321, e-mail maria@ex.com, CEP 06600-000.")
        r = anonimizar(t).texto
        for vazado in ("Maria", "123.456", "456789", "05/03", "98765", "maria@", "06600"):
            self.assertNotIn(vazado, r)
        self.assertFalse(contem_pii(r))

    def test_nome_com_titulo_mantem_titulo(self):
        self.assertEqual(anonimizar("Dr. Carlos Silva examinou").texto, "Dr. [NOME] examinou")

    def test_nao_altera_texto_clinico_sem_pii(self):
        t = "Paciente adulto com febre, lactato 4,2 e pressão 90x60."
        self.assertEqual(anonimizar(t).texto, t)

    def test_idempotente(self):
        t = "Paciente João Silva, CPF 111.222.333-44."
        uma = anonimizar(t).texto
        self.assertEqual(anonimizar(uma).texto, uma)

    def test_contagem(self):
        r = anonimizar("CPF 111.222.333-44 e CPF 555.666.777-88")
        self.assertEqual(r.contagem["CPF"], 2)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.bruto = gerar_faq_protocolos() + gerar_notas_clinicas(20, random.Random(1))

    def test_curadoria_remove_pii_e_duplicatas(self):
        base, rel_base = curar(self.bruto)
        curados, rel = curar(self.bruto + [dict(self.bruto[0])])
        self.assertEqual(len(curados), len(base))
        self.assertEqual(rel["descartados_duplicados"], rel_base["descartados_duplicados"] + 1)
        self.assertEqual(rel["descartados_pii_residual"], 0)
        for r in curados:
            self.assertFalse(contem_pii(r["input"] + r["output"] + r["instruction"]))

    def test_divisao_sem_vazamento_e_reprodutivel(self):
        curados, _ = curar(self.bruto)
        a = dividir(curados, seed=7)
        b = dividir(curados, seed=7)
        self.assertEqual(a, b)
        chaves = [{(r["instruction"], r["input"], r["output"]) for r in parte} for parte in a]
        self.assertFalse(chaves[0] & chaves[1] or chaves[0] & chaves[2] or chaves[1] & chaves[2])
        self.assertEqual(sum(len(p) for p in a), len(curados))


if __name__ == "__main__":
    unittest.main()
