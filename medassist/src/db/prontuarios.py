"""Base de prontuários SINTÉTICA, em SQLite, para contextualizar respostas com
'dados do paciente' sem precisar de um sistema hospitalar real (inexistente
neste projeto). É o requisito do PDF de "consultar bases estruturadas e
contextualizar com dados atualizados do paciente".

Nunca grava nome, CPF ou outro identificador direto — só um `codigo_leito`
(ex.: "leito-4"), o suficiente para o pipeline funcionar sem reintroduzir o
problema de privacidade já tratado na Etapa 1.
"""
from __future__ import annotations

import random
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.domain.protocolos import PROTOCOLOS, VERSAO

ESQUEMA = """
CREATE TABLE IF NOT EXISTS protocolos (
    id TEXT PRIMARY KEY,
    nome TEXT NOT NULL,
    versao TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY,
    codigo_leito TEXT NOT NULL UNIQUE,
    idade INTEGER NOT NULL,
    sexo TEXT NOT NULL CHECK (sexo IN ('feminino', 'masculino')),
    protocolo_id TEXT NOT NULL,
    admitido_em TEXT NOT NULL,
    FOREIGN KEY (protocolo_id) REFERENCES protocolos(id)
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY,
    paciente_id INTEGER NOT NULL,
    nome TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pendente', 'coletado', 'resultado_disponivel')),
    solicitado_em TEXT NOT NULL,
    FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);

CREATE TABLE IF NOT EXISTS eventos_auditoria (
    id INTEGER PRIMARY KEY,
    paciente_id INTEGER,
    tipo TEXT NOT NULL,
    conteudo TEXT NOT NULL,
    origem TEXT NOT NULL CHECK (origem IN ('assistente', 'medico')),
    criado_em TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class Paciente:
    id: int
    codigo_leito: str
    idade: int
    sexo: str
    protocolo_id: str


@contextmanager
def conectar(caminho: str | Path):
    con = sqlite3.connect(caminho)
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def criar_esquema(con: sqlite3.Connection) -> None:
    con.executescript(ESQUEMA)


def popular_protocolos(con: sqlite3.Connection) -> None:
    """Carrega o catálogo oficial (src/domain/protocolos.py) na tabela protocolos."""
    con.executemany(
        "INSERT OR IGNORE INTO protocolos (id, nome, versao) VALUES (?, ?, ?)",
        [(p["id"], p["nome"], VERSAO) for p in PROTOCOLOS],
    )


def popular_sinteticamente(con: sqlite3.Connection, n_pacientes: int = 20, seed: int = 42) -> None:
    """Preenche com pacientes e exames fictícios, sem nenhum dado pessoal direto."""
    popular_protocolos(con)
    rng = random.Random(seed)
    datas = [f"2026-09-{d:02d}" for d in range(1, 29)]
    for leito in range(1, n_pacientes + 1):
        p = rng.choice(PROTOCOLOS)
        cur = con.execute(
            "INSERT INTO pacientes (codigo_leito, idade, sexo, protocolo_id, admitido_em) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"leito-{leito}", rng.randint(18, 90), rng.choice(["feminino", "masculino"]),
             p["id"], rng.choice(datas)),
        )
        paciente_id = cur.lastrowid
        for exame in rng.sample(p["exames"], k=min(2, len(p["exames"]))):
            status = rng.choice(["pendente", "coletado", "resultado_disponivel"])
            con.execute(
                "INSERT INTO exames (paciente_id, nome, status, solicitado_em) VALUES (?, ?, ?, ?)",
                (paciente_id, exame, status, rng.choice(datas)),
            )


def buscar_paciente_por_leito(con: sqlite3.Connection, codigo_leito: str) -> Paciente | None:
    linha = con.execute(
        "SELECT id, codigo_leito, idade, sexo, protocolo_id FROM pacientes WHERE codigo_leito = ?",
        (codigo_leito,),
    ).fetchone()
    return Paciente(**dict(linha)) if linha else None


def exames_pendentes(con: sqlite3.Connection, paciente_id: int) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT nome, solicitado_em FROM exames WHERE paciente_id = ? AND status = 'pendente' "
        "ORDER BY solicitado_em",
        (paciente_id,),
    ).fetchall()


def registrar_evento_auditoria(
    con: sqlite3.Connection, tipo: str, conteudo: str, origem: str, paciente_id: int | None = None
) -> None:
    """Toda sugestão do assistente e toda decisão do médico ficam aqui (requisito de auditoria do PDF)."""
    con.execute(
        "INSERT INTO eventos_auditoria (paciente_id, tipo, conteudo, origem) VALUES (?, ?, ?, ?)",
        (paciente_id, tipo, conteudo, origem),
    )


def inicializar(caminho: str | Path, n_pacientes: int = 20, seed: int = 42) -> None:
    with conectar(caminho) as con:
        criar_esquema(con)
        ja_tem_dados = con.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0] > 0
        if not ja_tem_dados:
            popular_sinteticamente(con, n_pacientes, seed)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cria e popula a base de prontuários sintética.")
    ap.add_argument("--db", default="data/processed/prontuarios.db")
    ap.add_argument("--pacientes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    inicializar(args.db, args.pacientes, args.seed)
    print(f"Base pronta em {args.db}")
