"""Prepara o dataset de fine-tuning: carrega, anonimiza, cura e divide.

Entrada: arquivos .jsonl em data/raw/ com os campos
  instruction, input, output, source, type
Saída: data/processed/{train,val,test}.jsonl + relatorio_preparo.json

Uso: python -m src.preprocessing.build_dataset --raw data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .anonymizer import anonimizar, contem_pii

CAMPOS = ("instruction", "input", "output")


def carregar(pasta: Path) -> list[dict]:
    registros = []
    for arq in sorted(pasta.glob("*.jsonl")):
        with arq.open(encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    registros.append(json.loads(linha))
    return registros


def _normalizar(texto: str) -> str:
    return re.sub(r"[ \t]+", " ", texto.replace("\r\n", "\n")).strip()


def curar(registros: list[dict], min_saida: int = 20, max_total: int = 4000):
    """Anonimiza, limpa, filtra e remove duplicatas. Retorna (registros, relatório)."""
    rel = {"entrada": len(registros), "descartados_vazios": 0, "descartados_tamanho": 0,
           "descartados_duplicados": 0, "descartados_pii_residual": 0, "pii_substituida": Counter()}
    vistos: set[str] = set()
    saida = []
    for r in registros:
        novo = dict(r)
        for campo in CAMPOS:
            res = anonimizar(_normalizar(str(r.get(campo, ""))))
            novo[campo] = res.texto
            rel["pii_substituida"].update(res.contagem)
        if not novo["instruction"] or not novo["output"]:
            rel["descartados_vazios"] += 1
            continue
        if len(novo["output"]) < min_saida or sum(len(novo[c]) for c in CAMPOS) > max_total:
            rel["descartados_tamanho"] += 1
            continue
        chave = hashlib.sha256("\x1f".join(novo[c].lower() for c in CAMPOS).encode()).hexdigest()
        if chave in vistos:
            rel["descartados_duplicados"] += 1
            continue
        vistos.add(chave)
        if any(contem_pii(novo[c]) for c in CAMPOS):
            rel["descartados_pii_residual"] += 1
            continue
        saida.append(novo)
    rel["saida"] = len(saida)
    rel["pii_substituida"] = dict(rel["pii_substituida"])
    return saida, rel


def dividir(registros: list[dict], seed: int, val: float = 0.1, teste: float = 0.1):
    """Divisão estratificada por tipo, reprodutível pela seed."""
    rng = random.Random(seed)
    por_tipo: dict[str, list[dict]] = defaultdict(list)
    for r in registros:
        por_tipo[r.get("type", "outro")].append(r)
    treino, validacao, teste_set = [], [], []
    for tipo in sorted(por_tipo):
        grupo = por_tipo[tipo]
        rng.shuffle(grupo)
        n_val = max(1, round(len(grupo) * val)) if len(grupo) >= 3 else 0
        n_teste = max(1, round(len(grupo) * teste)) if len(grupo) >= 3 else 0
        validacao += grupo[:n_val]
        teste_set += grupo[n_val:n_val + n_teste]
        treino += grupo[n_val + n_teste:]
    for parte in (treino, validacao, teste_set):
        rng.shuffle(parte)
    return treino, validacao, teste_set


def _gravar(caminho: Path, registros: list[dict]) -> None:
    with caminho.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    brutos = carregar(Path(args.raw))
    if not brutos:
        raise SystemExit(f"Nenhum .jsonl encontrado em {args.raw}")
    curados, rel = curar(brutos)
    treino, val, teste = dividir(curados, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for nome, dados in (("train", treino), ("val", val), ("test", teste)):
        _gravar(out / f"{nome}.jsonl", dados)
    rel["divisao"] = {"train": len(treino), "val": len(val), "test": len(teste)}
    rel["tipos"] = dict(Counter(r["type"] for r in curados))
    (out / "relatorio_preparo.json").write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rel, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
