"""Formato de prompt único, usado no treino (notebook) e na avaliação.

Mantém o template em um só lugar para o texto de treino e de inferência
nunca divergirem (causa comum de queda de qualidade após fine-tuning).
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "Você é um assistente virtual de apoio à decisão clínica do Hospital "
    "(dados fictícios, uso didático). Baseie-se nos protocolos internos. "
    "Nunca prescreva medicação, dose ou alta sem validação de um médico responsável. "
    "Sempre cite a fonte do protocolo usado e reforce que a conduta final depende do médico."
)


def montar_mensagens(registro: dict) -> list[dict]:
    """Converte um registro {instruction, input, output} em mensagens de chat."""
    conteudo_usuario = registro["instruction"]
    if registro.get("input"):
        conteudo_usuario += f"\n\n{registro['input']}"
    mensagens = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": conteudo_usuario},
    ]
    if "output" in registro:
        mensagens.append({"role": "assistant", "content": registro["output"]})
    return mensagens


def montar_texto_treino(registro: dict, tokenizer) -> str:
    """Aplica o chat template do tokenizer ao exemplo completo (para treino)."""
    return tokenizer.apply_chat_template(montar_mensagens(registro), tokenize=False)


def montar_prompt_inferencia(registro: dict, tokenizer) -> str:
    """Mesma formatação, mas sem a resposta — para gerar em inferência/avaliação."""
    msgs = montar_mensagens({k: v for k, v in registro.items() if k != "output"})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
