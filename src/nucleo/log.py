"""
Log estruturado: legivel no terminal, em JSON no arquivo.

POR QUE ISTO EXISTE

O sistema tinha `print()` espalhado. Sem nivel, sem carimbo de tempo, sem
arquivo. Depurar dependia de estar olhando o terminal na hora — e vários
problemas de 07 e 08/08 so foram entendidos porque alguem tinha copiado a
saida por acaso.

DUAS SAIDAS, DE PROPOSITO

    terminal  para humano, curto, colorido por nivel
    arquivo   uma linha JSON por evento, para analise posterior

O arquivo e o que permite responder "o que aconteceu as 21h32" depois do fato.
E ele nao serve so para erro: cada mudanca de estado de camera vira registro.

REGRA

Erro nunca e engolido. Se algo falhar e o sistema seguir em frente, isso e uma
DECISAO — e a decisao aparece no log com o motivo.
"""

import json
import logging
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
PASTA_LOGS = RAIZ / "dados" / "logs"

CORES = {
    "DEBUG": "\033[90m", "INFO": "\033[0m", "WARNING": "\033[33m",
    "ERROR": "\033[31m", "CRITICAL": "\033[1;31m",
}
FIM = "\033[0m"


class FormatoTerminal(logging.Formatter):
    def format(self, r):
        cor = CORES.get(r.levelname, "")
        hora = time.strftime("%H:%M:%S", time.localtime(r.created))
        comp = getattr(r, "componente", r.name)
        msg = r.getMessage()
        extra = getattr(r, "dados", None)
        if extra:
            msg += "  " + " ".join(f"{k}={v}" for k, v in extra.items())
        return f"{cor}[{hora}] {comp:14} {msg}{FIM}"


class FormatoJson(logging.Formatter):
    def format(self, r):
        d = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r.created))
                 + f".{int(r.msecs):03d}",
            "nivel": r.levelname,
            "componente": getattr(r, "componente", r.name),
            "mensagem": r.getMessage(),
        }
        if getattr(r, "dados", None):
            d["dados"] = r.dados
        if r.exc_info:
            d["excecao"] = self.formatException(r.exc_info)
        return json.dumps(d, ensure_ascii=False, default=str)


_configurado = False


def configurar(nivel="INFO", arquivo=True):
    global _configurado
    if _configurado:
        return
    _configurado = True

    raiz = logging.getLogger("so")
    raiz.setLevel(getattr(logging, nivel.upper(), logging.INFO))
    raiz.propagate = False

    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(FormatoTerminal())
    raiz.addHandler(h)

    if arquivo:
        try:
            PASTA_LOGS.mkdir(parents=True, exist_ok=True)
            f = logging.FileHandler(PASTA_LOGS / "sistema.jsonl", encoding="utf-8")
            f.setFormatter(FormatoJson())
            raiz.addHandler(f)
        except OSError as e:
            # Log e telemetria: nao pode derrubar o programa. Mesma regra que
            # o Publicador aprendeu em 08/08, quando o arquivo aberto no editor
            # quebrou a sessao inteira.
            print(f"[log] sem arquivo de log: {e}")


class Log:
    """Registrador de um componente. `Log("camera.C920")`."""

    def __init__(self, componente):
        self.componente = componente
        self._l = logging.getLogger(f"so.{componente}")

    def _emitir(self, nivel, msg, exc=None, **dados):
        self._l.log(nivel, msg, exc_info=exc,
                    extra={"componente": self.componente, "dados": dados})

    def debug(self, msg, **d):
        self._emitir(logging.DEBUG, msg, **d)

    def info(self, msg, **d):
        self._emitir(logging.INFO, msg, **d)

    def aviso(self, msg, **d):
        self._emitir(logging.WARNING, msg, **d)

    def erro(self, msg, exc=None, **d):
        self._emitir(logging.ERROR, msg, exc=exc, **d)

    def critico(self, msg, exc=None, **d):
        self._emitir(logging.CRITICAL, msg, exc=exc, **d)
