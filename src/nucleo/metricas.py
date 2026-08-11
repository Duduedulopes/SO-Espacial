"""
Contadores. O que nao e medido nao existe.

O PRINCIPIO

Em 08/08 a auditoria nao conseguiu responder "quantos quadros foram
descartados?" porque nao havia contador. A resposta era fe, nao medida. Pior:
otimizei inferencia por duas rodadas enquanto metade do tempo ia para o
desenho — e so descobri quando um cronometro apareceu.

    Otimizar sem medir e adivinhar com trabalho.

Estas classes sao deliberadamente burras: somam e devolvem. Nenhuma decisao.
"""

import time
from collections import deque
from dataclasses import dataclass, field


class Media:
    """Media movel sobre as ultimas N amostras.

    Movel, nao acumulada: o que interessa e o comportamento AGORA. Uma media
    desde o inicio esconde a degradacao recente sob a boa fase inicial.
    """

    def __init__(self, memoria=60):
        self.amostras = deque(maxlen=memoria)

    def somar(self, v):
        self.amostras.append(float(v))

    @property
    def valor(self):
        return sum(self.amostras) / len(self.amostras) if self.amostras else 0.0

    @property
    def maximo(self):
        return max(self.amostras) if self.amostras else 0.0

    def __len__(self):
        return len(self.amostras)


@dataclass
class MetricasDeFonte:
    """Uma fonte de video, medida."""

    recebidos: int = 0
    descartados: int = 0
    falhas_leitura: int = 0
    reconexoes: int = 0

    _intervalos: Media = field(default_factory=lambda: Media(60))
    _latencias: Media = field(default_factory=lambda: Media(60))
    _brilhos: Media = field(default_factory=lambda: Media(20))

    ultimo_quadro_em: float = 0.0
    _t_anterior: float = 0.0

    def registrar_quadro(self, agora, brilho=None):
        if self._t_anterior:
            self._intervalos.somar(agora - self._t_anterior)
        self._t_anterior = agora
        self.ultimo_quadro_em = agora
        self.recebidos += 1
        if brilho is not None:
            self._brilhos.somar(brilho)

    def registrar_descarte(self, n=1):
        self.descartados += n

    def registrar_latencia(self, ms):
        self._latencias.somar(ms)

    @property
    def fps(self):
        m = self._intervalos.valor
        return 1.0 / m if m > 1e-9 else 0.0

    @property
    def latencia_ms(self):
        return self._latencias.valor

    @property
    def brilho(self):
        return self._brilhos.valor

    def silencio_s(self, agora=None):
        """Ha quanto tempo nao chega quadro. E o sinal que move a maquina de
        estados de ONLINE para DEGRADADA e depois para FALHA."""
        if not self.ultimo_quadro_em:
            return float("inf")
        return (agora or time.monotonic()) - self.ultimo_quadro_em

    def resumo(self):
        return {
            "recebidos": self.recebidos,
            "descartados": self.descartados,
            "falhas_leitura": self.falhas_leitura,
            "reconexoes": self.reconexoes,
            "fps": round(self.fps, 1),
            "latencia_ms": round(self.latencia_ms, 1),
            "brilho": round(self.brilho, 1),
        }


@dataclass
class MetricasDeTrabalhador:
    """Uma etapa de processamento, medida."""

    quadros: int = 0
    saidas: int = 0
    falhas: int = 0
    _duracoes: Media = field(default_factory=lambda: Media(60))

    def registrar(self, ms, n_saidas=0):
        self._duracoes.somar(ms)
        self.quadros += 1
        self.saidas += n_saidas

    @property
    def ms_medio(self):
        return self._duracoes.valor

    @property
    def ms_pior(self):
        return self._duracoes.maximo

    def resumo(self):
        return {
            "quadros": self.quadros,
            "saidas": self.saidas,
            "falhas": self.falhas,
            "ms_medio": round(self.ms_medio, 1),
            "ms_pior": round(self.ms_pior, 1),
        }


class Cronometro:
    """`with Cronometro() as c: ...` e depois `c.ms`."""

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.ms = (time.perf_counter() - self._t) * 1000
        return False
