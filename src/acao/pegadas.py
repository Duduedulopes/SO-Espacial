"""QUANTAS unidades. A terceira pergunta.

    Eu quero responder QUEM esta pegando, O QUE, e QUANTAS unidades.
                                                        — Eduardo

QUEM ja tem resposta (o id do rastro). O QUE ja tem resposta (o classificador
de prateleira). QUANTAS nao tinha nenhuma — e sem ela a demonstracao mostra que
alguem mexeu numa prateleira sem nunca dizer quanto saiu de la.

O QUE CONTA COMO UMA UNIDADE

Um gesto COMPLETO: o braco sai de baixo, sobe ate a prateleira, e volta.

Contar quadros em que o braco esta levantado daria dezenas por gesto. Contar a
subida daria a resposta certa quase sempre e um numero absurdo quando o braco
tremesse no limiar. Entao o que conta e o CICLO FECHADO — e ele so fecha na
descida, que e justamente quando a mao ja saiu da prateleira.

    Um evento so pode ser contado depois de terminar. Contar no comeco e
    contar intencao, e intencao se desfaz.

E o mesmo raciocinio da histerese em `classificador.py`: nao e o valor
instantaneo que decide, e a travessia confirmada.

O QUE ESTE CONTADOR NAO SABE

Ele nao sabe se a mao voltou VAZIA. Alguem que estende o braco, pensa e desiste
conta como uma unidade aqui. Resolver isso exige ver o produto, nao o corpo —
e e exatamente o que o Gemini do LOJA AUTONOMA PRO ja faz do outro lado. Este
numero e a contagem de GESTOS, e o nome do campo diz isso.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.acao.vocabulario import Braco

# Estados em que a mao esta em algum lugar que nao o proprio corpo.
ALCANCANDO = (Braco.LEVANTADO, Braco.ESTENDIDO)

# Quantos quadros seguidos confirmam a subida e a descida. Um braco que oscila
# no limiar entre dois quadros nao fecha ciclo — os dois lados exigem prova.
CONFIRMACOES = 2


@dataclass
class Pegada:
    """Um gesto completo: quem, de qual prateleira, com que braco."""
    pessoa_id: int
    prateleira: str
    lado: str


@dataclass
class _Lado:
    alcancando: bool = False
    seguidos: int = 0
    prateleira: str = ""


@dataclass
class ContadorDePegadas:
    """Conta gestos fechados por pessoa e por prateleira."""

    confirmacoes: int = CONFIRMACOES
    _estado: dict = field(default_factory=dict)
    _contagem: dict = field(default_factory=dict)
    _ultimas: list = field(default_factory=list)

    def observar(self, pessoa_id, braco_esq, braco_dir, prateleira=None):
        """Alimenta um quadro. Devolve as pegadas FECHADAS neste quadro."""
        fechadas = []
        for lado, estado in (("esq", braco_esq), ("dir", braco_dir)):
            chave = (pessoa_id, lado)
            s = self._estado.setdefault(chave, _Lado())
            alto = estado in ALCANCANDO

            if alto == s.alcancando:
                s.seguidos = 0
                # Enquanto a mao esta la em cima, o palpite pode melhorar. O
                # ultimo antes da descida e o que vale: e o mais informado.
                if alto and prateleira:
                    s.prateleira = prateleira
                continue

            s.seguidos += 1
            if s.seguidos < self.confirmacoes:
                continue

            s.seguidos = 0
            s.alcancando = alto
            if alto:                                  # subiu: abre o ciclo
                s.prateleira = prateleira or ""
                continue

            # Desceu: o ciclo fechou, e so agora ele conta.
            p = Pegada(pessoa_id, s.prateleira or "?", lado)
            self._contagem[(pessoa_id, p.prateleira)] = self.quantas(
                pessoa_id, p.prateleira) + 1
            self._ultimas.append(p)
            del self._ultimas[:-12]
            fechadas.append(p)
            s.prateleira = ""
        return fechadas

    def quantas(self, pessoa_id, prateleira=None):
        if prateleira is not None:
            return self._contagem.get((pessoa_id, prateleira), 0)
        return sum(n for (pid, _), n in self._contagem.items() if pid == pessoa_id)

    def por_prateleira(self, pessoa_id):
        return {pr: n for (pid, pr), n in sorted(self._contagem.items())
                if pid == pessoa_id}

    @property
    def ultimas(self):
        return list(self._ultimas)

    @property
    def total(self):
        return sum(self._contagem.values())

    def esquecer(self, vivos):
        """A contagem NAO morre com a pessoa: ela ja aconteceu.

        O estado do braco morre — ele descreve um corpo que saiu de cena. Mas
        apagar as unidades de quem foi embora seria apagar a venda junto com o
        cliente, e a resposta que interessa e justamente essa.
        """
        for pid, lado in list(self._estado):
            if pid not in vivos:
                del self._estado[(pid, lado)]
