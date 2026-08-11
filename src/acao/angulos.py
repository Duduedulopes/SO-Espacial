"""
Aritmetica de angulos. Uma fonte so, porque angulo dobra e ninguem percebe.

POR QUE UM ARQUIVO SO PARA ISTO

Angulo nao e numero comum: +179 e -179 estao a 2 graus um do outro, nao a 358.
Toda funcao que soma, subtrai ou faz media de angulos precisa saber disso, e
toda funcao que ESQUECE produz o mesmo defeito — o sistema anuncia uma
meia-volta que nunca aconteceu.

Em 10/08 esse cuidado ja estava escrito dentro do `classificador.py`. A camada
de corpo precisa exatamente do mesmo, e reimplementar seria repetir o defeito
de `para_o_mundo` vs `ancorar_no_chao`: duas copias que podem divergir em
silencio.

    Codigo duplicado deixa escolher a versao incompleta sem perceber.
"""

import math

import numpy as np


def diferenca_angular(a, b):
    """Menor angulo entre dois rumos, com sinal, em radianos.

    Sem isto, passar de +179 para -179 graus vira um giro de 358 em vez de 2.
    """
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def media_circular(angulos):
    """Direcao media de um conjunto de angulos, em radianos.

    POR QUE NAO `numpy.mean`

    A media aritmetica de 179 e -179 graus da ZERO — a direcao exatamente
    oposta a verdadeira, que e 180. Nao e um erro pequeno: e o pior erro
    possivel.

    A media circular soma VETORES UNITARIOS e devolve a direcao da soma. Nao
    tem como errar a volta, porque nunca trabalha com o numero do angulo.
    """
    a = np.asarray(angulos, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.arctan2(np.sin(a).sum(), np.cos(a).sum()))


def concentracao(angulos):
    """Quanto os angulos CONCORDAM entre si. 1 = identicos, 0 = espalhados.

    E o modulo do vetor medio, normalizado — a estatistica classica `R` de
    dados circulares.

    PARA QUE SERVE AQUI

    O mesmo papel que a dispersao de 30% cumpre no filtro de altura: dizer
    quando o estimador NAO conseguiu ajustar o proprio modelo, para que ele
    se abstenha em vez de responder qualquer coisa.

        Um estimador que nao consegue ajustar deve se abster, nao chutar.
    """
    a = np.asarray(angulos, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.hypot(np.sin(a).sum(), np.cos(a).sum()) / a.size)
