"""Tira o tremelique do desenho sem tirar o movimento.

    ele fica dando uns tremiliques tambem, precisa ser mais suave
                                                    — Eduardo, 12/08

O DESENHO TREME PORQUE A MEDIDA TREME, E A MEDIDA TREME PORQUE E MEDIDA.

A caixa que o detector devolve muda alguns pixels por quadro mesmo com a
pessoa imovel: o nariz entra e sai da caixa, a perna do fundo aparece, o
compressor da webcam decide outra coisa. Convertido em metros pela homografia,
isso vira um ou dois centimetros de salto por quadro. Ninguem ve dois
centimetros — mas todo mundo ve DEZ SALTOS POR SEGUNDO.

O que o olho lê como "tremendo" não é a amplitude: é a frequência. Então é a
frequência que este arquivo ataca.

POR QUE NAO CONSERTAR NO KALMAN, QUE JA EXISTE

O Kalman do `gemeo` estima ONDE A PESSOA ESTA — e ele deve continuar
respondendo rápido, porque quem consome aquela resposta é a lógica: quem
está na frente de qual prateleira, quem entrou na zona. Amaciar o Kalman
custaria atraso na DECISAO.

Aqui o consumidor é o olho, que tolera 200 ms de atraso e não tolera
tremelique. São dois requisitos opostos sobre o mesmo número — então são dois
filtros, e este fica do lado de quem desenha. A camada que decide continua
vendo a medida crua.

    O mesmo número serve a dois donos com exigências opostas.
    Duplicar o filtro é mais barato que escolher um dono.
"""
from __future__ import annotations

import math


def _mistura_angulo(atual, alvo, alfa):
    """Interpola dois ângulos pelo caminho curto.

    Sem isto, alguém virando de +179 graus para -179 graus (dois graus de
    giro real) faria o boneco girar 358 graus no sentido contrário — o pior
    tremelique possível, e justamente no momento em que o rumo mais importa.

    Interpolar os SENOS E COSSENOS não tem esse defeito, porque o círculo não
    tem costura. É a mesma razão pela qual a média de azimutes é feita em
    vetores e não em graus.
    """
    sx = (1 - alfa) * math.cos(atual) + alfa * math.cos(alvo)
    sy = (1 - alfa) * math.sin(atual) + alfa * math.sin(alvo)
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:      # opostos exatos: fica onde está
        return atual
    return math.atan2(sy, sx)


class Suavizador:
    """Um filtro exponencial por pessoa, para posição e rumo.

    `alfa` é o peso da medida nova. Perto de 0 fica lento e liso; perto de 1
    fica rápido e trêmulo. 0,25 a 10 fps significa que metade de um salto é
    absorvida em ~2,4 quadros: o suficiente para o olho não ver o pulo, pouco
    o bastante para o boneco não chegar atrasado onde a pessoa está.

    `salto_m` é a válvula de escape. Alguém que ANDA de verdade não deve ser
    amaciado até virar melado, e um id que o tracker reatribui de um canto da
    sala para o outro não deve arrastar o boneco pela cena. Acima deste salto o
    filtro desiste e vai direto — é a diferença entre alisar ruído e esconder
    movimento.
    """

    def __init__(self, alfa=0.25, alfa_rumo=0.15, salto_m=0.60):
        self.alfa = alfa
        self.alfa_rumo = alfa_rumo
        self.salto_m = salto_m
        self._estado = {}

    def suavizar(self, pessoa_id, x, y, rumo=None):
        """Devolve (x, y, rumo) amaciados. `rumo` em radianos, ou None."""
        antes = self._estado.get(pessoa_id)
        if antes is None:
            self._estado[pessoa_id] = (float(x), float(y), rumo)
            return float(x), float(y), rumo

        px, py, prumo = antes
        if math.hypot(x - px, y - py) > self.salto_m:
            self._estado[pessoa_id] = (float(x), float(y), rumo)
            return float(x), float(y), rumo

        nx = px + self.alfa * (x - px)
        ny = py + self.alfa * (y - py)

        # Rumo ausente NAO zera o rumo: mantém o último conhecido. Zerar faria
        # o boneco dar meia-volta toda vez que a câmera do alto perdesse um
        # quadro — inventando um giro que ninguém deu. É `DESCONHECIDO` de
        # novo: não saber e afirmar "norte" são coisas diferentes.
        if rumo is None:
            nrumo = prumo
        elif prumo is None:
            nrumo = float(rumo)
        else:
            nrumo = _mistura_angulo(prumo, float(rumo), self.alfa_rumo)

        self._estado[pessoa_id] = (nx, ny, nrumo)
        return nx, ny, nrumo

    def esquecer(self, vivos):
        for pid in list(self._estado):
            if pid not in vivos:
                del self._estado[pid]
