"""Tira o tremelique do desenho sem tirar o movimento — nem atrasar o boneco.

    ele fica dando uns tremiliques tambem, precisa ser mais suave
                                                    — Eduardo, 12/08
    o gemeo digital nao funciona totalmente certo, precisamos pensar em como
    melhorar o movimento e o acerto de movimento
                                                    — Eduardo, 19/08

O DESENHO TREME PORQUE A MEDIDA TREME, E A MEDIDA TREME PORQUE E MEDIDA.

A caixa que o detector devolve muda alguns pixels por quadro mesmo com a
pessoa imovel: o nariz entra e sai da caixa, a perna do fundo aparece, o
compressor da webcam decide outra coisa. Convertido em metros pela homografia,
isso vira um ou dois centimetros de salto por quadro. Ninguem ve dois
centimetros — mas todo mundo ve DEZ SALTOS POR SEGUNDO.

O que o olho le como "tremendo" nao e a amplitude: e a frequencia.

O DEFEITO DE 19/08: AMACIAR ERA ATRASAR

A primeira versao era media exponencial pura. Ela alisa o ruido e, no mesmo
gesto, deixa o boneco para tras de quem anda — o erro em regime permanente de
uma media exponencial contra um alvo em velocidade constante e

    atraso = (1 - alfa) / alfa * v * dt

que nao e pequeno. Medido, com alfa = 0,25:

     v (m/s)      6,5 fps      15 fps
      0,39         18 cm        8 cm
      0,80         37 cm       16 cm
      1,20         42 cm       24 cm

Mais a idade da propria medida: entre a camera capturar e a tela desenhar
passam o detector (130 ms medidos), a fila e o ciclo. A 1 m/s sao outros
15 cm. Num comodo de 4 m, meio metro de atraso e um oitavo da sala.

    Um filtro que so puxa a saida na direcao da entrada nunca alcanca uma
    entrada que anda. Ele nao esta mal afinado: esta resolvendo outro
    problema.

O QUE MUDOU: PREVER, DEPOIS CORRIGIR

O Kalman ja mede a velocidade e ela chegava aqui e era jogada fora. Agora:

    1. a velocidade e amaciada (ela muda devagar; o ruido dela, nao)
    2. a medida e trazida da hora em que foi TIRADA ate agora        -> alvo
    3. o boneco anda sozinho com a velocidade dele                   -> previsto
    4. a correcao amacia so a DIFERENCA entre os dois

Contra um alvo em velocidade constante o atraso vai a ZERO, com a mesma
suavidade de antes — porque o que sobrou para amaciar e so o ruido. Medido na
simulacao: 0,0 cm a 0,39, 0,80 e 1,20 m/s.

    Alisar o erro e diferente de alisar o sinal. So o primeiro e de graca.

E O TEMPO PASSOU A SER MEDIDO EM SEGUNDOS, NAO EM QUADROS

`alfa` por quadro amarra o comportamento ao fps: a mesma constante amacia o
dobro quando a maquina cai de 15 para 6,5 quadros por segundo. Como o proximo
passo do projeto e justamente subir o fps, o filtro mudaria de personalidade
sozinho. Aqui as constantes sao MEIAS-VIDAS EM SEGUNDOS e o alfa sai do dt
real de cada chamada.

    Constante por quadro e uma constante que depende da maquina.
"""
from __future__ import annotations

import math
import time

LN2 = math.log(2.0)


def _alfa(dt, meia_vida_s):
    """O peso da medida nova para este intervalo. dt=0 nao corrige nada."""
    if dt <= 0.0 or meia_vida_s <= 0.0:
        return 0.0 if dt <= 0.0 else 1.0
    return 1.0 - math.exp(-LN2 * dt / meia_vida_s)


def _mistura_angulo(atual, alvo, alfa):
    """Interpola dois angulos pelo caminho curto.

    Sem isto, alguem virando de +179 graus para -179 graus (dois graus de
    giro real) faria o boneco girar 358 graus no sentido contrario — o pior
    tremelique possivel, e justamente no momento em que o rumo mais importa.

    Interpolar os SENOS E COSSENOS nao tem esse defeito, porque o circulo nao
    tem costura. E a mesma razao pela qual a media de azimutes e feita em
    vetores e nao em graus.
    """
    sx = (1 - alfa) * math.cos(atual) + alfa * math.cos(alvo)
    sy = (1 - alfa) * math.sin(atual) + alfa * math.sin(alvo)
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:      # opostos exatos: fica onde esta
        return atual
    return math.atan2(sy, sx)


class Suavizador:
    """Prever, depois corrigir. Um por pessoa, para posicao e rumo.

    `meia_vida_s` — em quanto tempo METADE de um erro de posicao e absorvido.
    Curto demais deixa o ruido passar; longo demais atrasa a reacao a uma
    mudanca de direcao, que e a unica coisa que a previsao nao adivinha.

    `meia_vida_velocidade_s` — a velocidade e o que extrapola, entao o ruido
    dela vira posicao errada. Amaciada mais devagar que a posicao de proposito:
    a velocidade de quem caminha muda em segundos, o ruido dela em quadros.

    `avanco_maximo_m` — teto de quanto a previsao pode adiantar. Uma velocidade
    absurda (id reciclado, salto do rastreador) nao pode arremessar o boneco
    para o outro lado da sala antes que a correcao tenha chance de opinar.

    `salto_m` — a valvula de escape. Quem ANDA nao deve ser amaciado ate virar
    melado, e um id que o rastreador reatribui de um canto ao outro nao deve
    arrastar o boneco pela cena. Acima deste salto o filtro desiste e vai
    direto: e a diferenca entre alisar ruido e esconder movimento.


    `piso_de_velocidade_m_s` — abaixo de que velocidade o Kalman esta medindo
    o proprio ruido e nao movimento. Ver `_confianca_na_velocidade`.
    """

    def __init__(self, meia_vida_s=0.30, meia_vida_rumo_s=0.22,
                 meia_vida_velocidade_s=0.12, salto_m=0.60,
                 avanco_maximo_m=0.35, piso_de_velocidade_m_s=0.25):
        self.meia_vida_s = meia_vida_s
        self.meia_vida_rumo_s = meia_vida_rumo_s
        self.meia_vida_velocidade_s = meia_vida_velocidade_s
        self.salto_m = salto_m
        self.avanco_maximo_m = avanco_maximo_m
        self.piso_de_velocidade_m_s = piso_de_velocidade_m_s
        self._estado = {}

    def _confianca_na_velocidade(self, vx, vy):
        """Quanto desta velocidade e movimento, e quanto e ruido. De 0 a 1.

        MEDIDO EM 19/08: prever com a velocidade crua tirou os 37 cm de
        atraso e trouxe TREMOR — 3,5 cm parado, contra 0,8 cm do filtro
        antigo. Era previsivel depois de visto: velocidade e uma diferenca
        dividida por dt, entao o ruido dela e grande, e prever integra esse
        ruido direto na posicao. Quem esta parado ganhava movimento de graca.

            Abaixo de um piso, velocidade nao e evidencia de movimento: e a
            barra de erro do proprio estimador.

        O peso e `v2 / (v2 + piso2)` — encolhimento, nao limiar. Um limiar
        criaria um degrau bem na velocidade em que as pessoas passam mais
        tempo, e o boneco pularia entre prever e nao prever. Aqui a previsao
        entra devagar: 10% a 0,08 m/s, 71% a 0,39 m/s, 96% a 1,2 m/s.

        Custa 4 cm de atraso a 0,39 m/s. O filtro antigo custava 18 cm.
        """
        piso = self.piso_de_velocidade_m_s
        if piso <= 0.0:
            return 1.0
        v2 = vx * vx + vy * vy
        return v2 / (v2 + piso * piso)

    def suavizar(self, pessoa_id, x, y, rumo=None, vx=0.0, vy=0.0,
                 idade_s=0.0, agora=None):
        """Devolve (x, y, rumo) prontos para desenhar NESTE instante.

        `vx, vy`    velocidade medida pelo Kalman, em m/s
        `idade_s`   ha quanto tempo a camera capturou esta posicao
        `agora`     relogio monotonico; None usa o do sistema
        """
        agora = time.monotonic() if agora is None else float(agora)
        x, y = float(x), float(y)

        antes = self._estado.get(pessoa_id)
        if antes is None:
            self._estado[pessoa_id] = [x, y, rumo, float(vx), float(vy), agora]
            return x, y, rumo

        px, py, prumo, pvx, pvy, pt = antes
        dt = max(0.0, agora - pt)

        # 1. A VELOCIDADE, AMACIADA. E ela que vai extrapolar, entao o ruido
        #    dela custa posicao errada — e ruido de velocidade e maior que o
        #    de posicao, porque velocidade e uma diferenca dividida por dt.
        a_v = _alfa(dt, self.meia_vida_velocidade_s)
        vx = pvx + a_v * (float(vx) - pvx)
        vy = pvy + a_v * (float(vy) - pvy)

        # 1b. E SO A PARTE DELA QUE E MOVIMENTO EXTRAPOLA.
        #     Sem isto, quem esta parado ganha 3,5 cm de tremor de graca —
        #     o ruido da velocidade integrado direto na posicao.
        peso = self._confianca_na_velocidade(vx, vy)
        ux, uy = vx * peso, vy * peso

        # 2. O ALVO: onde a pessoa esta AGORA, e nao onde ela estava quando a
        #    camera a viu. Com teto, para uma velocidade absurda nao arremessar
        #    o boneco antes de a correcao ter chance de opinar.
        avx, avy = ux * max(0.0, idade_s), uy * max(0.0, idade_s)
        avanco = math.hypot(avx, avy)
        if avanco > self.avanco_maximo_m:
            escala = self.avanco_maximo_m / avanco
            avx, avy = avx * escala, avy * escala
        alvo_x, alvo_y = x + avx, y + avy

        # 3. ONDE O BONECO CHEGA SOZINHO. Este e o passo que tira o atraso:
        #    a saida acompanha o movimento por conta propria, e sobra para a
        #    correcao so a diferenca — que e ruido, e ruido se alisa de graca.
        prev_x, prev_y = px + ux * dt, py + uy * dt

        # 4. Salto grande: nem previsao nem media. Vai direto, e a velocidade
        #    de antes deixa de valer junto com a posicao de antes.
        if math.hypot(alvo_x - prev_x, alvo_y - prev_y) > self.salto_m:
            self._estado[pessoa_id] = [x, y, rumo, 0.0, 0.0, agora]
            return x, y, rumo

        # 5. A correcao, so sobre a diferenca.
        a = _alfa(dt, self.meia_vida_s)
        nx = prev_x + a * (alvo_x - prev_x)
        ny = prev_y + a * (alvo_y - prev_y)

        # Rumo ausente NAO zera o rumo: mantem o ultimo conhecido. Zerar faria
        # o boneco dar meia-volta toda vez que a camera do alto perdesse um
        # quadro — inventando um giro que ninguem deu. E `DESCONHECIDO` de
        # novo: nao saber e afirmar "norte" sao coisas diferentes.
        if rumo is None:
            nrumo = prumo
        elif prumo is None:
            nrumo = float(rumo)
        else:
            nrumo = _mistura_angulo(prumo, float(rumo),
                                    _alfa(dt, self.meia_vida_rumo_s))

        self._estado[pessoa_id] = [nx, ny, nrumo, vx, vy, agora]
        return nx, ny, nrumo

    def esquecer(self, vivos):
        for pid in list(self._estado):
            if pid not in vivos:
                del self._estado[pid]
