"""O ambiente que as cameras entregam. A estante e a regua.

    eu so preciso que esse ambiente virtual esteja sendo mostrado na minha
    interface da maneira correta e que eu consiga visualizar o gemeo, a
    estante, o chao e uma area para se movimentar de uma forma coerente
                                                    — Eduardo, 18/08

O QUE MUDOU, E POR QUE ELE ESTAVA CERTO

A versao anterior forcava a nuvem reconstruida a caber dentro do retangulo de
1,65 x 1,32 m da homografia. Seis corridas terminaram em zero ancoras, e o
motivo final foi humilde: a mascara de confianca do DUSt3R descarta piso liso,
e o retangulo calibrado e quase todo piso liso.

Mas o retangulo nunca foi necessario. Ele entrou porque eu tirava dele o
METRO — e o metro ja existia num lugar mais simples e mais firme:

    a estante mede 1,90 m de altura, medida com trena

Achar a coisa alta em cima do chao e dividir pela altura conhecida da a
escala. Nenhuma homografia, nenhuma ancora, nenhuma correspondencia.

    Quando um objeto de dimensao conhecida esta na cena, ele e a regua. Ir
    buscar a escala noutro instrumento e atravessar a rua para pegar o que
    esta na mao.

A ORDEM, E ELA CABE EM CINCO LINHAS

    1. plano dominante da nuvem      -> o CHAO, e ele vira z = 0
    2. o que fica ACIMA do chao      -> os objetos
    3. o maior aglomerado alto       -> a ESTANTE
    4. altura dela / 1,90            -> a ESCALA, e tudo vira metro
    5. o contorno do chao            -> a AREA de movimento

O que sai daqui e o ambiente que as cameras viram, em metros, pronto para
desenhar. Nao e o ambiente que eu queria que elas vissem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Espessura do plano, como FRACAO do tamanho da nuvem.
#
# Absoluto nao serve: a nuvem destas redes nasce sem unidade, e a escala so
# aparece no passo 4. Um numero fixo aqui vale centimetros numa cena e metros
# noutra.
#
#     Uma tolerancia absoluta sobre um dado sem unidade nao e frouxa nem
#     apertada: e indefinida ate alguem medir a escala.
FRACAO_DA_ESPESSURA = 0.004

# Fracao dos pontos que precisa cair no plano para ele ser o chao.
#
# Baixo de proposito. Numa cena com estante, mesa e parede, o piso pode ser
# minoria — e continuar sendo o piso. Exigir maioria e supor um quarto vazio.
FRACAO_DO_CHAO = 0.12

# A partir de que altura, em fracao da altura da estante, um ponto conta como
# "objeto" e nao como chao sujo.
ACIMA_DO_CHAO = 0.15


@dataclass
class Ambiente3D:
    """O lugar, em metros, do jeito que as cameras entregaram."""
    nuvem: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    escala: float = 0.0
    estante: tuple | None = None       # (x, y, rumo) em metros
    altura_da_cena: float = 0.0

    @property
    def pronto(self):
        return len(self.nuvem) > 100 and self.escala > 0

    @property
    def chao(self):
        """A area de movimento: (xmin, xmax, ymin, ymax) em metros.

        Sai do CONTORNO do piso reconstruido, e nao de um numero digitado. Se
        a camera enxergou tres metros de chao, a area tem tres metros.
        """
        if not len(self.nuvem):
            return None
        rasos = self.nuvem[self.nuvem[:, 2] < 0.10]
        if len(rasos) < 20:
            rasos = self.nuvem
        # percentis e nao extremos: um ponto solto no fundo do corredor
        # esticaria o piso ate ele
        x0, x1 = np.percentile(rasos[:, 0], [2, 98])
        y0, y1 = np.percentile(rasos[:, 1], [2, 98])
        return (float(x0), float(x1), float(y0), float(y1))


def plano_dominante(pontos, tolerancia=None, tentativas=300, semente=0):
    """O maior plano da nuvem — num quarto, o chao. RANSAC.

    Devolve (normal_unitaria, ponto_do_plano, mascara_dos_inliers), ou None.

    RANSAC e nao minimos quadrados porque a nuvem tem parede, movel e ruido:
    um ajuste que usa todos os pontos encontra o plano que agrada a todos e
    nao descreve nenhum.
    """
    p = np.asarray(pontos, dtype=float).reshape(-1, 3)
    if len(p) < 3:
        return None

    if tolerancia is None:
        diagonal = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
        tolerancia = max(diagonal * FRACAO_DA_ESPESSURA, 1e-9)

    rng = np.random.default_rng(semente)
    melhor, quantos = None, 0
    for _ in range(tentativas):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norma = np.linalg.norm(n)
        if norma < 1e-9:
            continue
        n = n / norma
        dentro = np.abs((p - a) @ n) <= tolerancia
        if dentro.sum() > quantos:
            melhor, quantos = (n, a, dentro), int(dentro.sum())

    if melhor is None or quantos < FRACAO_DO_CHAO * len(p):
        return None
    return melhor


def _de_pe(normal):
    """A rotacao que deita o plano do chao: normal -> (0, 0, 1)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    eixo = np.cross(n, np.array([0.0, 0.0, 1.0]))
    seno = float(np.linalg.norm(eixo))
    if seno < 1e-9:
        return np.eye(3)
    eixo = eixo / seno
    ang = math.atan2(seno, float(n @ np.array([0.0, 0.0, 1.0])))
    k = np.array([[0, -eixo[2], eixo[1]],
                  [eixo[2], 0, -eixo[0]],
                  [-eixo[1], eixo[0], 0]], dtype=float)
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)


def sem_paredes(pontos, quantas=3):
    """Tira os planos VERTICAIS da nuvem ja deitada. Sobra o que e movel.

    A PAREDE E MAIS ALTA QUE A ESTANTE, e por isso "o que sobe" nao basta.

    Um quarto tem 2,5 m de pe-direito e a estante tem 1,90. Procurar o ponto
    mais alto encontra a parede — e a pegada dela, sendo uma faixa longa na
    borda do quarto, arrasta o retangulo da estante para o lado.

    O que separa os dois nao e altura: e FORMA. Parede e um plano vertical
    grande; estante e um volume compacto. Depois de deitar o chao, plano
    vertical tem normal quase horizontal — a componente z dela e quase zero.

        Altura nao distingue movel de parede. Ambos sobem. O que distingue e
        que a parede e chapada e o movel tem os quatro lados.

    Remove ate `quantas` planos verticais grandes, um por vez.
    """
    p = np.asarray(pontos, dtype=float).reshape(-1, 3)

    # O CHAO SAI PRIMEIRO, e sem isso o resto nao acontece.
    #
    # A nuvem que chega aqui ainda tem o piso, e o piso E o plano dominante.
    # A primeira volta do laco o encontrava, via que ele e horizontal, e
    # parava — sem nunca chegar na parede.
    #
    #     Procurar a segunda coisa mais comum sem tirar a primeira e
    #     encontrar a primeira de novo.
    #
    # Depois de deitar, o chao esta em z ~ 0. Tirar a faixa rasa deixa so o
    # que sobe: parede e movel.
    alto_da_cena = float(np.percentile(p[:, 2], 99.0))
    if alto_da_cena > 0:
        p = p[p[:, 2] > alto_da_cena * 0.05]

    for _ in range(quantas):
        if len(p) < 200:
            break
        achado = plano_dominante(p)
        if achado is None:
            break
        normal, _, dentro = achado
        # vertical: a normal deita, entao |z| dela e pequeno
        if abs(float(normal[2])) > 0.35:
            break                       # ja nao e parede; para de tirar
        if dentro.sum() < 0.10 * len(p):
            break                       # plano pequeno demais para ser parede
        p = p[~dentro]
    return p


def achar_estante(pontos, largura_alvo, profundidade_alvo):
    """A estante na nuvem ja deitada: (x, y, rumo, altura) em unidades da nuvem.

    A ESTANTE E O VOLUME COMPACTO QUE SOBE, depois de tiradas as paredes.

    Procura o aglomerado alto com maior extensao horizontal e ajusta um
    retangulo girado sobre a pegada dele. O rumo sai do lado maior.

    Devolve None quando nao ha nada alto o bastante: dizer "nao achei" e uma
    resposta, e desenhar uma estante onde nao ha e que nao e.
    """
    import cv2

    p = sem_paredes(np.asarray(pontos, dtype=float).reshape(-1, 3))
    if len(p) < 50:
        return None

    teto = float(np.percentile(p[:, 2], 99.0))
    if teto <= 0:
        return None

    altos = p[p[:, 2] > teto * 0.40]
    if len(altos) < 30:
        return None

    pegada = altos[:, :2].astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pegada)
    if w < 1e-9 or h < 1e-9:
        return None

    # o lado maior e a largura da face; o rumo sai dele, na convencao do
    # projeto (normal = -sin r, cos r)
    rumo = math.radians(ang) if w >= h else math.radians(ang) + math.pi / 2
    return (float(cx), float(cy), float(rumo), teto)


def montar(nuvem, gabarito):
    """A nuvem crua vira ambiente em metros. E o passo unico deste modulo.

    `gabarito` precisa ter `.altura` — a altura da estante medida com trena.
    E ela que da o metro:

        escala = altura_de_trena / altura_da_estante_na_nuvem

    Devolve `Ambiente3D`, ou None quando nao ha chao reconhecivel.
    """
    p = np.asarray(nuvem, dtype=float).reshape(-1, 3)
    if len(p) < 100:
        return None

    # 1. o chao
    achado = plano_dominante(p)
    if achado is None:
        return None
    normal, no_plano, _ = achado

    # 2. deitar: o chao vira z = 0 e o resto fica acima
    giro = _de_pe(normal)
    deitada = (giro @ (p - no_plano).T).T
    # se o grosso da cena ficou abaixo de zero, a normal apontava para baixo
    if np.median(deitada[:, 2]) < 0:
        deitada[:, 2] *= -1.0

    # 3. e 4. a estante da a escala
    achada = achar_estante(deitada, gabarito.largura, gabarito.profundidade)
    if achada is None:
        return None
    cx, cy, rumo, alto_na_nuvem = achada
    if alto_na_nuvem <= 1e-9:
        return None
    escala = float(gabarito.altura) / alto_na_nuvem

    em_metros = deitada * escala
    estante = (cx * escala, cy * escala, rumo)

    return Ambiente3D(nuvem=em_metros, escala=escala, estante=estante,
                      altura_da_cena=float(em_metros[:, 2].max()))
