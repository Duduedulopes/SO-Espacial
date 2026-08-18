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
    escala_da_estante: float = 0.0     # a outra regua, para conferencia
    residuo_m: float = 0.0             # quanto a ponte discorda, em metros
    ancoras: int = 0                   # quantos pares casaram

    @property
    def no_mundo_do_gemeo(self):
        """A cena esta no sistema em que o gemeo e rastreado?

        Sem isso, o ambiente esta certo em forma e tamanho e flutuando num
        sistema proprio — e o boneco atravessa a estante.
        """
        return self.ancoras > 0

    @property
    def as_duas_reguas_concordam(self, tolerancia=0.20):
        """A escala da homografia bate com a da altura da estante?

        Duas reguas independentes medindo a mesma cena. Vinte por cento de
        folga porque a altura da estante na nuvem e estimada pelo percentil, e
        a homografia tem o erro dela.
        """
        if not (self.escala > 0 and self.escala_da_estante > 0):
            return None
        razao = self.escala / self.escala_da_estante
        return abs(razao - 1.0) <= tolerancia

    @property
    def pronto(self):
        return len(self.nuvem) > 100 and self.escala > 0

    @property
    def chao(self):
        """TUDO que as cameras enxergaram, em metros. (xmin, xmax, ymin, ymax).

        SEM RECORTE. Consertado em 18/08, e foi a terceira vez que eu recortei
        a visao delas sem ser pedido.

        As versoes anteriores cortavam duas vezes: so pontos com z < 0,10 m, e
        so entre os percentis 2 e 98. Cada corte tinha uma justificativa
        razoavel — "so o piso", "ponto solto estica o chao" — e as duas juntas
        devolviam um comodo de 1,19 x 1,93 m com a estante pendurada na quina.

        Mas nenhuma das duas foi pedida:

            a camera precisa reproduzir TODO O CHAO QUE ELA ENXERGAR e as
            outras cameras precisam reproduzir TUDO QUE ELAS ENXERGAREM
                                                    — Eduardo, 18/08

        E ele esta certo por um motivo que nao e so de gosto: um ambiente
        recortado torna a estante mais dificil de situar, porque ela passa a
        ficar na borda de um espaco que eu inventei em vez de dentro do
        espaco que existe.

            Filtrar a visao de um sensor antes de perguntar onde as coisas
            estao e responder sobre um mundo menor do que o que foi medido.

        O que sobra e o extremo do que as tres cameras reconstruiram. Se uma
        delas viu o corredor, o corredor entra.
        """
        if not len(self.nuvem):
            return None
        baixo, alto = self.nuvem.min(axis=0), self.nuvem.max(axis=0)
        return (float(baixo[0]), float(alto[0]), float(baixo[1]), float(alto[1]))


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


def similaridade_2d(origem, destino):
    """A transformacao que leva `origem` em `destino`: escala, giro, deslocamento.

    Umeyama. Devolve (escala, rotacao_2x2, deslocamento, residuo_medio) ou None.

    A escala sai como valor separado, e nao escondida dentro de uma matriz,
    porque ela e o numero que vamos conferir contra a altura da estante: duas
    reguas independentes medindo a mesma cena.
    """
    a = np.asarray(origem, dtype=float).reshape(-1, 2)
    b = np.asarray(destino, dtype=float).reshape(-1, 2)
    if len(a) < 3 or len(a) != len(b):
        return None

    ca, cb = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - ca, b - cb
    variancia = float((a0 ** 2).sum() / len(a))
    if variancia < 1e-12:
        return None

    u, sv, vt = np.linalg.svd((b0.T @ a0) / len(a))
    d = np.eye(2)
    # determinante negativo seria espelhamento — trocaria esquerda por direita
    # no mundo inteiro
    if np.linalg.det(u @ vt) < 0:
        d[1, 1] = -1.0
    r = u @ d @ vt
    escala = float(np.trace(np.diag(sv) @ d) / variancia)
    desloc = cb - escala * (r @ ca)
    residuo = float(np.linalg.norm(
        (escala * (r @ a.T).T + desloc) - b, axis=1).mean())
    return escala, r, desloc, residuo


def alinhar_com_a_homografia(mapa_do_alto, homografia, largura_m, altura_m,
                             tamanho_original, passos=9, z_maximo=0.06):
    """A ponte entre o mundo da rede e o mundo onde o gemeo anda.

    ESTA FUNCAO E O QUE FALTAVA, e a falta dela e o defeito que fez o boneco
    atravessar a estante.

    O gemeo e rastreado pela homografia: origem (0,0), area medida com trena.
    A estante vinha da reconstrucao: origem onde a rede quis, giro qualquer.
    Dois sistemas sem relacao nenhuma — os numeros caiam perto por acaso.

        Duas coisas desenhadas na mesma tela a partir de sistemas de
        coordenadas diferentes nao estao no mesmo lugar por engano: elas
        nunca estiveram no mesmo mundo.

    E COMECA PELOS METROS, NAO PELA NUVEM. Essa e a diferenca.

    As seis tentativas anteriores partiam da nuvem: escolhiam pontos de chao e
    perguntavam a homografia onde eles ficavam. Falhavam porque a maioria da
    nuvem esta FORA do retangulo aferido, e a homografia extrapola la.

    Aqui a direcao inverte. Percorre-se uma grade DENTRO do retangulo medido,
    e para cada ponto em metros pergunta-se qual pixel ele ocupa — pela
    homografia inversa, que e exata — e qual ponto 3D a rede pos ali.

        Amostrar do lado que se conhece garante que toda amostra esta na
        faixa em que o instrumento vale. Nao ha como cair fora.

    Cada par vira (metro, reconstrucao), e a similaridade entre eles e a
    ponte. Pontos que caem alto (movel, pessoa) sao descartados pela altura,
    porque o retangulo aferido e chao.

    Devolve (escala, rotacao_2x2, deslocamento, residuo_m, quantos), ou None.
    """
    h = np.asarray(homografia, dtype=float)
    if h.shape != (3, 3):
        return None
    try:
        inversa = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None

    alt_g, larg_g = mapa_do_alto.shape[:2]
    larg_o, alt_o = tamanho_original

    em_metros, na_nuvem = [], []
    for x in np.linspace(0.0, largura_m, passos):
        for y in np.linspace(0.0, altura_m, passos):
            # metro -> pixel, pela inversa. Exata dentro do retangulo.
            v = inversa @ np.array([x, y, 1.0])
            if abs(v[2]) < 1e-9:
                continue
            u_px, v_px = v[0] / v[2], v[1] / v[2]
            if not (0 <= u_px < larg_o and 0 <= v_px < alt_o):
                continue

            # pixel do quadro original -> indice na grade da rede
            col = int(round(u_px * larg_g / larg_o))
            lin = int(round(v_px * alt_g / alt_o))
            if not (0 <= col < larg_g and 0 <= lin < alt_g):
                continue

            ponto = mapa_do_alto[lin, col]
            if not np.all(np.isfinite(ponto)):
                continue
            # o retangulo aferido e CHAO: ponto alto ali e movel ou pessoa
            if abs(float(ponto[2])) > z_maximo:
                continue

            em_metros.append((x, y))
            na_nuvem.append(ponto[:2])

    if len(na_nuvem) < 6:
        return None
    casado = similaridade_2d(na_nuvem, em_metros)
    if casado is None:
        return None
    escala, r, desloc, residuo = casado
    return escala, r, desloc, residuo, len(na_nuvem)


def montar(nuvem, gabarito, mapa_do_alto=None, homografia=None,
           calib=None, tamanho_original=(640, 480)):
    """A nuvem crua vira o ambiente, NO MUNDO ONDE O GEMEO ANDA.

    Duas reguas, e as duas importam por motivos diferentes:

        a ALTURA DA ESTANTE   da a escala, e nada mais. Sempre disponivel.
        a HOMOGRAFIA          da a escala, o giro e a origem — ou seja, poe
                              a cena no mesmo sistema em que o gemeo e
                              rastreado. So existe se `homografia` vier.

    Sem a segunda, o ambiente sai correto em forma e tamanho e FLUTUANDO num
    sistema proprio: foi assim que o boneco passou a atravessar a estante.
    Com ela, os dois mundos viram um.

    E as duas escalas sao comparadas. Se discordarem muito, o programa avisa —
    porque duas reguas independentes discordando e a informacao mais util que
    esta cena produz.

        Uma medida sozinha nunca esta errada. Ela so passa a estar quando ha
        uma segunda.
    """
    p = np.asarray(nuvem, dtype=float).reshape(-1, 3)
    if len(p) < 100:
        return None

    achado = plano_dominante(p)
    if achado is None:
        return None
    normal, no_plano, _ = achado

    giro = _de_pe(normal)
    deitada = (giro @ (p - no_plano).T).T
    para_cima = 1.0
    if np.median(deitada[:, 2]) < 0:
        deitada[:, 2] *= -1.0
        para_cima = -1.0

    achada = achar_estante(deitada, gabarito.largura, gabarito.profundidade)
    if achada is None:
        return None
    cx, cy, rumo, alto_na_nuvem = achada
    if alto_na_nuvem <= 1e-9:
        return None
    escala_estante = float(gabarito.altura) / alto_na_nuvem

    # --- a ponte com o mundo do gemeo, quando ela existe ---
    ponte = None
    if mapa_do_alto is not None and homografia is not None and calib:
        alto_deitado = np.empty_like(np.asarray(mapa_do_alto, dtype=float))
        forma = alto_deitado.shape
        planos = (giro @ (np.asarray(mapa_do_alto, dtype=float).reshape(-1, 3)
                          - no_plano).T).T
        planos[:, 2] *= para_cima
        alto_deitado = planos.reshape(forma)
        ponte = alinhar_com_a_homografia(
            alto_deitado, homografia,
            float(calib.get("largura_m") or 0.0),
            float(calib.get("altura_m") or 0.0), tamanho_original)

    if ponte is not None:
        escala, r2, desloc, residuo, quantos = ponte
        r3 = np.eye(3)
        r3[:2, :2] = r2
        em_metros = escala * (r3 @ deitada.T).T
        em_metros[:, 0] += desloc[0]
        em_metros[:, 1] += desloc[1]

        centro = escala * (r2 @ np.array([cx, cy])) + desloc
        # o rumo gira junto com a cena
        rumo = rumo + math.atan2(r2[1, 0], r2[0, 0])
        estante = (float(centro[0]), float(centro[1]),
                   float(math.atan2(math.sin(rumo), math.cos(rumo))))
        return Ambiente3D(nuvem=em_metros, escala=escala, estante=estante,
                          altura_da_cena=float(em_metros[:, 2].max()),
                          escala_da_estante=escala_estante,
                          residuo_m=residuo, ancoras=quantos)

    # sem a ponte: forma e tamanho certos, mundo proprio
    em_metros = deitada * escala_estante
    return Ambiente3D(nuvem=em_metros, escala=escala_estante,
                      estante=(cx * escala_estante, cy * escala_estante, rumo),
                      altura_da_cena=float(em_metros[:, 2].max()),
                      escala_da_estante=escala_estante)
