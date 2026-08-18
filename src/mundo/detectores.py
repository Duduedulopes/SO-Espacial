"""Os tres detectores: o que cada camera extrai da estante.

    siga por todos os detectores, as 3 cameras pegam algum angulo da estante
                                                        — Eduardo, 13/08

CADA CAMERA VE UM ANGULO, E CADA ANGULO RESPONDE UMA COISA DIFERENTE.

    ALTO      ve a PLANTA         -> onde ela esta e como esta girada
    FRONTAL   ve a FACE           -> largura e as prateleiras, de frente
    LATERAL   ve o PERFIL         -> profundidade e as prateleiras, de lado

Nenhum destes detectores decide nada. Cada um devolve o que enxergou, e a fusao
com o gabarito acontece em `ambiente.py`. A separacao nao e arrumacao: e o que
permite testar a decisao dificil sem camera, e trocar um detector sem tocar na
regra.

O QUE HA DE COMUM AOS TRES: A ESTANTE E FEITA DE RETAS.

Uma estante de aco e um movel de linhas — montantes verticais, prateleiras
horizontais, arestas retas. Isso e sorte, e vale usar: linha reta e das poucas
coisas que a visao computacional classica acha de forma barata e confiavel, sem
treinar nada. Canny para as bordas, Hough para as retas, e a geometria decide.

    Nao ha modelo nenhum aqui, e nao e economia: e que o objeto ja se
    descreve em retas, e retas tem solucao fechada.

E O QUE NENHUM DELES FAZ: DECIDIR SOZINHO.

Todo detector aqui devolve CANDIDATOS, no plural, ou None. Quem escolhe entre
eles e o gabarito — o retangulo de 0,92 x 0,30 medido com trena. Detector que
devolve uma resposta unica esconde a duvida que ele tinha, e a duvida e o dado
mais util que ele produz.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from percepcao.chao import para_metros
from src.mundo.ambiente import VistaDeFrente, VistaDoAlto

# ---------------------------------------------------------------- comum
CANNY_BAIXO, CANNY_ALTO = 60, 180


def _bordas(bgr, desfoque=5):
    """Cinza -> desfoque -> Canny. O desfoque tira o granulado da webcam.

    Sem ele, o ruido de sensor vira micro-borda e o Hough acha dezenas de retas
    de tres pixels que nao sao nada. Cinco pixels de desfoque custam quase nada
    e limpam o que viria depois.
    """
    cinza = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (desfoque, desfoque), 0)
    return cv2.Canny(cinza, CANNY_BAIXO, CANNY_ALTO)


# ---------------------------------------------------------------- ALTO
def _retangulos(bordas, area_minima_px=800):
    """Contornos fechados que se parecem com retangulos, em pixels."""
    bordas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    fora = []
    for c in contornos:
        if cv2.contourArea(c) < area_minima_px:
            continue
        # minAreaRect da o retangulo GIRADO — e a estante quase nunca esta
        # alinhada com os eixos da imagem.
        fora.append(cv2.minAreaRect(c))
    return fora


def candidatos_do_alto(bgr, H, limite_area_m2=(0.10, 1.20)):
    """Retangulos vistos de cima, JA MEDIDOS EM METROS no plano do chao.

    `H` e a matriz da homografia — a mesma de `calibracao/homografia.json`,
    lida por `percepcao.chao.carregar_homografia`. Converter
    ANTES de filtrar e o que torna este detector diferente de um genérico: os
    candidatos chegam a `ambiente.reconhecer` com dimensoes reais, e o gabarito
    pode simplesmente comparar.

        Filtrar em pixels seria filtrar por distancia da camera. Filtrar em
        metros e filtrar pelo objeto.

    A faixa de area em m2 elimina de saida o que nao tem tamanho de movel: um
    livro no chao (0,03 m2) e a area inteira (2 m2) somem antes de qualquer
    comparacao mais fina.

    O DEFEITO QUE ESTA FUNCAO TEM, MEDIDO EM 18/08. LEIA ANTES DE CONFIAR.

    A homografia so vale para pontos NO CHAO. Esta escrito em
    `calibracao/homografia.py` desde o primeiro dia:

        "so pontos SOBRE O CHAO sao mapeados corretamente. A cabeca de uma
         pessoa nao esta no chao; os pes estao."

    Da estante, a camera do teto enxerga a BANDEJA DE CIMA, a 1,90 m. Os pes
    ficam escondidos debaixo das cinco prateleiras. Entao o contorno que
    chega aqui nao e a pegada do movel: e o topo dele, e passar o topo pela
    homografia do chao devolve um retangulo que nao existe em lugar nenhum.

        Para uma camera a 2,5 m, um ponto a 1,90 aparece cerca de QUATRO
        VEZES mais longe do que esta. Nao e imprecisao: e o plano errado.

    A assinatura ficou no dado:

        proporcao real da pegada    0,92 / 0,30 = 3,07
        proporcao medida            1,01 / 0,23 = 4,41

    Um eixo esticou, o outro encolheu, e o centro caiu em x=1,79 — fora da
    area calibrada, empurrado para longe da camera. Ruido erraria os dois
    lados juntos; projecao de um plano alto no plano do chao erra assim.

        Erro que cresce numa direcao e encolhe na outra nao e imprecisao: e
        outra regra sendo aplicada.

    Por que as dimensoes tortas ja nao contaminam mais nada: `reconhecer`
    grava as medidas de trena. O que AINDA sai daqui e vai para o mundo sao
    a posicao e o rumo — e os dois herdam este erro.
    """
    if bgr is None or H is None:
        return []

    fora = []
    for (cx, cy), (w, h), ang in _retangulos(_bordas(bgr)):
        if w < 2 or h < 2:
            continue
        # Os quatro cantos em pixels -> quatro pontos no chao, em metros.
        cantos_px = cv2.boxPoints(((cx, cy), (w, h), ang))
        cantos_m = []
        for px, py in cantos_px:
            try:
                cantos_m.append(para_metros(H, float(px), float(py)))
            except Exception:
                break
        if len(cantos_m) != 4 or any(p is None for p in cantos_m):
            continue

        c = np.array(cantos_m, dtype=float)
        lados = [float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4)]
        maior = (lados[0] + lados[2]) / 2.0
        menor = (lados[1] + lados[3]) / 2.0
        if menor > maior:
            maior, menor = menor, maior
            eixo = c[1] - c[2]
        else:
            eixo = c[1] - c[0]

        area = maior * menor
        if not (limite_area_m2[0] <= area <= limite_area_m2[1]):
            continue

        fora.append(VistaDoAlto(centro=tuple(c.mean(axis=0)),
                                lado_maior=maior, lado_menor=menor,
                                angulo=float(math.atan2(eixo[1], eixo[0]))))
    return fora


# ------------------------------------------------------- FRONTAL e LATERAL
def _linhas_horizontais(bordas, comprimento_min, inclinacao_max_graus=12.0):
    """Segmentos quase horizontais. Devolve [(y_medio_px, comprimento_px)].

    A tolerancia de inclinacao existe porque a camera nunca esta perfeitamente
    nivelada, e uma prateleira a 5 graus continua sendo uma prateleira. Doze
    graus e generoso o bastante para o desalinhamento real e apertado o
    bastante para nao aceitar a diagonal de uma caixa.
    """
    segs = cv2.HoughLinesP(bordas, 1, np.pi / 180, threshold=60,
                           minLineLength=comprimento_min, maxLineGap=14)
    if segs is None:
        return []
    fora = []
    # `segs[:, 0]` supoe a forma (N, 1, 4) que o OpenCV devolvia. Versoes
    # recentes devolvem (N, 4), e a mesma linha passa a desempacotar um
    # inteiro. `reshape(-1, 4)` vale para as duas.
    #
    #     Codigo que depende da forma exata de um retorno de terceiro quebra
    #     numa atualizacao que nao e sua.
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        if abs(math.degrees(math.atan2(dy, dx))) > inclinacao_max_graus:
            continue
        fora.append(((y1 + y2) / 2.0, math.hypot(dx, dy)))
    return fora


def _agrupar(alturas_px, tolerancia_px=10):
    """Junta segmentos da MESMA prateleira numa altura so.

    Uma prateleira nunca vira um segmento: vira cinco ou seis pedacos, cortados
    por produto, por reflexo, pelo proprio Hough. Sem agrupar, o detector
    reportaria seis prateleiras onde ha uma, e o casamento com o gabarito
    fracassaria por excesso, nao por falta.
    """
    if not alturas_px:
        return []
    ordenadas = sorted(alturas_px, key=lambda p: p[0])
    grupos = [[ordenadas[0]]]
    for y, comp in ordenadas[1:]:
        if y - grupos[-1][-1][0] <= tolerancia_px:
            grupos[-1].append((y, comp))
        else:
            grupos.append([(y, comp)])
    # a altura do grupo e a media ponderada pelo comprimento: o segmento longo
    # descreve a prateleira melhor que o toco de vinte pixels
    fora = []
    for g in grupos:
        peso = sum(c for _, c in g) or 1.0
        fora.append((sum(y * c for y, c in g) / peso, peso))
    return fora


def alturas_de_frente(bgr, altura_de, comprimento_min_px=70, minimo=2):
    """Prateleiras vistas de frente ou de perfil, em METROS acima do chao.

    `altura_de` e uma FUNCAO que recebe um y em pixels e devolve a altura em
    metros — a mesma escala
    vertical que ja mede a altura da mao. Reutiliza-la aqui nao e economia de
    codigo: e garantia de que a prateleira detectada e a altura da mao vivem no
    MESMO sistema. Se as duas viessem de conversoes diferentes, comparar uma
    com a outra seria comparar duas reguas.

    Devolve None quando ha menos que `minimo` linhas: uma linha horizontal
    solta pode ser a quina da parede, a mesa, o rodape. Duas ja formam um
    padrao de prateleira.
    """
    if bgr is None or altura_de is None:
        return None
    linhas = _linhas_horizontais(_bordas(bgr), comprimento_min_px)
    grupos = _agrupar(linhas)
    if len(grupos) < minimo:
        return None

    alturas, comprimentos = [], []
    for y_px, peso in grupos:
        h = altura_de(y_px)
        if h is None or not (0.0 <= h <= 2.5):
            continue
        alturas.append(float(h))
        comprimentos.append(peso)
    if len(alturas) < minimo:
        return None

    # A largura aparente e o comprimento do segmento mais longo, em pixels:
    # serve de pista para a frontal, e a fusao decide se usa.
    return VistaDeFrente(alturas=sorted(alturas),
                         largura_aparente=max(comprimentos) if comprimentos
                         else None)


# ---------------------------------------------------------------- juntar
def olhar_o_ambiente(quadros, H=None, escalas=None):
    """Roda os tres detectores sobre um instante. Devolve o que cada um viu.

    `quadros`   {"alto": bgr, "frontal": bgr, "lateral": bgr}
    `H`         matriz da homografia da camera do alto
    `escalas`   {"frontal": f(y_px)->m, "lateral": f(y_px)->m}

    Devolve (candidatos_do_alto, vista_frontal, vista_lateral) — sempre os
    tres, sempre podendo ser vazio ou None. Quem junta com o gabarito e
    `ambiente.reconhecer`.
    """
    quadros = quadros or {}
    escalas = escalas or {}
    return (candidatos_do_alto(quadros.get("alto"), H),
            alturas_de_frente(quadros.get("frontal"), escalas.get("frontal")),
            alturas_de_frente(quadros.get("lateral"), escalas.get("lateral")))
