"""A pessoa andando E o objeto de calibracao. Sem papel, sem fita, sem clique.

    nao vou imprimir nada, a gente perdeu dias tentando fazer com que a
    camera fizesse esse trabalho                     — Eduardo, 20/08

Ele disse isso desde o comeco e estava certo. O tabuleiro de xadrez e um
objeto de dimensoes conhecidas posto na cena — e ja ha um objeto de dimensoes
conhecidas na cena: a pessoa. Ela tem 1,80 m, fica em pe, e as tres cameras
olham para ela ao mesmo tempo.

    Quando o proprio objeto de interesse tem dimensao conhecida, ele e o
    padrao de calibracao. Trazer outro e trazer um problema a mais.

NAO E INVENCAO. [Camera Calibration from Video of a Walking Human], PAMI
2006, e [Automatic Multi-Camera Extrinsic Calibration Based on Pedestrian
Torsors], 2019: detectar a pose, modelar quem anda como um BASTAO VERTICAL, e
tirar as extrinsecas do topo e da base em varias vistas. Relatam erros de
triangulacao de poucos centimetros com menos de um minuto de caminhada.

AQUI E MAIS FACIL QUE NO ARTIGO, PORQUE UMA CAMERA JA ESTA CALIBRADA

O artigo parte do zero e precisa resolver todas as cameras de uma vez. Nos
temos a do teto amarrada ao mundo pela trena. Entao, a cada instante:

    a camera do alto      diz ONDE a pessoa esta, em metros    (homografia)
    a estatura medida     diz a que altura fica a cabeca dela  (escala.json)
    as outras cameras     dizem em que PIXEL cada uma cai

Isso e uma correspondencia 3D->2D por junta, por quadro. Cem quadros de
caminhada dao duzentas — e `cv2.calibrateCamera` resolve K, distorcao, R e t
com isso.

    Duas coisas conhecidas e uma desconhecida nao formam um problema
    dificil: formam um sistema.

POR QUE PE E CABECA, E NAO OS 17 PONTOS

Os dois estao sobre a MESMA VERTICAL e a distancia entre eles e conhecida. O
resto do esqueleto se mexe: o cotovelo nao tem altura fixa, o ombro depende
de estar de lado. Pe e cabeca sao os unicos dois pontos de uma pessoa cuja
posicao 3D e deduzivel de fora.

E eles caem em DOIS PLANOS — z=0 e z=1,80. Isso importa: pontos coplanares
nao determinam a focal, e um piso sozinho e um plano so. A cabeca e o que
tira o problema da degenerescencia.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Quantos pares 3D->2D bastam para tentar. Abaixo disso o resultado sai, e
# sai instavel — a calibracao aceita seis pontos e nao merece confianca com
# menos de algumas dezenas.
MINIMO_DE_PARES = 40

# Erro de reprojecao acima do qual a calibracao nao presta.
#
# ESTE NUMERO NAO E O DO TABULEIRO, E COPIAR AQUELE FOI UM ERRO MEU.
#
# A regra de bolso da calibracao por xadrez e "abaixo de 2 px". Ela vale
# porque o canto de um tabuleiro e detectado com precisao de sub-pixel — uns
# 0,1 px. Aqui o "canto" e um TORNOZELO achado por um detector, e ele oscila
# cerca de 3 px entre quadros com a pessoa parada.
#
# E o erro de reprojecao nao pode ficar abaixo do ruido da entrada. Medido:
#
#     ruido 1 px  ->  erro 1,35 px       (raiz(2) x 1 = 1,41)
#     ruido 3 px  ->  erro 4,05 px       (raiz(2) x 3 = 4,24)
#     ruido 5 px  ->  erro 6,74 px       (raiz(2) x 5 = 7,07)
#
# Com 2 px de limiar, uma calibracao PERFEITA sobre dados reais seria
# reprovada — e a busca pelo defeito comecaria no lugar errado.
#
#     Um limiar copiado de outro instrumento mede o outro instrumento.
#
# Seis px e o piso de 3 px de ruido (4,24) com folga, e ainda pega o que
# importa: rastro que trocou de pessoa, caminhada curta demais, pose torta.
ERRO_MAXIMO_PX = 6.0


@dataclass
class Coleta:
    """Os pares 3D->2D juntados enquanto a pessoa anda.

    Um por junta, por quadro, por camera. Guardar em vez de resolver a cada
    quadro nao e preguica: a calibracao melhora com a VARIEDADE das poses, e
    variedade so existe depois de um tempo andando.
    """
    mundo: list = field(default_factory=list)      # (X, Y, Z) em metros
    pixel: list = field(default_factory=list)      # (u, v)
    quadros: int = 0

    def __len__(self):
        return len(self.mundo)

    def juntar(self, ponto3d, ponto2d):
        self.mundo.append(tuple(float(v) for v in ponto3d))
        self.pixel.append(tuple(float(v) for v in ponto2d))

    @property
    def alturas(self):
        """Quantos niveis de z distintos ha. Menos de dois nao calibra."""
        return len({round(z, 2) for _x, _y, z in self.mundo})

    @property
    def espalhamento(self):
        """Quanto de chao a caminhada cobriu, em metros. (largura, fundura).

        Uma pessoa que anda em cima de um ponto so da cem pares identicos.
        Cem pares identicos valem um par.
        """
        if not self.mundo:
            return 0.0, 0.0
        p = np.array(self.mundo)
        return (float(p[:, 0].max() - p[:, 0].min()),
                float(p[:, 1].max() - p[:, 1].min()))


def pares_do_instante(pe_no_chao, estatura, pixeis):
    """Um instante de caminhada -> pares 3D->2D para cada camera.

    `pe_no_chao`  (x, y) em metros, da camera do alto
    `estatura`    metros, medida (config/escala.json)
    `pixeis`      {papel: {"pe": (u,v), "cabeca": (u,v)}}

    A CABECA NAO ESTA EM CIMA DO PE, E FINGIR QUE ESTA E O ERRO OBVIO.

    Quem anda inclina o tronco. Mas a inclinacao e pequena — poucos graus — e
    o erro que ela introduz e de centimetros num braco de 1,80 m. O bastao
    vertical e uma aproximacao, e e a aproximacao que o artigo usa e mede.

        Uma aproximacao declarada e medida vale mais que uma exatidao que
        depende de dados que nao existem.
    """
    x, y = float(pe_no_chao[0]), float(pe_no_chao[1])
    fora = {}
    for papel, juntas in (pixeis or {}).items():
        pares = []
        if juntas.get("pe") is not None:
            pares.append(((x, y, 0.0), juntas["pe"]))
        if juntas.get("cabeca") is not None and estatura:
            pares.append(((x, y, float(estatura)), juntas["cabeca"]))
        if pares:
            fora[papel] = pares
    return fora


def resolver(coleta, tamanho, fixar_centro=True):
    """Os pares -> (K, dist, R, t, erro_px). Ou None.

    A MATEMATICA: `cv2.calibrateCamera` sobre UMA vista de uma nuvem 3D.

    O uso classico da funcao e varias fotos de um tabuleiro plano. Aqui e o
    contrario: uma "foto" so, de uma nuvem de pontos que NAO e plana — os pes
    em z=0 e as cabecas em z=1,80. Funciona pelo mesmo motivo, e precisa de
    nao-planaridade pela mesma razao: um plano unico deixa a focal
    indeterminada.

    `fixar_centro` prende o ponto principal no meio da imagem e zera a
    distorcao tangencial. Com uma vista so, deixar tudo livre e convidar o
    otimizador a explicar ruido com distorcao — o resultado ajusta melhor os
    dados que tem e pior todos os outros.

        Parametro livre demais nao aumenta a precisao: aumenta a confianca
        no que se mediu por acaso.
    """
    import cv2

    if len(coleta) < MINIMO_DE_PARES:
        return None
    if coleta.alturas < 2:
        return None                      # tudo no chao: focal indeterminada

    obj = np.array(coleta.mundo, dtype=np.float32).reshape(-1, 1, 3)
    img = np.array(coleta.pixel, dtype=np.float32).reshape(-1, 1, 2)
    larg, alt = int(tamanho[0]), int(tamanho[1])

    bandeiras = 0
    if fixar_centro:
        bandeiras = (cv2.CALIB_FIX_PRINCIPAL_POINT
                     | cv2.CALIB_ZERO_TANGENT_DIST
                     | cv2.CALIB_FIX_K3)
    palpite = np.array([[float(larg), 0, larg / 2.0],
                        [0, float(larg), alt / 2.0],
                        [0, 0, 1.0]])
    if fixar_centro:
        bandeiras |= cv2.CALIB_USE_INTRINSIC_GUESS

    try:
        erro, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            [obj], [img], (larg, alt), palpite, None, flags=bandeiras)
    except cv2.error:
        return None
    if not rvecs:
        return None

    R, _ = cv2.Rodrigues(rvecs[0])
    t = np.asarray(tvecs[0], dtype=float).ravel()

    # A CAMERA TEM QUE FICAR ACIMA DO CHAO. Se nao ficar, a solucao e a
    # imagem espelhada da verdadeira e nao descreve aparelho nenhum.
    if float((-R.T @ t)[2]) <= 0:
        return None
    return K, np.asarray(dist, dtype=float).ravel(), R, t, float(erro)


def homografia_da_pose(K, R, t):
    """A homografia pixel -> metro do plano z=0, tirada da pose.

    `P = K [r1 r2 t]` leva (x, y) do chao a pixel; a inversa faz o caminho de
    volta. Assim a camera nova entra no mesmo formato das que ja existem, e
    nada a jusante precisa saber de onde ela veio.
    """
    G = K @ np.column_stack([R[:, 0], R[:, 1], t])
    if abs(np.linalg.det(G)) < 1e-12:
        return None
    return np.linalg.inv(G / G[2, 2])


def diagnostico(coleta, resultado, tamanho):
    """O que dizer a quem calibrou, em vez de um 'ok'.

    Tres numeros decidem se vale gravar, e nenhum deles e opiniao:

        pares          quantos 3D->2D entraram
        espalhamento   quanto de chao a caminhada cobriu
        erro           reprojecao em pixels
    """
    larg, _alt = tamanho
    linhas = [f"  pares            {len(coleta)}",
              f"  quadros usados   {coleta.quadros}"]
    lx, ly = coleta.espalhamento
    linhas.append(f"  caminhada cobriu {lx:.2f} x {ly:.2f} m de chao")
    linhas.append(f"  alturas          {coleta.alturas}  (pe e cabeca)")

    if resultado is None:
        linhas.append("  NAO RESOLVEU — ande mais, e por uma area maior.")
        return linhas, False

    K, _dist, _R, t, erro = resultado
    import math
    diag = 2 * math.degrees(math.atan(
        math.hypot(*tamanho) / 2 / float(K[0, 0])))
    linhas += [
        f"  focal            {K[0, 0]:.0f} px  ->  {diag:.0f} graus de diagonal",
        f"  camera a         {float(np.linalg.norm(t)):.2f} m da origem",
        f"  ERRO DE REPROJECAO  {erro:.2f} px",
    ]
    bom = erro <= ERRO_MAXIMO_PX and 40.0 <= diag <= 130.0
    if erro > ERRO_MAXIMO_PX:
        linhas.append(f"  acima de {ERRO_MAXIMO_PX:.0f} px nao presta — "
                      f"a caminhada foi curta ou o rastro trocou de pessoa.")
    if not (40.0 <= diag <= 130.0):
        linhas.append("  esse campo de visao nao e de webcam nenhuma.")
    if lx < 0.8 or ly < 0.8:
        linhas.append("  ande por uma area MAIOR: pontos concentrados dao "
                      "uma solucao que so vale onde voce esteve.")
        bom = False
    return linhas, bom
