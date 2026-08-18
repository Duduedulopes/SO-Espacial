"""As tres cameras mapeiam o lugar. Sem clique, sem gabarito de calibracao.

    Faca com que as 3 cameras facam um mapeamento do lugar, recriem em tempo
    real e captem o gemeo digital nesse ambiente     — Eduardo, 18/08

O METODO, E POR QUE ESTE E NAO OUTRO

VGGT (Visual Geometry Grounded Transformer) recebe imagens de cameras SEM
CALIBRACAO NENHUMA e devolve, numa passagem so, a pose de cada uma e uma nuvem
de pontos densa da cena. Nao pede tabuleiro de xadrez, nao pede objeto de
dimensao conhecida, nao pede correspondencia marcada a mao.

Isso importa aqui por um motivo concreto: as tres cameras olham a estante de
angulos muito diferentes e quase nao compartilham textura. O COLMAP — o
caminho classico de structure-from-motion — e justamente onde ele sofre: poucas
imagens e parede lisa. As redes desta familia foram feitas para sobreposicao
escassa e ganham do COLMAP com poucas vistas.

    Tres tentativas foram gastas pedindo que uma pessoa marcasse cantos na
    tela. Marcar a mao nao era o projeto: era o atalho que eu tomei por nao
    ter procurado o metodo.

O QUE ELE DEVOLVE, E O QUE FALTA NELE

VGGT resolve a geometria a menos de uma SIMILARIDADE: escala, rotacao e
translacao globais ficam indeterminadas. A nuvem esta certa em forma e errada
em tamanho e orientacao, porque nada na imagem diz quantos metros tem um pixel.

E aqui este projeto tem uma vantagem que quase ninguem tem: a homografia da
camera do alto JA define um sistema de coordenadas em metros, medido com trena
em 1,65 x 1,32 m. Entao a ancora nao precisa ser inventada.

    Um numero que ja existe em algum lugar nao deve ser reescrito noutro.
    Deve ser LIDO de onde ele mora.

Amarrar o mapa a homografia, em vez de criar um sistema novo, e o que faz o
gemeo digital e o ambiente mapeado viverem no MESMO mundo — que era o problema
desde o comeco.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Fracao dos pontos que precisa cair no plano para ele ser o chao.
#
# O chao e a maior superficie plana de um quarto vazio. Trinta por cento e
# folgado o bastante para um quarto com moveis e apertado o bastante para nao
# aceitar uma parede como chao.
FRACAO_DO_CHAO = 0.30

# Tolerancia do plano, em unidades da nuvem (adimensionais antes da escala).
TOLERANCIA_PLANO = 0.02

# Escala minima aceitavel entre a nuvem e o mundo. Abaixo disso a similaridade
# degenerou e o mapa nao esta amarrado a coisa nenhuma.
ESCALA_MINIMA = 1e-6


@dataclass
class Mapa:
    """O lugar, mapeado e ja em metros no sistema da homografia."""
    poses: dict = field(default_factory=dict)      # papel -> (posicao, olhar)
    nuvem: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    escala: float = 0.0
    residuo_m: float = 0.0
    pontos_ancora: int = 0

    @property
    def pronto(self):
        return len(self.poses) >= 2 and self.escala > ESCALA_MINIMA

    @property
    def chao(self):
        """Os limites do piso reconstruido, em metros: (xmin, xmax, ymin, ymax)."""
        if not len(self.nuvem):
            return None
        baixo, alto = self.nuvem.min(axis=0), self.nuvem.max(axis=0)
        return (float(baixo[0]), float(alto[0]), float(baixo[1]), float(alto[1]))


def plano_dominante(pontos, tolerancia=TOLERANCIA_PLANO, tentativas=200,
                    semente=0):
    """O maior plano da nuvem — num quarto, o chao. RANSAC.

    Devolve (normal_unitaria, ponto_do_plano, mascara_dos_inliers).

    RANSAC e nao minimos quadrados porque a nuvem tem parede, movel e ruido: um
    ajuste que usa todos os pontos encontra o plano que agrada a todos e nao
    descreve nenhum.
    """
    p = np.asarray(pontos, dtype=float).reshape(-1, 3)
    if len(p) < 3:
        return None

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


def similaridade(origem, destino):
    """A transformacao 2D que leva `origem` em `destino`: escala, giro, deslocamento.

    Umeyama. Devolve (escala, rotacao_2x2, deslocamento, residuo_medio).

    Sem escala isolada nao ha metro: a nuvem do VGGT nasce adimensional, e e
    exatamente este numero que a converte. Por isso ele sai como valor de
    primeira classe, e nao escondido dentro de uma matriz 3x3.
    """
    a = np.asarray(origem, dtype=float).reshape(-1, 2)
    b = np.asarray(destino, dtype=float).reshape(-1, 2)
    if len(a) < 2 or len(a) != len(b):
        return None

    ca, cb = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - ca, b - cb
    variancia = float((a0 ** 2).sum() / len(a))
    if variancia < 1e-12:
        return None

    u, s, vt = np.linalg.svd((b0.T @ a0) / len(a))
    d = np.eye(2)
    # O determinante negativo seria um espelhamento — e espelhar um mapa nao e
    # um movimento rigido: seria trocar esquerda por direita no mundo inteiro.
    if np.linalg.det(u @ vt) < 0:
        d[1, 1] = -1.0
    r = u @ d @ vt
    escala = float(np.trace(np.diag(s) @ d) / variancia)
    deslocamento = cb - escala * (r @ ca)
    residuo = float(np.linalg.norm(
        (escala * (r @ a.T).T + deslocamento) - b, axis=1).mean())
    return escala, r, deslocamento, residuo


def _de_pe(normal, pontos):
    """Gira a nuvem para o plano do chao virar z = 0 e o resto ficar acima."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    eixo = np.cross(n, np.array([0.0, 0.0, 1.0]))
    seno = np.linalg.norm(eixo)
    if seno < 1e-9:
        return np.eye(3)
    eixo = eixo / seno
    ang = math.atan2(seno, float(n @ np.array([0.0, 0.0, 1.0])))
    k = np.array([[0, -eixo[2], eixo[1]],
                  [eixo[2], 0, -eixo[0]],
                  [-eixo[1], eixo[0], 0]], dtype=float)
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)


def amarrar(nuvem, poses_vggt, ancoras_nuvem, ancoras_mundo):
    """Poe o mapa em metros, no sistema de coordenadas que ja existe.

    `ancoras_nuvem`   pontos do chao NA NUVEM, adimensionais
    `ancoras_mundo`   os mesmos pontos em METROS, pela homografia

    O VGGT resolve a geometria a menos de uma similaridade — a forma esta
    certa, o tamanho e a orientacao nao. Estes dois conjuntos sao a unica coisa
    que amarra um no outro, e eles vem da calibracao que ja existia.

    Devolve `Mapa`, ou None quando a amarracao nao fecha.
    """
    p = np.asarray(nuvem, dtype=float).reshape(-1, 3)
    if not len(p) or len(ancoras_nuvem) < 2:
        return None

    achado = plano_dominante(p)
    if achado is None:
        return None
    normal, no_plano, _ = achado

    giro = _de_pe(normal, p)
    p_reto = (giro @ (p - no_plano).T).T
    ancoras_retas = (giro @ (np.asarray(ancoras_nuvem, dtype=float)
                             - no_plano).T).T

    casado = similaridade(ancoras_retas[:, :2], ancoras_mundo)
    if casado is None:
        return None
    escala, r2, desloc, residuo = casado
    if escala < ESCALA_MINIMA:
        return None

    r3 = np.eye(3)
    r3[:2, :2] = r2
    final = escala * (r3 @ p_reto.T).T
    final[:, 0] += desloc[0]
    final[:, 1] += desloc[1]

    poses = {}
    for papel, (posicao, olhar) in poses_vggt.items():
        c = escala * (r3 @ (giro @ (np.asarray(posicao, dtype=float)
                                    - no_plano)))
        c[0] += desloc[0]
        c[1] += desloc[1]
        poses[papel] = (c, (r3 @ (giro @ np.asarray(olhar, dtype=float))))

    return Mapa(poses=poses, nuvem=final, escala=escala,
                residuo_m=residuo * escala, pontos_ancora=len(ancoras_mundo))
