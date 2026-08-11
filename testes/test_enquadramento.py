"""A junta que caiu fora da imagem nao foi vista, por mais segura que a rede esteja.

Este arquivo existe por causa de UMA medicao, feita em 11/08 contra a estante
de aco, e vale repetir os numeros porque eles sao o teste:

    prateleira 5, verdade 1,90 m   lido 1,15 m em 27 quadros, dispersao 5 cm
    prateleira 1, verdade 0,15 m   lido 0,93 m,               dispersao 20 cm
    prateleira 3, verdade 0,95 m   SEM LEITURA
    prateleira 4, verdade 1,35 m   SEM LEITURA

As duas alturas FACEIS — as que a camera enxerga sem esforco — nao deram
numero. As duas IMPOSSIVEIS deram, com dispersao de medicao boa e erro de
quase um metro, em direcoes opostas.

    Erro que se inverte conforme a dificuldade nao e ruido. E sinal lido de
    um lugar que nao existe.

A causa: `visibility` do MediaPipe modela OCLUSAO, nao ENQUADRAMENTO. Pulso
escondido dentro da prateleira -> visibility baixa -> o sistema recusou, certo.
Pulso FORA da imagem -> a rede extrapola a partir do resto do esqueleto e
devolve visibility ALTA, porque nada o esta ocultando: ele so nao esta la.

O conserto e geometrico e nao tem como discordar da realidade: a projecao 2D
cai dentro do retangulo capturado, ou nao cai.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from percepcao.pose3d import (                         # noqa: E402
    MP_PARA_COCO, Pose3D, dentro_do_quadro,
)

# COCO-17
OMBRO_DIR, PULSO_DIR, TORNOZELO_ESQ = 6, 10, 15
FORMA = (480, 640, 3)          # altura, largura, canais — como vem do OpenCV


def _todas_no_centro():
    return np.tile([320.0, 240.0], (17, 1))


# --------------------------------------------------------------- o caso medido
def test_pulso_acima_do_quadro_com_visibilidade_alta_nao_conta():
    """Prateleira 5, a 1,90 m: o pulso sobe e sai pelo topo da imagem.

    Era este quadro que virava `1.15 m` — 27 vezes seguidas, com 5 cm de
    dispersao. A rede estava CONFIANTE, e e justamente por isso que o teste
    poe a visibilidade em 1.0: se a geometria so vetasse quando o modelo ja
    duvidasse, ela nao serviria para nada.
    """
    px2d = _todas_no_centro()
    px2d[PULSO_DIR] = [330.0, -37.0]

    visivel = np.ones(17, dtype=bool)
    final = visivel & dentro_do_quadro(px2d, FORMA)

    assert not final[PULSO_DIR], "pulso fora do quadro nao pode contar como visto"
    assert final[OMBRO_DIR], "o resto do corpo continua sendo visto"


def test_pulso_abaixo_do_quadro_tambem_nao_conta():
    """Prateleira 1, a 0,15 m: agachado, o pulso sai por baixo.

    O erro medido foi de sinal OPOSTO ao da prateleira 5 (+0,78 contra -0,75),
    e as duas bordas precisam ser recusadas pelo mesmo motivo.
    """
    px2d = _todas_no_centro()
    px2d[PULSO_DIR] = [312.0, 501.0]        # abaixo de altura=480

    final = np.ones(17, dtype=bool) & dentro_do_quadro(px2d, FORMA)
    assert not final[PULSO_DIR]


@pytest.mark.parametrize("ponto", [
    (-1.0, 240.0),      # esquerda
    (640.0, 240.0),     # direita, exatamente na largura
    (320.0, -0.5),      # topo
    (320.0, 480.0),     # base, exatamente na altura
])
def test_qualquer_borda_veta(ponto):
    px2d = _todas_no_centro()
    px2d[PULSO_DIR] = list(ponto)
    assert not dentro_do_quadro(px2d, FORMA)[PULSO_DIR]


@pytest.mark.parametrize("ponto", [
    (0.0, 0.0),         # canto superior esquerdo, inclusivo
    (639.0, 479.0),     # ultimo pixel real
    (320.0, 240.0),
])
def test_dentro_continua_dentro(ponto):
    px2d = _todas_no_centro()
    px2d[PULSO_DIR] = list(ponto)
    assert dentro_do_quadro(px2d, FORMA)[PULSO_DIR]


# ------------------------------------------------- o veto so tira, nunca poe
def test_enquadramento_nunca_cria_visibilidade():
    """Estar dentro do quadro nao prova que a junta foi vista.

    Uma junta pode estar dentro da imagem e OCULTA — o braco atras do tronco,
    a mao dentro da prateleira. Nesse caso quem sabe e o MediaPipe, e a
    geometria nao tem o que dizer.

        O enquadramento tem voto de VETO, nao voto de aprovacao.

    Sem isto, "corrigir" a visibilidade acabaria ressuscitando as juntas que a
    rede corretamente recusou — que sao, ironicamente, as prateleiras 3 e 4
    do teste de 11/08.
    """
    px2d = _todas_no_centro()               # todas dentro
    visivel = np.zeros(17, dtype=bool)      # o modelo nao viu nenhuma

    final = visivel & dentro_do_quadro(px2d, FORMA)
    assert not final.any()


def test_todas_dentro_preserva_a_leitura_do_modelo():
    """Sem nada fora do quadro, o veto nao pode mudar resultado nenhum."""
    px2d = _todas_no_centro()
    visivel = np.array([i % 2 == 0 for i in range(17)])

    final = visivel & dentro_do_quadro(px2d, FORMA)
    assert (final == visivel).all()


# ------------------------------------------------------------------ a forma
def test_aceita_forma_com_e_sem_canais():
    """`frame.shape` vem (h, w, 3) em cor e (h, w) em cinza. Os dois servem."""
    px2d = _todas_no_centro()
    px2d[TORNOZELO_ESQ] = [320.0, 700.0]

    assert not dentro_do_quadro(px2d, (480, 640, 3))[TORNOZELO_ESQ]
    assert not dentro_do_quadro(px2d, (480, 640))[TORNOZELO_ESQ]


# ------------------------------------------------------------------- a ligacao
#
# Os testes acima provam que `dentro_do_quadro` esta certa. Nenhum deles prova
# que ela e CHAMADA — e uma funcao correta que ninguem invoca protege tanto
# quanto uma funcao que nao existe.
#
#     Testar a peca e testar a montagem sao duas coisas, e a maioria dos
#     defeitos deste projeto morava na montagem.
#
# Como `estimar` so precisa de tres coisas do MediaPipe (criar a imagem,
# chamar o detector, ler o resultado), da para poe-las na mao e exercitar o
# caminho real sem carregar rede nenhuma.

class _Marca:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _Resultado:
    def __init__(self, mundo, tela):
        self.pose_world_landmarks = [mundo]
        self.pose_landmarks = [tela]


class _MpFalso:
    class ImageFormat:
        SRGB = "srgb"

    @staticmethod
    def Image(image_format=None, data=None):
        return data


def _pose_com(tela_norm, visibilidade=1.0):
    """Um Pose3D com o MediaPipe substituido por um resultado escolhido."""
    mundo = [_Marca(0.0, -0.5 + i * 0.01, 0.0, visibilidade) for i in range(33)]
    tela = [_Marca(x, y) for x, y in tela_norm]

    pose = object.__new__(Pose3D)
    pose.mp = _MpFalso
    pose._t_ms = 0
    pose.detector = type("D", (), {
        "detect_for_video": lambda self, img, t: _Resultado(mundo, tela)})()
    return pose


def test_estimar_realmente_aplica_o_veto_de_enquadramento():
    """O caso de 11/08 rodando pelo caminho de verdade, ponta a ponta.

    Se alguem remover a linha que combina o enquadramento com a visibilidade,
    ESTE teste quebra — os outros nao quebrariam, porque testam a funcao
    isolada.
    """
    tela = [(0.5, 0.5)] * 33
    tela[MP_PARA_COCO[PULSO_DIR]] = (0.52, -0.08)      # acima do topo

    frame = np.zeros((480, 640, 3), np.uint8)
    _, visivel, px2d = _pose_com(tela).estimar(frame)

    assert px2d[PULSO_DIR][1] < 0, "o cenario precisa por o pulso fora mesmo"
    assert not visivel[PULSO_DIR], "estimar() nao aplicou o veto"
    assert visivel[OMBRO_DIR], "o resto do corpo continua visivel"


def test_estimar_nao_veta_ninguem_com_o_corpo_todo_no_quadro():
    """Contraprova: sem nada fora, a leitura do modelo passa inteira."""
    frame = np.zeros((480, 640, 3), np.uint8)
    _, visivel, _ = _pose_com([(0.5, 0.5)] * 33).estimar(frame)
    assert visivel.all()


def test_quadro_e_o_da_imagem_inteira_nao_o_do_recorte():
    """As coordenadas chegam ja remapeadas para a imagem inteira.

    `estimar` recorta a pessoa antes de estimar a pose, mas devolve `px2d` em
    pixels da imagem ORIGINAL. Testar contra o recorte recusaria juntas
    perfeitamente visiveis so por estarem fora da caixa do detector — que e
    exatamente o caso do braco esticado para o lado.
    """
    px2d = _todas_no_centro()
    px2d[PULSO_DIR] = [600.0, 240.0]        # fora de um recorte central...

    assert dentro_do_quadro(px2d, FORMA)[PULSO_DIR]   # ...mas dentro da imagem
