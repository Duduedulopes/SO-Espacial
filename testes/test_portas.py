"""A entrada e a saida saem da estante, nao do teclado.

    quero que ao lado da prateleira seja a entrada e a saida
                                                — Eduardo, 13/08

O que estes testes protegem nao e o numero: e a DEPENDENCIA. Se alguem um dia
resolver escrever as zonas a mao no `quarto.json`, elas param de acompanhar a
estante — e o defeito so aparece no dia em que o movel for empurrado meio
metro, muito depois de o commit ter sido esquecido.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ferramentas.achar_ambiente import _extrapolado, _portas   # noqa: E402
from src.mundo.ambiente import Ambiente                        # noqa: E402

CALIB = {"largura_m": 1.65, "altura_m": 1.32}

CHAO = (-2.0, 2.0, -2.0, 2.0)


def _estante(x=0.0, y=0.0, rumo=0.0, largura=0.92):
    return Ambiente(x=x, y=y, rumo_da_face=rumo, largura=largura,
                    profundidade=0.30, altura=1.90,
                    prateleiras=[("p1", 0.15)], cameras=("alto",))


def _centro(z):
    return np.array([(z["x0"] + z["x1"]) / 2, (z["y0"] + z["y1"]) / 2])


def test_saem_duas_e_uma_de_cada_lado():
    zs = _portas(_estante(), CHAO)
    assert [z["id"] for z in zs] == ["entrada", "saida"]


def test_ficam_dos_dois_lados_da_estante():
    """Uma de cada lado, e nao as duas juntas — senao nao ha entrada e saida."""
    e, s = _portas(_estante(), CHAO)
    ao_longo = np.array([1.0, 0.0])          # rumo zero
    assert (_centro(e) - np.array([0.0, 0.0])) @ ao_longo < 0
    assert (_centro(s) - np.array([0.0, 0.0])) @ ao_longo > 0


def test_nao_encostam_na_estante():
    """Porta colada no movel conta como alcance, nao como passagem."""
    est = _estante()
    for z in _portas(est, CHAO):
        distancia = abs(_centro(z)[0])       # rumo zero: ao longo e o eixo x
        assert distancia > est.largura / 2


def test_acompanham_a_estante_quando_ela_anda():
    """A prova da dependencia: mover o movel tem que mover as portas junto."""
    antes = [_centro(z) for z in _portas(_estante(x=0.0), CHAO)]
    depois = [_centro(z) for z in _portas(_estante(x=0.8), CHAO)]
    for a, b in zip(antes, depois):
        assert b[0] - a[0] == pytest.approx(0.8)


def test_giram_com_a_face():
    """Girada 90 graus, o que era deslocamento em x vira deslocamento em y."""
    zs = _portas(_estante(rumo=math.pi / 2), CHAO)
    for z in zs:
        c = _centro(z)
        assert abs(c[1]) > abs(c[0]), "as portas nao acompanharam o giro"


def test_ficam_do_lado_de_quem_anda():
    """Adiante da face, e nao atras: atras da estante e parede."""
    est = _estante()
    for z in _portas(est, CHAO):
        adiante = float(_centro(z) @ est.normal)
        assert adiante > 0


def test_porta_que_cairia_fora_do_chao_nao_e_gravada():
    """Melhor nao existir do que existir com area zero e nunca acusar ninguem."""
    espremido = (-0.30, 0.30, -0.30, 0.30)
    assert _portas(_estante(), espremido) == []


def test_cabem_no_chao_quando_cabem():
    xmin, xmax, ymin, ymax = CHAO
    for z in _portas(_estante(), CHAO):
        assert xmin <= z["x0"] < z["x1"] <= xmax
        assert ymin <= z["y0"] < z["y1"] <= ymax


# ------------------------------------------- fora da area que foi calibrada
#
# Medido em 18/08, com as tres cameras ligadas. A ferramenta achou a estante
# em (+1,79, +0,29) — e a homografia foi ajustada num retangulo de
# 1,65 x 1,32 m com origem em (0,0). O centro caiu 14 cm ALEM da borda.
#
# O resultado apareceu nas dimensoes: 1,01 x 0,23 m contra 0,92 x 0,30 de
# trena. Um eixo esticou 10%, o outro encolheu 24% — que e a assinatura da
# extrapolacao projetiva, e nao de ruido, que erraria os dois para o mesmo
# lado.
#
#     Erro que cresce numa direcao e encolhe na outra nao e imprecisao: e
#     outra regra sendo aplicada.

def test_dentro_da_area_nao_acusa():
    assert _extrapolado(0.80, 0.60, CALIB) is False


def test_o_canto_exato_ainda_esta_dentro():
    assert _extrapolado(1.65, 1.32, CALIB) is False


def test_a_estante_de_18_08_e_acusada():
    """O caso real que motivou o aviso."""
    assert _extrapolado(1.79, 0.29, CALIB) is True


def test_um_pouco_atras_da_origem_ainda_passa():
    """Cinco centimetros de margem: a borda medida nao e uma parede."""
    assert _extrapolado(-0.04, 0.50, CALIB) is False


def test_dez_centimetros_atras_da_origem_e_acusado():
    assert _extrapolado(-0.10, 0.50, CALIB) is True


def test_sem_calibracao_declarada_nao_acusa_nada():
    """Nao ha como dizer 'fora' sem saber onde e o dentro."""
    assert _extrapolado(99.0, 99.0, {}) is False
    assert _extrapolado(99.0, 99.0, {"largura_m": 0, "altura_m": 0}) is False
