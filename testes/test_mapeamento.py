"""O ambiente que as cameras entregam, provado contra verdade conhecida.

Monta-se um quarto sintetico — chao, parede, uma estante de 1,90 m — aplica-se
uma similaridade qualquer (escala, giro, tombo) para simular o que a rede
devolve, e exige-se que `montar` desfaca aquilo usando SO a altura da estante.

    Verdade conhecida nao e uma aproximacao melhor do real: e a unica
    situacao em que um erro pode ser MEDIDO em vez de estimado.
"""
import math

import numpy as np
import pytest

from src.mundo.ambiente import Gabarito
from src.mundo.mapeamento import (Ambiente3D, achar_estante, montar,
                                  plano_dominante, _de_pe)

GAB = Gabarito.de_arquivo("loja/estante.json")


def _quarto(n=3000, semente=1):
    """Chao 3x3 m, uma parede, e a estante de 0,92 x 0,30 x 1,90 em (1.0, 2.2)."""
    rng = np.random.default_rng(semente)
    chao = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 3, n),
                            np.zeros(n)])
    parede = np.column_stack([rng.uniform(0, 3, n // 5), np.full(n // 5, 3.0),
                              rng.uniform(0, 2.5, n // 5)])
    est = np.column_stack([rng.uniform(0.54, 1.46, n // 4),
                           rng.uniform(2.05, 2.35, n // 4),
                           rng.uniform(0, 1.90, n // 4)])
    return np.vstack([chao, parede, est])


def _como_a_rede_entrega(pontos, escala=1 / 18.3, giro=0.7, tomba=0.25,
                         desloc=(3.1, -1.4, 0.9)):
    """Forma certa, tamanho e orientacao arbitrarios."""
    cs, sn = math.cos(tomba), math.sin(tomba)
    rx = np.array([[1, 0, 0], [0, cs, -sn], [0, sn, cs]], dtype=float)
    c, s = math.cos(giro), math.sin(giro)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return escala * ((rx @ rz) @ np.asarray(pontos, float).T).T + np.array(desloc)


# ------------------------------------------------------------------ o chao
def test_o_chao_e_achado_em_qualquer_escala():
    """A espessura sai do dado, entao a escala nao importa."""
    for escala in (0.02, 1.0, 50.0):
        achado = plano_dominante(_quarto() * escala)
        assert achado is not None, f"nao achou chao na escala {escala}"
        normal = achado[0]
        assert abs(abs(float(normal @ [0, 0, 1.0])) - 1.0) < 0.05


def test_o_chao_pode_ser_minoria_e_continuar_sendo_o_chao():
    """Exigir maioria seria supor um quarto vazio."""
    rng = np.random.default_rng(3)
    chao = np.column_stack([rng.uniform(0, 3, 400), rng.uniform(0, 3, 400),
                            np.zeros(400)])
    tralha = rng.uniform(0, 2.5, (1200, 3))
    assert plano_dominante(np.vstack([chao, tralha])) is not None


def test_nuvem_sem_plano_nenhum_e_recusada():
    rng = np.random.default_rng(0)
    assert plano_dominante(rng.uniform(0, 1, (400, 3)),
                           tolerancia=0.0005) is None


def test_o_giro_deita_o_chao():
    n = np.array([0.3, -0.2, 0.93]); n /= np.linalg.norm(n)
    assert (_de_pe(n) @ n) == pytest.approx([0, 0, 1], abs=1e-6)


# --------------------------------------------------------------- a estante
def test_a_estante_e_o_que_sobe():
    q = _quarto()
    achada = achar_estante(q, GAB.largura, GAB.profundidade)
    assert achada is not None
    cx, cy, _, alto = achada
    assert (cx, cy) == pytest.approx((1.0, 2.2), abs=0.15)
    assert alto == pytest.approx(1.90, abs=0.05)


def test_sem_nada_alto_nao_ha_estante():
    rng = np.random.default_rng(1)
    plano = np.column_stack([rng.uniform(0, 3, 500), rng.uniform(0, 3, 500),
                             np.zeros(500)])
    assert achar_estante(plano, GAB.largura, GAB.profundidade) is None


def test_poucos_pontos_nao_dao_estante():
    assert achar_estante(np.zeros((10, 3)), 0.92, 0.30) is None


# ------------------------------------------------------------ o ambiente
def test_a_escala_volta_pela_altura_da_estante():
    """O teste central: SO a altura de trena, e o metro volta."""
    bruta = _como_a_rede_entrega(_quarto())
    amb = montar(bruta, GAB)

    assert amb is not None and amb.pronto
    assert amb.escala == pytest.approx(18.3, rel=0.08)
    assert amb.altura_da_cena == pytest.approx(2.5, abs=0.25)


def test_a_estante_sai_em_metros():
    amb = montar(_como_a_rede_entrega(_quarto()), GAB)
    x, y, _ = amb.estante
    # a posicao e no referencial da nuvem deitada, mas a DISTANCIA ate o
    # centro do quarto tem que sobreviver a escala
    assert 0.5 < math.hypot(x, y) < 6.0


def test_a_area_de_movimento_tem_tamanho_de_quarto():
    """Nao um numero digitado: o contorno do piso que a camera enxergou."""
    amb = montar(_como_a_rede_entrega(_quarto()), GAB)
    x0, x1, y0, y1 = amb.chao
    assert 2.0 < (x1 - x0) < 5.0, f"largura de {x1 - x0:.1f} m"
    assert 2.0 < (y1 - y0) < 5.0, f"profundidade de {y1 - y0:.1f} m"


def test_funciona_com_a_cena_de_cabeca_para_baixo():
    """A normal do RANSAC pode vir invertida; o quarto nao pode."""
    de_ponta_cabeca = _como_a_rede_entrega(_quarto(), tomba=math.pi - 0.3)
    amb = montar(de_ponta_cabeca, GAB)
    assert amb is not None
    assert amb.altura_da_cena > 1.0, "a cena ficou abaixo do chao"


def test_nuvem_pequena_demais_e_recusada():
    assert montar(np.zeros((10, 3)), GAB) is None


def test_area_de_movimento_ignora_ponto_solto_no_fundo():
    """Percentis e nao extremos: um ponto perdido esticaria o piso ate ele."""
    amb = Ambiente3D(nuvem=np.vstack([
        np.column_stack([np.random.default_rng(2).uniform(0, 2, 500),
                         np.random.default_rng(3).uniform(0, 2, 500),
                         np.zeros(500)]),
        [[40.0, 40.0, 0.0]]]), escala=1.0)
    x0, x1, y0, y1 = amb.chao
    assert x1 < 5.0, "o ponto solto esticou o piso"
