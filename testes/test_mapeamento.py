"""O mapeamento provado contra verdade conhecida.

Monta-se um quarto sintetico — chao, parede, uma estante — aplica-se uma
similaridade conhecida (escala, giro, deslocamento) para simular o que o VGGT
devolve, e exige-se que `amarrar` desfaca exatamente aquilo.

    Verdade conhecida nao e uma aproximacao melhor do real: e a unica
    situacao em que um erro pode ser MEDIDO em vez de estimado.
"""
import math

import numpy as np
import pytest

from src.mundo.mapeamento import (amarrar, plano_dominante, similaridade,
                                  _de_pe)


def _quarto(n=600, semente=1):
    """Chao 2x2 m, uma parede e uma estante de 0,92 x 0,30 x 1,90."""
    rng = np.random.default_rng(semente)
    chao = np.column_stack([rng.uniform(0, 2, n), rng.uniform(0, 2, n),
                            np.zeros(n)])
    parede = np.column_stack([rng.uniform(0, 2, n // 4), np.full(n // 4, 2.0),
                              rng.uniform(0, 2.4, n // 4)])
    est = np.column_stack([rng.uniform(0.6, 1.52, n // 4),
                           rng.uniform(1.5, 1.8, n // 4),
                           rng.uniform(0, 1.9, n // 4)])
    return np.vstack([chao, parede, est])


def _embaralhar(pontos, escala=0.37, giro=0.8, desloc=(3.1, -1.4, 0.9),
                tomba=0.25):
    """O que o VGGT entrega: forma certa, escala e orientacao arbitrarias."""
    cs, sn = math.cos(tomba), math.sin(tomba)
    rx = np.array([[1, 0, 0], [0, cs, -sn], [0, sn, cs]], dtype=float)
    c, s = math.cos(giro), math.sin(giro)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    r = rx @ rz
    return escala * (r @ np.asarray(pontos, dtype=float).T).T + np.array(desloc)


# ----------------------------------------------------------------- o plano
def test_o_chao_e_o_maior_plano():
    achado = plano_dominante(_quarto(), tolerancia=0.02)
    assert achado is not None
    normal, _, dentro = achado
    assert abs(abs(float(normal @ np.array([0.0, 0.0, 1.0]))) - 1.0) < 0.05
    assert dentro.sum() > 400


def test_nuvem_sem_plano_nenhum_e_recusada():
    rng = np.random.default_rng(0)
    assert plano_dominante(rng.uniform(0, 1, (300, 3)), tolerancia=0.001) is None


def test_poucos_pontos_nao_dao_plano():
    assert plano_dominante(np.zeros((2, 3))) is None


def test_o_giro_deixa_o_chao_deitado():
    normal = np.array([0.3, -0.2, 0.93])
    normal /= np.linalg.norm(normal)
    assert (_de_pe(normal, None) @ normal) == pytest.approx([0, 0, 1], abs=1e-6)


# ---------------------------------------------------------- a similaridade
def test_a_similaridade_volta_exata():
    rng = np.random.default_rng(2)
    a = rng.uniform(-1, 1, (12, 2))
    e, ang, t = 2.7, 0.9, np.array([4.0, -3.0])
    c, s = math.cos(ang), math.sin(ang)
    r = np.array([[c, -s], [s, c]])
    b = e * (r @ a.T).T + t

    escala, rot, desloc, residuo = similaridade(a, b)
    assert escala == pytest.approx(e)
    assert rot == pytest.approx(r)
    assert desloc == pytest.approx(t)
    assert residuo < 1e-9


def test_a_similaridade_nao_espelha():
    """Espelhar um mapa trocaria esquerda por direita no mundo inteiro."""
    a = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    b = np.array([[0, 0], [1, 0], [0, -1], [1, -1]], dtype=float)
    _, r, _, _ = similaridade(a, b)
    assert np.linalg.det(r) > 0


def test_pontos_de_mais_a_menos_sao_recusados():
    assert similaridade(np.zeros((3, 2)), np.zeros((2, 2))) is None
    assert similaridade(np.zeros((1, 2)), np.zeros((1, 2))) is None


def test_ancoras_todas_no_mesmo_ponto_sao_recusadas():
    """Sem espalhamento nao ha escala: seria dividir por zero com fe."""
    iguais = np.zeros((5, 2))
    assert similaridade(iguais, np.arange(10).reshape(5, 2).astype(float)) is None


# ------------------------------------------------------------- o mapa todo
def test_o_mapa_volta_para_metros():
    """O teste central: desfazer a similaridade que o VGGT deixou."""
    verdade = _quarto()
    nuvem = _embaralhar(verdade)

    # as ancoras: pontos do chao, na nuvem e em metros pela homografia
    chao = np.where(verdade[:, 2] < 1e-6)[0][:40]
    mapa = amarrar(nuvem, {}, nuvem[chao], verdade[chao, :2])

    assert mapa is not None and mapa.pronto is False   # sem poses ainda
    assert mapa.escala == pytest.approx(1 / 0.37, rel=0.02)
    assert mapa.residuo_m < 0.02

    erro = np.linalg.norm(mapa.nuvem[chao][:, :2] - verdade[chao, :2], axis=1)
    assert erro.mean() < 0.02, f"o chao voltou errado: {erro.mean():.3f} m"


def test_a_altura_da_estante_volta_a_1_90():
    """A prova de que a ESCALA esta certa, e nao so o alinhamento."""
    verdade = _quarto()
    nuvem = _embaralhar(verdade)
    chao = np.where(verdade[:, 2] < 1e-6)[0][:40]
    mapa = amarrar(nuvem, {}, nuvem[chao], verdade[chao, :2])
    assert mapa.nuvem[:, 2].max() == pytest.approx(2.4, abs=0.08)


def test_as_poses_vao_junto_para_metros():
    verdade = _quarto()
    nuvem = _embaralhar(verdade)
    chao = np.where(verdade[:, 2] < 1e-6)[0][:40]

    camera_no_mundo = np.array([1.0, -1.2, 2.35])
    camera_na_nuvem = _embaralhar(camera_no_mundo.reshape(1, 3))[0]
    mapa = amarrar(nuvem, {"alto": (camera_na_nuvem, [0.0, 1.0, -1.0])},
                   nuvem[chao], verdade[chao, :2])

    assert mapa.pronto is False          # uma camera so nao e mapa
    posicao, _ = mapa.poses["alto"]
    assert posicao == pytest.approx(camera_no_mundo, abs=0.05)


def test_duas_cameras_ja_e_mapa():
    verdade = _quarto()
    nuvem = _embaralhar(verdade)
    chao = np.where(verdade[:, 2] < 1e-6)[0][:40]
    poses = {p: (_embaralhar(np.array([[1.0, -1.2, 2.3]]))[0], [0, 1, -1])
             for p in ("alto", "lateral")}
    assert amarrar(nuvem, poses, nuvem[chao], verdade[chao, :2]).pronto


def test_o_chao_do_mapa_cobre_o_quarto():
    verdade = _quarto()
    nuvem = _embaralhar(verdade)
    chao = np.where(verdade[:, 2] < 1e-6)[0][:40]
    x0, x1, y0, y1 = amarrar(nuvem, {}, nuvem[chao], verdade[chao, :2]).chao
    assert x0 == pytest.approx(0.0, abs=0.1) and x1 == pytest.approx(2.0, abs=0.1)
    assert y0 == pytest.approx(0.0, abs=0.1) and y1 == pytest.approx(2.0, abs=0.1)


def test_sem_ancora_nao_ha_mapa():
    assert amarrar(_quarto(), {}, np.zeros((1, 3)), np.zeros((1, 2))) is None


def test_nuvem_vazia_nao_ha_mapa():
    assert amarrar(np.zeros((0, 3)), {}, np.zeros((4, 3)),
                   np.zeros((4, 2))) is None


# ------------------------------------------- a espessura sai do proprio dado
#
# O DEFEITO QUE CUSTOU O DIA 18/08.
#
# `plano_dominante` tinha tolerancia ABSOLUTA de 0,02. Mas a nuvem destas
# redes nao tem unidade: a escala e arbitraria e so vira metro depois de
# `amarrar`. Na corrida real a escala era 18,3 — entao 0,02 valia 37 cm de
# espessura, e o "plano" engolia chao, base da estante, caixa e parede na
# mesma fatia. `_de_pe` girava aquilo e achatava a cena: o quarto saiu com
# 4,5 m de largura e 0,66 de altura.
#
#     Uma tolerancia absoluta sobre um dado sem unidade nao e frouxa nem
#     apertada: e indefinida ate alguem medir a escala, que e justamente o
#     que ainda nao aconteceu.

def _quarto_em(escala, semente=3):
    return _quarto(semente=semente) * escala


def test_o_chao_e_achado_em_qualquer_escala():
    """A prova: a mesma cena, tres escalas, a mesma normal."""
    for escala in (0.02, 1.0, 50.0):
        achado = plano_dominante(_quarto_em(escala))
        assert achado is not None, f"nao achou plano na escala {escala}"
        normal, _, dentro = achado
        assert abs(abs(float(normal @ np.array([0.0, 0.0, 1.0]))) - 1.0) < 0.05, \
            f"na escala {escala} o plano achado nao e horizontal"


def test_a_espessura_absoluta_engorda_o_plano_conforme_a_escala():
    """O defeito, medido no que ele tem de objetivo.

    Numa nuvem de escala 1/18, a tolerancia absoluta de 0,02 equivale a 37 cm
    de espessura. O plano deixa de descrever uma superficie e passa a engolir
    uma fatia da cena — e quanto menor a escala, mais ele engole.

    Num quarto sintetico limpo o chao ainda ganha; numa cena real, com movel
    e parede perto do piso, e assim que ele deixa de ser o chao.
    """
    cima = np.array([0.0, 0.0, 1.0])
    engolidos = []
    for escala in (1.0, 1 / 18.3):
        nuvem = _quarto(semente=7) * escala
        relativa = plano_dominante(nuvem)
        assert abs(abs(float(relativa[0] @ cima)) - 1.0) < 0.05, \
            f"a espessura relativa errou o chao na escala {escala:.3f}"
        engolidos.append(int(plano_dominante(nuvem, tolerancia=0.02)[2].sum()))

    assert engolidos[1] > engolidos[0], (
        "com a mesma tolerancia absoluta, a escala menor tinha que engolir "
        "mais pontos — se nao engole, o defeito de 18/08 nao existe")


def test_o_mapa_volta_para_metros_em_escala_pequena():
    """A escala real do DUSt3R foi 18,3. O caminho tem que aguentar isso."""
    verdade = _quarto(semente=5)
    nuvem = _embaralhar(verdade, escala=1 / 18.3, giro=0.4, tomba=0.2)
    chao = np.where(verdade[:, 2] < 1e-6)[0][:40]
    mapa = amarrar(nuvem, {}, nuvem[chao], verdade[chao, :2])

    assert mapa is not None
    assert mapa.escala == pytest.approx(18.3, rel=0.05)
    assert mapa.residuo_m < 0.05, f"residuo de {mapa.residuo_m * 100:.0f} cm"
    assert mapa.nuvem[:, 2].max() == pytest.approx(2.4, abs=0.1), \
        "a cena continua achatada"
