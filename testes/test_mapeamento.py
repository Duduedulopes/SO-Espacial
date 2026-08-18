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


def test_a_area_cobre_TUDO_que_as_cameras_viram():
    """Sem recorte. O quarto tem 3 m de lado; a caixa dele pode ser maior.

    `montar` desentorta o TOMBO — deita o chao — mas nao gira a cena no
    plano: nada na nuvem diz qual parede e o norte. Entao um quarto de 3x3
    girado 40 graus tem caixa alinhada de 3 x (cos + sin) = 4,2 m.

    Isso nao e erro. A area precisa CONTER o que foi visto, e conter um
    quadrado girado custa mais que o lado dele.
    """
    amb = montar(_como_a_rede_entrega(_quarto()), GAB)
    x0, x1, y0, y1 = amb.chao
    for lado in (x1 - x0, y1 - y0):
        assert 2.8 <= lado <= 4.6, f"lado de {lado:.2f} m"


def test_a_estante_fica_DENTRO_da_area():
    """Consequencia de nao recortar: nada do que foi visto fica de fora."""
    amb = montar(_como_a_rede_entrega(_quarto()), GAB)
    x0, x1, y0, y1 = amb.chao
    ex, ey, _ = amb.estante
    assert x0 <= ex <= x1 and y0 <= ey <= y1, (
        f"estante em ({ex:.2f}, {ey:.2f}) fora de "
        f"({x0:.2f}..{x1:.2f}, {y0:.2f}..{y1:.2f})")


def test_nada_do_que_a_camera_viu_fica_de_fora():
    """A prova direta: TODO ponto da nuvem cabe na area."""
    amb = montar(_como_a_rede_entrega(_quarto()), GAB)
    x0, x1, y0, y1 = amb.chao
    assert amb.nuvem[:, 0].min() >= x0 and amb.nuvem[:, 0].max() <= x1
    assert amb.nuvem[:, 1].min() >= y0 and amb.nuvem[:, 1].max() <= y1


def test_funciona_com_a_cena_de_cabeca_para_baixo():
    """A normal do RANSAC pode vir invertida; o quarto nao pode."""
    de_ponta_cabeca = _como_a_rede_entrega(_quarto(), tomba=math.pi - 0.3)
    amb = montar(de_ponta_cabeca, GAB)
    assert amb is not None
    assert amb.altura_da_cena > 1.0, "a cena ficou abaixo do chao"


def test_nuvem_pequena_demais_e_recusada():
    assert montar(np.zeros((10, 3)), GAB) is None


def test_ponto_distante_entra_porque_a_camera_o_viu():
    """O contrario do que eu tinha feito, e de proposito.

    Um ponto longe pode ser ruido — ou pode ser o corredor que a lateral
    enxerga. O programa nao sabe distinguir, e recortar por precaucao foi
    justamente o que encolheu o comodo tres vezes seguidas.
    """
    amb = Ambiente3D(nuvem=np.vstack([
        np.column_stack([np.random.default_rng(2).uniform(0, 2, 500),
                         np.random.default_rng(3).uniform(0, 2, 500),
                         np.zeros(500)]),
        [[6.0, 6.0, 0.0]]]), escala=1.0)
    x0, x1, y0, y1 = amb.chao
    assert x1 == pytest.approx(6.0), "recortou o que a camera viu"


# ============================================================ A PONTE
#
# O DEFEITO QUE FEZ O BONECO ATRAVESSAR A ESTANTE.
#
# O gemeo e rastreado pela homografia: origem (0,0), area 1,65 x 1,32 medida
# com trena. A estante vinha da reconstrucao: origem onde a rede quis, giro
# qualquer. Medido na corrida de 18/08:
#
#     boneco    x  0,00 a 1,65    y  0,00 a 1,32
#     estante   x -0,62 a 1,01    y -0,95 a 0,75
#
# Dois sistemas sem relacao nenhuma. Os numeros caiam perto por acaso.
#
#     Duas coisas desenhadas na mesma tela a partir de sistemas de coordenadas
#     diferentes nao estao no mesmo lugar por engano: elas nunca estiveram no
#     mesmo mundo.

from src.mundo.mapeamento import (alinhar_com_a_homografia,   # noqa: E402
                                  similaridade_2d)

AREA = (1.65, 1.32)


def _camera_do_teto(altura=2.4, alvo=(0.8, 0.6)):
    """Uma camera olhando o chao em angulo, e a homografia dela.

    Devolve (projetar, H) — a funcao que leva metro em pixel, e a homografia
    que leva pixel em metro. Sao inversas uma da outra por construcao, que e
    exatamente a relacao que a calibracao real produz.
    """
    import cv2
    c = np.array([alvo[0] - 0.9, alvo[1] - 1.6, altura])
    frente = np.array([*alvo, 0.0]) - c
    frente /= np.linalg.norm(frente)
    direita = np.cross(frente, [0, 0, 1.0]); direita /= np.linalg.norm(direita)
    baixo = np.cross(frente, direita)
    r = np.vstack([direita, baixo, frente])
    k = np.array([[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1.0]])

    def projetar(x, y):
        p = k @ (r @ (np.array([x, y, 0.0]) - c))
        return p[0] / p[2], p[1] / p[2]

    cantos_m = np.array([[0, 0], [AREA[0], 0], AREA, [0, AREA[1]]], float)
    cantos_px = np.array([projetar(*m) for m in cantos_m], float)
    h, _ = cv2.findHomography(cantos_px, cantos_m)
    return projetar, h


def _mapa_do_alto(projetar, mundo_deitado, forma=(384, 512),
                  tamanho=(640, 480)):
    """O que a rede devolveria para a camera do alto: um ponto 3D por pixel.

    Preenche a grade projetando o chao de volta. Onde nao ha chao visivel,
    fica NaN — como acontece de verdade.
    """
    alt_g, larg_g = forma
    mapa = np.full((alt_g, larg_g, 3), np.nan)
    for x in np.linspace(-1.0, 3.0, 220):
        for y in np.linspace(-1.0, 3.0, 220):
            u, v = projetar(x, y)
            col = int(round(u * larg_g / tamanho[0]))
            lin = int(round(v * alt_g / tamanho[1]))
            if 0 <= col < larg_g and 0 <= lin < alt_g:
                mapa[lin, col] = mundo_deitado(x, y)
    return mapa


def test_a_similaridade_2d_volta_exata():
    rng = np.random.default_rng(4)
    a = rng.uniform(-1, 1, (20, 2))
    e, ang, t = 3.1, 0.8, np.array([2.0, -5.0])
    c, s = math.cos(ang), math.sin(ang)
    r = np.array([[c, -s], [s, c]])
    escala, rot, desloc, residuo = similaridade_2d(a, e * (r @ a.T).T + t)
    assert escala == pytest.approx(e)
    assert rot == pytest.approx(r)
    assert desloc == pytest.approx(t)
    assert residuo < 1e-9


def test_a_similaridade_nao_espelha():
    a = np.array([[0., 0], [1, 0], [0, 1], [1, 1]])
    b = np.array([[0., 0], [1, 0], [0, -1], [1, -1]])
    assert np.linalg.det(similaridade_2d(a, b)[1]) > 0


def test_a_ponte_desfaz_uma_transformacao_CONHECIDA():
    """O teste central: a rede entrega girado e em outra escala; a ponte
    devolve exatamente os metros da homografia."""
    projetar, h = _camera_do_teto()
    e_verdade, ang, t = 1 / 14.0, 0.9, np.array([5.0, -2.0])
    c, s = math.cos(ang), math.sin(ang)
    rv = np.array([[c, -s], [s, c]])

    def na_nuvem(x, y):
        xy = e_verdade * (rv @ np.array([x, y])) + t
        return np.array([xy[0], xy[1], 0.0])

    ponte = alinhar_com_a_homografia(_mapa_do_alto(projetar, na_nuvem), h,
                                     *AREA, (640, 480))
    assert ponte is not None, "nao achou pares"
    escala, r2, desloc, residuo, quantos = ponte

    assert quantos >= 6
    assert escala == pytest.approx(1 / e_verdade, rel=0.02)
    assert residuo < 0.02, f"residuo de {residuo * 100:.1f} cm"

    # e a prova direta: um ponto conhecido volta ao lugar
    volta = escala * (r2 @ na_nuvem(1.0, 0.5)[:2]) + desloc
    assert volta == pytest.approx([1.0, 0.5], abs=0.03)


def test_a_ponte_ignora_o_que_esta_alto():
    """O retangulo aferido e chao. Ponto alto ali e movel ou pessoa."""
    projetar, h = _camera_do_teto()

    def tudo_no_alto(x, y):
        return np.array([x, y, 1.5])

    assert alinhar_com_a_homografia(_mapa_do_alto(projetar, tudo_no_alto), h,
                                    *AREA, (640, 480)) is None


def test_sem_homografia_nao_ha_ponte():
    assert alinhar_com_a_homografia(np.zeros((10, 10, 3)), np.eye(2),
                                    *AREA, (640, 480)) is None


def test_o_ambiente_fica_NO_MUNDO_DO_GEMEO():
    """Ponta a ponta: a estante cai dentro da area onde o gemeo anda."""
    projetar, h = _camera_do_teto()
    verdade = _quarto()
    e_verdade, ang, t = 1 / 14.0, 0.9, np.array([5.0, -2.0])
    c, s = math.cos(ang), math.sin(ang)
    rv = np.array([[c, -s], [s, c]])

    def na_nuvem(x, y, z=0.0):
        xy = e_verdade * (rv @ np.array([x, y])) + t
        return np.array([xy[0], xy[1], z * e_verdade])

    nuvem = np.array([na_nuvem(*pt) for pt in verdade])
    mapa = _mapa_do_alto(projetar, lambda x, y: na_nuvem(x, y))
    calib = {"largura_m": AREA[0], "altura_m": AREA[1]}

    amb = montar(nuvem, GAB, mapa_do_alto=mapa, homografia=h, calib=calib)
    assert amb is not None and amb.no_mundo_do_gemeo, "a ponte nao foi feita"
    assert amb.residuo_m < 0.05, f"{amb.residuo_m * 100:.0f} cm de residuo"

    # a estante do quarto sintetico esta em (1.0, 2.2) no mundo dos metros
    ex, ey, _ = amb.estante
    assert (ex, ey) == pytest.approx((1.0, 2.2), abs=0.25)


def test_as_duas_reguas_sao_comparadas():
    """A escala da homografia e a da altura da estante tem que concordar."""
    projetar, h = _camera_do_teto()
    verdade = _quarto()
    e_verdade = 1 / 14.0

    def na_nuvem(x, y, z=0.0):
        return np.array([x, y, z]) * e_verdade

    nuvem = np.array([na_nuvem(*pt) for pt in verdade])
    mapa = _mapa_do_alto(projetar, lambda x, y: na_nuvem(x, y))
    amb = montar(nuvem, GAB, mapa_do_alto=mapa, homografia=h,
                 calib={"largura_m": AREA[0], "altura_m": AREA[1]})
    assert amb.as_duas_reguas_concordam is True, (
        f"homografia diz {amb.escala:.2f}, estante diz "
        f"{amb.escala_da_estante:.2f}")


def test_sem_a_ponte_o_ambiente_avisa():
    """Sem homografia o ambiente sai flutuando, e ele DIZ isso."""
    amb = montar(_como_a_rede_entrega(_quarto()), GAB)
    assert amb is not None
    assert not amb.no_mundo_do_gemeo
