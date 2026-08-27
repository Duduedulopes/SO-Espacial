"""Os detectores medem em METROS antes de filtrar. Sem imagem real, com verdade
conhecida: desenhamos a estante e conferimos que ela sai com o tamanho certo."""

import cv2
import numpy as np
import pytest

from src.mundo.ambiente import Gabarito, reconhecer
from src.mundo.detectores import (alturas_de_frente, candidatos_do_alto,
                                  olhar_o_ambiente)

# Homografia sintetica: 200 px por metro, origem no canto da imagem.
PX_POR_M = 200.0
H = np.array([[1 / PX_POR_M, 0, 0],
              [0, 1 / PX_POR_M, 0],
              [0, 0, 1.0]])
GAB = Gabarito.de_arquivo("loja/estante.json")


def _vista_de_cima(largura_m=0.92, prof_m=0.30, cx_m=1.0, cy_m=1.1, giro=0):
    """Desenha, de cima, um retangulo do tamanho da estante."""
    img = np.zeros((480, 640, 3), np.uint8)
    cx, cy = cx_m * PX_POR_M, cy_m * PX_POR_M
    caixa = ((cx, cy), (largura_m * PX_POR_M, prof_m * PX_POR_M), giro)
    cv2.drawContours(img, [cv2.boxPoints(caixa).astype(np.int32)], 0,
                     (210, 210, 210), -1)
    return img


def test_a_estante_sai_medida_em_metros():
    c = candidatos_do_alto(_vista_de_cima(), H)
    assert c, "nao achou o retangulo"
    m = max(c, key=lambda v: v.lado_maior * v.lado_menor)
    assert m.lado_maior == pytest.approx(0.92, abs=0.06)
    assert m.lado_menor == pytest.approx(0.30, abs=0.06)
    assert m.centro[0] == pytest.approx(1.0, abs=0.05)


def test_o_candidato_medido_casa_com_o_gabarito():
    """A ponta a ponta: detectar de cima e reconhecer contra a trena."""
    c = candidatos_do_alto(_vista_de_cima(), H)
    a = reconhecer(GAB, do_alto=max(c, key=lambda v: v.lado_maior))
    assert a is not None
    assert a.largura == pytest.approx(0.92, abs=0.06)


def test_a_mesa_e_medida_e_recusada():
    """O filtro nao esta no detector: esta no gabarito. E ele recusa."""
    c = candidatos_do_alto(_vista_de_cima(largura_m=1.40, prof_m=0.80,
                                          cx_m=0.9, cy_m=0.6), H)
    assert c, "a mesa TEM que ser detectada — e depois recusada"
    assert reconhecer(GAB, do_alto=max(c, key=lambda v: v.lado_maior)) is None


def test_sem_homografia_nao_ha_medida():
    assert candidatos_do_alto(_vista_de_cima(), None) == []


def test_livro_no_chao_some_pela_area():
    assert candidatos_do_alto(_vista_de_cima(0.20, 0.15, 0.5, 0.5), H) == []


# ------------------------------------------------------ frontal / lateral
def _prateleiras(alturas_px=(90, 160, 230, 300, 370), largura=420):
    img = np.zeros((480, 640, 3), np.uint8)
    for y in alturas_px:
        cv2.line(img, (110, y), (110 + largura, y), (215, 215, 215), 4)
    return img


def test_as_prateleiras_viram_alturas_em_metros():
    # y=440 e o chao; 200 px por metro, para cima
    v = alturas_de_frente(_prateleiras(), lambda y: (440 - y) / 200.0)
    assert v is not None
    assert len(v.alturas) >= 4
    assert min(v.alturas) >= 0.0 and max(v.alturas) <= 2.0


def test_uma_linha_solta_nao_e_prateleira():
    """Pode ser o rodape, a quina da parede, a mesa."""
    assert alturas_de_frente(_prateleiras(alturas_px=(230,)),
                             lambda y: (440 - y) / 200.0) is None


def test_segmentos_quebrados_viram_uma_prateleira_so():
    """Produto corta a prateleira em pedacos. Continua sendo uma."""
    img = np.zeros((480, 640, 3), np.uint8)
    for x0 in (110, 260, 400):
        cv2.line(img, (x0, 230), (x0 + 110, 231), (215, 215, 215), 4)
    cv2.line(img, (110, 300), (530, 300), (215, 215, 215), 4)
    # comprimento_min menor: o teste e sobre AGRUPAR pedacos, nao sobre o
    # tamanho minimo de um segmento
    v = alturas_de_frente(img, lambda y: (440 - y) / 200.0,
                          comprimento_min_px=50)
    assert v is not None and len(v.alturas) == 2, v.alturas


def test_sem_escala_nao_responde():
    assert alturas_de_frente(_prateleiras(), None) is None


def test_olhar_o_ambiente_devolve_os_tres():
    alto, frontal, lateral = olhar_o_ambiente(
        {"alto": _vista_de_cima(), "frontal": _prateleiras(),
         "lateral": _prateleiras()},
        H=H, escalas={"frontal": lambda y: (440 - y) / 200.0,
                      "lateral": lambda y: (440 - y) / 200.0})
    assert alto and frontal is not None and lateral is not None


def test_o_ambiente_completo_com_as_tres_cameras():
    alto, frontal, lateral = olhar_o_ambiente(
        {"alto": _vista_de_cima(), "frontal": _prateleiras(),
         "lateral": _prateleiras()},
        H=H, escalas={"frontal": lambda y: (1.90 - (y - 90) * 0.00479),
                      "lateral": lambda y: (1.90 - (y - 90) * 0.00479)})
    a = reconhecer(GAB, do_alto=max(alto, key=lambda v: v.lado_maior),
                   da_frente=frontal, da_lateral=lateral)
    assert a is not None
    assert a.confiavel, f"esperava 2+ cameras, veio {a.cameras}"
