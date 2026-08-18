"""A estante posta pela trena, e a conferencia que paga a etapa.

    A camera e o instrumento do que se move. A trena e o instrumento do que
    fica parado. Trocar os dois de lugar custa um dia.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ferramentas.por_estante import estante_de       # noqa: E402
from src.mundo.ambiente import Gabarito              # noqa: E402
from visual.cena3d import Cena3D                     # noqa: E402

GAB = Gabarito.de_arquivo("loja/estante.json")


def test_as_dimensoes_sao_as_da_trena():
    e, _ = estante_de((0.30, 1.00), (1.22, 1.00), GAB)
    assert (e.largura, e.profundidade, e.altura) == (GAB.largura,
                                                     GAB.profundidade,
                                                     GAB.altura)


def test_o_centro_fica_ATRAS_dos_pes_da_frente():
    """Os pes sao a frente; o centro esta meia profundidade atras deles.

    Confundir os dois poe a estante 15 cm para dentro da area onde a pessoa
    anda, e a conta de 'esta na frente da estante' passa a comecar dentro
    dela.
    """
    e, _ = estante_de((0.30, 1.00), (1.22, 1.00), GAB)
    meio_da_frente = np.array([0.76, 1.00])
    d = np.array([e.x, e.y]) - meio_da_frente
    assert np.linalg.norm(d) == pytest.approx(GAB.profundidade / 2, abs=1e-6)
    assert float(d @ e.normal) < 0, "o centro foi parar na frente dos pes"


def test_a_face_olha_para_onde_a_pessoa_anda():
    """Para a origem, porque atras da estante ha parede."""
    for esq, dir_ in ((( 0.30, 1.00), (1.22, 1.00)),
                      (( 1.22, 1.00), (0.30, 1.00)),
                      (( 1.40, 0.30), (1.40, 1.22))):
        e, _ = estante_de(esq, dir_, GAB)
        para_origem = np.array([0.0, 0.0]) - np.array([e.x, e.y])
        assert float(e.normal @ para_origem) > 0, f"face virada ao contrario"


def test_a_largura_medida_sai_junto_para_conferencia():
    _, medida = estante_de((0.30, 1.00), (1.22, 1.00), GAB)
    assert medida == pytest.approx(0.92, abs=1e-6)


def test_medida_torta_e_denunciada_pela_largura():
    """A conferencia: se os pes nao distam a largura, algo saiu errado."""
    _, medida = estante_de((0.30, 1.00), (1.60, 1.00), GAB)
    assert abs(medida - GAB.largura) > 0.06


def test_estante_girada_devolve_o_rumo_certo():
    e, _ = estante_de((1.40, 0.30), (1.40, 1.22), GAB)
    ao_longo = np.array([math.cos(e.rumo_da_face), math.sin(e.rumo_da_face)])
    assert abs(abs(float(ao_longo @ np.array([0.0, 1.0]))) - 1.0) < 1e-6


def test_quem_esta_na_frente_e_reconhecido():
    """A prova de que a posicao serve para o que ela existe."""
    e, _ = estante_de((0.30, 1.00), (1.22, 1.00), GAB)
    assert e.de_frente(0.76, 0.60), "quem esta diante dela nao foi visto"
    assert not e.de_frente(0.76, 1.80), "quem esta atras da parede foi aceito"


def test_a_cena_desenha_a_estante_posta():
    e, _ = estante_de((0.30, 1.00), (1.22, 1.00), GAB)
    cena = Cena3D(320, 240, chao=(-0.2, 1.85, -0.2, 1.85))
    cena.add_movel(e.x, e.y, e.largura, e.profundidade, e.altura, "Estante",
                   rumo=e.rumo_da_face, prateleiras=e.prateleiras)
    assert cena.desenhar([]) is not None
    assert cena.moveis[0][7] == (0.15, 0.55, 0.95, 1.35, 1.90)
