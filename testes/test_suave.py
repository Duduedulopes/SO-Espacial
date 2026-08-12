"""O filtro que tira o tremelique tem que tirar SO o tremelique."""
import math

import pytest

from src.gemeo.suave import Suavizador, _mistura_angulo


def test_primeira_amostra_passa_intacta():
    s = Suavizador()
    assert s.suavizar(1, 2.0, 3.0, 0.5) == (2.0, 3.0, 0.5)


def test_ruido_pequeno_e_amaciado():
    """Alguem parado com a caixa tremendo 2 cm nao pode andar 2 cm por quadro."""
    s = Suavizador(alfa=0.25)
    s.suavizar(1, 0.0, 0.0)
    x, _, _ = s.suavizar(1, 0.02, 0.0)
    assert 0.0 < x < 0.02
    assert x == pytest.approx(0.005)


def test_o_boneco_alcanca_quem_anda():
    """Amaciar nao pode virar atrasar para sempre."""
    s = Suavizador(alfa=0.25)
    s.suavizar(1, 0.0, 0.0)
    for _ in range(40):
        x, y, _ = s.suavizar(1, 0.5, 0.0)
    assert x == pytest.approx(0.5, abs=0.005)


def test_salto_grande_nao_arrasta_o_boneco_pela_cena():
    """Id reciclado num canto oposto: vai direto, sem cruzar a sala."""
    s = Suavizador(alfa=0.25, salto_m=0.60)
    s.suavizar(1, 0.0, 0.0)
    x, y, _ = s.suavizar(1, 3.0, 0.0)
    assert (x, y) == (3.0, 0.0)


def test_rumo_ausente_mantem_o_ultimo_conhecido():
    """Perder um quadro nao pode fazer o boneco dar meia-volta."""
    s = Suavizador()
    s.suavizar(1, 0.0, 0.0, 1.2)
    _, _, r = s.suavizar(1, 0.0, 0.0, None)
    assert r == pytest.approx(1.2)


def test_rumo_cruza_a_costura_pelo_caminho_curto():
    """De +179 para -179 sao 2 graus de giro, nao 358."""
    a, b = math.radians(179), math.radians(-179)
    r = _mistura_angulo(a, b, 0.5)
    assert abs(r) > math.radians(179.0)          # ficou perto de 180, nao de 0


def test_esquecer_apaga_quem_saiu():
    s = Suavizador()
    s.suavizar(1, 0.0, 0.0)
    s.suavizar(2, 5.0, 5.0)
    s.esquecer({2})
    assert s.suavizar(1, 9.0, 9.0) == (9.0, 9.0, None)      # tratado como novo
