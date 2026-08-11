"""
Testes da escala vertical — a camera do alto medindo estatura em metros.

A IDEIA QUE ORIGINOU ESTE ARQUIVO, DO EDUARDO, 11/08

    nenhuma delas vai captar 100% de tudo, as 3 ja existem ao mesmo tempo
    para uma complementar a outra

A frontal e a lateral ficam sobre a mesa e nunca verao um tornozelo — medido
naquele dia: 0% nas duas. A do alto ve, e o `FiltroDePlausibilidade` ja
calculava, desde o bloco 1, a razao que multiplicada pela altura da camera E a
estatura.

    O dado que faltava ja estava sendo calculado para outra finalidade.

    python -m pytest testes/test_escala.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acao.escala import (                                      # noqa: E402
    QUADRIL_POR_ESTATURA, EscalaVertical,
)

CAMERA = 2.40      # altura da camera do alto, em metros
EDUARDO = 1.78     # estatura de referencia


def razao_de(estatura, altura_camera=CAMERA):
    """A razao que a camera veria para alguem daquela altura."""
    return estatura / altura_camera


# ------------------------------------------------------------- calibracao
def test_uma_pessoa_conhecida_calibra_a_camera():
    """Sem trena na parede, sem angulo, sem subir em cadeira: alguem que sabe
    a propria altura aparece uma vez e a geometria devolve Hc."""
    achado = EscalaVertical.calibrar(EDUARDO, razao_de(EDUARDO))

    assert abs(achado - CAMERA) < 0.01


def test_calibracao_recusa_estatura_impossivel():
    """1,78 cm em vez de 1,78 m e o erro de digitacao mais provavel, e ele
    envenenaria toda medida seguinte sem dar sintoma."""
    import pytest

    for absurdo in (0.0178, 17.8, -1.78):
        with pytest.raises(ValueError):
            EscalaVertical.calibrar(absurdo, 0.74)


def test_calibracao_recusa_razao_invalida():
    import pytest

    with pytest.raises(ValueError):
        EscalaVertical.calibrar(EDUARDO, None)


# ---------------------------------------------------------------- medicao
def test_mede_a_estatura_de_qualquer_um_depois_de_calibrada():
    """Uma pessoa calibra; todas as outras sao MEDIDAS, nao assumidas."""
    e = EscalaVertical(altura_camera_m=CAMERA)

    for pid, verdade in ((1, 1.78), (2, 1.62), (3, 1.10)):
        for _ in range(10):
            e.observar(pid, razao_de(verdade))
        assert abs(e.estatura(pid) - verdade) < 0.01, pid


def test_sem_calibracao_nao_responde():
    """Nao calibrada, a escala nao chuta uma altura de camera plausivel — ela
    se cala, e a altura da mao volta a ser estimada pelo tronco e marcada."""
    e = EscalaVertical()

    for _ in range(20):
        e.observar(1, razao_de(EDUARDO))

    assert e.estatura(1) is None
    assert e.altura_do_quadril(1) is None
    assert "NAO CALIBRADA" in e.diagnostico


def test_quem_agacha_nao_envenena_a_estatura():
    """A caixa de quem agacha e menor DE VERDADE, e uma amostra dali diria que
    a pessoa mede 1,10 m. O sinal de postura vem do classificador, que decidiu
    pela coxa — um caminho independente da caixa."""
    e = EscalaVertical(altura_camera_m=CAMERA)

    for _ in range(20):
        e.observar(1, razao_de(EDUARDO), em_pe=True)
    for _ in range(20):
        e.observar(1, razao_de(1.05), em_pe=False)      # agachado

    assert abs(e.estatura(1) - EDUARDO) < 0.01


def test_a_mediana_absorve_o_tremor_da_caixa():
    """A caixa do detector treme alguns pixels a cada inferencia. A razao de um
    quadro so carrega esse ruido; a mediana de dezenas nao."""
    import random

    random.seed(11)
    e = EscalaVertical(altura_camera_m=CAMERA)

    for _ in range(60):
        ruido = random.gauss(0, 0.05)
        e.observar(1, razao_de(EDUARDO + ruido))

    assert abs(e.estatura(1) - EDUARDO) < 0.02


def test_estatura_absurda_nao_entra():
    """Uma caixa de 4 m nao e uma pessoa alta: e deteccao ruim, e ela nao pode
    virar a referencia de altura de ninguem."""
    e = EscalaVertical(altura_camera_m=CAMERA)

    for _ in range(20):
        e.observar(1, razao_de(4.0))

    assert e.estatura(1) is None


def test_o_quadril_sai_da_estatura_MEDIDA():
    """A proporcao continua sendo modelo — mas aplicada sobre uma estatura
    medida, e nao sobre um tronco estimado. O erro cai de ~8 cm para ~3 cm."""
    e = EscalaVertical(altura_camera_m=CAMERA)
    for _ in range(10):
        e.observar(1, razao_de(EDUARDO))

    assert abs(e.altura_do_quadril(1) - EDUARDO * QUADRIL_POR_ESTATURA) < 0.01


def test_a_memoria_some_com_o_rastro():
    """Estatura aprendida da pessoa 1 nao pode responder pela pessoa 2."""
    e = EscalaVertical(altura_camera_m=CAMERA)
    for _ in range(10):
        e.observar(1, razao_de(EDUARDO))

    e.esquecer({2})
    assert e.estatura(1) is None


def test_camera_perpendicular_nao_derruba_a_escala():
    """Sem horizonte nao ha razao, e `razao()` devolve None. A escala tem que
    aceitar isso em silencio — foi um ZeroDivisionError em producao antes de a
    guarda entrar dentro do proprio filtro."""
    e = EscalaVertical(altura_camera_m=CAMERA)

    for _ in range(20):
        e.observar(1, None)

    assert e.estatura(1) is None
