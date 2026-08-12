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

FATOR = 5.25       # fator empirico medido em 11/08 com a C920 do alto
EDUARDO = 1.78     # estatura de referencia


def razao_de(estatura, fator=FATOR):
    """A razao que a camera veria para alguem daquela altura."""
    return estatura / fator


# ------------------------------------------------------------- calibracao
def test_uma_pessoa_conhecida_calibra_a_escala():
    """Sem trena na parede, sem angulo, sem subir em cadeira: alguem que sabe a
    propria altura aparece uma vez e a razao observada fecha a conta.

    O NUMERO NAO E A ALTURA DA CAMERA, E EU ERREI AO CHAMA-LO ASSIM.

    MEDIDO EM 11/08: Eduardo a 1,80 m deu razao 0,343 e fator 5,25 — e a camera
    nao esta a cinco metros do chao. A relacao `razao = altura / Hc` vale para
    camera SEM inclinacao; a do alto olha o chao quase de cima e a pessoa
    aparece encurtada. O fator absorveu a inclinacao junto.

        Uma constante empirica nao precisa ter nome fisico. Precisa ser
        estavel, e precisa ser medida do mesmo jeito que sera usada.
    """
    achado = EscalaVertical.calibrar(EDUARDO, razao_de(EDUARDO))

    assert abs(achado - FATOR) < 0.01


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
    e = EscalaVertical(fator=FATOR)

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
    e = EscalaVertical(fator=FATOR)

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
    e = EscalaVertical(fator=FATOR)

    for _ in range(60):
        ruido = random.gauss(0, 0.05)
        e.observar(1, razao_de(EDUARDO + ruido))

    assert abs(e.estatura(1) - EDUARDO) < 0.02


def test_estatura_absurda_nao_entra():
    """Uma caixa de 4 m nao e uma pessoa alta: e deteccao ruim, e ela nao pode
    virar a referencia de altura de ninguem."""
    e = EscalaVertical(fator=FATOR)

    for _ in range(20):
        e.observar(1, razao_de(4.0))

    assert e.estatura(1) is None


def test_o_quadril_sai_da_estatura_MEDIDA():
    """A proporcao continua sendo modelo — mas aplicada sobre uma estatura
    medida, e nao sobre um tronco estimado. O erro cai de ~8 cm para ~3 cm."""
    e = EscalaVertical(fator=FATOR)
    for _ in range(10):
        e.observar(1, razao_de(EDUARDO))

    assert abs(e.altura_do_quadril(1) - EDUARDO * QUADRIL_POR_ESTATURA) < 0.01


def test_a_memoria_some_com_o_rastro():
    """Estatura aprendida da pessoa 1 nao pode responder pela pessoa 2."""
    e = EscalaVertical(fator=FATOR)
    for _ in range(10):
        e.observar(1, razao_de(EDUARDO))

    e.esquecer({2})
    assert e.estatura(1) is None


def test_camera_perpendicular_nao_derruba_a_escala():
    """Sem horizonte nao ha razao, e `razao()` devolve None. A escala tem que
    aceitar isso em silencio — foi um ZeroDivisionError em producao antes de a
    guarda entrar dentro do proprio filtro."""
    e = EscalaVertical(fator=FATOR)

    for _ in range(20):
        e.observar(1, None)

    assert e.estatura(1) is None


# ----------------------------------------- a camera do alto mede o quadril
#
#     A camera superior consegue captar SIM a imagem da primeira prateleira.
#     O que nao esta acontecendo e as tres trabalharem juntas.
#                                                       — Eduardo, 12/08
#
# No dia anterior eu tinha escrito que "agachado sem tornozelo a vista nao ha
# resposta", e apresentei como recusa honesta. Nao era: era o limite das duas
# cameras de mesa sendo chamado de principio.
#
# A formula e a mesma metrologia de vista unica que `chao.py` ja usa para a
# estatura, aplicada a outro par de pontos — e nao assume postura nenhuma.

import numpy as np                                              # noqa: E402

from src.acao.escala import (                                   # noqa: E402
    altura_do_quadril_vista_de_cima as altura_do_alto,
)

FATOR = 5.25
HORIZONTE = -400.0          # acima do topo da imagem, como numa camera de teto


def cena(quadril_m, y2=460.0, fator=FATOR, horizonte=HORIZONTE):
    """Monta juntas 2D e CAIXA coerentes com um quadril `quadril_m` do chao.

    `y2` e a base da caixa — o mesmo ponto contra o qual o `fator` foi
    calibrado em `chao.py`. Usar o tornozelo aqui foi o defeito de 12/08: a
    ancora leu 0,47 m para alguem em pe, e todas as prateleiras desabaram.

    Inverte a propria formula para gerar a entrada: se a conta estiver certa,
    ela devolve exatamente o valor pedido. Gerar com a formula e conferir com
    ela testaria tautologia — por isso os testes seguintes mexem em UMA coisa
    de cada vez e olham o SENTIDO da mudanca, nao so o valor.
    """
    v_quadril = y2 - quadril_m * (y2 - horizonte) / fator
    p = np.zeros((17, 2))
    p[11] = [300.0, v_quadril]
    p[12] = [340.0, v_quadril]
    p[15] = [310.0, y2 - 18]        # tornozelo ACIMA da sola, como na vida
    p[16] = [330.0, y2 - 30]
    caixa = (280.0, v_quadril - 300.0, 360.0, y2)
    return p, np.ones(17), caixa


def horizonte_fixo(u):
    return HORIZONTE


def test_mede_o_quadril_de_quem_esta_em_pe():
    p, c, caixa = cena(0.95)
    assert abs(altura_do_alto(p, c, caixa, horizonte_fixo, FATOR) - 0.95) < 0.01


def test_mede_o_quadril_de_quem_agachou():
    """O caso que a constante `estatura x 0,53` nao consegue ver.

    Medido em 11/08: 1 cm de erro na prateleira de 1,90 m e 75 cm na de 0,15 m,
    porque a ancora era uma constante de pessoa em pe.
    """
    p, c, caixa = cena(0.45)
    assert abs(altura_do_alto(p, c, caixa, horizonte_fixo, FATOR) - 0.45) < 0.01


def test_quadril_mais_alto_na_imagem_significa_pessoa_mais_ereta():
    """Sentido da relacao, sem depender do valor exato."""
    p_agachado, c, caixa = cena(0.45)
    p_em_pe, _, caixa_em_pe = cena(0.95)
    assert p_em_pe[11][1] < p_agachado[11][1], "em pe, o quadril sobe na imagem"
    assert (altura_do_alto(p_em_pe, c, caixa_em_pe, horizonte_fixo, FATOR)
            > altura_do_alto(p_agachado, c, caixa, horizonte_fixo, FATOR))


def test_o_chao_e_a_BASE_DA_CAIXA_e_nao_o_tornozelo():
    """O defeito de 12/08, virado em teste.

    O `fator` foi calibrado como `estatura / ((y2-y1)/(y2-horizonte))`, com
    `y2` = base da caixa. Aplicar contra o tornozelo — dezenas de pixels acima
    da sola — encolhe numerador e denominador e derruba a razao.

        Constante empirica so vale medida do MESMO jeito que sera usada.

    Aqui o tornozelo esta 18 px acima da base. Se a conta usasse ele, o
    resultado cairia bem abaixo de 0,95 — foi o que produziu 0,47 no gabarito.
    """
    p, c, caixa = cena(0.95)
    assert abs(altura_do_alto(p, c, caixa, horizonte_fixo, FATOR) - 0.95) < 0.01

    # E a prova de que a diferenca importa: refazendo a conta com o tornozelo.
    v_torn = max(p[15][1], p[16][1])
    v_quadril = p[11][1]
    pelo_tornozelo = FATOR * (v_torn - v_quadril) / (v_torn - HORIZONTE)
    assert pelo_tornozelo < 0.93, (
        "se dar quase o mesmo, o cenario nao reproduz o defeito medido")


def test_pe_no_ar_nao_atrapalha_mais():
    """Com a caixa como chao, tornozelo nem entra na conta."""
    p, c, caixa = cena(0.95)
    p[15][1] = p[16][1] = 100.0
    c[15] = c[16] = 0.0
    assert abs(altura_do_alto(p, c, caixa, horizonte_fixo, FATOR) - 0.95) < 0.01


def test_sem_caixa_nao_responde():
    p, c, _ = cena(0.95)
    assert altura_do_alto(p, c, None, horizonte_fixo, FATOR) is None


def test_quadril_fora_da_propria_caixa_e_recusado():
    """Reconstrucao ruim poe o quadril acima da cabeca ou abaixo do pe."""
    p, c, caixa = cena(0.95)
    p[11][1] = p[12][1] = caixa[3] + 50.0      # abaixo da base
    assert altura_do_alto(p, c, caixa, horizonte_fixo, FATOR) is None


def test_sem_os_dois_quadris_nao_responde():
    """Um visto e outro extrapolado da uma media com cara de medida."""
    p, c, caixa = cena(0.95)
    c[12] = 0.0
    assert altura_do_alto(p, c, caixa, horizonte_fixo, FATOR) is None


def test_sem_calibracao_nao_responde():
    """Sem o fator nao ha metro nenhum a devolver."""
    p, c, caixa = cena(0.95)
    assert altura_do_alto(p, c, caixa, horizonte_fixo, None) is None


def test_horizonte_no_infinito_nao_responde():
    """Lente sem inclinacao e configuracao legitima, nao erro.

    `FiltroDePlausibilidade.v_horizonte` devolve None nesse caso — e foi
    exatamente o ZeroDivisionError que apareceu ao ligar esta funcao.
    """
    p, c, caixa = cena(0.95)
    assert altura_do_alto(p, c, caixa, lambda u: None, FATOR) is None


def test_pe_acima_do_horizonte_e_impossivel():
    p, c, caixa = cena(0.95)
    assert altura_do_alto(p, c, caixa, lambda u: 9999.0, FATOR) is None


def test_resultado_fora_da_faixa_humana_e_recusado():
    """Reconstrucao ruim nao vira quadril de tres metros."""
    p, c, caixa = cena(0.95)
    p[11][1] = p[12][1] = -5000.0        # quadril absurdamente alto
    assert altura_do_alto(p, c, caixa, horizonte_fixo, FATOR) is None
