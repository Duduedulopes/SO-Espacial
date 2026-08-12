"""QUANTAS unidades: so conta gesto que terminou."""
from src.acao.pegadas import ContadorDePegadas
from src.acao.vocabulario import Braco

AO_LADO, ALTO = Braco.AO_LADO, Braco.LEVANTADO


def _gesto(c, pid=1, prateleira="p3", quadros_no_alto=6):
    for _ in range(3):
        c.observar(pid, AO_LADO, AO_LADO, prateleira)
    for _ in range(quadros_no_alto):
        c.observar(pid, AO_LADO, ALTO, prateleira)
    return [p for _ in range(3)
            for p in c.observar(pid, AO_LADO, AO_LADO, prateleira)]


def test_braco_no_alto_ainda_nao_conta():
    c = ContadorDePegadas()
    for _ in range(30):
        c.observar(1, AO_LADO, ALTO, "p3")
    assert c.quantas(1) == 0, "intencao nao e unidade"


def test_o_ciclo_fechado_conta_uma_vez():
    c = ContadorDePegadas()
    fechadas = _gesto(c)
    assert len(fechadas) == 1
    assert c.quantas(1, "p3") == 1
    assert fechadas[0].lado == "dir"


def test_tres_gestos_tres_unidades():
    c = ContadorDePegadas()
    for _ in range(3):
        _gesto(c)
    assert c.quantas(1, "p3") == 3
    assert c.total == 3


def test_tremor_no_limiar_nao_conta():
    """Um quadro solto para cada lado nao fecha ciclo nenhum."""
    c = ContadorDePegadas(confirmacoes=2)
    for _ in range(20):
        c.observar(1, AO_LADO, ALTO, "p3")
        c.observar(1, AO_LADO, AO_LADO, "p3")
    assert c.quantas(1) == 0


def test_cada_prateleira_conta_separado():
    c = ContadorDePegadas()
    _gesto(c, prateleira="p1")
    _gesto(c, prateleira="p4")
    _gesto(c, prateleira="p1")
    assert c.por_prateleira(1) == {"p1": 2, "p4": 1}


def test_o_palpite_que_vale_e_o_ultimo_antes_de_descer():
    """O classificador melhora enquanto a mao esta la em cima."""
    c = ContadorDePegadas()
    for _ in range(3):
        c.observar(1, AO_LADO, AO_LADO, None)
    for _ in range(4):
        c.observar(1, AO_LADO, ALTO, "p2")
    for _ in range(4):
        c.observar(1, AO_LADO, ALTO, "p3")
    fechadas = [p for _ in range(3)
                for p in c.observar(1, AO_LADO, AO_LADO, None)]
    assert fechadas[0].prateleira == "p3"


def test_os_dois_bracos_contam_separado():
    c = ContadorDePegadas()
    for _ in range(3):
        c.observar(1, AO_LADO, AO_LADO, "p3")
    for _ in range(5):
        c.observar(1, ALTO, ALTO, "p3")
    fechadas = [p for _ in range(3)
                for p in c.observar(1, AO_LADO, AO_LADO, "p3")]
    assert {p.lado for p in fechadas} == {"esq", "dir"}
    assert c.quantas(1, "p3") == 2


def test_a_venda_nao_vai_embora_com_o_cliente():
    c = ContadorDePegadas()
    _gesto(c)
    c.esquecer(set())
    assert c.quantas(1, "p3") == 1
