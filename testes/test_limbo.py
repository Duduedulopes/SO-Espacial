"""Uma pessoa, uma identidade — mesmo depois de o rastro morrer.

O DEFEITO, MEDIDO NA TELA DE 12/08

    #2  {'p1': 3, 'p3': 6, 'p4': 3, 'p5': 5}
    #3  {'p1': 1}
    #4  {'p2': 2, 'p3': 1, 'p4': 2}

Uma pessoa so na sala, tres identidades. Cada troca joga fora a estatura ja
fechada e reinicia a contagem de unidades do zero — foi o que inflou o total
para 23 unidades numa sessao com muito menos gestos.

A CAUSA

A recostura por proximidade ja existia e funcionava, mas so enquanto o rastro
estivesse VIVO. Passados `max_coasting_s`, ele era apagado — e com ele sumia a
unica coisa capaz de reconhecer a pessoa mais tarde: onde ela estava e quanto
ela media.

    Descartar o rastro perdido para economizar memoria e jogar fora justamente
    a prova de que ele era o mesmo.

O CONSERTO: o rastro morto vira uma ficha barata e espera no limbo.
"""
import pytest

from estado.rastreio import GerenciadorDeRastros


def _andar(g, id_ext, x, y, quadros=1, dt=1 / 30):
    for _ in range(quadros):
        g.atualizar([(id_ext, x, y)], dt)
    return g


def _sumir(g, segundos, dt=1 / 30):
    for _ in range(int(segundos / dt)):
        g.atualizar([], dt)
    return g


# ------------------------------------------------------- o que ja funcionava
def test_rastro_vivo_ainda_ganha_da_ficha_morta():
    """Recostura tem prioridade. Readotar quem ainda esta na sala funde duas."""
    g = GerenciadorDeRastros()
    _andar(g, 10, 0.5, 0.5, quadros=5)
    _sumir(g, 0.2)
    g.atualizar([(11, 0.55, 0.5)], 1 / 30)
    assert g.de_externo[11] == 1
    assert g.recosturas == 1
    assert g.readocoes == 0


# ------------------------------------------------------- o limbo
def test_rastro_apagado_vai_para_o_limbo_e_nao_para_o_nada():
    g = GerenciadorDeRastros(max_coasting_s=1.0)
    _andar(g, 10, 0.5, 0.5, quadros=5)
    _sumir(g, 1.5)
    assert g.rastros == {}
    assert 1 in g.limbo


def test_quem_volta_depois_do_coasting_recebe_o_ID_ANTIGO():
    """O conserto. Antes isto criava a identidade #2."""
    g = GerenciadorDeRastros(max_coasting_s=1.0)
    _andar(g, 10, 0.5, 0.5, quadros=5)
    _sumir(g, 1.5)
    g.atualizar([(77, 0.6, 0.5)], 1 / 30)
    assert g.de_externo[77] == 1, "ganhou identidade nova de novo"
    assert g.readocoes == 1
    assert g.proximo_id == 2, "nao pode ter consumido um id novo"


def test_o_limbo_expira():
    """Vinte segundos e memoria; memoria eterna e outra coisa."""
    g = GerenciadorDeRastros(max_coasting_s=1.0, max_limbo_s=5.0)
    _andar(g, 10, 0.5, 0.5, quadros=5)
    _sumir(g, 8.0)
    assert g.limbo == {}
    g.atualizar([(77, 0.5, 0.5)], 1 / 30)
    assert g.de_externo[77] == 2, "devia ser pessoa nova depois de expirar"


# ------------------------------------------------------- a estatura decide
def test_estatura_diferente_nao_readota():
    """A protecao que torna o limbo aceitavel: altura e a assinatura."""
    g = GerenciadorDeRastros(max_coasting_s=1.0)
    _andar(g, 10, 0.5, 0.5, quadros=5)
    g.informar_estatura(1, 1.83)
    _sumir(g, 1.5)

    # chega alguem no mesmo lugar, mas 30 cm mais baixo
    g.atualizar([(77, 0.5, 0.5)], 1 / 30)
    novo = g.de_externo[77]
    g.informar_estatura(novo, 1.53)
    assert novo == 1, "sem estatura conhecida ainda, decide a proximidade"

    # agora com as duas estaturas na mesa, a recusa tem que valer
    g2 = GerenciadorDeRastros(max_coasting_s=1.0)
    _andar(g2, 10, 0.5, 0.5, quadros=5)
    g2.informar_estatura(1, 1.83)
    _sumir(g2, 1.5)
    g2.estaturas[2] = 1.53          # o proximo id ja chega medido
    ficha = g2.limbo[1]
    assert not g2._combina_estatura(2, ficha)


def test_estatura_igual_readota():
    g = GerenciadorDeRastros(max_coasting_s=1.0)
    _andar(g, 10, 0.5, 0.5, quadros=5)
    g.informar_estatura(1, 1.83)
    _sumir(g, 1.5)
    g.estaturas[2] = 1.85           # 2 cm de diferenca: a mesma pessoa
    assert g._combina_estatura(2, g.limbo[1])


def test_sem_estatura_medida_decide_a_proximidade():
    """Recusar por falta de medida faria o conserto depender do que ele salva."""
    g = GerenciadorDeRastros(max_coasting_s=1.0)
    _andar(g, 10, 0.5, 0.5, quadros=5)
    _sumir(g, 1.5)
    assert g._combina_estatura(2, g.limbo[1]) is True


def test_informar_estatura_ignora_none():
    g = GerenciadorDeRastros()
    g.informar_estatura(1, None)
    assert 1 not in g.estaturas


# ------------------------------------------------------- longe demais
def test_quem_volta_do_outro_lado_do_mundo_nao_e_a_mesma_pessoa():
    g = GerenciadorDeRastros(max_coasting_s=1.0, vel_max=0.5)
    _andar(g, 10, 0.0, 0.0, quadros=5)
    _sumir(g, 1.2)
    g.atualizar([(77, 40.0, 40.0)], 1 / 30)
    assert g.de_externo[77] == 2


def test_a_ausencia_conta_desde_a_ultima_vez_que_a_PESSOA_foi_vista():
    """E nao desde a morte do rastro.

    Entre a ultima medicao e a morte passam `max_coasting_s` — e nesse
    intervalo a pessoa esteve andando por ai sem ser vista. Datar a ficha
    pelo obito encolhe o raio justamente no trecho que mais importa.
    """
    g = GerenciadorDeRastros(max_coasting_s=1.0, vel_max=1.0)
    _andar(g, 10, 0.0, 0.0, quadros=5)
    _sumir(g, 1.2)
    ausencia = g.relogio - g.limbo[1]["t"]
    assert ausencia == pytest.approx(1.2, abs=0.1), "datou pelo obito"


def test_o_raio_cresce_com_a_ausencia():
    """Quanto mais tempo sem ver, mais longe ela pode estar."""
    perto = GerenciadorDeRastros(max_coasting_s=1.0, vel_max=1.0)
    _andar(perto, 10, 0.0, 0.0, quadros=5)
    _sumir(perto, 1.2)
    perto.atualizar([(77, 1.0, 0.0)], 1 / 30)
    assert perto.de_externo[77] == 1

    longe = GerenciadorDeRastros(max_coasting_s=1.0, vel_max=1.0)
    _andar(longe, 10, 0.0, 0.0, quadros=5)
    _sumir(longe, 1.2)
    longe.atualizar([(77, 12.0, 0.0)], 1 / 30)
    assert longe.de_externo[77] == 2


def test_uma_sessao_inteira_com_tres_sumicos_da_uma_identidade_so():
    """O teste que descreve o defeito do jeito que ele apareceu."""
    g = GerenciadorDeRastros(max_coasting_s=1.0)
    _andar(g, 10, 0.4, 0.4, quadros=30)
    g.informar_estatura(1, 1.83)

    for n, ext in enumerate((21, 32, 43), start=1):
        _sumir(g, 2.0)
        g.atualizar([(ext, 0.45, 0.42)], 1 / 30)
        g.informar_estatura(g.de_externo[ext], 1.83)
        assert g.de_externo[ext] == 1, f"perdeu a identidade no sumico {n}"

    assert g.proximo_id == 2, "criou identidade nova alguma vez"
