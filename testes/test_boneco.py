"""O boneco não consegue fazer o impossível — e estes testes TENTAM obrigá-lo.

    para que serviu eu falar para vc das ações que eram impossiveis? que só
    teria a opção de estar em pé e agachar se o esqueleto nao faz isso????
                                                        — Eduardo, 12/08

Ele estava certo. O vocabulário fechado existia na camada que LÊ e não existia
na camada que DESENHA: `rodar.py` continuava desenhando `p.esqueleto`, as
juntas cruas da reconstrução, com todos os defeitos que dois dias de medição
tinham isolado.

`src/acao/vocabulario.py` prometia isto desde 10/08:

    Se "deitado" nao esta no vocabulario, o boneco nao consegue deitar.
    A classe inteira de defeito desaparece por construcao, e nao por conserto.

"Por construção" é uma afirmação forte, e afirmação forte pede teste hostil.
Os testes abaixo alimentam absurdos — mão a nove metros, estatura negativa,
postura que não existe — e exigem que o corpo continue possível.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acao.vocabulario import Braco, Locomocao, Postura      # noqa: E402
from src.gemeo import boneco                                    # noqa: E402
from src.gemeo.boneco import (                                  # noqa: E402
    ALCANCE_MAXIMO, NARIZ, OMBRO_DIR, OMBRO_ESQ, PULSO_DIR, PULSO_ESQ,
    QUADRIL_DIR, QUADRIL_ESQ, TORNOZELO_DIR, TORNOZELO_ESQ, montar,
)

E = 1.80


def altura_do_pe(j):
    return min(j[TORNOZELO_ESQ][2], j[TORNOZELO_DIR][2])


def altura_do_quadril(j):
    return (j[QUADRIL_ESQ][2] + j[QUADRIL_DIR][2]) / 2


# ------------------------------------------------- as invariantes do corpo
TODAS_AS_POSTURAS = list(Postura.TODAS)
TODOS_OS_BRACOS = list(Braco.TODOS)
TODAS_AS_LOCOMOCOES = list(Locomocao.TODAS)


@pytest.mark.parametrize("postura", TODAS_AS_POSTURAS)
@pytest.mark.parametrize("braco", TODOS_OS_BRACOS)
@pytest.mark.parametrize("locomocao", TODAS_AS_LOCOMOCOES)
def test_nenhuma_combinacao_do_vocabulario_atravessa_o_chao(
        postura, braco, locomocao):
    """O vocabulário inteiro, combinado, e nenhum corpo dentro do piso."""
    j = montar(estatura=E, postura=postura, locomocao=locomocao,
               braco_esq=braco, braco_dir=braco)
    assert j[:, 2].min() >= 0.0, f"junta a {j[:, 2].min():.3f} m do chao"


@pytest.mark.parametrize("postura", TODAS_AS_POSTURAS)
def test_sempre_ha_um_pe_apoiado(postura):
    """Quem não voa se apoia. Pé pairando é o corpo perdendo o chão."""
    j = montar(estatura=E, postura=postura)
    assert altura_do_pe(j) < 0.12 * E


@pytest.mark.parametrize("braco", TODOS_OS_BRACOS)
def test_o_pulso_nunca_sai_do_alcance_do_braco(braco):
    j = montar(estatura=E, braco_esq=braco, braco_dir=braco)
    for ombro, pulso in ((OMBRO_ESQ, PULSO_ESQ), (OMBRO_DIR, PULSO_DIR)):
        d = float(np.linalg.norm(j[pulso] - j[ombro]))
        assert d <= ALCANCE_MAXIMO * E + 1e-6, f"braco de {d:.2f} m"


@pytest.mark.parametrize("postura", TODAS_AS_POSTURAS)
def test_a_cabeca_nunca_fica_abaixo_do_quadril(postura):
    """Foi um esqueleto DEITADO que motivou o vocabulário fechado, em 10/08."""
    j = montar(estatura=E, postura=postura)
    assert j[NARIZ][2] > altura_do_quadril(j)


@pytest.mark.parametrize("postura", TODAS_AS_POSTURAS)
def test_com_os_bracos_baixos_nada_passa_da_estatura(postura):
    """Braco pendurado: a junta mais alta e a cabeca, e ela e a estatura.

    Com o braco LEVANTADO a mao sobe acima da cabeca — isso e anatomia, nao
    defeito, e tem teste proprio abaixo com o limite de alcance humano.
    """
    j = montar(estatura=E, postura=postura)
    assert j[:, 2].max() <= E + 1e-6


# ----------------------------------------------- descrição absurda, corpo possível
@pytest.mark.parametrize("altura", [9.0, -3.0, 0.0, 100.0])
def test_altura_de_mao_absurda_nao_produz_braco_absurdo(altura):
    """A descrição pede; a anatomia decide.

        O corpo obedece à anatomia antes de obedecer à leitura.

    É a diferença entre construir o corpo e copiar a medida: uma checagem
    depois só descobriria o braço de três metros já desenhado.
    """
    j = montar(estatura=E, braco_dir=Braco.LEVANTADO, altura_mao_dir=altura)
    d = float(np.linalg.norm(j[PULSO_DIR] - j[OMBRO_DIR]))

    assert d <= ALCANCE_MAXIMO * E + 1e-6, f"braco de {d:.2f} m"
    assert j[:, 2].min() >= 0.0
    # Ombro (0,82) + alcance do braco (0,335) = 1,155 da estatura. Acima
    # disso nao existe pessoa, so erro de desenho.
    assert j[:, 2].max() <= 1.16 * E + 1e-6, f"mao a {j[:, 2].max():.2f} m"


@pytest.mark.parametrize("estatura", [-1.0, 0.0, 12.0, None])
def test_estatura_impossivel_e_trazida_para_a_faixa_humana(estatura):
    j = montar(estatura=estatura)
    alto = j[:, 2].max()
    # A junta mais alta do COCO-17 e o olho (0,935 da estatura): nao existe
    # ponto "topo da cabeca" no padrao. Entao a faixa util e 0,935 da faixa
    # humana, e nao a faixa humana.
    assert 0.93 * 0.80 <= alto <= 0.94 * 2.20, f"pessoa de {alto:.2f} m"
    assert j[:, 2].min() >= 0.0


def test_estado_desconhecido_nao_quebra_nada():
    """`desconhecido` é valor de primeira classe no vocabulário."""
    j = montar(estatura=E, postura=Postura.DESCONHECIDA,
               locomocao=Locomocao.DESCONHECIDA,
               braco_esq=Braco.DESCONHECIDO, braco_dir=Braco.DESCONHECIDO)
    assert j[:, 2].min() >= 0.0
    assert np.isfinite(j).all()


def test_rotulo_que_nao_existe_no_vocabulario_vira_repouso():
    """Entrada fora do contrato não pode produzir corpo fora da anatomia."""
    j = montar(estatura=E, postura="voando", braco_dir="quebrado")
    assert j[:, 2].min() >= 0.0
    assert j[:, 2].max() <= E + 1e-6


# ------------------------------------------------------- a descrição é obedecida
def test_agachar_abaixa_o_quadril_e_mantem_os_pes_no_chao():
    em_pe = montar(estatura=E, postura=Postura.EM_PE)
    agachado = montar(estatura=E, postura=Postura.AGACHADO)

    assert altura_do_quadril(agachado) < altura_do_quadril(em_pe) * 0.75
    assert abs(altura_do_pe(agachado) - altura_do_pe(em_pe)) < 0.01, (
        "agachar move o quadril, nao o chao")


def test_agachar_leva_o_joelho_para_a_frente():
    """Perna que encolhe sem o joelho sair do lugar atravessa o proprio corpo."""
    from src.gemeo.boneco import JOELHO_DIR

    em_pe = montar(estatura=E, postura=Postura.EM_PE)
    agachado = montar(estatura=E, postura=Postura.AGACHADO)
    assert agachado[JOELHO_DIR][1] > em_pe[JOELHO_DIR][1] + 0.05


def test_levantar_o_braco_poe_o_pulso_acima_do_ombro():
    j = montar(estatura=E, braco_dir=Braco.LEVANTADO)
    assert j[PULSO_DIR][2] > j[OMBRO_DIR][2]


def test_braco_ao_lado_poe_o_pulso_abaixo_do_ombro():
    j = montar(estatura=E, braco_dir=Braco.AO_LADO)
    assert j[PULSO_DIR][2] < j[OMBRO_DIR][2]


def test_braco_estendido_vai_para_a_FRENTE_do_corpo():
    j = montar(estatura=E, braco_dir=Braco.ESTENDIDO, rumo=0.0)
    assert j[PULSO_DIR][1] > j[OMBRO_DIR][1] + 0.15


def test_o_rumo_gira_o_corpo_inteiro_e_nao_muda_altura():
    """Gravidade não gira. Girar o corpo não pode mexer em altura nenhuma."""
    frente = montar(estatura=E, braco_dir=Braco.ESTENDIDO, rumo=0.0)
    lado = montar(estatura=E, braco_dir=Braco.ESTENDIDO, rumo=np.pi / 2)

    assert np.allclose(frente[:, 2], lado[:, 2], atol=1e-9)

    # O braco continua ESTICADO, so que apontando para outro lado. Testar a
    # distancia em vez do eixo evita amarrar o teste a uma convencao de sinal
    # — e convencao deduzida foi o que causou o erro de 180 graus em 11/08.
    def alcance(j):
        return float(np.linalg.norm(j[PULSO_DIR] - j[OMBRO_DIR]))

    assert abs(alcance(frente) - alcance(lado)) < 1e-9
    assert abs(lado[PULSO_DIR][0] - lado[OMBRO_DIR][0]) > 0.15, (
        "girado 90 graus, o braco estendido tem que sair do eixo y")
    assert abs(lado[PULSO_DIR][1] - lado[OMBRO_DIR][1]) < 0.15


def test_a_posicao_no_chao_desloca_o_corpo_inteiro():
    aqui = montar(estatura=E)
    ali = montar(estatura=E, x=1.5, y=-2.0)

    assert np.allclose(ali[:, 0] - aqui[:, 0], 1.5)
    assert np.allclose(ali[:, 1] - aqui[:, 1], -2.0)
    assert np.allclose(ali[:, 2], aqui[:, 2])


def test_andar_nao_mexe_as_pernas():
    """A PERNA SO AGACHA.

        eu nao quero que as pernas do boneco fiquem se mechendo para andar
                                                        — Eduardo, 12/08

    Este teste afirmava o CONTRARIO ate 12/08: exigia que os tornozelos se
    alternassem durante a caminhada, para o boneco nao deslizar. Era um teste
    correto para uma decisao errada — a cadencia era uma constante inventada
    em `rodar.py`, nao a passada de ninguem.
    """
    andando_a = montar(estatura=E, locomocao=Locomocao.FRENTE, fase=0.25)
    andando_b = montar(estatura=E, locomocao=Locomocao.FRENTE, fase=0.75)
    assert np.allclose(andando_a[TORNOZELO_ESQ], andando_b[TORNOZELO_ESQ])
    assert np.allclose(andando_a[TORNOZELO_DIR], andando_b[TORNOZELO_DIR])

    parado = montar(estatura=E, locomocao=Locomocao.PARADO, fase=0.25)
    assert np.allclose(parado[TORNOZELO_ESQ], andando_a[TORNOZELO_ESQ]), (
        "andar e parar tem que dar o mesmo par de pes")


def test_o_corpo_respira_e_os_pes_ficam_no_chao():
    """Flutuar suave sem descolar do chao: milimetros, e so acima do pe."""
    a = montar(estatura=E, fase=0.0)
    b = montar(estatura=E, fase=1.0)          # meio ciclo de respiro

    assert not np.allclose(a[QUADRIL_ESQ], b[QUADRIL_ESQ])
    assert abs(a[QUADRIL_ESQ][2] - b[QUADRIL_ESQ][2]) < 0.02, "respiro, nao pulo"
    assert np.allclose(a[TORNOZELO_ESQ], b[TORNOZELO_ESQ]), (
        "pe que flutua sai do chao")


def test_a_altura_da_mao_pedida_e_atendida_quando_cabe_no_corpo():
    """Dentro do alcance, a descrição manda. Fora dele, a anatomia manda."""
    j = montar(estatura=E, braco_dir=Braco.ESTENDIDO, altura_mao_dir=1.20)
    assert abs(j[PULSO_DIR][2] - 1.20) < 0.06, j[PULSO_DIR][2]


# --------------------------------------------------- a ponte com a descricao
class AcaoFalsa:
    x, y = 2.0, 3.0
    rumo_corpo = 0.0
    postura = Postura.AGACHADO
    locomocao = Locomocao.PARADO
    braco_esquerdo = Braco.AO_LADO
    braco_direito = Braco.LEVANTADO
    altura_mao_esq = None
    altura_mao_dir = 1.60


def test_de_acao_monta_o_corpo_sem_receber_junta_nenhuma():
    """A `AcaoFalsa` não tem esqueleto: só o vocabulário e a posição.

    É exatamente esse o ponto — se o boneco precisasse de juntas, ele estaria
    copiando medida de novo.
    """
    j = boneco.de_acao(AcaoFalsa(), estatura=E)

    assert j.shape == (17, 3)
    assert j[:, 2].min() >= 0.0
    assert j[PULSO_DIR][2] > j[OMBRO_DIR][2], "braco direito estava levantado"
    assert abs(j[QUADRIL_DIR][0] - 2.0) < 0.2 and abs(j[QUADRIL_DIR][1] - 3.0) < 0.2


def test_de_acao_aceita_descricao_incompleta():
    """Meia descrição é o caso normal: uma câmera cega não pode derrubar o
    desenho."""
    class Vazia:
        pass

    j = boneco.de_acao(Vazia(), estatura=E)
    assert j.shape == (17, 3)
    assert j[:, 2].min() >= 0.0
