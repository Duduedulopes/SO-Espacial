"""Sem braco medido nao ha palpite de prateleira.

O DEFEITO, COMO ELE APARECEU NA TELA DE 12/08

    bracos   esq ? dir ?      sem fonte      e mesmo assim:  P4

Os dois bracos em DESCONHECIDO, nenhuma camera reivindicando a leitura, e o
classificador opinando "prateleira 4" com a mesma confianca de sempre.

A causa era um valor padrao. `lado_que_alcanca` devolvia "direita" quando nao
media nada — o lado em que as assinaturas tinham sido aprendidas. Parece um
padrao razoavel, e e uma invencao: ela afirmava que o braco direito estava
alcancando, e o `_palpitar` acreditava.

    Um valor padrao que substitui a ausencia de medida transforma "nao sei"
    em "sei", e e a mentira mais barata que um programa consegue contar.

O que estes testes fixam e a fronteira: meia evidencia continua valendo, e
nenhuma evidencia passa a nao votar.
"""

from src.acao.corpo import LeituraDoCorpo
from src.acao.prateleira import lado_que_alcanca
from src.espacial.motor import SpatialEngine


def _leitura(**kw):
    return LeituraDoCorpo(**kw)


# ------------------------------------------------- qual braco esta agindo
def test_dois_bracos_mudos_nao_escolhem_lado():
    """O conserto. Antes isto devolvia 'direita' e o quadro entrava na votacao."""
    assert lado_que_alcanca(_leitura()) is None


def test_um_braco_so_continua_valendo():
    """Meia evidencia e evidencia. Recusar aqui seria o exagero oposto."""
    assert lado_que_alcanca(_leitura(alcance_dir=0.30)) == "direita"
    assert lado_que_alcanca(_leitura(alcance_esq=0.30)) == "esquerda"


def test_ganha_o_braco_que_mais_se_afastou():
    assert lado_que_alcanca(
        _leitura(alcance_esq=0.40, alcance_dir=0.05)) == "esquerda"
    assert lado_que_alcanca(
        _leitura(alcance_esq=0.05, alcance_dir=0.40)) == "direita"


def test_a_medida_2d_tem_prioridade_sobre_a_3d():
    """Regra que ja existia; entra aqui para nao se perder no conserto."""
    assert lado_que_alcanca(
        _leitura(alcance_esq=0.90, alcance_2d_esq=0.01,
                 alcance_dir=0.02, alcance_2d_dir=0.50)) == "direita"


# ------------------------------------------------- o motor recusa o quadro
class _Classificador:
    pronto = True

    def __init__(self):
        self.quadros = []

    def observar(self, pessoa_id, evidencia):
        self.quadros.append((pessoa_id, evidencia))


class _MotorDeMentira:
    """So o que `_palpitar` toca. Montar o motor inteiro traria camera junto."""

    def __init__(self):
        self.prateleiras = _Classificador()
        self._encolhimento = {}
        self._quadril_na_caixa = {}


def _palpitar(leitura):
    m = _MotorDeMentira()
    SpatialEngine._palpitar(m, 1, leitura)
    return m


def test_quadro_sem_braco_nenhum_nao_vota():
    m = _palpitar(_leitura())
    assert m.prateleiras.quadros == []
    assert m._palpites_sem_braco == 1


def test_quadro_com_braco_mas_sem_camera_nao_vota():
    """`fonte_braco_*` vazio e a coluna 'sem fonte' da tela de 12/08.

    O alcance pode ter vindo de uma reconstrucao antiga; se nenhuma vista
    reivindica a leitura DESTE quadro, ela nao e evidencia deste quadro.
    """
    m = _palpitar(_leitura(alcance_dir=0.35, fonte_braco_dir=""))
    assert m.prateleiras.quadros == []
    assert m._palpites_sem_fonte == 1


def test_quadro_com_braco_e_camera_vota():
    """A prova de que o conserto nao emudeceu o palpite inteiro."""
    m = _palpitar(_leitura(alcance_dir=0.35, fonte_braco_dir="frontal"))
    assert len(m.prateleiras.quadros) == 1
    assert m.prateleiras.quadros[0][0] == 1


def test_a_fonte_conferida_e_a_do_lado_escolhido():
    """Braco esquerdo alcancando com fonte, direito sem: tem que votar."""
    m = _palpitar(_leitura(alcance_esq=0.40, fonte_braco_esq="lateral",
                           alcance_dir=0.02, fonte_braco_dir=""))
    assert len(m.prateleiras.quadros) == 1


def test_fonte_do_outro_lado_nao_serve():
    """Direito alcancando sem fonte, esquerdo parado com fonte: recusa."""
    m = _palpitar(_leitura(alcance_dir=0.40, fonte_braco_dir="",
                           alcance_esq=0.02, fonte_braco_esq="frontal"))
    assert m.prateleiras.quadros == []


def test_classificador_nao_pronto_continua_nao_fazendo_nada():
    m = _MotorDeMentira()
    m.prateleiras.pronto = False
    SpatialEngine._palpitar(m, 1, _leitura(alcance_dir=0.35,
                                           fonte_braco_dir="frontal"))
    assert m.prateleiras.quadros == []
