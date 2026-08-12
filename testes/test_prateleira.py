"""Cinco opcoes, evidencia fraca de tres cameras, e sempre uma resposta.

    o cliente so tem essas opcao, pegar algo da prateleira 1,2,3,4,5...
    NAO EXISTE OUTRA OPCAO                                — Eduardo, 12/08

O caminho da regua — medir a altura da mao em metros e comparar com faixas —
falhou tres vezes por tres motivos diferentes, sempre produzindo um numero
plausivel e errado. Classificar entre cinco opcoes aguenta o erro que medir
nao aguenta.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acao.prateleira import (                              # noqa: E402
    Assinatura, ClassificadorDePrateleira, Evidencia, evidencia_de,
)


def assinaturas_da_estante():
    """As cinco, com assinaturas plausiveis para alguem de 1,80 m.

    Os numeros vem da geometria do corpo, nao de medicao: pulso na altura do
    ombro e alcance 1.0, na altura do quadril e 0.0. Servem para exercitar o
    metodo; os reais saem do gabarito.
    """
    return [
        Assinatura("p1", "chao", 0.15, alcance=(-0.6, 0.35), coxa=(0.35, 0.20),
                   bracos={"ao_lado": 0.9}, visto_frontal=0.2, visto_lateral=0.7),
        Assinatura("p2", "joelho", 0.55, alcance=(-0.3, 0.30), coxa=(0.60, 0.20),
                   bracos={"ao_lado": 0.8}, visto_frontal=0.4, visto_lateral=0.8),
        Assinatura("p3", "quadril", 0.95, alcance=(0.05, 0.25), coxa=(0.90, 0.15),
                   bracos={"estendido": 0.7}, visto_frontal=0.9, visto_lateral=0.8),
        Assinatura("p4", "peito", 1.35, alcance=(0.65, 0.25), coxa=(1.00, 0.12),
                   bracos={"estendido": 0.9}, visto_frontal=0.9, visto_lateral=0.7),
        Assinatura("p5", "topo", 1.90, alcance=(1.35, 0.30), coxa=(1.00, 0.12),
                   bracos={"levantado": 0.95}, visto_frontal=0.5, visto_lateral=0.6),
    ]


def classificador(janela=20):
    return ClassificadorDePrateleira(assinaturas_da_estante(), janela=janela)


def alimentar(c, ev, n=10, pid=1):
    for _ in range(n):
        p = c.observar(pid, ev)
    return p


# ------------------------------------------------------- as cinco se separam
@pytest.mark.parametrize("esperada,ev", [
    ("p1", Evidencia(alcance=-0.60, coxa=0.35, braco="ao_lado",
                     viu_frontal=False, viu_lateral=True)),
    ("p2", Evidencia(alcance=-0.30, coxa=0.60, braco="ao_lado",
                     viu_frontal=False, viu_lateral=True)),
    ("p3", Evidencia(alcance=0.05, coxa=0.90, braco="estendido",
                     viu_frontal=True, viu_lateral=True)),
    ("p4", Evidencia(alcance=0.65, coxa=1.00, braco="estendido",
                     viu_frontal=True, viu_lateral=True)),
    ("p5", Evidencia(alcance=1.35, coxa=1.00, braco="levantado",
                     viu_frontal=True, viu_lateral=False)),
])
def test_cada_prateleira_e_reconhecida(esperada, ev):
    p = alimentar(classificador(), ev)
    assert p.prateleira == esperada, p.ranking


# ------------------------------------------- camera cega nao derruba ninguem
def test_uma_camera_so_ainda_responde():
    """O criterio de projeto: nenhuma precisa enxergar tudo.

    So o alcance, sem coxa, sem braco, sem quem viu. Um campo `None` nao pesa
    contra ninguem — ele nao vota.
    """
    p = alimentar(classificador(), Evidencia(alcance=1.35))
    assert p.prateleira == "p5"


def test_so_a_camera_do_alto_ainda_responde():
    """Sem pulso nenhum: so o encolhimento da caixa vista de cima.

    Nao separa as cinco — mas separa agachado de em pe, e isso ja e
    informacao. O sistema responde com confianca baixa em vez de calar.
    """
    c = ClassificadorDePrateleira([
        Assinatura("p1", encolhimento=(0.55, 0.15)),
        Assinatura("p5", encolhimento=(1.00, 0.10)),
    ])
    assert alimentar(c, Evidencia(encolhimento=0.55)).prateleira == "p1"
    assert alimentar(c, Evidencia(encolhimento=1.00), pid=2).prateleira == "p5"


def test_o_que_NAO_foi_visto_tambem_vota():
    """Perder o pulso e evidencia, nao falha.

    Duas assinaturas identicas em tudo, menos em qual camera costuma enxergar.
    So esse sinal tem que decidir — e ate 12/08 ele era jogado fora.
    """
    c = ClassificadorDePrateleira([
        Assinatura("alta", alcance=(1.0, 0.5), visto_frontal=0.1),
        Assinatura("media", alcance=(1.0, 0.5), visto_frontal=0.9),
    ])
    assert alimentar(c, Evidencia(alcance=1.0, viu_frontal=False)).prateleira == "alta"
    assert alimentar(c, Evidencia(alcance=1.0, viu_frontal=True), pid=2).prateleira == "media"


# ------------------------------------------------------ nunca fica sem dizer
def test_evidencia_fora_de_qualquer_assinatura_ainda_escolhe():
    """A regra de 09/08: se nada corresponde, escolher a mais provavel.

    Um alcance de 2.5 nao bate com prateleira nenhuma. O sistema tem que
    apontar a mais alta — e avisar que nao esta firme.
    """
    p = alimentar(classificador(), Evidencia(alcance=2.5, coxa=1.0,
                                             braco="levantado"))
    assert p.prateleira == "p5"
    assert not p.firme, "evidencia fora de faixa nao pode sair como certeza"


def test_evidencia_ambigua_sai_marcada_como_nao_firme():
    """Entre a 3 e a 4, ninguem deve dizer 'certo'."""
    p = alimentar(classificador(), Evidencia(alcance=0.35, coxa=0.95,
                                             braco="estendido"))
    assert p.prateleira in ("p3", "p4")
    assert not p.firme, f"margem {p.margem:.3f} deveria ser apertada"


def test_evidencia_limpa_sai_firme():
    p = alimentar(classificador(), Evidencia(alcance=1.35, coxa=1.00,
                                             braco="levantado",
                                             viu_frontal=True,
                                             viu_lateral=False))
    assert p.firme, f"conf {p.confianca:.2f} margem {p.margem:.2f}"


def test_sem_evidencia_nenhuma_nao_ha_palpite():
    """Diferente de 'nao sei': e 'ninguem alcancou nada ainda'."""
    c = classificador()
    assert c.observar(1, Evidencia()) is None
    assert c.observar(1, None) is None


# ------------------------------------------------------ decide por GESTO
def test_um_quadro_ruim_nao_derruba_o_gesto():
    """Decidir por quadro produz um sistema que gagueja."""
    c = classificador()
    bom = Evidencia(alcance=1.35, coxa=1.0, braco="levantado")
    ruim = Evidencia(alcance=-0.6, coxa=0.35, braco="ao_lado")

    for _ in range(9):
        c.observar(1, bom)
    p = c.observar(1, ruim)          # um quadro discordante no meio

    assert p.prateleira == "p5", p.ranking


def test_a_janela_deixa_a_prateleira_mudar_quando_o_gesto_muda():
    """Compromisso temporal nao pode virar teimosia."""
    c = classificador(janela=6)
    for _ in range(6):
        c.observar(1, Evidencia(alcance=1.35, coxa=1.0, braco="levantado"))
    assert c.palpite(1).prateleira == "p5"

    for _ in range(6):
        p = c.observar(1, Evidencia(alcance=-0.6, coxa=0.35, braco="ao_lado"))
    assert p.prateleira == "p1", p.ranking


def test_reiniciar_limpa_o_gesto_anterior():
    """Quem pega da 5 e depois da 1 nao pode ter a segunda contaminada."""
    c = classificador()
    alimentar(c, Evidencia(alcance=1.35, coxa=1.0, braco="levantado"))
    c.reiniciar(1)
    assert c.palpite(1) is None


def test_esquecer_some_com_quem_saiu():
    c = classificador()
    alimentar(c, Evidencia(alcance=1.35))
    c.esquecer(vivos={2})
    assert c.palpite(1) is None


# ------------------------------------------------ a ponte com a leitura real
class LeituraFalsa:
    fonte_braco_dir = "lateral"
    fonte_braco_esq = ""
    braco_direito = "levantado"
    braco_esquerdo = "desconhecido"
    alcance_dir = 1.30
    alcance_esq = None
    verticalidade_coxa = 0.98


def test_traduz_a_leitura_combinada_em_evidencia():
    ev = evidencia_de(LeituraFalsa(), lado="direita", encolhimento=0.99)

    assert ev.alcance == 1.30
    assert ev.braco == "levantado"
    assert ev.viu_lateral is True and ev.viu_frontal is False
    assert ev.encolhimento == 0.99


def test_braco_desconhecido_nao_vota():
    """`desconhecido` nao e um rotulo do vocabulario: e ausencia de leitura."""
    ev = evidencia_de(LeituraFalsa(), lado="esquerda")
    assert ev.braco is None
    assert ev.viu_frontal is None and ev.viu_lateral is None
