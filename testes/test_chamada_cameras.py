"""A falta de uma camera aparece ANTES do teste, nao no boletim.

O CASO, 11/08

O tablet caiu da rede durante a inicializacao. Todas as camadas fizeram o que
deviam: a fonte registrou a falha, agendou nova tentativa e dobrou o intervalo.
E mesmo assim o conferidor chegou ao `ENTER para comecar` sem dizer nada — o
ultimo aviso estava a doze linhas de log de distancia da pergunta.

Um ENTER distraido ali gastaria cinco prateleiras e alguns minutos de
agachamento para produzir um boletim de DUAS cameras, e a falta so seria
contada no fim.

    A hora de descobrir que falta uma camera e antes de a pessoa agachar
    cinco vezes, nao depois.

Rodar degradado continua permitido — mede o que duas cameras medem, e as vezes
e isso mesmo que se quer saber. O que nao pode e acontecer sozinho.

    Degradar em silencio produz um numero que ninguem sabe interpretar.
    Degradar com consentimento produz uma medicao com escopo declarado.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from ferramentas.conferir_altura import _mostrar_chamada    # noqa: E402

TODAS_DE_PE = {"alto": "online", "frontal": "online", "lateral": "online"}


def test_com_todas_de_pe_nao_ha_o_que_confirmar():
    assert _mostrar_chamada(dict(TODAS_DE_PE)) == []


@pytest.mark.parametrize("papel", ["alto", "frontal", "lateral"])
@pytest.mark.parametrize("estado", ["falha", "conectando", "offline"])
def test_qualquer_estado_que_nao_seja_online_e_ausencia(papel, estado):
    """`conectando` conta como fora. Ela ainda pode voltar — e pode nao voltar.

    Tratar "esta tentando" como presenca seria decidir pelo otimismo justamente
    no momento em que a decisao e da pessoa.
    """
    estados = dict(TODAS_DE_PE, **{papel: estado})
    assert _mostrar_chamada(estados) == [papel]


def test_varias_fora_saem_todas(capsys):
    estados = dict(TODAS_DE_PE, frontal="falha", lateral="falha")
    assert set(_mostrar_chamada(estados)) == {"frontal", "lateral"}


def test_a_chamada_diz_o_que_cada_ausencia_custa(capsys):
    """Aviso generico nao ajuda a decidir; saber QUAL pergunta fica sem
    resposta, sim."""
    _mostrar_chamada(dict(TODAS_DE_PE, lateral="falha"))
    saida = capsys.readouterr().out

    assert "lateral" in saida
    assert "NAO VAI RESPONDER NADA" in saida
    assert "complementaridade" in saida.lower()


def test_sem_o_alto_o_aviso_fala_de_escala(capsys):
    """Perder o alto e o caso mais caro: a altura da mao volta a ser estimada."""
    _mostrar_chamada(dict(TODAS_DE_PE, alto="falha"))
    saida = capsys.readouterr().out
    assert "estimada" in saida.lower()


def test_camera_online_nao_recebe_marca_de_ausente(capsys):
    _mostrar_chamada(dict(TODAS_DE_PE, lateral="falha"))
    linhas = capsys.readouterr().out.splitlines()

    for linha in linhas:
        if linha.strip().startswith("frontal"):
            assert "NAO VAI RESPONDER" not in linha
