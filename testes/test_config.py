"""Calibracao gravada tem que chegar viva ate o motor.

O DEFEITO QUE CUSTOU UM DIA, 11/08

A escala vertical foi calibrada com 148 amostras e 5% de dispersao, e gravada
em `config/escala.json`. Depois disso o sistema rodou a tarde inteira dizendo
`escala NAO CALIBRADA`, e a altura da mao saiu estimada pelo tronco (+-8 cm)
quando havia uma medicao de +-3 cm disponivel no disco.

O motivo cabia numa linha:

    return float(d.get("fator", d["altura_camera_m"]))

Python avalia o argumento default ANTES de chamar `.get`. O arquivo novo nao
tem a chave antiga, entao `d["altura_camera_m"]` levantava `KeyError` antes do
`.get` rodar — e um `except Exception: return None` engolia.

    O codigo escrito para dar COMPATIBILIDADE com o formato antigo quebrava
    exatamente o formato novo.

E o `except` mudo transformou um defeito num comportamento: `None` significava
ao mesmo tempo "nunca calibrei" e "calibrei e voce perdeu", e as duas frases
saiam com a mesma cara no painel.

    Um bloco except que nao registra nada nao trata o erro. Apaga o erro.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.espacial import motor                          # noqa: E402


@pytest.fixture(autouse=True)
def log_visivel(monkeypatch):
    """Devolve a propagacao do logger `so` para o `caplog` enxergar.

    ISTO NAO E ENFEITE DE TESTE — E CONTAMINACAO ENTRE TESTES, E ELA APARECEU AQUI

    `log.configurar()` faz `logging.getLogger("so").propagate = False`, o que
    esta certo em producao: as mensagens vao para os handlers do projeto e nao
    duplicam na raiz.

    Mas `logging` e estado GLOBAL do processo. Qualquer teste que chame
    `configurar()` antes deste arquivo desliga a propagacao para todos os
    seguintes — e o `caplog` do pytest escuta na raiz.

    O sintoma foi exato e vale registrar: os quatro testes de aviso passavam
    sozinhos e falhavam na suite inteira, com `caplog.text` vazio. Nada no
    codigo testado tinha mudado.

        Teste que passa sozinho e falha acompanhado nao e teste instavel.
        E estado global sendo compartilhado sem ninguem ter pedido.

    `monkeypatch` devolve os dois atributos no fim, entao a ordem dos arquivos
    deixa de importar nos dois sentidos.
    """
    import logging

    so = logging.getLogger("so")
    monkeypatch.setattr(so, "propagate", True)
    monkeypatch.setattr(so, "level", logging.NOTSET)


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Aponta RAIZ para um diretorio temporario e devolve `escreve(nome, d)`."""
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(motor, "RAIZ", tmp_path)

    def escreve(nome, conteudo):
        alvo = tmp_path / "config" / nome
        if isinstance(conteudo, str):
            alvo.write_text(conteudo, encoding="utf-8")
        else:
            alvo.write_text(json.dumps(conteudo), encoding="utf-8")
        return alvo

    return escreve


# ------------------------------------------------------ o defeito, ao contrario
def test_escala_com_a_chave_nova_e_lida(config):
    """ESTE E O TESTE QUE FALTAVA. Arquivo real de 11/08, so a chave nova."""
    config("escala.json", {
        "fator": 5.2475,
        "estatura_de_referencia_m": 1.8,
        "razao_mediana": 0.34302,
        "amostras": 148,
        "dispersao": 0.054,
    })
    assert motor._fator_de_escala() == pytest.approx(5.2475)


def test_escala_com_a_chave_antiga_continua_sendo_lida(config):
    """A compatibilidade e real e precisa continuar valendo — sem sabotar."""
    config("escala.json", {"altura_camera_m": 4.10})
    assert motor._fator_de_escala() == pytest.approx(4.10)


def test_a_chave_nova_ganha_da_antiga(config):
    config("escala.json", {"fator": 5.25, "altura_camera_m": 4.10})
    assert motor._fator_de_escala() == pytest.approx(5.25)


# ------------------------------------------------------------- ausente e calado
def test_arquivo_ausente_e_none_sem_reclamar(config, caplog):
    """Nunca calibrar e um estado legitimo: o sistema roda estimando.

    Reclamar aqui treinaria quem le o log a ignorar o aviso — e ai o aviso
    que importa passa despercebido junto.
    """
    assert motor._fator_de_escala() is None
    assert motor._sinal_do_rumo_fixado() is None
    assert motor._azimute_calibrado() is None
    assert "config" not in caplog.text.lower() or not caplog.records


# ----------------------------------------------------------- quebrado e barulho
@pytest.mark.parametrize("conteudo", [
    "{isto nao e json",              # arquivo corrompido
    {"chave_errada": 1},             # json valido, chave que nao existe
    {"fator": "muito alto"},         # tipo impossivel de converter
])
def test_arquivo_quebrado_nao_derruba_e_deixa_rastro(config, caplog, conteudo):
    """Seguir sem a config e certo. Seguir CALADO foi o que custou o dia."""
    config("escala.json", conteudo)

    with caplog.at_level("WARNING"):
        assert motor._fator_de_escala() is None

    assert caplog.text, "config quebrada tem que aparecer no log"
    assert "escala.json" in caplog.text


def test_rumo_e_azimute_usam_o_mesmo_caminho(config, caplog):
    """As tres leitoras tinham o mesmo defeito de forma. Agora tem uma so forma."""
    config("rumo.json", {"sinal": -1})
    config("azimute.json", {"azimute_rad": 1.25})

    assert motor._sinal_do_rumo_fixado() == -1
    assert motor._azimute_calibrado() == pytest.approx(1.25)

    config("rumo.json", {"sinal_errado": -1})
    with caplog.at_level("WARNING"):
        assert motor._sinal_do_rumo_fixado() is None
    assert "rumo.json" in caplog.text


def test_motor_nao_e_o_unico_a_precisar_disso(config):
    """O leitor generico e publico o bastante para ser testado sozinho.

    Se amanha aparecer `config/prateleiras.json`, ele nasce ja com o aviso —
    e nao repete o `except` mudo por copia e cola, que foi como o defeito
    chegou a existir em tres lugares ao mesmo tempo.
    """
    config("qualquer.json", {"valor": 7})
    assert motor._ler_config("qualquer.json", lambda d: d["valor"]) == 7
    assert motor._ler_config("nao_existe.json", lambda d: d["valor"]) is None
