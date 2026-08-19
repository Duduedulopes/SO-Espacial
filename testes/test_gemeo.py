"""
Testes do EventEngine e do DigitalTwin — sem camera, sem modelo, sem janela.

O QUE ESTES TESTES DEFENDEM

O gemeo e o dono unico da verdade. Se ele emitir um evento a mais por quadro,
o painel vira ruido; se emitir a menos, uma entrada em zona se perde. Se o mapa
de calor acumular posicao PREVISTA, o sistema passa a registrar permanencia num
lugar onde ninguem foi visto — que e exatamente o tipo de mentira silenciosa
que corrompe um relatorio inteiro sem dar erro nenhum.

Nada aqui depende de hardware. Todos os estados sao construidos a mao.

    python testes/test_gemeo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estado.ocupacao import Zona                       # noqa: E402
from estado.planta import Planta                       # noqa: E402
from src.espacial.estado import EstadoDePessoa         # noqa: E402
from src.eventos.motor import EventEngine, Tipo        # noqa: E402
from src.gemeo.gemeo import DigitalTwin                # noqa: E402
from src.nucleo import log as logmod                   # noqa: E402

logmod.configurar("ERROR")


# ---------------------------------------------------------------- cenario
def planta_de_teste():
    """Chao de 4x3 m com uma zona no canto. Sem arquivo: o teste nao pode
    quebrar porque alguem editou loja/bancada.json."""
    esquerda = Zona("Esquerda", 0.0, 1.0, 0.0, 1.0)
    esquerda.id = "esquerda"
    direita = Zona("Direita", 3.0, 4.0, 0.0, 1.0)
    direita.id = "direita"
    return Planta(id="teste", nome="Sala de teste",
                  chao=(0.0, 4.0, 0.0, 3.0), moveis=[],
                  zonas=[esquerda, direita])


def gemeo():
    ev = EventEngine()
    return DigitalTwin(planta_de_teste(), ev, meia_vida_calor=1e6), ev


def p(pid, x, y, prevendo=0):
    return EstadoDePessoa(id=pid, x=x, y=y, prevendo=prevendo)


# ---------------------------------------------------------------- eventos
def test_evento_guarda_historico_e_contagem():
    ev = EventEngine()
    ev.emitir(Tipo.TRACK_STARTED, {"pessoa": 1})
    ev.emitir(Tipo.TRACK_STARTED, {"pessoa": 2})
    ev.emitir(Tipo.TRACK_LOST, {"pessoa": 1})

    assert len(ev.historico) == 3
    assert ev.resumo()[Tipo.TRACK_STARTED] == 2
    assert ev.resumo()[Tipo.TRACK_LOST] == 1
    assert list(ev.resumo())[0] == Tipo.TRACK_STARTED, "resumo deve vir ordenado"


def test_assinante_recebe_do_seu_tipo_e_do_curinga():
    ev = EventEngine()
    so_perdas, tudo = [], []
    ev.assinar(Tipo.TRACK_LOST, so_perdas.append)
    ev.assinar("*", tudo.append)

    ev.emitir(Tipo.TRACK_STARTED, {"pessoa": 1})
    ev.emitir(Tipo.TRACK_LOST, {"pessoa": 1})

    assert len(so_perdas) == 1, "assinante de tipo recebeu o que nao pediu"
    assert len(tudo) == 2, "curinga tem que ver tudo"


def test_assinante_quebrado_nao_derruba_quem_emitiu():
    """Regra aprendida em 08/08: canal lateral nao para o nucleo."""
    ev = EventEngine()
    depois = []

    def explode(_e):
        raise RuntimeError("painel morreu")

    ev.assinar("*", explode)
    ev.assinar("*", depois.append)

    e = ev.emitir(Tipo.SYSTEM_STARTED, {})           # nao pode levantar
    assert e.tipo == Tipo.SYSTEM_STARTED
    assert len(depois) == 1, "assinante seguinte foi pulado pelo que quebrou"


def test_memoria_limitada_nao_cresce_para_sempre():
    ev = EventEngine(memoria=3)
    for i in range(10):
        ev.emitir(Tipo.OBJECT_DETECTED, {"n": i})

    assert len(ev.historico) == 3, "historico tem que ter teto"
    assert ev.ultimos(3)[-1].dados["n"] == 9, "guardou os antigos, nao os novos"
    assert ev.contagem[Tipo.OBJECT_DETECTED] == 10, "contagem nao pode esquecer"


def test_filtro_por_tipo_nos_ultimos():
    ev = EventEngine()
    ev.emitir(Tipo.TRACK_STARTED, {"pessoa": 1})
    ev.emitir(Tipo.CAMERA_DEGRADED, {"camera": "alto"})
    ev.emitir(Tipo.TRACK_STARTED, {"pessoa": 2})

    inicios = ev.ultimos(10, tipo=Tipo.TRACK_STARTED)
    assert len(inicios) == 2
    assert all(e.tipo == Tipo.TRACK_STARTED for e in inicios)


# ---------------------------------------------------------------- rastros
def test_inicio_e_perda_de_rastro_saem_uma_vez_so():
    g, ev = gemeo()

    for _ in range(5):
        g.atualizar([p(1, 2.0, 2.0)], {}, 0.1)       # 5 quadros, 1 evento
    for _ in range(3):
        g.atualizar([], {}, 0.1)                     # sumiu, 1 evento

    assert ev.contagem.get(Tipo.TRACK_STARTED) == 1, "repetiu o inicio por quadro"
    assert ev.contagem.get(Tipo.TRACK_LOST) == 1, "repetiu a perda por quadro"


def test_pessoa_que_volta_com_id_novo_gera_rastro_novo():
    g, ev = gemeo()
    g.atualizar([p(1, 2.0, 2.0)], {}, 0.1)
    g.atualizar([], {}, 0.1)
    g.atualizar([p(2, 2.0, 2.0)], {}, 0.1)

    assert ev.contagem[Tipo.TRACK_STARTED] == 2
    assert ev.contagem[Tipo.TRACK_LOST] == 1


# ------------------------------------------------------------------ zonas
def test_entrada_e_saida_de_zona_uma_vez_por_travessia():
    g, ev = gemeo()

    for _ in range(4):
        g.atualizar([p(1, 2.0, 2.0)], {}, 0.1)       # meio do chao, sem zona
    for _ in range(6):
        g.atualizar([p(1, 0.5, 0.5)], {}, 0.1)       # dentro da esquerda
    for _ in range(4):
        g.atualizar([p(1, 2.0, 2.0)], {}, 0.1)       # saiu

    assert ev.contagem.get(Tipo.PERSON_ENTERED_ZONE) == 1
    assert ev.contagem.get(Tipo.PERSON_LEFT_ZONE) == 1

    entrada = ev.ultimos(20, tipo=Tipo.PERSON_ENTERED_ZONE)[0]
    assert entrada.dados["zona"] == "esquerda", "evento deve usar o id da zona"


def test_zona_acumula_tempo_por_rastro_e_nao_por_quadro():
    g, _ = gemeo()
    for _ in range(10):
        g.atualizar([p(1, 0.5, 0.5)], {}, 0.1)       # 10 quadros x 0,1 s = 1 s

    esquerda = g.zonas[0]
    assert esquerda.visitas == 1
    assert abs(esquerda.tempo_total - 1.0) < 1e-6, f"{esquerda.tempo_total}"
    assert esquerda.ocupacao == 1


def test_rastro_perdido_dentro_da_zona_nao_deixa_fantasma():
    """Se o esquecimento falhar, a pessoa reaparece 'saindo' de uma zona onde
    nunca mais esteve — e a contagem de saidas passa a inventar.

    Os cinco quadros de permanencia sao por causa da confirmacao de 0,4 s que
    a zona passou a exigir em 19/08 (ver `test_zona_fronteira.py`). Um quadro
    solto ja nao conta como entrar, e nao deve mesmo: era isso que fazia uma
    pessoa sentada gerar quinze visitas.
    """
    g, ev = gemeo()
    for _ in range(5):
        g.atualizar([p(1, 0.5, 0.5)], {}, 0.1)
    g.atualizar([], {}, 0.1)                          # perdeu dentro da zona

    assert 1 not in g._zonas_anteriores, "memoria de zona nao foi limpa"
    assert ev.contagem.get(Tipo.PERSON_LEFT_ZONE) is None

    for _ in range(5):
        g.atualizar([p(1, 3.5, 0.5)], {}, 0.1)        # volta na outra ponta
    entradas = ev.ultimos(20, tipo=Tipo.PERSON_ENTERED_ZONE)
    assert [e.dados["zona"] for e in entradas] == ["esquerda", "direita"]


def test_duas_pessoas_em_zonas_diferentes_nao_se_confundem():
    g, ev = gemeo()
    for _ in range(5):
        g.atualizar([p(1, 0.5, 0.5), p(2, 3.5, 0.5)], {}, 0.1)

    zonas = {e.dados["pessoa"]: e.dados["zona"]
             for e in ev.ultimos(20, tipo=Tipo.PERSON_ENTERED_ZONE)}
    assert zonas == {1: "esquerda", 2: "direita"}
    assert g.zonas[0].ocupacao == 1 and g.zonas[1].ocupacao == 1


# ------------------------------------------------------------------- calor
def test_calor_so_acumula_o_que_foi_medido():
    """Posicao prevista pelo Kalman e estimativa, nao presenca observada."""
    medido, _ = gemeo()
    for _ in range(10):
        medido.atualizar([p(1, 2.0, 1.5)], {}, 0.1)

    previsto, _ = gemeo()
    for _ in range(10):
        previsto.atualizar([p(1, 2.0, 1.5, prevendo=3)], {}, 0.1)

    assert medido.calor.grade.sum() > 0.9, "quadro medido nao entrou no calor"
    assert previsto.calor.grade.sum() == 0.0, "posicao prevista virou permanencia"


def test_calor_ignora_posicao_fora_do_chao():
    g, _ = gemeo()
    g.atualizar([p(1, 99.0, 99.0)], {}, 0.1)
    assert g.calor.grade.sum() == 0.0


# -------------------------------------------------------------- publicacao
def test_instantaneo_e_serializavel_e_nao_carrega_imagem():
    g, _ = gemeo()
    import numpy as np
    pessoa = p(1, 1.2, 0.8)
    pessoa.esqueleto = np.zeros((17, 3))               # array nao serializa
    g.atualizar([pessoa], {"alto": {"fps": 12.0}}, 0.1)

    texto = json.dumps(g.instantaneo(), ensure_ascii=False)
    d = json.loads(texto)

    assert d["loja"]["id"] == "teste"
    assert d["pessoas"][0]["tem_esqueleto"] is True, "sinaliza sem carregar"
    assert "esqueleto" not in d["pessoas"][0], "o gemeo e estado, nao desenho"
    assert d["cameras"]["alto"]["fps"] == 12.0
    assert len(d["zonas"]) == 2


def test_gemeo_sem_motor_de_eventos_continua_funcionando():
    """Um teste, um script de analise ou uma reproducao de gravacao podem
    querer o gemeo sem o barramento. Nao pode explodir."""
    g = DigitalTwin(planta_de_teste(), eventos=None)
    g.atualizar([p(1, 0.5, 0.5)], {}, 0.1)
    g.atualizar([], {}, 0.1)
    assert g.resumo()["pessoas"] == 0
    assert g.quadros == 2


def test_resumo_conta_zonas_ocupadas():
    g, _ = gemeo()
    for _ in range(5):                    # a zona confirma em 0,4 s
        g.atualizar([p(1, 0.5, 0.5)], {}, 0.1)
    assert g.resumo() == {"pessoas": 1, "quadros": 5, "zonas_ocupadas": 1}


def test_trilha_guarda_o_percurso_e_esquece_quem_saiu():
    """Calor responde onde as pessoas FICAM, em agregado. Trilha responde por
    onde ESTA pessoa veio. Sao perguntas diferentes."""
    g, _ = gemeo()
    for i in range(5):
        g.atualizar([p(1, 0.5 + i * 0.2, 1.0)], {}, 0.1)

    assert list(g.trilhas[1])[0] == (0.5, 1.0)
    assert len(g.trilhas[1]) == 5

    g.atualizar([], {}, 0.1)
    assert 1 not in g.trilhas, "rastro perdido nao pode deixar trilha para tras"


def test_trilha_tem_teto():
    """Sem teto, uma sessao longa acumula memoria sem limite."""
    g = DigitalTwin(planta_de_teste(), eventos=None, memoria_trilha=10)
    for i in range(50):
        g.atualizar([p(1, 1.0 + i * 0.01, 1.0)], {}, 0.05)
    assert len(g.trilhas[1]) == 10


# ---------------------------------------------------------------- execucao
if __name__ == "__main__":
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    falhas = 0
    for t in testes:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            falhas += 1
            print(f"  FALHA {t.__name__}: {e}")
        except Exception as e:
            falhas += 1
            print(f"  ERRO  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(testes) - falhas}/{len(testes)} passaram")
    sys.exit(1 if falhas else 0)
