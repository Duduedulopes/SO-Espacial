"""
Testes da camada de acao — trajetorias sinteticas, resultado conhecido.

POR QUE SEM CAMERA

A etapa A foi desenhada para ser provada sem hardware: locomocao e postura
saem de numeros que o sistema ja calcula. Aqui as trajetorias sao escritas a
mao, entao "andou para frente" nao e impressao — e o que a trajetoria diz.

    python testes/test_acao.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acao.classificador import ClassificadorDeAcao, Descritor  # noqa: E402
from src.acao.corpo import LeituraDoCorpo                          # noqa: E402
from src.acao.vocabulario import (                                 # noqa: E402
    Acao, Braco, Estavel, Locomocao, Postura,
)
from src.espacial.estado import EstadoDePessoa                     # noqa: E402


def pessoa(pid=1, x=0.0, y=0.0, vx=0.0, vy=0.0, rumo=0.0, prevendo=0):
    return EstadoDePessoa(id=pid, x=x, y=y, vx=vx, vy=vy, rumo=rumo,
                          prevendo=prevendo)


def girar(c, passos, graus_por_passo, dt=0.1, v=0.6):
    """Anda girando. Devolve a ultima acao."""
    rumo = 0.0
    acao = None
    for _ in range(passos):
        rumo += math.radians(graus_por_passo)
        acao, _ = c.classificar(
            pessoa(vx=v * math.cos(rumo), vy=v * math.sin(rumo), rumo=rumo), dt)
    return acao


# ------------------------------------------------------------- vocabulario
def test_estavel_so_muda_depois_de_repetir():
    """Ruido nao se repete; mudanca de verdade sim."""
    e = Estavel(Locomocao.PARADO, minimo_s=0.3)

    assert not e.propor(Locomocao.ANDANDO, 0.1)
    assert not e.propor(Locomocao.ANDANDO, 0.1)
    assert e.propor(Locomocao.ANDANDO, 0.1), "0,3 s deviam ter mudado"
    assert e.valor == Locomocao.ANDANDO

    # um quadro discordante no meio zera o acumulo
    e2 = Estavel(Locomocao.PARADO, minimo_s=0.3)
    e2.propor(Locomocao.ANDANDO, 0.1)
    e2.propor(Locomocao.ANDANDO, 0.1)
    e2.propor(Locomocao.MEIA_VOLTA, 0.1)
    assert not e2.propor(Locomocao.ANDANDO, 0.1)
    assert e2.valor == Locomocao.PARADO, "ruido nao pode mudar o estado"


def test_estabilidade_e_em_TEMPO_e_nao_em_quadros():
    """A 30 fps e a 10 fps o mesmo limiar tem que significar a mesma coisa.

    Em quadros, `3` vale 0,1 s numa maquina e 0,5 s noutra — a mesma armadilha
    do `1/30` cravado no raio de recostura, que quebrou a 4 fps em 08/08.
    """
    rapido = Estavel(Locomocao.PARADO, minimo_s=0.3)
    mudou_rapido = sum(rapido.propor(Locomocao.ANDANDO, 1 / 30)
                       for _ in range(9))

    lento = Estavel(Locomocao.PARADO, minimo_s=0.3)
    mudou_lento = sum(lento.propor(Locomocao.ANDANDO, 1 / 10)
                      for _ in range(3))

    assert mudou_rapido == 1 and mudou_lento == 1
    assert rapido.valor == lento.valor == Locomocao.ANDANDO


def test_acao_e_serializavel_e_nao_carrega_junta():
    d = Acao(locomocao=Locomocao.FRENTE, postura=Postura.EM_PE).para_dicionario()
    assert d["locomocao"] == Locomocao.FRENTE
    assert "esqueleto" not in d and "juntas" not in d


# -------------------------------------------------------------- locomocao
def test_parado_e_andando_com_histerese():
    """Limiar unico faz alguem parado piscar entre os dois estados."""
    c = ClassificadorDeAcao(parar_abaixo_de=0.15, andar_acima_de=0.25,
                            estabilidade_s=0.2)

    for _ in range(6):
        a, _ = c.classificar(pessoa(vx=0.02), 0.1)
    assert a.locomocao == Locomocao.PARADO

    for _ in range(6):
        a, _ = c.classificar(pessoa(vx=0.80), 0.1)
    assert a.locomocao == Locomocao.ANDANDO

    # 0,20 m/s esta na zona morta: quem ja anda continua andando
    for _ in range(6):
        a, _ = c.classificar(pessoa(vx=0.20), 0.1)
    assert a.locomocao == Locomocao.ANDANDO, "histerese nao segurou"

    for _ in range(6):
        a, _ = c.classificar(pessoa(vx=0.05), 0.1)
    assert a.locomocao == Locomocao.PARADO


def test_sem_rumo_do_corpo_a_resposta_e_apenas_andando():
    """LIMITE DECLARADO: o rumo vem da direcao do movimento, entao andar de
    lado e indistinguivel de andar para frente. A resposta honesta e ANDANDO,
    nao um chute entre frente e lado."""
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    for _ in range(6):
        a, _ = c.classificar(pessoa(vx=0.8, rumo=0.0), 0.1)

    assert a.locomocao == Locomocao.ANDANDO
    assert a.locomocao not in (Locomocao.FRENTE, Locomocao.ESQUERDA)
    assert a.confianca < 0.7, "confianca tem que refletir o que falta"


def test_com_rumo_do_corpo_separa_frente_de_lado():
    c = ClassificadorDeAcao(estabilidade_s=0.2)

    for _ in range(6):
        a, _ = c.classificar(pessoa(vx=0.8, rumo=0.0), 0.1, leitura=LeituraDoCorpo(rumo_corpo=0.0))
    assert a.locomocao == Locomocao.FRENTE

    c2 = ClassificadorDeAcao(estabilidade_s=0.2)
    for _ in range(6):     # anda para o norte com o corpo apontando para leste
        a2, _ = c2.classificar(
            pessoa(vy=0.8, rumo=math.pi / 2), 0.1, leitura=LeituraDoCorpo(rumo_corpo=0.0))
    assert a2.locomocao == Locomocao.ESQUERDA, a2.locomocao

    c3 = ClassificadorDeAcao(estabilidade_s=0.2)
    for _ in range(6):     # anda para tras
        a3, _ = c3.classificar(
            pessoa(vx=-0.8, rumo=math.pi), 0.1, leitura=LeituraDoCorpo(rumo_corpo=0.0))
    assert a3.locomocao == Locomocao.TRAS


def test_virar_e_meia_volta():
    c = ClassificadorDeAcao(estabilidade_s=0.2, girar_acima_de=45)
    a = girar(c, 6, graus_por_passo=12, dt=0.1)      # 120 graus/s
    assert a.locomocao == Locomocao.VIRANDO_ESQ, a.locomocao

    c2 = ClassificadorDeAcao(estabilidade_s=0.2, girar_acima_de=45)
    a2 = girar(c2, 6, graus_por_passo=-12, dt=0.1)
    assert a2.locomocao == Locomocao.VIRANDO_DIR, a2.locomocao

    # 20 graus por quadro: o limiar de 150 e cruzado no 9o, e o `Estavel`
    # confirma no 10o. O primeiro quadro so registra o rumo inicial e nao
    # acumula — por isso a conta comeca no segundo.
    c3 = ClassificadorDeAcao(estabilidade_s=0.2, meia_volta_graus=150)
    a3 = girar(c3, 10, graus_por_passo=20, dt=0.1)   # 180 graus acumulados
    assert a3.locomocao == Locomocao.MEIA_VOLTA, a3.locomocao


def test_cruzar_o_meridiano_nao_inventa_meia_volta():
    """De +179 para -179 graus sao 2 graus, nao 358. Sem o cuidado com o
    embrulho do angulo, o sistema anunciaria uma meia-volta que nao houve."""
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    rumo = math.radians(179)
    for _ in range(10):
        rumo += math.radians(1)
        if rumo > math.pi:
            rumo -= 2 * math.pi
        a, _ = c.classificar(
            pessoa(vx=0.6 * math.cos(rumo), vy=0.6 * math.sin(rumo),
                   rumo=rumo), 0.1)
    assert a.locomocao != Locomocao.MEIA_VOLTA
    assert abs(a.giro_graus_s) < 30, a.giro_graus_s


def test_posicao_prevista_derruba_a_confianca():
    """Prever onde alguem deveria estar nao e o mesmo que ve-lo ali."""
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    for _ in range(6):
        medida, _ = c.classificar(pessoa(vx=0.02), 0.1)
    prevista, _ = c.classificar(pessoa(vx=0.02, prevendo=4), 0.1)
    assert prevista.confianca < medida.confianca


# ---------------------------------------------------------------- postura
def test_postura_reaproveita_o_k_do_filtro_de_altura():
    c = ClassificadorDeAcao(estabilidade_s=0.2, agachado_abaixo_de=0.78)

    for _ in range(4):
        a, _ = c.classificar(pessoa(), 0.1, razao_altura=0.25,
                                k_referencia=0.25)
    assert a.postura == Postura.EM_PE

    for _ in range(4):
        a, _ = c.classificar(pessoa(), 0.1, razao_altura=0.15,
                                k_referencia=0.25)
    assert a.postura == Postura.AGACHADO, a.razao_altura


def test_sem_k_confiavel_a_postura_e_desconhecida():
    """Quando o filtro de altura esta ABSTIDO, `k` nao vale. Sem base, nao
    se opina — a mesma regra que o proprio filtro passou a seguir em 10/08."""
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    for _ in range(6):
        a, _ = c.classificar(pessoa(), 0.1, razao_altura=0.20,
                                k_referencia=None)
    assert a.postura == Postura.DESCONHECIDA


# --------------------------------------------------------------- descritor
def test_descritor_guarda_historia_por_pessoa():
    """Um classificador global misturaria os rumos de duas pessoas e
    anunciaria giros que ninguem fez."""
    d = Descritor(estabilidade_s=0.2)
    for _ in range(6):
        r = d.atualizar([pessoa(1, vx=0.8, rumo=0.0),
                         pessoa(2, vx=0.01, rumo=0.0)], 0.1)
    assert r[1][0].locomocao == Locomocao.ANDANDO
    assert r[2][0].locomocao == Locomocao.PARADO


def test_descritor_esquece_quem_saiu():
    d = Descritor(estabilidade_s=0.2)
    d.atualizar([pessoa(1), pessoa(2)], 0.1)
    assert len(d._por_pessoa) == 2
    d.atualizar([pessoa(1)], 0.1)
    assert set(d._por_pessoa) == {1}, "rastro perdido deixou classificador"


def test_mudanca_e_anunciada_uma_vez_so():
    """O mesmo principio dos eventos de zona: uma vez por travessia."""
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    mudancas = 0
    for _ in range(12):
        _, _mud = c.classificar(pessoa(vx=0.8), 0.1)
        mudancas += int(_mud["locomocao"])
    assert mudancas == 1, f"anunciou {mudancas} vezes a mesma mudanca"


def test_ruido_em_velocidade_baixa_nao_vira_giro():
    """MEDIDO EM 10/08: `andando 0,23 m/s -193 graus/s` — meia volta por
    segundo com a pessoa quase parada. Nao era giro, era ruido.

    Um vetor velocidade curto tem angulo mal definido: o modulo mal se mexe e
    a direcao gira loucamente. Direcao nao confiavel nao pode virar "virando".
    """
    import random
    random.seed(11)
    c = ClassificadorDeAcao(estabilidade_s=0.2, parar_abaixo_de=0.15,
                            andar_acima_de=0.25)

    estados = set()
    for _ in range(40):
        # 0,20 m/s: acima de parar, abaixo de andar. Rumo totalmente aleatorio.
        rumo = random.uniform(-math.pi, math.pi)
        a, _ = c.classificar(
            pessoa(vx=0.20 * math.cos(rumo), vy=0.20 * math.sin(rumo),
                   rumo=rumo), 0.08)
        estados.add(a.locomocao)

    assert Locomocao.VIRANDO_ESQ not in estados, estados
    assert Locomocao.VIRANDO_DIR not in estados, estados
    assert Locomocao.MEIA_VOLTA not in estados, estados


def test_giro_de_quem_anda_de_verdade_continua_sendo_detectado():
    """O contraste: acima do limiar de andar, o giro vale."""
    c = ClassificadorDeAcao(estabilidade_s=0.2, andar_acima_de=0.25)
    a = girar(c, 8, graus_por_passo=12, dt=0.1, v=0.7)
    assert a.locomocao == Locomocao.VIRANDO_ESQ, a.locomocao


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
