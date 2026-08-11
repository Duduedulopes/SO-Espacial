"""
Testes do gabarito — e do proprio aparato de medicao.

POR QUE O APARATO PRECISA SER TESTADO

O conferidor existe para julgar o sistema. Se ele mesmo estiver errado, ele vai
reprovar codigo que funciona ou aprovar codigo que nao funciona — e nos dois
casos manda consertar o lugar errado, que e o defeito mais caro que este
projeto ja pagou.

Em 10/08 tres arquivos da fase 7 tinham sido escritos e nao executados, e rodar
encontrou tres defeitos em minutos. Um conferidor nao executado seria o quarto.

O TESTE QUE VALE MAIS: O ATOR SINTETICO

`test_o_ator_sintetico_tira_nota_alta` faz um corpo de medidas conhecidas
executar o roteiro inteiro e passa isso pelo SpatialEngine de verdade —
homografia, Kalman, filtros, analisador de corpo, classificador. O conferidor
pontua o resultado.

Isso separa duas causas que, no hardware, chegam misturadas:

    nota baixa AQUI       a logica de leitura esta errada
    nota baixa NO HARDWARE  a logica esta certa; a camera nao esta entregando

Sem essa separacao, uma sessao ruim com as tres cameras nao diz onde mexer — e
foi exatamente assim que a hipotese errada sobre a camera lateral sobreviveu a
tres execucoes em 10/08.

    python -m pytest testes/test_gabarito.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.acao.gabarito import (                                    # noqa: E402
    CERTO, ERRADO, POBRE, SEM_LEITURA, Passo, Placar, roteiro_padrao,
)
from src.acao.vocabulario import Acao, Braco, Locomocao, Postura   # noqa: E402
from src.espacial.motor import SpatialEngine                       # noqa: E402
from src.visao.observacao import Observacao                        # noqa: E402
from test_corpo import corpo, tudo_visivel                         # noqa: E402

PASSO_ANDAR = Passo("andar_frente", "ANDE", eixo="locomocao",
                    certo=(Locomocao.FRENTE,), pobre=(Locomocao.ANDANDO,),
                    segundos=5, acomodacao_s=1.0)


def acoes(**kw):
    """{id: (Acao, mudancas)}, no formato que o SpatialEngine publica."""
    return {1: (Acao(**kw), {})}


# ---------------------------------------------------------------- placar
def test_acomodacao_nao_conta():
    """Sem esta janela, todo passo seria penalizado pela propria transicao.

    O `Estavel` do classificador exige 0,35 s de concordancia antes de mudar de
    estado. Contar esse intervalo mediria tempo de reacao, nao acerto — e a
    penalidade seria maior nos passos curtos, fazendo a nota depender da
    duracao escolhida no roteiro.
    """
    p = Placar()
    for t in (0.0, 0.5, 0.99):
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.PARADO), t)

    assert not p.contagens, "quadro dentro da acomodacao foi contado"

    p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.FRENTE), 1.5)
    assert p.contagens["andar_frente"].total == 1


def test_pobre_nao_e_errado():
    """`andando` quando a pessoa andou para frente e abstencao, nao erro.

    O sistema respondeu MENOS porque o azimute nao convergiu. Somar isso com
    erro empurraria para consertar um sistema que se comportou exatamente como
    projetado — e a abstencao e projeto, decidido em 10/08.
    """
    p = Placar()
    for _ in range(10):
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.ANDANDO), 2.0)

    c = p.contagens["andar_frente"]
    assert c.veredictos[POBRE] == 10
    assert c.veredictos[ERRADO] == 0
    assert c.nota == 0.0, "nota so premia acerto exato"
    assert c.aproveitamento == 1.0, "mas nao houve contradicao nenhuma"


def test_nota_e_aproveitamento_separam_dois_diagnosticos():
    """Nota 0 com aproveitamento 100% = falta materia-prima, nao ha bug.
    Nota 0 com aproveitamento 0% = o sistema esta afirmando outra coisa."""
    absteve, errou = Placar(), Placar()
    for _ in range(10):
        absteve.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.ANDANDO), 2.0)
        errou.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.PARADO), 2.0)

    assert absteve.contagens["andar_frente"].aproveitamento == 1.0
    assert errou.contagens["andar_frente"].aproveitamento == 0.0


def test_sem_leitura_e_categoria_propria():
    """'Nao vi ninguem' e falha de DETECCAO. 'Vi e classifiquei errado' e falha
    de CLASSIFICACAO. Os consertos ficam em arquivos diferentes."""
    p = Placar()
    for _ in range(10):
        p.anotar(PASSO_ANDAR, {}, 2.0)

    c = p.contagens["andar_frente"]
    assert c.veredictos[SEM_LEITURA] == 10
    assert c.veredictos[ERRADO] == 0


def test_quadro_previsto_nao_pontua():
    """Mesma regra do mapa de calor: so acumula o que foi MEDIDO.

    Prever onde alguem deveria estar nao e ve-lo ali. Pontuar a previsao
    misturaria erro de classificacao com falta de medicao.
    """
    p = Placar()
    for _ in range(5):
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.PARADO), 2.0,
                 prevendo={1: 3})

    assert not p.contagens.get("andar_frente") or \
        p.contagens["andar_frente"].total == 0
    assert p.previstos_ignorados == 5


def test_pior_confusao_aponta_o_conserto():
    """Numero de erro sozinho nao diz onde mexer. Em 10/08 o painel disse
    `rejeitadas plausibilidade 358` e a pergunta que importava — de quantas, e
    em favor de que — nao tinha resposta na tela."""
    p = Placar()
    for _ in range(7):
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.PARADO), 2.0)
    for _ in range(2):
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.TRAS), 2.0)
    for _ in range(1):
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.FRENTE), 2.0)

    confusao, frac = p.contagens["andar_frente"].pior_confusao
    assert confusao == Locomocao.PARADO
    assert abs(frac - 0.7) < 0.01


def test_passo_sem_eixo_nao_pontua():
    """O passo de aquecimento existe para o azimute aprender, nao para valer
    nota. Pontuar um sistema que ainda esta aprendendo o reprova pelo roteiro,
    nao pelo comportamento."""
    p = Placar()
    aquecer = Passo("aquecer", "ANDE", eixo=None, certo=(), segundos=10)
    for _ in range(20):
        p.anotar(aquecer, acoes(locomocao=Locomocao.PARADO), 5.0)

    assert not p.contagens


def test_boletim_e_serializavel():
    """Sem gravar nao ha comparacao, e sem comparacao ajustar e fe."""
    import json

    p = Placar()
    p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.FRENTE,
                                altura_mao_dir=1.2), 2.0)
    d = p.para_dicionario()

    assert json.loads(json.dumps(d))["acoes"]["andar_frente"]["nota"] == 1.0


def test_roteiro_padrao_cobre_o_que_foi_pedido():
    """As acoes que o Eduardo listou tem que estar todas la."""
    acoes_r = {p.acao for p in roteiro_padrao()}

    for exigida in ("parado", "andar_frente", "andar_tras", "virar_direita",
                    "virar_esquerda", "agachar", "braco_dir_levantado",
                    "braco_esq_levantado"):
        assert exigida in acoes_r, exigida


def test_roteiro_comeca_andando():
    """O azimute so aprende com quem anda. Pedir 'vire' antes de qualquer
    caminhada mediria um estimador sem materia-prima — e reprovaria o sistema
    por uma coisa que o roteiro causou."""
    r = roteiro_padrao()
    assert r[0].eixo is None, "o primeiro passo tem que ser aquecimento"

    def indice(nome):
        return next(i for i, p in enumerate(r) if p.acao == nome)

    assert indice("andar_frente") < indice("virar_direita")
    assert indice("andar_frente") < indice("andar_lado")


# ------------------------------------------------------ ator sintetico
class Ator:
    """Um corpo de medidas conhecidas executando o roteiro.

    A camera esta girada 35 graus em relacao ao mundo. Esse numero nao aparece
    em lugar nenhum do sistema: o `EstimadorDeAzimute` tem que descobri-lo
    sozinho vendo o ator andar. Se ele nao descobrir, `andar_frente` sai como
    `andando` e o teste acusa.
    """

    AZIMUTE = math.radians(35)
    PIXEL_POR_METRO = 100.0

    def __init__(self):
        self.x, self.y = 1.0, 1.0
        self.rumo = 0.0                  # para onde o CORPO aponta, no mundo
        self.mao_esq = self.mao_dir = None

    def mover(self, direcao_mundo, metros):
        self.x += metros * math.cos(direcao_mundo)
        self.y += metros * math.sin(direcao_mundo)

    def observar(self, t):
        px = self.x * self.PIXEL_POR_METRO
        py = self.y * self.PIXEL_POR_METRO
        juntas_2d = np.tile([px, py], (17, 1)).astype(float)

        return [
            Observacao(camera_id="alto", papel="alto", t_mono=t,
                       caixa=(px - 30, py - 240, px + 30, py),
                       id_externo=1, confianca=0.9,
                       juntas_2d=juntas_2d, conf_2d=np.ones(17)),
            Observacao(camera_id="frontal", papel="frontal", t_mono=t,
                       juntas_3d=corpo(rumo_camera=self.rumo - self.AZIMUTE,
                                       mao_esq=self.mao_esq,
                                       mao_dir=self.mao_dir),
                       conf_2d=tudo_visivel(), confianca=0.9),
        ]


def encenar(motor, ator, placar, passo, quadros=60, dt=0.1,
            passo_m=0.06, direcao=None, giro_por_quadro=0.0):
    """Roda um passo do roteiro e anota. `direcao` e relativa ao rumo do corpo.

    O CARIMBO DE TEMPO E O RELOGIO DE VERDADE, E NAO UM NUMERO INVENTADO.

    A primeira versao deste ajudante usava `t = 100.0 + n * dt`. O motor
    descarta pose com mais de 0,5 s de idade comparando `t_mono` com
    `time.monotonic()` — entao TODA pose chegava com milhares de segundos de
    atraso e era jogada fora. O azimute ficou com zero amostras e o teste
    reprovou codigo que estava certo.

    O defeito era do teste, e ele e instrutivo: `t_mono` nao e um rotulo
    qualquer, e uma leitura do MESMO relogio que o motor consulta. Simular o
    tempo exige simular os dois lados, e nao vale a pena aqui — `dt` continua
    sendo passado a mao, que e o que controla a fisica.
    """
    import time

    for n in range(quadros):
        if direcao is not None:
            ator.mover(ator.rumo + direcao, passo_m)
        ator.rumo += giro_por_quadro

        motor.atualizar(ator.observar(time.monotonic()), dt)
        placar.anotar(passo, motor.acoes, n * dt)


def montar():
    H = np.array([[1 / Ator.PIXEL_POR_METRO, 0, 0],
                  [0, 1 / Ator.PIXEL_POR_METRO, 0], [0, 0, 1.0]])
    return SpatialEngine(H, usar_plausibilidade=False, min_tornozelo=1), Ator()


def test_o_ator_sintetico_tira_nota_alta():
    """A prova de que a cadeia de leitura esta certa.

    Se este teste passar e o hardware reprovar, o defeito esta na camera ou na
    pose — nao na logica. Essa separacao e o que faltava para poder julgar uma
    sessao ruim sem adivinhar.
    """
    motor, ator = montar()
    placar = Placar()
    por_acao = {p.acao: p for p in roteiro_padrao()}

    # aquecimento: anda para frente e para tras, o azimute aprende
    for _ in range(2):
        encenar(motor, ator, placar, por_acao["aquecer"], 40,
                    direcao=0.0)
        ator.rumo += math.pi
        encenar(motor, ator, placar, por_acao["aquecer"], 40,
                    direcao=0.0)
    ator.rumo = 0.0

    assert motor.corpo.azimute.confiavel, motor.corpo.diagnostico
    erro = abs(math.degrees(motor.corpo.azimute.valor) - 35)
    assert erro < 3, f"azimute aprendido {math.degrees(motor.corpo.azimute.valor):.0f}"

    # cada passo do roteiro, encenado
    encenar(motor, ator, placar, por_acao["andar_frente"], 40,
                direcao=0.0)
    encenar(motor, ator, placar, por_acao["andar_tras"], 40,
                direcao=math.pi)
    encenar(motor, ator, placar, por_acao["andar_lado"], 40,
                direcao=math.pi / 2)

    ator.mao_dir = (1.75, 0.10)
    encenar(motor, ator, placar, por_acao["braco_dir_levantado"], 30,
                direcao=0.0)

    ator.mao_dir, ator.mao_esq = None, (1.75, 0.10)
    encenar(motor, ator, placar, por_acao["braco_esq_levantado"], 30,
                direcao=0.0)

    ator.mao_esq = None
    ator.mao_dir = (1.47, 0.45)
    encenar(motor, ator, placar, por_acao["braco_dir_estendido"], 30,
                direcao=0.0)

    boletim = "\n".join(placar.linhas())
    for acao in ("andar_frente", "andar_tras", "andar_lado",
                 "braco_dir_levantado", "braco_esq_levantado",
                 "braco_dir_estendido"):
        nota = placar.contagens[acao].nota
        assert nota > 0.85, f"{acao} tirou {nota:.0%}\n\n{boletim}"


def test_o_ator_parado_e_lido_como_parado():
    motor, ator = montar()
    placar = Placar()
    parado = next(p for p in roteiro_padrao() if p.acao == "parado")

    encenar(motor, ator, placar, parado, 40, direcao=None)

    assert placar.contagens["parado"].nota > 0.9


def test_a_altura_da_mao_atravessa_o_conferidor():
    """O numero que a etapa D vai comparar com a faixa da prateleira."""
    motor, ator = montar()
    placar = Placar()
    passo = next(p for p in roteiro_padrao()
                 if p.acao == "braco_dir_levantado")

    ator.mao_dir = (1.42, 0.10)
    encenar(motor, ator, placar, passo, 40, direcao=0.0)

    alturas = placar.contagens["braco_dir_levantado"].alturas
    assert alturas, "nenhuma altura chegou ao boletim"
    assert abs(np.median(alturas) - 1.42) < 0.03, np.median(alturas)


def test_o_id_da_pessoa_e_registrado():
    """O Eduardo pediu o ID em cada leitura: sem ele, um boletim com duas
    pessoas nao diz de quem e a nota."""
    motor, ator = montar()
    placar = Placar()
    parado = next(p for p in roteiro_padrao() if p.acao == "parado")

    encenar(motor, ator, placar, parado, 30, direcao=None)

    assert placar.contagens["parado"].ids == {1}
