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
from src.espacial.estado import EstadoDePessoa                     # noqa: E402
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
    coasting = {1: EstadoDePessoa(id=1, x=0, y=0, prevendo=3)}
    for _ in range(5):
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.PARADO), 2.0, coasting)

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
    posicionar = next(p for p in roteiro_padrao() if p.reposiciona)

    # OS PASSOS DE REPOSICIONAMENTO SAO O AQUECIMENTO.
    #
    # O roteiro do Eduardo nao tem passo dedicado a isso, e nao precisa: em
    # `VA ATE A BORDA DE TRAS`, `VA PARA O MEIO` e `VOLTE PARA O MEIO` a pessoa
    # anda olhando para onde vai, que e exatamente a hipotese do estimador.
    # Aqui o ator faz o mesmo — vai e volta virando o corpo.
    for _ in range(2):
        encenar(motor, ator, placar, posicionar, 40, direcao=0.0)
        ator.rumo += math.pi
        encenar(motor, ator, placar, posicionar, 40, direcao=0.0)
    ator.rumo = 0.0

    assert motor.corpo.azimute.confiavel, motor.corpo.diagnostico
    erro = abs(math.degrees(motor.corpo.azimute.valor) - 35)
    assert erro < 3, f"azimute aprendido {math.degrees(motor.corpo.azimute.valor):.0f}"

    # cada passo do roteiro, encenado
    encenar(motor, ator, placar, por_acao["andar_frente"], 40,
                direcao=0.0)
    # ANDAR DE RE: a minoria que a moda tem que descartar.
    #
    # Estas amostras chegam ao azimute 180 graus fora. Com media elas puxavam a
    # resposta para o vazio entre os grupos; com moda, sao minoria e caem fora.
    # Se este passo quebrar o azimute, `andar_esquerda` abaixo reprova junto —
    # que e exatamente o efeito em cascata visto no hardware em 11/08.
    encenar(motor, ator, placar, por_acao["andar_tras"], 40,
                direcao=math.pi)
    encenar(motor, ator, placar, por_acao["andar_esquerda"], 40,
                direcao=math.pi / 2)

    assert motor.corpo.azimute.confiavel, (
        f"andar de re e de lado derrubaram o azimute: "
        f"{motor.corpo.diagnostico}")

    ator.mao_dir = (1.75, 0.10)
    encenar(motor, ator, placar, por_acao["braco_dir_levantado"], 30,
                direcao=0.0)

    ator.mao_dir, ator.mao_esq = None, (1.75, 0.10)
    encenar(motor, ator, placar, por_acao["braco_esq_levantado"], 30,
                direcao=0.0)

    boletim = "\n".join(placar.linhas())
    for acao in ("andar_frente", "andar_tras", "andar_esquerda",
                 "braco_dir_levantado", "braco_esq_levantado"):
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


# ------------------------------------------------- a fita metrica do chao
def _andar(placar, passo, v, metros_por_quadro, n=20):
    for i in range(n):
        p = EstadoDePessoa(id=1, x=i * metros_por_quadro, y=0.0, vx=v)
        placar.anotar(passo, acoes(locomocao=Locomocao.PARADO,
                                   velocidade_ms=v), 2.0, {1: p})


def test_boletim_mostra_a_velocidade_que_o_sistema_MEDIU():
    """`parado` em 66% de quem andou tem duas explicacoes opostas.

    A pessoa mal se deslocou, ou a homografia encolhe a distancia. A nota nao
    separa as duas; o numero cru separa. Sem ele, escolher entre recalibrar e
    refazer a sessao seria chute — e chute com trabalho ja custou duas rodadas
    de otimizacao no lugar errado.
    """
    p = Placar()
    _andar(p, PASSO_ANDAR, v=0.08, metros_por_quadro=0.02)

    c = p.contagens["andar_frente"]
    assert abs(c.velocidade_mediana - 0.08) < 0.001
    assert abs(c.deslocamento - 0.38) < 0.01

    texto = "\n".join(p.linhas())
    assert "MOVIMENTO MEDIDO" in texto
    assert "homografia esta encolhendo" in texto, texto


def test_velocidade_saudavel_nao_dispara_o_alerta():
    """O alerta so vale se ele ficar calado quando nao ha o que alertar."""
    p = Placar()
    _andar(p, PASSO_ANDAR, v=0.90, metros_por_quadro=0.09)

    texto = "\n".join(p.linhas())
    assert "MOVIMENTO MEDIDO" in texto
    assert "homografia esta encolhendo" not in texto


def test_deslocamento_usa_extremos_e_nao_soma_de_passinhos():
    """Somar trechos infla com ruido: uma pessoa PARADA percorreria metros.

    Com extremos, o tremor do Kalman em torno de um ponto continua sendo
    aproximadamente zero de deslocamento — que e a verdade.
    """
    p = Placar()
    for i in range(40):
        tremor = 0.01 * (1 if i % 2 else -1)
        p.anotar(PASSO_ANDAR, acoes(locomocao=Locomocao.PARADO), 2.0,
                 {1: EstadoDePessoa(id=1, x=tremor, y=0.0)})

    assert p.contagens["andar_frente"].deslocamento < 0.03


def test_numeros_crus_valem_mesmo_quando_a_classificacao_erra():
    """E principalmente quando erra: e ali que se precisa deles."""
    p = Placar()
    _andar(p, PASSO_ANDAR, v=0.05, metros_por_quadro=0.01)

    c = p.contagens["andar_frente"]
    assert c.nota == 0.0, "a classificacao errou mesmo"
    assert c.velocidades and c.posicoes, "e os numeros crus ficaram"


# ---------------------------------------------------- pares que se desfazem







# ------------------------------------- o roteiro tem que caber em 140x140 cm
AREA_M = 1.40      # o espaco real do Eduardo, medido em 11/08
LIMIAR = 0.25      # `andar_acima` do ClassificadorDeAcao


def test_todo_passo_de_caminhada_cabe_na_area_acima_do_limiar():
    """A duracao do passo, num espaco fechado, E um limiar de velocidade.

    MEDIDO EM 11/08: com 6 segundos, atravessar 1,40 m exige 0,23 m/s — ABAIXO
    dos 0,25 que separam `parado` de `andando`. O roteiro PROIBIA o resultado
    certo: mesmo andando de ponta a ponta, o passo sairia `parado`, e a nota
    culparia o classificador por uma conta do roteiro.

    Este teste nao mede o sistema. Mede se o roteiro e possivel de cumprir.
    """
    for p in roteiro_padrao():
        if not (p.desloca and p.acao.startswith("andar")):
            continue
        exigida = AREA_M / p.segundos
        assert exigida > LIMIAR * 1.2, (
            f"{p.acao}: {p.segundos}s para {AREA_M} m exige {exigida:.2f} m/s, "
            f"perto demais do limiar de {LIMIAR}")


def test_a_acomodacao_nao_come_a_janela_curta():
    """Com passos de 4 s, 1,2 s de acomodacao comeria um terco. O que sobra
    ainda precisa cobrir os 0,35 s de que o `Estavel` precisa para se
    comprometer, com folga para medir depois disso."""
    for p in roteiro_padrao():
        if p.eixo is None:
            continue
        sobra = p.segundos - p.acomodacao_s
        assert sobra >= 2.0, f"{p.acao}: sobram so {sobra:.1f}s de medicao"




# --------------------------------------------- o roteiro escrito pelo Eduardo
def test_cada_acao_volta_ao_neutro_antes_da_seguinte():
    """A correcao do Eduardo, 11/08, e a razao dela.

    Ninguem emenda "ande de lado" em "agache" sem passar por ficar em pe no
    meio. Quando o roteiro pede isso, a pessoa improvisa a transicao — e a
    transicao entra na medicao.

        Roteiro que emenda movimentos mede as emendas.
    """
    nomes = [p.acao for p in roteiro_padrao()]

    for ida, volta in (("agachar", "levantar"),
                       ("braco_dir_levantado", "braco_dir_baixado"),
                       ("braco_esq_levantado", "braco_esq_baixado")):
        assert nomes.index(volta) == nomes.index(ida) + 1, nomes


def test_cada_movimento_de_braco_vale_nota_propria():
    """Na minha versao a baixada vinha embutida na instrucao seguinte —
    'BAIXE O DIREITO e LEVANTE O ESQUERDO'. Dois movimentos, uma nota so: se o
    sistema perdesse a descida do direito, nada acusaria."""
    por = {p.acao: p for p in roteiro_padrao()}

    assert por["braco_dir_baixado"].certo == (Braco.AO_LADO,)
    assert por["braco_esq_baixado"].certo == (Braco.AO_LADO,)
    assert por["braco_dir_baixado"].eixo == "braco_direito"
    assert por["braco_esq_baixado"].eixo == "braco_esquerdo"


def test_o_deslocamento_e_sempre_seguido_de_reposicionamento():
    """Sem isso, cada passo empurra a pessoa para fora do enquadramento."""
    r = roteiro_padrao()

    for i, p in enumerate(r[:-1]):
        if p.desloca and not r[i + 1].desloca:
            assert r[i + 1].reposiciona or r[i + 1].eixo != "locomocao", (
                f"{p.acao} desloca e o seguinte ({r[i+1].acao}) nao "
                f"reposiciona nem e outro deslocamento")


def test_o_roteiro_dispensa_passo_de_aquecimento_dedicado():
    """Os proprios `VA ATE A BORDA` sao o aquecimento: a pessoa anda olhando
    para onde vai, que e a hipotese do EstimadorDeAzimute. O passo de 14 s
    existia porque eu nao tinha reparado nisso."""
    r = roteiro_padrao()
    caminhadas = [p for p in r if p.reposiciona]

    assert len(caminhadas) >= 4, "poucos trechos de caminhada util"
    assert r[0].reposiciona, "o roteiro comeca posicionando, e ja andando"


def test_termina_parado_e_com_os_bracos_ao_lado():
    """`parado` no fim pergunta se o sistema VOLTA ao neutro depois de tudo —
    a mesma cobranca que `levantar` faz da postura."""
    r = roteiro_padrao()

    assert r[-1].acao == "parado_fim"
    assert r[-1].certo == (Locomocao.PARADO,)
    assert r[-2].acao == "braco_esq_baixado"


def test_os_dois_parados_sao_contados_separadamente():
    """Dois passos com o mesmo nome colidiriam no placar e a nota do fim
    sobrescreveria a do meio — apagando justamente a comparacao entre elas."""
    nomes = [p.acao for p in roteiro_padrao() if p.eixo]

    assert len(nomes) == len(set(nomes)), f"nome repetido: {nomes}"


def test_todo_passo_cabe_em_140cm_acima_do_limiar():
    """A duracao do passo, num espaco fechado, E um limiar de velocidade."""
    for p in roteiro_padrao():
        if not (p.desloca and p.acao.startswith("andar")):
            continue
        exigida = 1.40 / p.segundos
        assert exigida > 0.25 * 1.2, (
            f"{p.acao}: {p.segundos}s para 1,40 m exige {exigida:.2f} m/s")


# ------------------------------------------------------------ modo travado
def test_o_tempo_separa_lento_de_errado():
    """Ideia do Eduardo, 11/08.

    A nota responde "que fracao dos quadros estava certa" — e isso mistura um
    sistema LENTO com um sistema ERRADO. Os dois dao 40%, e os consertos sao
    opostos: um pede ajuste de estabilidade, o outro pede conserto de logica.

        Nota baixa nao diz se o sistema e lento ou se esta errado.
    """
    p = Placar()
    por = {x.acao: x for x in roteiro_padrao()}

    for _ in range(10):
        p.anotar(por["andar_frente"],
                 acoes(locomocao=Locomocao.FRENTE), 999)
    p.marcar_tempo(por["andar_frente"], 4.2, 5.0, 5.0)      # lento

    for _ in range(10):
        p.anotar(por["agachar"], acoes(postura=Postura.EM_PE), 999)
    p.marcar_tempo(por["agachar"], None, None, 25.0)        # errado

    texto = "\n".join(p.linhas_de_tempo())
    assert "LENTO" in texto
    assert "NAO RECONHECEU" in texto
    assert "em_pe" in texto, "precisa dizer EM FAVOR DE QUE ele errou"


def test_nunca_confirmar_e_registrado_como_falha_e_nao_como_zero():
    p = Placar()
    passo = next(x for x in roteiro_padrao() if x.acao == "agachar")
    p.anotar(passo, acoes(postura=Postura.EM_PE), 999)
    p.marcar_tempo(passo, None, None, 25.0)

    c = p.contagens["agachar"]
    assert not c.confirmou
    assert c.espera_s == 25.0
    assert "NUNCA" in "\n".join(p.linhas_de_tempo())


def test_confirmar_exige_leitura_SUSTENTADA():
    """Um unico quadro certo no meio do ruido nao e reconhecimento. E a mesma
    regra do `Estavel` do classificador, aplicada ao gabarito."""
    from ferramentas.conferir import _esta_certo

    passo = next(x for x in roteiro_padrao() if x.acao == "agachar")
    certo = {1: (Acao(postura=Postura.AGACHADO), {})}
    errado = {1: (Acao(postura=Postura.EM_PE), {})}

    assert _esta_certo(passo, certo, {})
    assert not _esta_certo(passo, errado, {})
    assert not _esta_certo(passo, {}, {}), "sem ninguem nao confirma"


def test_posicao_prevista_nao_confirma():
    """Prever onde alguem deveria estar nao e ve-lo ali. Confirmar um passo com
    posicao prevista deixaria o roteiro avancar sobre um palpite."""
    from ferramentas.conferir import _esta_certo

    passo = next(x for x in roteiro_padrao() if x.acao == "agachar")
    certo = {1: (Acao(postura=Postura.AGACHADO), {})}
    coasting = {1: EstadoDePessoa(id=1, x=0, y=0, prevendo=4)}

    assert _esta_certo(passo, certo, {})
    assert not _esta_certo(passo, certo, coasting)


def test_o_limite_existe_para_a_sessao_nao_travar():
    """Sem limite, uma acao que o sistema nao sabe ler prende a sessao para
    sempre e NADA e medido — nem os passos seguintes, que talvez funcionem."""
    import inspect

    from ferramentas.conferir import rodar_travado

    assinatura = inspect.signature(rodar_travado)
    assert assinatura.parameters["limite_s"].default > 0


# ----------------------------------------- o que se sustenta e o que nao
def test_caminhada_nao_e_travavel():
    """MEDIDO EM VIDEO, 11/08: num passo de 25 s pedindo `ande para frente`, o
    Eduardo ficou DOZE SEGUNDOS no mesmo lugar. Nao por falta de esforco — em
    1,4 m ele atravessa em dois segundos, chega na parede, e so resta esperar.

    A mediana da velocidade daquele passo ficou em 0,03 m/s e o sistema
    respondeu `parado`, corretamente.

        O instrumento pediu uma acao fisicamente insustentavel e depois
        reprovou quem nao a sustentou.
    """
    for p in roteiro_padrao():
        if p.eixo == "locomocao" and p.acao.startswith("andar"):
            assert not p.travavel, f"{p.acao} nao pode ser travado"


def test_postura_e_bracos_continuam_travaveis():
    """Da para ficar agachado dez segundos e segurar o braco no alto. Tirar a
    trava deles perderia o unico modo que mede tempo de reconhecimento."""
    for p in roteiro_padrao():
        if p.eixo in ("postura", "braco_direito", "braco_esquerdo"):
            assert p.travavel, f"{p.acao} deveria ser travavel"


def test_parado_e_travavel_mesmo_sendo_locomocao():
    """`parado` e locomocao e SE SUSTENTA — a distincao nao e pelo eixo, e
    pela fisica do movimento."""
    por = {p.acao: p for p in roteiro_padrao()}

    assert por["parado"].travavel
    assert por["parado_fim"].travavel
