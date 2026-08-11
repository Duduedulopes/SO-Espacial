"""
Testes da camada de corpo — esqueleto sintetico, verdade conhecida.

POR QUE SINTETICO, E POR QUE ISSO NAO E TRAPACA

Com camera de verdade nao existe gabarito: ninguem sabe se a mao estava a
1,20 m ou a 1,26 m no quadro 341. Sem gabarito, so da para olhar a tela e
achar que parece certo — e achar que parece certo foi o que sustentou por tres
execucoes a hipotese errada sobre a camera lateral em 10/08, ate o campo
`erro` mudar de lugar e revelar que a imagem chegava preta.

Aqui o corpo e CONSTRUIDO com a mao a 1,20 m. Se o sistema responder 1,18, o
erro e 2 cm e esta escrito. Se responder 0,42, alguma coisa esta errada e o
teste diz na hora qual.

    Sem registro do que aconteceu, nao ha como julgar o que o sistema disse
    que aconteceu.                                          — caderno, 10/08

O ROTEIRO DE CADA TESTE

    1. montar um corpo com medidas conhecidas
    2. gira-lo pelos angulos de uma camera conhecida
    3. entregar so isso ao analisador
    4. conferir se ele devolve os numeros do passo 1

Nenhum precisa de camera, de modelo, de rede ou de janela.

    python -m pytest testes/test_corpo.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acao.angulos import (                                     # noqa: E402
    concentracao, diferenca_angular, media_circular,
)
from src.acao.classificador import ClassificadorDeAcao             # noqa: E402
from src.acao.corpo import AnalisadorDeCorpo, EstimadorDeAzimute   # noqa: E402
from src.acao.vocabulario import Braco, Locomocao, Postura         # noqa: E402
from src.espacial.estado import EstadoDePessoa                     # noqa: E402

QUADRIL = 0.95          # altura do quadril acima do chao, em metros
OMBRO = 0.52            # altura do ombro ACIMA do quadril
LARGURA = 0.38          # distancia entre os ombros
OMBRO_DO_CHAO = QUADRIL + OMBRO      # 1,47 m


def corpo(rumo_camera=-math.pi / 2, quadril=QUADRIL,
          mao_esq=None, mao_dir=None, largura_ombros=LARGURA,
          agachado=False):
    """Monta um esqueleto COCO-17 de medidas conhecidas.

    `rumo_camera`  para onde o corpo aponta, no referencial da lente.
                   -pi/2 e de frente para a camera.
    `mao_*`        (altura_do_chao_em_metros, avanco_em_metros).
                   None = braco pendurado ao lado.
    `agachado`     dobra a perna DE VERDADE: coxa quase horizontal.

    Devolve (17,3) com origem no quadril e z para cima — que e exatamente o
    que o MediaPipe entrega DEPOIS de a inclinacao da lente ser desfeita.

    POR QUE `agachado` E UM PARAMETRO SEPARADO DE `quadril`

    A primeira versao simulava agachamento so baixando o quadril, e mantinha o
    joelho na metade da distancia ate o chao — ou seja, a coxa continuava
    VERTICAL. Isso nao e um agachamento: e uma pessoa em pe encolhida, que nao
    existe.

    O erro so apareceu quando a postura passou a ser lida pela verticalidade
    da coxa: o teste reprovou codigo que estava certo, porque a ENTRADA nao
    descrevia a situacao que o nome do teste prometia.

        Corpo sintetico que nao respeita a anatomia testa a propria fantasia.
    """
    frente = np.array([math.cos(rumo_camera), math.sin(rumo_camera), 0.0])
    lado = np.array([-math.sin(rumo_camera), math.cos(rumo_camera), 0.0])
    meia = lado * (largura_ombros / 2)

    def pulso(spec, sinal):
        if spec is None:                       # pendurado junto ao corpo
            return sinal * meia + np.array([0, 0, -0.05])
        altura_chao, avanco = spec
        return (sinal * meia + frente * avanco
                + np.array([0, 0, altura_chao - quadril]))

    j = np.zeros((17, 3))
    j[0] = [0, 0, OMBRO + 0.25]                                # nariz
    j[1] = j[2] = j[3] = j[4] = j[0]                           # olhos, orelhas
    j[5] = meia + np.array([0, 0, OMBRO])                      # ombro esq
    j[6] = -meia + np.array([0, 0, OMBRO])                     # ombro dir
    j[9] = pulso(mao_esq, +1)                                  # pulso esq
    j[10] = pulso(mao_dir, -1)                                 # pulso dir
    j[7] = (j[5] + j[9]) / 2                                   # cotovelos
    j[8] = (j[6] + j[10]) / 2
    j[11] = meia * 0.5                                         # quadris
    j[12] = -meia * 0.5

    if agachado:
        # Coxa quase horizontal: o joelho sobe ate quase a altura do quadril e
        # avanca. E o que o corpo faz de verdade — o femur nao encolhe, gira.
        joelho = frente * 0.42 + np.array([0, 0, -0.08])
        tornozelo = frente * 0.10 + np.array([0, 0, -quadril])
    else:
        joelho = np.array([0, 0, -quadril / 2])
        tornozelo = np.array([0, 0, -quadril])

    j[13] = meia * 0.5 + joelho                                # joelhos
    j[14] = -meia * 0.5 + joelho
    j[15] = meia * 0.5 + tornozelo                             # tornozelos
    j[16] = -meia * 0.5 + tornozelo
    return j


def inclinar(juntas, graus):
    """Simula a lente inclinada: gira o corpo para o referencial da CAMERA.

    E a operacao INVERSA de `desfazer_inclinacao`. Se as duas nao se
    cancelarem exatamente, o teste acusa — e foi um erro de sinal exatamente
    aqui que deixou o boneco deitado em 10/08.
    """
    r = math.radians(-graus)
    c, s = math.cos(r), math.sin(r)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return (Rx @ np.asarray(juntas).T).T


def tudo_visivel(exceto=()):
    v = np.ones(17, dtype=float)
    for i in exceto:
        v[i] = 0.0
    return v


def convergir_azimute(a, offset_graus, n=40, rumo_mundo=0.3):
    """Faz o analisador aprender o giro da camera vendo alguem andar.

    Anda em linha reta: rumo do corpo = rumo do deslocamento. E a hipotese que
    o estimador assume, e a mesma do estimador de inclinacao — ninguem caminha
    de lado por quinze metros.
    """
    rumo_cam = rumo_mundo - math.radians(offset_graus)
    for _ in range(n):
        a.ler(1, corpo(rumo_camera=rumo_cam), tudo_visivel(),
              rumo_mundo=rumo_mundo, velocidade=0.9)
    return a


# --------------------------------------------------------------- aritmetica
def test_media_circular_nao_colapsa_no_meridiano():
    """A media aritmetica de +179 e -179 da ZERO: a direcao OPOSTA a verdade.

    Nao e um erro pequeno — e o pior erro possivel. Uma camera montada perto
    de 180 graus produziria exatamente este caso.
    """
    angulos = [math.radians(179), math.radians(-179)]

    ingenua = math.degrees(sum(angulos) / len(angulos))
    correta = math.degrees(media_circular(angulos))

    assert abs(ingenua) < 1, "a media ingenua deveria mesmo dar ~0"
    assert abs(abs(correta) - 180) < 1, f"media circular deu {correta}"


def test_concentracao_separa_concordancia_de_bagunca():
    juntos = [0.01, -0.01, 0.02, 0.0]
    espalhados = [0.0, math.pi / 2, math.pi, -math.pi / 2]

    assert concentracao(juntos) > 0.99
    assert concentracao(espalhados) < 0.1


# ------------------------------------------------------------------ azimute
def test_azimute_aprende_o_giro_da_camera_sozinho():
    """Nenhum transferidor, nenhum campo novo no JSON: so gente andando."""
    a = AnalisadorDeCorpo()
    convergir_azimute(a, offset_graus=40)

    assert a.azimute.confiavel, a.azimute.diagnostico
    assert abs(math.degrees(a.azimute.valor) - 40) < 2, a.azimute.diagnostico


def test_azimute_se_abstem_enquanto_nao_sabe():
    """Antes de ver alguem andar, a resposta certa e nao responder.

    Um azimute chutado nao produz erro visivel: produz um `andando_frente`
    confiante e errado. Estado errado com cara de medido e o defeito que a
    arquitetura v3 inteira existe para evitar.
    """
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(), tudo_visivel(), rumo_mundo=0.0, velocidade=0.0)

    assert leitura.rumo_corpo is None
    assert leitura.rumo_corpo_camera is not None, "os ombros foram vistos"
    assert "azimute" in leitura.motivo


def test_azimute_nao_aprende_com_quem_esta_parado():
    """Perto de velocidade zero o vetor do Kalman e quase todo ruido, e a
    DIRECAO dele gira loucamente enquanto o modulo mal se mexe. Aprender ali
    seria aprender ruido e chamar de calibracao."""
    e = EstimadorDeAzimute(vel_minima=0.25)
    aceitas = sum(e.observar(0.5, 0.9, velocidade=0.05) for _ in range(50))

    assert aceitas == 0
    assert not e.confiavel


def test_azimute_recusa_amostra_absurda_depois_de_convergir():
    """Um quadro em que o MediaPipe troca os ombros de lado nao pode entrar na
    conta como se fosse medida boa."""
    e = EstimadorDeAzimute()
    for _ in range(40):
        e.observar(0.0, math.radians(30), velocidade=0.9)
    antes = e.valor

    for _ in range(10):
        e.observar(math.pi, math.radians(30), velocidade=0.9)   # 180 fora

    assert e.descartadas == 10
    assert abs(diferenca_angular(e.valor, antes)) < math.radians(1)


def test_azimute_atravessa_o_meridiano():
    """Camera montada perto de 180 graus e o caso que quebra media ingenua."""
    a = AnalisadorDeCorpo()
    convergir_azimute(a, offset_graus=179, rumo_mundo=0.0)

    assert a.azimute.confiavel
    erro = abs(diferenca_angular(a.azimute.valor, math.radians(179)))
    assert erro < math.radians(2), a.azimute.diagnostico


# ------------------------------------------------------------- rumo do corpo
def test_rumo_do_corpo_sai_no_referencial_do_MUNDO():
    a = AnalisadorDeCorpo()
    convergir_azimute(a, offset_graus=40, rumo_mundo=0.3)

    alvo = math.radians(70)
    leitura = a.ler(1, corpo(rumo_camera=alvo - math.radians(40)),
                    tudo_visivel(), rumo_mundo=alvo, velocidade=0.9)

    assert leitura.rumo_corpo is not None
    assert abs(diferenca_angular(leitura.rumo_corpo, alvo)) < math.radians(3)


def test_ombros_de_perfil_nao_produzem_rumo():
    """Com os ombros alinhados com a profundidade, a projecao horizontal some
    e o angulo passa a ser definido por ruido. Mesmo motivo geometrico pelo
    qual o giro exige velocidade: vetor curto tem angulo mal definido."""
    a = AnalisadorDeCorpo()
    convergir_azimute(a, offset_graus=0)

    de_perfil = corpo(largura_ombros=0.02)
    leitura = a.ler(1, de_perfil, tudo_visivel(), rumo_mundo=0.0,
                    velocidade=0.9)

    assert leitura.rumo_corpo is None
    assert leitura.rumo_corpo_camera is None
    assert "ombros" in leitura.motivo


def test_ombros_nao_vistos_nao_produzem_rumo():
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(), tudo_visivel(exceto=(5, 6)),
                    rumo_mundo=0.0, velocidade=0.9)

    assert leitura.rumo_corpo_camera is None
    assert leitura.braco_esquerdo == Braco.DESCONHECIDO


# ------------------------------------------------------- ALTURA DA MAO
def test_altura_da_mao_bate_com_a_verdade():
    """O numero que decide qual prateleira foi tocada."""
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(1.20, 0.30)), tudo_visivel())

    assert leitura.altura_mao_dir is not None
    assert abs(leitura.altura_mao_dir - 1.20) < 0.01, leitura.altura_mao_dir


def test_altura_sobrevive_a_camera_inclinada():
    """A correcao de inclinacao paga o proprio custo, ou nao serve.

    Este teste prova por CONTRASTE: com a correcao, 1 cm de erro; sem ela, o
    erro e grande o bastante para responder a prateleira errada. Foi essa
    diferenca — uma unica rotacao — que separou `ancorar_no_chao` de
    `para_o_mundo` e deitou o boneco por uma sessao inteira.
    """
    verdade = 1.20
    juntas_cam = inclinar(corpo(mao_dir=(verdade, 0.30)), graus=-30)

    com = AnalisadorDeCorpo().ler(
        1, juntas_cam, tudo_visivel(), inclinacao_rad=math.radians(-30))
    sem = AnalisadorDeCorpo().ler(
        1, juntas_cam, tudo_visivel(), inclinacao_rad=0.0)

    assert abs(com.altura_mao_dir - verdade) < 0.01, com.altura_mao_dir
    assert abs(sem.altura_mao_dir - verdade) > 0.10, (
        "sem a correcao o erro deveria ser grande; se nao for, o teste nao "
        "esta provando nada")


def test_altura_sobrevive_a_perna_sair_do_quadro():
    """O caso real: webcam que pega do peito para cima.

    O MediaPipe SEMPRE devolve os tornozelos, inclusive extrapolados. A altura
    do quadril foi MEDIDA enquanto as pernas estavam a vista e guardada; e ela
    que responde depois. Nao ha invencao neste caminho — ha memoria.
    """
    a = AnalisadorDeCorpo()
    for _ in range(10):                       # pernas a vista: aprende
        a.ler(1, corpo(mao_dir=(1.20, 0.30)), tudo_visivel())

    # As pernas saem do quadro. O corpo continua igual; SO os tornozelos
    # passam a ser extrapolados — e o valor que o MediaPipe inventa erra 40 cm.
    mentira = corpo(mao_dir=(1.20, 0.30))
    mentira[15, 2] = mentira[16, 2] = -0.55

    leitura = a.ler(1, mentira, tudo_visivel(exceto=(15, 16)))

    assert abs(leitura.altura_quadril - QUADRIL) < 0.01, (
        "usou o tornozelo inventado em vez do que foi medido antes")
    assert abs(leitura.altura_mao_dir - 1.20) < 0.02, leitura.altura_mao_dir


def test_sem_tornozelo_a_altura_e_ESTIMADA_e_marcada():
    """MEDIDO EM 11/08: nenhuma das duas cameras enxerga os pes — frontal 0%,
    lateral 0%. E nao e ajuste de enquadramento: uma webcam de ~60 graus a
    1,4 m cobre 1,6 m de altura, e uma pessoa em pe tem 1,75. Nao cabe.

    Recusar-se a responder deixaria a altura da mao — o numero que decide qual
    prateleira — sem resposta para sempre naquela sala. A saida nao e inventar
    um chao: e usar o TRONCO, que o MediaPipe mede em metros, com a proporcao
    antropometrica.

        Responder por modelo e legitimo. Responder por modelo sem dizer que e
        modelo e o defeito que este projeto inteiro combate.
    """
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(1.20, 0.30)),
                    tudo_visivel(exceto=(15, 16)))

    assert not leitura.altura_medida, "sem pe, nao pode dizer que mediu"
    assert leitura.altura_quadril is not None, "mas tem que responder"
    assert abs(leitura.altura_quadril - QUADRIL) < 0.08, (
        f"estimou {leitura.altura_quadril:.2f}, verdade {QUADRIL}")
    assert abs(leitura.altura_mao_dir - 1.20) < 0.10


def test_a_medicao_ganha_da_estimativa_sempre_que_existe():
    """Um unico quadro com pe a vista ja produz a mediana medida, e ela e
    melhor. A estimativa so entra quando NENHUM quadro teve tornozelo."""
    a = AnalisadorDeCorpo()

    estimada = a.ler(1, corpo(), tudo_visivel(exceto=(15, 16)))
    assert not estimada.altura_medida

    for _ in range(5):
        medida = a.ler(1, corpo(), tudo_visivel())
    assert medida.altura_medida
    assert abs(medida.altura_quadril - QUADRIL) < 0.01


def test_tronco_encurtado_nao_vira_estimativa():
    """Tronco curto e pessoa curvada ou reconstrucao ruim, e nos dois casos a
    proporcao nao vale."""
    a = AnalisadorDeCorpo()
    j = corpo()
    j[5][2] = j[6][2] = 0.10          # ombros quase na altura do quadril

    leitura = a.ler(1, j, tudo_visivel(exceto=(15, 16)))

    assert leitura.altura_quadril is None
    assert leitura.altura_mao_dir is None


def test_quadril_de_proporcao_impossivel_e_recusado():
    """Quadril a 2,4 m do chao nao e pessoa alta: e reconstrucao ruim.

    Vale para a medida E para a estimativa — um tronco absurdo produziria uma
    altura absurda, e a guarda tem que pegar as duas.
    """
    a = AnalisadorDeCorpo()
    gigante = corpo(quadril=2.40)
    gigante[5][2] = gigante[6][2] = 1.40      # tronco de 1,4 m

    leitura = a.ler(1, gigante, tudo_visivel())

    assert leitura.altura_quadril is None
    assert not leitura.altura_medida


def test_altura_impossivel_da_mao_e_recusada():
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(3.40, 0.0)), tudo_visivel())

    assert leitura.altura_mao_dir is None
    assert leitura.altura_quadril is not None, "o quadril continua medido"


def test_pulso_nao_visto_nao_vira_braco_nem_altura():
    """Desenhar o que nao foi visto e mentira com aparencia de dado."""
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(1.20, 0.30)), tudo_visivel(exceto=(10,)))

    assert leitura.braco_direito == Braco.DESCONHECIDO
    assert leitura.altura_mao_dir is None
    assert leitura.braco_esquerdo != Braco.DESCONHECIDO, "o outro lado e vivo"


def test_visibilidade_ausente_nao_conta_como_visto():
    """`None` significa 'sem informacao'. Sem informacao, nao se afirma."""
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(1.20, 0.30)), None)

    assert leitura.braco_direito == Braco.DESCONHECIDO
    assert leitura.rumo_corpo_camera is None


# ------------------------------------------------------------------- bracos
def test_os_tres_estados_do_braco():
    a = AnalisadorDeCorpo()
    v = tudo_visivel()

    ao_lado = a.ler(1, corpo(), v).braco_direito
    levantado = a.ler(1, corpo(mao_dir=(1.75, 0.10)), v).braco_direito
    estendido = a.ler(1, corpo(mao_dir=(OMBRO_DO_CHAO, 0.45)), v).braco_direito

    assert ao_lado == Braco.AO_LADO
    assert levantado == Braco.LEVANTADO
    assert estendido == Braco.ESTENDIDO


def test_levantado_ganha_de_estendido():
    """Quando os dois valem, a ALTURA e que responde qual prateleira."""
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(1.80, 0.50)), tudo_visivel())

    assert leitura.braco_direito == Braco.LEVANTADO


def test_bracos_sao_independentes():
    """Sem eixos independentes, o vocabulario precisaria de um item por
    combinacao e explodiria."""
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_esq=None, mao_dir=(1.75, 0.10)),
                    tudo_visivel())

    assert leitura.braco_esquerdo == Braco.AO_LADO
    assert leitura.braco_direito == Braco.LEVANTADO


def test_prateleira_baixa_aparece_na_altura_e_nao_no_rotulo():
    """Nao ha 'abaixado' no vocabulario, de proposito. Pegar embaixo e
    `ao_lado` com altura pequena — e a altura responde."""
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(0.35, 0.25)), tudo_visivel())

    assert leitura.braco_direito == Braco.AO_LADO
    assert abs(leitura.altura_mao_dir - 0.35) < 0.02


def test_o_estado_do_braco_nao_muda_quando_a_pessoa_gira():
    """Medir o pulso contra o OMBRO dela, e nao contra a imagem, e o que faz a
    resposta valer com a pessoa virada para qualquer lado."""
    a = AnalisadorDeCorpo()
    estados, alturas = set(), []

    for graus in range(0, 360, 30):
        leitura = a.ler(1, corpo(rumo_camera=math.radians(graus),
                                 mao_dir=(1.75, 0.10)), tudo_visivel())
        estados.add(leitura.braco_direito)
        alturas.append(leitura.altura_mao_dir)

    assert estados == {Braco.LEVANTADO}, estados
    assert max(alturas) - min(alturas) < 0.01


# ---------------------------------------------------------------- integracao
def test_andar_de_lado_deixa_de_ser_andar_para_frente():
    """O ganho central da etapa B, ponta a ponta.

    Ate hoje rumo e deslocamento eram a mesma coisa por construcao, e o
    sistema so sabia dizer `andando`. Com a linha dos ombros os dois se
    separam: aqui o corpo aponta para +x e a pessoa se desloca para +y.
    """
    a = AnalisadorDeCorpo()
    convergir_azimute(a, offset_graus=0, rumo_mundo=0.0)

    c = ClassificadorDeAcao(estabilidade_s=0.2)
    acao = None
    for _ in range(8):
        leitura = a.ler(1, corpo(rumo_camera=0.0), tudo_visivel(),
                        rumo_mundo=math.pi / 2, velocidade=0.8)
        pessoa = EstadoDePessoa(id=1, x=0, y=0, vy=0.8, rumo=math.pi / 2)
        acao, _ = c.classificar(pessoa, 0.1, leitura=leitura)

    assert acao.locomocao == Locomocao.ESQUERDA, (
        f"{acao.locomocao} — o corpo aponta para +x e ela anda para +y")


def test_sem_azimute_a_locomocao_degrada_para_andando():
    """Degradar para a resposta mais pobre e melhor que responder sem base."""
    a = AnalisadorDeCorpo()
    c = ClassificadorDeAcao(estabilidade_s=0.2)

    acao = None
    for _ in range(8):
        leitura = a.ler(1, corpo(), tudo_visivel(), rumo_mundo=0.0,
                        velocidade=0.0)
        acao, _ = c.classificar(
            EstadoDePessoa(id=1, x=0, y=0, vx=0.8, rumo=0.0), 0.1,
            leitura=leitura)

    assert acao.locomocao == Locomocao.ANDANDO
    assert acao.braco_direito == Braco.AO_LADO, "os bracos respondem mesmo assim"


def test_braco_nao_estrobla_ao_cruzar_o_limiar():
    """O pulso passa EXATAMENTE pela altura do ombro durante a subida. Sem
    histerese temporal, um unico gesto de pegar produto geraria uma dezena de
    BRACO_MUDOU e o painel viraria ruido."""
    a = AnalisadorDeCorpo()
    c = ClassificadorDeAcao(estabilidade_braco_s=0.2)
    rng = np.random.default_rng(7)

    mudancas = 0
    for _ in range(40):
        altura = OMBRO_DO_CHAO + 0.10 + rng.normal(0, 0.02)   # tremendo no limiar
        leitura = a.ler(1, corpo(mao_dir=(altura, 0.0)), tudo_visivel())
        _, m = c.classificar(EstadoDePessoa(id=1, x=0, y=0), 0.05,
                             leitura=leitura)
        mudancas += int(m["braco_direito"])

    assert mudancas <= 2, f"{mudancas} mudancas de braco em um gesto so"


def test_o_gesto_de_pegar_produz_a_sequencia_certa():
    """Levantar e baixar = tres mudancas, nao vinte. E na ordem certa.

    Contar mudancas nao basta: tres eventos na ordem errada passariam num
    teste de contagem e produziriam um boneco fazendo o gesto invertido. A
    SEQUENCIA e o que precisa ser travado.

    E a primeira mudanca e `desconhecido -> ao_lado`, que nao e ruido: e o
    sistema anunciando que passou a enxergar aquele braco. Quem consome o
    fluxo precisa desse marco para saber a partir de quando os proximos
    eventos daquele lado significam alguma coisa.
    """
    a = AnalisadorDeCorpo()
    c = ClassificadorDeAcao(estabilidade_braco_s=0.2)
    v = tudo_visivel()

    sequencia = []
    for altura in [QUADRIL - 0.05] * 10 + [1.75] * 10 + [QUADRIL - 0.05] * 10:
        leitura = a.ler(1, corpo(mao_dir=(altura, 0.0)), v)
        acao, m = c.classificar(EstadoDePessoa(id=1, x=0, y=0), 0.05,
                                leitura=leitura)
        if m["braco_direito"]:
            sequencia.append(acao.braco_direito)

    assert sequencia == [Braco.AO_LADO, Braco.LEVANTADO, Braco.AO_LADO], (
        sequencia)


def test_memoria_de_quadril_some_com_o_rastro():
    """Altura aprendida da pessoa 1 nao pode responder pela pessoa 2."""
    a = AnalisadorDeCorpo()
    for _ in range(5):
        a.ler(1, corpo(), tudo_visivel())
    assert 1 in a._quadris

    a.esquecer({2})
    assert 1 not in a._quadris


def test_custo_e_ruido_diante_do_detector():
    """156 ms do detector contra o que esta camada custa.

    O numero exato varia com a maquina; a ordem de grandeza e o que importa.
    Se um dia isto falhar, alguem pos trabalho de verdade aqui dentro — e o
    lugar de trabalho de verdade e o detector, nao a descricao.
    """
    import time

    a = AnalisadorDeCorpo()
    juntas, v = corpo(mao_dir=(1.20, 0.30)), tudo_visivel()

    t = time.perf_counter()
    for _ in range(1000):
        a.ler(1, juntas, v, rumo_mundo=0.3, velocidade=0.9)
    ms = (time.perf_counter() - t)

    assert ms < 1.0, f"{ms * 1000:.0f} us por leitura — esperado dezenas"


# ------------------------------------------------------------------ postura
def test_agachar_e_visto_pela_altura_do_quadril():
    """O defeito de 11/08: `agachar` lido como `em_pe` em 100% dos quadros.

    A postura vinha da altura da CAIXA da camera do alto. Uma camera olhando
    de cima quase nao ve mudanca de estatura — a caixa naquela vista e
    dominada pela pegada da pessoa no chao, nao pela altura dela.

    O sinal estava errado na ORIGEM. Nenhum ajuste de limiar corrigiria, e
    mexer no limiar teria sido a terceira rodada de ajuste as cegas deste
    projeto.
    """
    a = AnalisadorDeCorpo()
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    v = tudo_visivel()
    p = EstadoDePessoa(id=1, x=0, y=0)

    for _ in range(30):                       # em pe: aprende o padrao
        acao, _ = c.classificar(p, 0.1, leitura=a.ler(1, corpo(), v))
    assert acao.postura == Postura.EM_PE, acao.postura

    for _ in range(20):                       # agacha de verdade: coxa dobrada
        acao, _ = c.classificar(
            p, 0.1, leitura=a.ler(1, corpo(quadril=0.50, agachado=True), v))

    assert acao.postura == Postura.AGACHADO, acao.postura


def test_levantar_devolve_em_pe():
    """Um estado que so entra e nunca sai nao e estado, e armadilha."""
    a = AnalisadorDeCorpo()
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    v = tudo_visivel()
    p = EstadoDePessoa(id=1, x=0, y=0)

    for _ in range(30):
        c.classificar(p, 0.1, leitura=a.ler(1, corpo(), v))
    for _ in range(20):
        c.classificar(p, 0.1,
                      leitura=a.ler(1, corpo(quadril=0.50, agachado=True), v))
    for _ in range(20):
        acao, _ = c.classificar(p, 0.1, leitura=a.ler(1, corpo(), v))

    assert acao.postura == Postura.EM_PE


def test_a_mediana_do_quadril_nao_sente_o_agachamento():
    """A referencia da ALTURA DA MAO tem que ignorar o agachamento; a POSTURA
    tem que senti-lo. Mesmo dado, duas leituras — e e por isso que existem
    dois campos em vez de um."""
    a = AnalisadorDeCorpo()
    v = tudo_visivel()

    for _ in range(40):
        a.ler(1, corpo(), v)
    leitura = a.ler(1, corpo(quadril=0.50, agachado=True), v)

    assert abs(leitura.altura_quadril - QUADRIL) < 0.05, "a mediana cedeu"
    assert abs(leitura.altura_quadril_agora - 0.50) < 0.01
    assert leitura.encolhimento < 0.6


def test_sem_pose_frontal_a_postura_volta_para_a_caixa():
    """A fonte antiga nao foi removida: ela nao esta errada, esta cega para
    ESTE movimento nesta montagem. Sem quadril, ela e melhor que nada."""
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    p = EstadoDePessoa(id=1, x=0, y=0)

    for _ in range(20):
        acao, _ = c.classificar(p, 0.1, razao_altura=0.15, k_referencia=0.30)

    assert acao.postura == Postura.AGACHADO


def test_azimute_acusa_amostras_bimodais():
    """O caso real de 11/08: tres execucoes, tres valores, mesma camera.

    +7, +85 e -123 graus com concentracao entre 35% e 83%. Nao era ruido: era
    ir e voltar no mesmo eixo SEM VIRAR O CORPO. Metade das amostras dizia
    "corpo aponta para X, anda para +X"; a outra metade, "-X". Dois grupos a
    180 graus, e a media circular deles cai onde o acaso da contagem mandar.

        Abster-se sem explicar transfere o problema para quem le.
    """
    e = EstimadorDeAzimute()
    for i in range(40):
        # anda para frente e para tras, sempre de frente para a camera
        rumo_mundo = 0.0 if i % 2 else math.pi
        e.observar(0.0, rumo_mundo, velocidade=0.9)

    assert not e.confiavel, "com dois grupos opostos ele nao pode responder"
    assert e.bimodal, e.diagnostico
    assert "dois grupos opostos" in e.diagnostico


def test_amostras_coerentes_nao_sao_bimodais():
    """O aviso so vale se ele ficar calado quando nao ha o que avisar."""
    e = EstimadorDeAzimute()
    for _ in range(40):
        e.observar(0.0, math.radians(30), velocidade=0.9)

    assert e.confiavel
    assert not e.bimodal
    assert "grupos opostos" not in e.diagnostico


def test_bagunca_pura_nao_e_confundida_com_bimodal():
    """Espalhado em todas as direcoes e outro diagnostico: nao ha conserto de
    postura da pessoa, ha dado ruim. Chamar isso de bimodal mandaria virar o
    corpo quando o problema e outro."""
    import random

    random.seed(3)
    e = EstimadorDeAzimute()
    for _ in range(60):
        e.observar(random.uniform(-math.pi, math.pi), 0.0, velocidade=0.9)

    assert not e.confiavel
    assert not e.bimodal, e.diagnostico


def test_azimute_sobrevive_a_uma_minoria_andando_de_re():
    """O ganho que permitiu o roteiro do Eduardo existir.

    O roteiro dele tem `VOLTE DE RE` e `ANDE DE LADO` — passos legitimos, que
    produzem amostras 180 e 90 graus fora. Com MEDIA, elas puxavam a resposta
    para um ponto entre os grupos onde nao havia amostra nenhuma; foi o que
    deu +7, +85 e -123 graus em tres execucoes com a camera parada.

    Com MODA, a minoria e descartada por ser minoria.

        A media de dois grupos separados aponta para um lugar vazio entre eles.
    """
    e = EstimadorDeAzimute()
    verdade = math.radians(35)

    for i in range(60):
        if i % 4 == 3:                       # 25% andando de re
            e.observar(0.0, verdade + math.pi, velocidade=0.9)
        else:
            e.observar(0.0, verdade, velocidade=0.9)

    assert e.confiavel, e.diagnostico
    assert abs(diferenca_angular(e.valor, verdade)) < math.radians(3), \
        e.diagnostico


def test_a_moda_nao_e_recalculada_a_cada_amostra():
    """O(n^2) com 240 amostras sao 57 mil comparacoes por quadro. A camada de
    acao existe por nao custar nada; recalcular sempre custaria seis vezes
    mais que ela inteira."""
    e = EstimadorDeAzimute()
    for _ in range(40):
        e.observar(0.0, 0.5, velocidade=0.9)

    antes = e._desde_o_calculo
    e.observar(0.0, 0.5, velocidade=0.9)

    assert e._desde_o_calculo == antes + 1, "recalculou na hora"
    assert e.confiavel, "e mesmo assim continua respondendo"


def test_agachar_e_visto_SEM_o_tornozelo_no_quadro():
    """O caso real de 11/08, e a razao da coxa ter ganhado das outras fontes.

    `agachar` NUNCA foi reconhecido, com 36% dos quadros sem leitura nenhuma.
    Quem agacha tira as pernas do enquadramento — e as duas fontes anteriores
    dependiam justamente do que sumia:

        altura da CAIXA      de cima quase nao se ve mudanca de estatura
        altura do QUADRIL    precisa do tornozelo para saber onde e o chao

    A coxa e uma razao entre duas juntas VIZINHAS. Se o joelho aparece, o
    quadril aparece.

        Um sinal que precisa do chao morre quando o chao sai do quadro.
    """
    a = AnalisadorDeCorpo()
    c = ClassificadorDeAcao(estabilidade_s=0.2)
    sem_pe = tudo_visivel(exceto=(15, 16))
    p = EstadoDePessoa(id=1, x=0, y=0)

    for _ in range(20):
        acao, _ = c.classificar(p, 0.1, leitura=a.ler(1, corpo(), sem_pe))
    assert acao.postura == Postura.EM_PE, acao.postura

    for _ in range(20):
        acao, _ = c.classificar(
            p, 0.1, leitura=a.ler(1, corpo(agachado=True), sem_pe))

    assert acao.postura == Postura.AGACHADO, acao.postura


def test_a_coxa_nao_depende_do_tamanho_da_pessoa():
    """Razao adimensional: nao precisa de metros, de calibracao nem de saber a
    altura. Funciona igual para adulto e para crianca."""
    a = AnalisadorDeCorpo()
    v = tudo_visivel()

    adulto = a.ler(1, corpo(quadril=0.95), v).verticalidade_coxa
    crianca = a.ler(2, corpo(quadril=0.55), v).verticalidade_coxa

    assert abs(adulto - crianca) < 0.01
    assert adulto > 0.95, "em pe a coxa e vertical"


def test_a_coxa_sobrevive_a_camera_inclinada():
    """Sem desfazer a inclinacao, a coxa de quem esta em pe apareceria dobrada
    — e o sistema anunciaria agachamento em toda sessao com a lente torta."""
    a = AnalisadorDeCorpo()
    juntas = inclinar(corpo(), graus=-35)

    com = a.ler(1, juntas, tudo_visivel(), inclinacao_rad=math.radians(-35))
    sem = AnalisadorDeCorpo().ler(1, juntas, tudo_visivel())

    assert com.verticalidade_coxa > 0.95
    assert sem.verticalidade_coxa < 0.90, (
        "sem a correcao a coxa deveria parecer dobrada; se nao parecer, "
        "este teste nao esta provando nada")


def test_fica_com_a_perna_mais_ESTICADA():
    """Quem se abaixa para pegar algo frequentemente estende uma perna. Ficar
    com a mais dobrada faria qualquer passada larga virar agachamento."""
    a = AnalisadorDeCorpo()
    j = corpo()
    j[13] = j[11] + np.array([0.40, 0, -0.05])   # esquerda dobrada
    j[14] = j[12] + np.array([0, 0, -0.45])      # direita esticada

    leitura = a.ler(1, j, tudo_visivel())

    assert leitura.verticalidade_coxa > 0.95, leitura.verticalidade_coxa


def test_sem_joelho_visivel_cai_na_altura_do_quadril():
    """A fonte antiga nao foi removida. Nenhuma vista ve tudo sempre."""
    a = AnalisadorDeCorpo()
    sem_joelho = tudo_visivel(exceto=(13, 14))

    for _ in range(20):
        a.ler(1, corpo(), sem_joelho)
    leitura = a.ler(1, corpo(quadril=0.50, agachado=True), sem_joelho)

    assert leitura.verticalidade_coxa is None
    assert leitura.encolhimento is not None, "a reserva tem que responder"
    assert leitura.encolhimento < 0.7


# --------------------------------------------------- varias vistas
def test_o_braco_vem_da_vista_que_ENXERGOU_o_pulso():
    """O defeito de 11/08, e o mais caro dos tres.

    Levantar o braco levava 9 a 10 s e lia `ao_lado` em 65 a 87% dos quadros;
    BAIXAR levava 2 s. Assimetria e assinatura de mao saindo do quadro: a
    webcam do notebook pega do peito para cima, o pulso levantado sobe alem da
    borda, e o MediaPipe extrapola para baixo.

    A lateral entregava 100% de pose e nunca era consultada, porque a escolha
    da vista era uma ORDEM FIXA.

        A pergunta certa nao e "qual camera e melhor", e "qual delas viu ESTA
        junta".
    """
    a = AnalisadorDeCorpo()
    levantado = corpo(mao_dir=(1.75, 0.10))

    frontal = (levantado, tudo_visivel(exceto=(10,)))   # perdeu o pulso direito
    lateral = (levantado, tudo_visivel())               # viu tudo

    so_frontal = AnalisadorDeCorpo().ler(1, *frontal)
    assert so_frontal.braco_direito == Braco.DESCONHECIDO

    junto = a.ler_varias(1, [frontal, lateral])
    assert junto.braco_direito == Braco.LEVANTADO, junto.braco_direito


def test_a_altura_viaja_junto_com_o_braco_que_a_sustenta():
    """Separar os dois deixaria o estado vindo de uma camera e a altura de
    outra — e e a altura que sustenta o estado."""
    a = AnalisadorDeCorpo()
    for _ in range(10):                       # aprende o quadril nas duas
        a.ler_varias(1, [(corpo(), tudo_visivel()),
                         (corpo(), tudo_visivel())])

    # As duas vistas veem a mao em alturas diferentes — na pratica isso e erro
    # de reconstrucao, e aqui e proposital: se a altura publicada for a da
    # frontal enquanto o ESTADO veio da lateral, o teste acusa.
    frontal = (corpo(mao_dir=(1.90, 0.10)), tudo_visivel(exceto=(10,)))
    lateral = (corpo(mao_dir=(1.70, 0.10)), tudo_visivel())

    junto = a.ler_varias(1, [frontal, lateral])

    assert junto.braco_direito == Braco.LEVANTADO
    assert abs(junto.altura_mao_dir - 1.70) < 0.03, (
        "a altura veio de uma vista e o estado de outra")


def test_uma_vista_cega_nao_apaga_a_outra():
    a = AnalisadorDeCorpo()
    boa = (corpo(mao_esq=(1.75, 0.1)), tudo_visivel())
    cega = (None, None)

    junto = a.ler_varias(1, [cega, boa])

    assert junto.braco_esquerdo == Braco.LEVANTADO


def test_o_azimute_aprende_de_UMA_vista_so():
    """Ele mede o giro DAQUELA lente. Alimenta-lo com duas cameras misturaria
    duas constantes distintas — o caso bimodal que ele existe para recusar."""
    a = AnalisadorDeCorpo()
    verdade = math.radians(35)
    frontal = (corpo(rumo_camera=-verdade), tudo_visivel())
    lateral = (corpo(rumo_camera=-verdade + math.pi / 2), tudo_visivel())

    for _ in range(40):
        a.ler_varias(1, [frontal, lateral], rumo_mundo=0.0, velocidade=0.9)

    assert a.azimute.confiavel, a.azimute.diagnostico
    assert abs(diferenca_angular(a.azimute.valor, verdade)) < math.radians(3)
    assert len(a.azimute.amostras) == 40, "a segunda vista tambem alimentou"


# ------------------------------------------------ direcao por deslocamento
def test_ruido_parado_nao_produz_direcao():
    """MEDIDO EM 11/08: `parado` teve pico de 0,23 m/s so de tremor do Kalman —
    exatamente a mediana da caminhada real do Eduardo naquele espaco.

    Em velocidade os dois empatam. Em deslocamento nao: o ruido e centrado em
    zero e se CANCELA ao longo da janela; a caminhada se ACUMULA.

        Ruido nao vai a lugar nenhum. Caminhada vai.
    """
    import random

    from src.acao.corpo import DirecaoPorDeslocamento

    random.seed(5)
    d = DirecaoPorDeslocamento(janela_s=2.0, deslocamento_minimo=0.25)

    rumo = None
    for i in range(30):
        # tremendo em torno de um ponto, com passos do tamanho de 0,23 m/s
        x = random.gauss(0, 0.03)
        y = random.gauss(0, 0.03)
        rumo, desloc = d.observar(1, x, y, i * 0.1)

    assert rumo is None, "tremor virou direcao"
    assert desloc < 0.25


def test_caminhada_curta_produz_direcao_bem_definida():
    """0,68 m em 4 s foi o que o Eduardo andou. Nao passa do limiar de
    velocidade e passa com folga no de deslocamento."""
    import math

    from src.acao.corpo import DirecaoPorDeslocamento

    d = DirecaoPorDeslocamento(janela_s=2.0, deslocamento_minimo=0.25)
    alvo = math.radians(40)

    rumo = None
    for i in range(20):
        p = i * 0.023                       # 0,23 m/s a 10 fps
        rumo, desloc = d.observar(1, p * math.cos(alvo), p * math.sin(alvo),
                                  i * 0.1)

    assert rumo is not None, "caminhada real foi recusada"
    assert abs(diferenca_angular(rumo, alvo)) < math.radians(3)


def test_o_azimute_converge_com_caminhada_lenta():
    """O caso que falhou no hardware: caminhada abaixo do limiar de velocidade.

    Com deslocamento no lugar da velocidade instantanea, o azimute recebe
    amostras de toda a caminhada em vez dos poucos picos de ruido que passavam.
    """
    import math

    from src.acao.corpo import DirecaoPorDeslocamento

    a = AnalisadorDeCorpo()
    d = DirecaoPorDeslocamento(janela_s=2.0, deslocamento_minimo=0.25)
    offset = math.radians(35)
    rumo_real = math.radians(20)

    for i in range(60):
        p = i * 0.023                       # devagar: 0,23 m/s
        rumo, andou = d.observar(1, p * math.cos(rumo_real),
                                 p * math.sin(rumo_real), i * 0.1)
        a.ler(1, corpo(rumo_camera=rumo_real - offset), tudo_visivel(),
              rumo_mundo=rumo, velocidade=andou)

    assert a.azimute.confiavel, a.azimute.diagnostico
    assert abs(diferenca_angular(a.azimute.valor, offset)) < math.radians(3)


def test_a_trilha_some_com_o_rastro():
    from src.acao.corpo import DirecaoPorDeslocamento

    d = DirecaoPorDeslocamento()
    for i in range(10):
        d.observar(1, i * 0.05, 0, i * 0.1)

    d.esquecer({2})
    assert 1 not in d._trilhas


# ------------------------------------------------- azimute calibrado
def test_calibrado_manda_no_aprendido():
    """MEDIDO EM 11/08: o automatico convergiu para o grupo ERRADO — 55% de
    maioria numa hipotese falsa. `andar_frente` saiu como `andando_tras`.

    A hipotese "quem anda olha para onde vai" e falsa numa sala de 1,4 m
    diante de um computador: a pessoa se desloca olhando para a tela.

        Aumentar a amostra de uma hipotese falsa nao a torna verdadeira;
        torna o erro confiante.

    Misturar o calibrado com o aprendido daria um terceiro numero pior que a
    medida honesta, e ainda esconderia qual dos dois estava errado.
    """
    e = EstimadorDeAzimute()
    for _ in range(60):                      # aprende algo errado, com folga
        e.observar(0.0, math.pi, velocidade=0.9)
    assert e.confiavel and abs(e.valor - math.pi) < 0.1

    e.calibrado = math.radians(35)

    assert abs(diferenca_angular(e.offset, math.radians(35))) < 1e-9
    mundo = e.para_o_mundo(0.0)
    assert abs(diferenca_angular(mundo, math.radians(35))) < 1e-9


def test_calibrado_responde_mesmo_sem_amostra_nenhuma():
    """Uma medida deliberada nao precisa esperar aprendizado para valer."""
    e = EstimadorDeAzimute()
    e.calibrado = math.radians(-20)

    assert e.confiavel, "calibrado tem que responder na hora"
    assert "CALIBRADO" in e.diagnostico


def test_o_painel_mostra_os_dois_e_acusa_discordancia():
    """Se o calibrado e o automatico discordarem muito depois de gravado, ou a
    camera foi movida ou o ambiente mudou. E o unico jeito de a calibracao ser
    questionada depois de virar arquivo."""
    e = EstimadorDeAzimute()
    for _ in range(60):
        e.observar(0.0, math.radians(150), velocidade=0.9)
    e.calibrado = math.radians(20)

    d = e.diagnostico
    assert "CALIBRADO" in d
    assert "aprenderia" in d
    assert "camera foi movida" in d


def test_sem_discordancia_o_painel_nao_alarma():
    """O aviso so vale se ficar calado quando os dois concordam."""
    e = EstimadorDeAzimute()
    for _ in range(60):
        e.observar(0.0, math.radians(30), velocidade=0.9)
    e.calibrado = math.radians(32)

    assert "camera foi movida" not in e.diagnostico


# ------------------------------------------- rumo do corpo pela camera do alto
def _ombros_no_chao(rumo_mundo, x=1.0, y=1.0, largura=0.38, px_por_m=100.0):
    """Juntas 2D como a camera do alto veria: ombros deitados no chao."""
    lado = np.array([-math.sin(rumo_mundo), math.cos(rumo_mundo)])
    centro = np.array([x, y])
    esq = (centro + lado * largura / 2) * px_por_m
    dir_ = (centro - lado * largura / 2) * px_por_m

    j = np.tile(centro * px_por_m, (17, 1)).astype(float)
    j[5], j[6] = esq, dir_
    return j


def test_o_rumo_do_corpo_sai_direto_da_camera_do_alto():
    """A simplificacao que apaga o problema inteiro.

    O caminho pela frontal precisava do azimute — e ele nao convergiu por dois
    caminhos independentes em 11/08: aprendido sozinho caiu no grupo errado
    (180 graus fora), e calibrado a mao teve tres travessias discordando em
    105 e 148 graus.

    A causa da segunda e geometrica: no referencial da lente, a linha dos
    ombros de quem esta DE PERFIL deita sobre o eixo de PROFUNDIDADE, que e o
    mais fraco do MediaPipe. O rumo virava ruido justamente no caso que se
    queria medir.

    De cima, a linha dos ombros esta DEITADA NO PLANO DO CHAO — o mesmo plano
    que a homografia converte em metros desde o bloco 1.

        Quando uma constante nao converge por dois caminhos diferentes, vale
        perguntar se ela precisa existir.
    """
    from percepcao.chao import para_metros
    from src.acao.corpo import rumo_do_alto

    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])

    for graus in range(0, 360, 30):
        alvo = math.radians(graus)
        j = _ombros_no_chao(alvo)
        lido = rumo_do_alto(j, tudo_visivel(), para_metros, H)

        assert lido is not None, graus
        assert abs(diferenca_angular(lido, alvo)) < math.radians(2), (
            f"pedido {graus}, lido {math.degrees(lido):.0f}")


def test_sem_ombros_no_alto_nao_ha_rumo():
    from percepcao.chao import para_metros
    from src.acao.corpo import rumo_do_alto

    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])
    j = _ombros_no_chao(0.0)

    assert rumo_do_alto(j, tudo_visivel(exceto=(5, 6)), para_metros, H) is None
    assert rumo_do_alto(None, tudo_visivel(), para_metros, H) is None


def test_ombros_colados_no_alto_nao_produzem_rumo():
    """Largura no chao nao pode encolher, so girar. Ombros colados sao
    reconstrucao ruim — e vetor curto tem angulo mal definido."""
    from percepcao.chao import para_metros
    from src.acao.corpo import rumo_do_alto

    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])
    j = _ombros_no_chao(0.0, largura=0.02)

    assert rumo_do_alto(j, tudo_visivel(), para_metros, H) is None


def test_o_alto_manda_no_rumo_do_corpo():
    """A frontal continua como reserva, para quando o alto perder os ombros —
    mas quando os dois respondem, quem manda e quem nao precisa de constante
    nenhuma."""
    a = AnalisadorDeCorpo()
    convergir_azimute(a, offset_graus=40)          # a frontal aprendeu algo

    do_alto = math.radians(-77)
    leitura = a.ler_varias(1, [(corpo(), tudo_visivel())],
                           rumo_do_alto=do_alto)

    assert abs(diferenca_angular(leitura.rumo_corpo, do_alto)) < 1e-9


# ------------------------------------------------------ o sinal do rumo
def test_o_sinal_aprende_que_o_rumo_esta_invertido():
    """MEDIDO EM 11/08: com o rumo vindo do alto, quem andava para frente saiu
    como `andando_tras` — 180 graus exatos, nao ruido.

    A formula `frente = (dy, -dx)` supoe um sistema de coordenadas DESTRO. Se
    a calibracao da homografia produziu um canhoto, o ombro esquerdo aparece
    onde eu espero o direito e o rumo sai virado.

        Deduzi uma convencao que so o dado pode responder.
    """
    from src.acao.corpo import SinalDoRumo

    s = SinalDoRumo()
    for i in range(20):
        andado = math.radians(30 + i)          # a pessoa anda para la
        lido = diferenca_angular(andado + math.pi, 0)   # e o rumo sai virado
        s.votar(lido, andado)

    assert s.decidido
    assert s.sinal == -1, s.diagnostico
    assert "INVERTIDO" in s.diagnostico

    corrigido = s.aplicar(math.radians(200))
    assert abs(diferenca_angular(corrigido, math.radians(20))) < 1e-9


def test_o_sinal_confirma_quando_esta_certo():
    from src.acao.corpo import SinalDoRumo

    s = SinalDoRumo()
    for i in range(20):
        a = math.radians(10 * i)
        s.votar(a, a)

    assert s.decidido and s.sinal == 1
    assert s.aplicar(0.7) == 0.7


def test_um_BIT_sobrevive_a_ruido_que_derruba_um_angulo():
    """A razao de a pergunta ter virado binaria.

    Todo o esforco anterior tentou aprender o AZIMUTE, um numero continuo. Com
    ruido, a moda escolheu o grupo errado e o dia inteiro se foi nisso.

    Aqui cada amostra so precisa acertar de que LADO esta. Com 30% das
    amostras completamente erradas, a maioria ainda decide certo.

        Uma pergunta binaria sobrevive a um dado que nao sustenta uma
        pergunta continua.
    """
    import random

    from src.acao.corpo import SinalDoRumo

    random.seed(9)
    s = SinalDoRumo()
    for i in range(60):
        andado = random.uniform(-math.pi, math.pi)
        if i % 10 < 3:                          # 30% de lixo total
            s.votar(random.uniform(-math.pi, math.pi), andado)
        else:
            s.votar(diferenca_angular(andado + math.pi, 0), andado)

    assert s.decidido and s.sinal == -1, s.diagnostico


def test_sem_maioria_ele_usa_o_direto_e_avisa():
    """Metade e metade nao decide nada. Usar o palpite original e melhor que
    nao responder: a votacao SO acontece com alguem andando, que e o unico
    momento em que a resposta importa."""
    from src.acao.corpo import SinalDoRumo

    s = SinalDoRumo()
    for i in range(20):
        a = math.radians(10 * i)
        s.votar(a if i % 2 else diferenca_angular(a + math.pi, 0), a)

    assert not s.decidido
    assert s.sinal == 1
    assert "sem maioria" in s.diagnostico


def test_nao_vota_sem_caminhada():
    """`rumo_andado` so existe quando houve deslocamento na janela. Sem ele,
    nao ha com o que comparar — e votar seria inventar."""
    from src.acao.corpo import SinalDoRumo

    s = SinalDoRumo()
    for _ in range(30):
        s.votar(0.5, None)
        s.votar(None, 0.5)

    assert s.total == 0
    assert not s.decidido


def test_a_faixa_ambigua_e_descartada():
    """Quem anda DE LADO tem o corpo perpendicular ao deslocamento. Esse quadro
    nao responde nem `certo` nem `invertido` — contar como voto so injeta ruido
    numa decisao binaria que deveria ser facil.

    PROVA NOS DADOS QUE JA EXISTIAM: as duas calibracoes de azimute que
    falharam davam -175, -138, -70 e -165, +78, -134. Como angulo, discordancia
    de 105 e 148 graus. Como bit, 2 x 1 nas duas — e o voto discordante de cada
    uma e justamente o que cai perto de 90 graus.
    """
    from src.acao.corpo import SinalDoRumo

    s = SinalDoRumo(minimo_votos=1)
    for graus in (-175.5, -138.3, -70.3):
        s.votar(math.radians(graus), 0.0)

    assert s.ignorados == 1, "o -70 deveria cair na faixa cega"
    assert (s.certo, s.invertido) == (0, 2), "e os outros dois, unanimes"
    assert s.sinal == -1


def test_o_bit_fixado_manda_no_aprendido():
    """Uma sessao boa resolve para sempre. O aprendizado continua rodando ao
    lado justamente para poder DISCORDAR do arquivo."""
    from src.acao.corpo import SinalDoRumo

    s = SinalDoRumo(fixado=-1)
    for i in range(30):                       # aprende o contrario
        a = math.radians(10 * i)
        s.votar(a, a)

    assert s.sinal == -1, "o arquivo tem que mandar"
    assert "FIXADO" in s.diagnostico
    assert "DISCORDAM" in s.diagnostico, "e a discordancia tem que aparecer"


def test_bit_fixado_sem_discordancia_nao_alarma():
    from src.acao.corpo import SinalDoRumo

    s = SinalDoRumo(fixado=1)
    for i in range(30):
        a = math.radians(10 * i)
        s.votar(a, a)

    assert s.sinal == 1
    assert "DISCORDAM" not in s.diagnostico
