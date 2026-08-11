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
          mao_esq=None, mao_dir=None, largura_ombros=LARGURA):
    """Monta um esqueleto COCO-17 de medidas conhecidas.

    `rumo_camera`  para onde o corpo aponta, no referencial da lente.
                   -pi/2 e de frente para a camera.
    `mao_*`        (altura_do_chao_em_metros, avanco_em_metros).
                   None = braco pendurado ao lado.

    Devolve (17,3) com origem no quadril e z para cima — que e exatamente o
    que o MediaPipe entrega DEPOIS de a inclinacao da lente ser desfeita.
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
    j[13] = meia * 0.5 + [0, 0, -quadril / 2]                  # joelhos
    j[14] = -meia * 0.5 + [0, 0, -quadril / 2]
    j[15] = meia * 0.5 + [0, 0, -quadril]                      # tornozelos
    j[16] = -meia * 0.5 + [0, 0, -quadril]
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


def test_sem_nunca_ter_visto_tornozelo_nao_ha_altura():
    """Sem base, nao se responde. `None` nao e falta de implementacao."""
    a = AnalisadorDeCorpo()
    leitura = a.ler(1, corpo(mao_dir=(1.20, 0.30)),
                    tudo_visivel(exceto=(15, 16)))

    assert leitura.altura_quadril is None
    assert leitura.altura_mao_dir is None
    assert leitura.braco_direito != Braco.DESCONHECIDO, (
        "o ESTADO do braco nao depende do tornozelo; so a altura depende")


def test_quadril_de_proporcao_impossivel_e_recusado():
    """Quadril a 2,4 m do chao nao e pessoa alta: e reconstrucao ruim."""
    a = AnalisadorDeCorpo()
    a.ler(1, corpo(quadril=2.40), tudo_visivel())

    assert a.ler(1, corpo(quadril=2.40), tudo_visivel()).altura_quadril is None


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

    for _ in range(20):                       # agacha: quadril a 0,50 m
        acao, _ = c.classificar(
            p, 0.1, leitura=a.ler(1, corpo(quadril=0.50), v))

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
        c.classificar(p, 0.1, leitura=a.ler(1, corpo(quadril=0.50), v))
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
    leitura = a.ler(1, corpo(quadril=0.50), v)

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
