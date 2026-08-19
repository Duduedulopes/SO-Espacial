"""O filtro tem que tirar o tremelique E nao deixar o boneco para tras.

    o gemeo digital nao funciona totalmente certo, precisamos pensar em como
    melhorar o movimento e o acerto de movimento    — Eduardo, 19/08

Ate 19/08 este filtro era media exponencial pura: alisava o ruido e, no mesmo
gesto, deixava o boneco atras de quem anda. Medido, com alfa = 0,25:

    a 6,5 fps e 1,20 m/s        42 cm de atraso permanente

Nao era ma afinacao. Media exponencial NAO ALCANCA um alvo em movimento — o
erro em regime e (1-alfa)/alfa * v * dt, e so vai a zero quando v vai.

    Um filtro que so puxa a saida na direcao da entrada nunca alcanca uma
    entrada que anda. Ele nao esta mal afinado: esta resolvendo outro
    problema.

Estes testes exercitam o filtro com RELOGIO EXPLICITO. Sem isso o resultado
dependeria da velocidade da maquina que roda a suite — e um teste que muda de
resposta conforme o computador nao mede nada.
"""
import math

import pytest

from src.espacial.motor import SpatialEngine
from src.gemeo.suave import Suavizador, _alfa, _mistura_angulo


def _andando(s, v, fps, segundos=8.0, pid=1, idade_s=0.0):
    """Alguem caminhando em linha reta. Devolve (atraso_em_metros, saida)."""
    dt = 1.0 / fps
    t, x = 0.0, 0.0
    saida = 0.0
    s.suavizar(pid, x, 0.0, vx=v, vy=0.0, idade_s=idade_s, agora=t)
    for _ in range(int(segundos * fps)):
        t += dt
        x += v * dt
        # a camera viu `idade_s` atras: a medida entregue e a posicao de la
        medida = x - v * idade_s
        saida, _, _ = s.suavizar(pid, medida, 0.0, vx=v, vy=0.0,
                                 idade_s=idade_s, agora=t)
    return x - saida, saida


# --------------------------------------------------------- o defeito central
@pytest.mark.parametrize("v", [0.20, 0.39, 0.80, 1.20])
@pytest.mark.parametrize("fps", [6.5, 10.0, 15.0])
def test_o_boneco_nao_fica_atras_de_quem_anda(v, fps):
    """O teste que a versao anterior falhava em todas as combinacoes.

    Nao e zero, e nao deveria ser: abaixo de 0,25 m/s a previsao entra
    encolhida de proposito, porque ali a velocidade medida e quase toda ruido
    (ver `_confianca_na_velocidade`). Sobram 4 a 6 cm — um palmo — contra os
    9 a 55 cm de antes.
    """
    atraso, _ = _andando(Suavizador(), v, fps)
    assert abs(atraso) < 0.07, f"{atraso * 100:.0f} cm de atraso"

    # A comparacao com o filtro antigo so vale acima do piso de velocidade.
    # A 0,20 m/s a previsao entra encolhida DE PROPOSITO — e la o filtro
    # antigo ja era bom, porque quem quase nao anda quase nao fica para tras.
    if v >= 0.39:
        antes = 3.0 * v / fps                # (1-alfa)/alfa * v * dt, alfa=0,25
        assert abs(atraso) < 0.6 * antes, "nao melhorou o suficiente para valer"


@pytest.mark.parametrize("v", [0.39, 0.80, 1.20])
def test_o_atraso_nao_depende_mais_do_fps(v):
    """Constante por quadro e uma constante que depende da maquina.

    O passo seguinte do projeto e subir o fps. Com alfa por quadro o atraso
    caia pela metade sozinho e o filtro mudava de personalidade sem ninguem
    mexer nele — o que torna impossivel saber a que atribuir a melhora.
    """
    atrasos = [_andando(Suavizador(), v, fps)[0] for fps in (6.5, 10.0, 15.0)]
    assert max(atrasos) - min(atrasos) < 0.01, (
        f"{[round(a * 100, 1) for a in atrasos]} cm")


def test_a_media_exponencial_pura_falharia_este_teste():
    """A prova de que o teste acima mede algo. Sem previsao, o atraso volta.

    Reproduz aqui o filtro antigo — alfa fixo por quadro, sem velocidade — e
    exige que ele fique para tras. Se um dia alguem tirar a previsao achando
    que e enfeite, este teste e que denuncia.
    """
    alfa, v, dt = 0.25, 0.80, 1 / 6.5
    x = saida = 0.0
    for _ in range(300):
        x += v * dt
        saida += alfa * (x - saida)
    assert (x - saida) > 0.30, "o filtro antigo deveria atrasar uns 37 cm"


def test_a_idade_da_medida_e_compensada():
    """A medida descreve o passado; o desenho e do presente.

    Entre a camera capturar e a tela desenhar passam o detector (130 ms
    medidos), a fila e o ciclo. A 1 m/s sao 15 cm — que apareceriam somados
    ao atraso do filtro se ninguem contasse o tempo.
    """
    com = abs(_andando(Suavizador(), 1.0, 10.0, idade_s=0.15)[0])
    assert com < 0.04, f"{com * 100:.0f} cm de atraso"

    # sem compensar, a idade entra inteira no atraso
    sem = Suavizador()
    t, x, dt, v, idade = 0.0, 0.0, 0.1, 1.0, 0.15
    sem.suavizar(1, 0.0, 0.0, vx=v, vy=0.0, agora=t)
    for _ in range(80):
        t += dt
        x += v * dt
        saida, _, _ = sem.suavizar(1, x - v * idade, 0.0, vx=v, vy=0.0,
                                   idade_s=0.0, agora=t)
    assert (x - saida) > 0.12, "a idade nao compensada deveria custar ~15 cm"


# ------------------------------------------------------------- o tremelique
def test_ruido_de_quem_esta_parado_continua_sendo_alisado():
    """Tirar o atraso nao pode ter custado a suavidade — era o pedido de 12/08."""
    import random
    rnd = random.Random(7)
    s = Suavizador()
    t = 0.0
    s.suavizar(1, 0.0, 0.0, vx=0.0, vy=0.0, agora=t)
    entradas, saidas = [], []
    for _ in range(200):
        t += 1 / 6.5
        ruido = rnd.gauss(0, 0.02)               # 2 cm de tremor por quadro
        entradas.append(ruido)
        x, _, _ = s.suavizar(1, ruido, 0.0, vx=0.0, vy=0.0, agora=t)
        saidas.append(x)

    def desvio(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))

    assert desvio(saidas) < 0.45 * desvio(entradas), (
        "o filtro deixou passar mais da metade do tremor")


def test_velocidade_ruidosa_nao_vira_tremor_de_posicao():
    """A velocidade e que extrapola: o ruido dela custaria posicao errada."""
    import random
    rnd = random.Random(3)
    s = Suavizador()
    t = 0.0
    s.suavizar(1, 0.0, 0.0, vx=0.0, vy=0.0, agora=t)
    saidas = []
    for _ in range(120):
        t += 1 / 6.5
        # parada, mas o Kalman oscila +-0,3 m/s
        saidas.append(s.suavizar(1, 0.0, 0.0, vx=rnd.gauss(0, 0.3), vy=0.0,
                                 agora=t)[0])
    assert max(abs(v) for v in saidas[20:]) < 0.10, (
        "a velocidade ruidosa empurrou o boneco para longe do lugar")


# -------------------------------------------------- o que nao pode mudar
def test_primeira_amostra_passa_intacta():
    s = Suavizador()
    assert s.suavizar(1, 2.0, 3.0, 0.5, agora=0.0) == (2.0, 3.0, 0.5)


def test_salto_grande_nao_arrasta_o_boneco_pela_cena():
    """Id reciclado num canto oposto: vai direto, sem cruzar a sala."""
    s = Suavizador(salto_m=0.60)
    s.suavizar(1, 0.0, 0.0, agora=0.0)
    x, y, _ = s.suavizar(1, 3.0, 0.0, agora=0.1)
    assert (x, y) == (3.0, 0.0)


def test_depois_de_um_salto_a_velocidade_antiga_deixa_de_valer():
    """A posicao velha e a velocidade velha morrem juntas.

    Guardar a velocidade de antes faria o boneco chegar no lugar novo ja
    correndo na direcao que ele seguia no lugar velho.
    """
    s = Suavizador(salto_m=0.60)
    s.suavizar(1, 0.0, 0.0, vx=2.0, vy=0.0, agora=0.0)
    s.suavizar(1, 3.0, 0.0, vx=2.0, vy=0.0, agora=0.1)
    x, _, _ = s.suavizar(1, 3.0, 0.0, vx=0.0, vy=0.0, agora=0.2)
    assert x == pytest.approx(3.0, abs=0.05)


def test_velocidade_absurda_nao_arremessa_o_boneco():
    """Teto na previsao: a correcao precisa ter chance de opinar."""
    s = Suavizador(avanco_maximo_m=0.35)
    s.suavizar(1, 0.0, 0.0, agora=0.0)
    x, _, _ = s.suavizar(1, 0.0, 0.0, vx=50.0, vy=0.0, idade_s=1.0, agora=0.15)
    assert x <= 0.36, f"o boneco foi parar em {x:.2f} m"


def test_rumo_ausente_mantem_o_ultimo_conhecido():
    """Perder um quadro nao pode fazer o boneco dar meia-volta."""
    s = Suavizador()
    s.suavizar(1, 0.0, 0.0, 1.2, agora=0.0)
    _, _, r = s.suavizar(1, 0.0, 0.0, None, agora=0.1)
    assert r == pytest.approx(1.2)


def test_rumo_cruza_a_costura_pelo_caminho_curto():
    """De +179 para -179 sao 2 graus de giro, nao 358."""
    a, b = math.radians(179), math.radians(-179)
    r = _mistura_angulo(a, b, 0.5)
    assert abs(r) > math.radians(179.0)          # ficou perto de 180, nao de 0


def test_esquecer_apaga_quem_saiu():
    s = Suavizador()
    s.suavizar(1, 0.0, 0.0, agora=0.0)
    s.suavizar(2, 5.0, 5.0, agora=0.0)
    s.esquecer({2})
    assert s.suavizar(1, 9.0, 9.0, agora=0.1) == (9.0, 9.0, None)


# ------------------------------------------------- o tempo em segundos
def test_a_suavizacao_nao_muda_quando_o_fps_muda():
    """Constante por quadro e uma constante que depende da maquina.

    O proximo passo do projeto e subir o fps. Com alfa por quadro, o filtro
    mudaria de personalidade sozinho no dia em que a maquina ficasse rapida.
    """
    def erro_restante(fps):
        s = Suavizador()
        t, dt = 0.0, 1.0 / fps
        s.suavizar(1, 0.0, 0.0, agora=t)
        for _ in range(int(0.30 * fps)):         # 300 ms de perseguicao
            t += dt
            x, _, _ = s.suavizar(1, 1.0, 0.0, agora=t)
        return 1.0 - x

    assert erro_restante(6.5) == pytest.approx(erro_restante(30.0), abs=0.06)


def test_meia_vida_e_meia_vida_mesmo():
    """Um dt de uma meia-vida absorve metade do erro. Por definicao."""
    assert _alfa(0.10, 0.10) == pytest.approx(0.5)
    assert _alfa(0.20, 0.10) == pytest.approx(0.75)


def test_dt_zero_nao_corrige_nada():
    """Duas chamadas no mesmo instante nao sao duas observacoes.

    Com 5 m de diferenca a valvula de salto dispararia primeiro e o teste
    mediria outra coisa. Dez centimetros ficam dentro dela.
    """
    assert _alfa(0.0, 0.10) == 0.0
    s = Suavizador()
    s.suavizar(1, 0.0, 0.0, agora=1.0)
    assert s.suavizar(1, 0.10, 0.0, agora=1.0)[0] == pytest.approx(0.0)


# ------------------------------------------- o preco, medido e declarado
def test_o_pico_ao_parar_de_repente_cabe_num_palmo():
    """Todo previsor ultrapassa quando o alvo para. A questao e quanto.

    Ninguem para instantaneamente: uma desaceleracao de 0,35 s e o que um
    corpo humano faz. Medido assim, o pico fica em torno de 5 cm — contra os
    37 cm de atraso PERMANENTE que a previsao removeu.

        Erro transitorio de 5 cm troca por erro permanente de 37. A conta nao
        e apertada.
    """
    s = Suavizador()
    fps, v, dt, desacel = 6.5, 0.80, 1 / 6.5, 0.35
    t = x = 0.0
    s.suavizar(1, 0.0, 0.0, vx=v, vy=0.0, agora=t)
    for _ in range(int(4 * fps)):
        t += dt
        x += v * dt
        s.suavizar(1, x, 0.0, vx=v, vy=0.0, agora=t)
    parou_em = x
    pico = 0.0
    for n in range(int(3 * fps)):
        t += dt
        vel = v * max(0.0, 1 - (n * dt) / desacel)
        x += vel * dt
        saida, _, _ = s.suavizar(1, x, 0.0, vx=vel, vy=0.0, agora=t)
        pico = max(pico, saida - (parou_em + v * desacel / 2))
    assert pico < 0.09, f"ultrapassou {pico * 100:.0f} cm"


def test_quem_esta_parado_fica_parado_mesmo_com_o_kalman_oscilando():
    """O encolhimento da velocidade e o que compra isto.

    Sem ele, prever integrava o ruido do Kalman direto na posicao: 3,5 cm de
    tremor em quem nao saiu do lugar, pior que o filtro antigo.
    """
    import random
    rnd = random.Random(11)
    s = Suavizador()
    t = 0.0
    s.suavizar(1, 0.0, 0.0, vx=0.0, vy=0.0, agora=t)
    saidas = []
    for _ in range(200):
        t += 1 / 6.5
        saidas.append(s.suavizar(1, rnd.gauss(0, 0.02), 0.0,
                                 vx=rnd.gauss(0, 0.12), vy=rnd.gauss(0, 0.12),
                                 idade_s=0.15, agora=t)[0])
    assert max(abs(v) for v in saidas[20:]) < 0.06, "o parado andou"


def test_a_confianca_na_velocidade_e_suave_e_nao_um_degrau():
    """Um limiar criaria um degrau na velocidade em que se anda mais."""
    s = Suavizador(piso_de_velocidade_m_s=0.25)
    passo = 0.01
    pesos = [s._confianca_na_velocidade(k * passo, 0.0) for k in range(301)]
    assert pesos[0] == 0.0
    assert pesos[25] == pytest.approx(0.5)         # no piso, meio a meio
    assert all(a <= b for a, b in zip(pesos, pesos[1:]))
    assert pesos[-1] > 0.98
    # 1 cm/s de velocidade a mais nunca muda o peso em mais de 3 pontos.
    # Um limiar mudaria 100 pontos de uma vez, na velocidade errada.
    assert max(b - a for a, b in zip(pesos, pesos[1:])) < 0.03


# ------------------------------------------------- a fiacao, de ponta a ponta
def test_o_motor_entrega_a_hora_em_que_a_CAMERA_viu():
    """`t_medido` tem que atravessar o motor sem virar a hora do processamento.

    Este e o elo que faltava: `_montar_estados` carimbava `t_mono` com
    `_agora()` — a hora em que o SISTEMA concluiu — e a idade da medida se
    perdia ali. Quem desenha ficava sem saber quanto tempo extrapolar.

        Um numero que existe na origem e nao e carregado ate o consumidor
        equivale a nao existir, e custa mais caro porque parece existir.
    """
    import time

    from testes.test_espacial import homografia_sintetica, obs_alto

    motor = SpatialEngine(homografia_sintetica(), usar_plausibilidade=False)
    t0 = time.monotonic() - 0.42                 # a camera viu 420 ms atras
    estados = []
    for k in range(8):
        estados = motor.atualizar([obs_alto(1, (300, 180, 340, 300),
                                            t=t0 + k * 0.01)], dt=0.1)

    assert estados, "ninguem sobreviveu ao funil"
    e = estados[0]
    assert e.t_medido == pytest.approx(t0 + 0.07, abs=0.01)
    assert 0.3 < e.idade_s < 0.5, f"idade {e.idade_s:.2f} s"


def test_sem_hora_de_captura_a_idade_e_zero_e_nao_um_numero_gigante():
    """`t_medido` ausente vale 0. Subtrair dele daria a idade do Unix.

    Extrapolar por 1,7 bilhao de segundos poria o boneco fora da galaxia. O
    caso importa porque camera falsa e teste antigo nao carimbam a hora.
    """
    from src.espacial.estado import EstadoDePessoa
    assert EstadoDePessoa(id=1, x=0, y=0, t_mono=1e9).idade_s == 0.0
    assert EstadoDePessoa(id=1, x=0, y=0, t_medido=1e9).idade_s == 0.0


def test_relogio_ao_contrario_nao_puxa_o_boneco_para_tras():
    """Concluir ANTES de capturar e impossivel; o codigo nao pode acreditar."""
    from src.espacial.estado import EstadoDePessoa
    e = EstadoDePessoa(id=1, x=0, y=0, t_mono=100.0, t_medido=100.5)
    assert e.idade_s == 0.0
