"""
Testes do SpatialEngine — sem camera, sem modelo.

As observacoes sao construidas a mao, com geometria conhecida. Isso permite
afirmar coisas que com hardware seriam so impressao: "a cadeira foi rejeitada",
"a pessoa continuou existindo durante 1,6 s de ausencia", "o esqueleto ficou
em pe no chao".

    python testes/test_espacial.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.espacial.motor import SpatialEngine        # noqa: E402
from src.visao.observacao import Observacao         # noqa: E402


def homografia_sintetica():
    """Camera inclinada olhando o chao. Mapeia pixel -> metro de forma
    conhecida, para os testes poderem conferir numeros."""
    origem = np.float32([[200, 200], [440, 200], [500, 420], [140, 420]])
    destino = np.float32([[0, 0], [1.0, 0], [1.0, 1.0], [0, 1.0]])
    import cv2
    return cv2.getPerspectiveTransform(origem, destino)


def obs_alto(tid, caixa, tornozelos=True, t=None):
    x1, y1, x2, y2 = caixa
    j2d = c2d = None
    if tornozelos:
        j2d = np.zeros((17, 2))
        j2d[15] = [(x1 + x2) / 2 - 8, y2 - 4]
        j2d[16] = [(x1 + x2) / 2 + 8, y2 - 4]
        c2d = np.zeros(17)
        c2d[15] = c2d[16] = 0.9
    return Observacao(camera_id="alto", papel="alto",
                      t_mono=t if t is not None else time.monotonic(),
                      caixa=caixa, id_externo=tid, confianca=0.9,
                      juntas_2d=j2d, conf_2d=c2d)


def esqueleto_rel():
    """Pose relativa ao quadril, na convencao do MediaPipe.

    ATENCAO AO EIXO — este teste ja falhou uma vez por causa disto.

        MediaPipe:  x = direita   y = PARA BAIXO na imagem   z = profundidade

    Ou seja: quem esta ACIMA tem y NEGATIVO. Montar o esqueleto com a altura
    em z faz a `fundir` tratar altura como profundidade e descartar — o
    resultado foi um esqueleto todo em z=0, achatado no chao.

    O codigo estava certo; o dado de teste e que usava a convencao errada.
    """
    j = np.zeros((17, 3))
    j[0] = [0.00, -0.72, 0]                              # nariz, acima
    j[5] = [-0.20, -0.50, 0]; j[6] = [0.20, -0.50, 0]    # ombros
    j[11] = [-0.11, 0.00, 0]; j[12] = [0.11, 0.00, 0]    # quadril = origem
    j[13] = [-0.11, 0.45, 0]; j[14] = [0.11, 0.45, 0]    # joelhos
    j[15] = [-0.11, 0.88, 0]; j[16] = [0.11, 0.88, 0]    # tornozelos, abaixo
    return j


def obs_pose(papel, t=None):
    return Observacao(camera_id=papel, papel=papel,
                      t_mono=t if t is not None else time.monotonic(),
                      juntas_3d=esqueleto_rel(), confianca=0.9)


def motor(**kw):
    return SpatialEngine(homografia_sintetica(), **kw)


# ---------------------------------------------------------------- basico
def test_pessoa_vira_posicao_em_metros():
    m = motor(usar_plausibilidade=False)
    caixa = (280, 180, 360, 400)
    e = []
    for _ in range(6):
        e = m.atualizar([obs_alto(1, caixa)], 1 / 15)
    assert len(e) == 1
    p = e[0]
    assert 0.0 <= p.x <= 1.5 and 0.0 <= p.y <= 1.5, f"({p.x}, {p.y})"


def test_sem_observacao_nenhuma_pessoa():
    m = motor(usar_plausibilidade=False)
    assert m.atualizar([], 1 / 15) == []


def test_observacao_sem_tornozelo_e_barrada():
    """O filtro exige que o rastro prove ser gente antes de entrar."""
    m = motor(usar_plausibilidade=False, min_tornozelo=3)
    caixa = (280, 180, 360, 400)
    e = m.atualizar([obs_alto(1, caixa, tornozelos=False)], 1 / 15)
    assert e == []
    assert m.rejeitadas["tornozelo"] >= 1


# ---------------------------------------------------------------- persistencia
def test_pessoa_sobrevive_a_ausencia():
    """O caso que fragmentava identidade: 1,6 s fora do quadro."""
    m = motor(usar_plausibilidade=False)
    caixa = (280, 180, 360, 400)
    for _ in range(10):
        m.atualizar([obs_alto(1, caixa)], 1 / 15)

    for _ in range(24):                       # 1,6 s sem ver ninguem
        e = m.atualizar([], 1 / 15)

    assert len(e) == 1, "a pessoa nao pode sumir por 1,6 s de ausencia"
    assert e[0].prevendo > 0
    assert e[0].incerteza > 0.05, "a incerteza tem que crescer sem medicao"


def test_identidade_recosturada_com_id_externo_novo():
    """Ao voltar, o rastreador da um ID novo. O sistema tem que reconhecer."""
    m = motor(usar_plausibilidade=False)
    caixa = (280, 180, 360, 400)
    for _ in range(10):
        e = m.atualizar([obs_alto(1, caixa)], 1 / 15)
    antes = e[0].id

    for _ in range(6):
        m.atualizar([], 1 / 15)

    for _ in range(4):
        e = m.atualizar([obs_alto(99, caixa)], 1 / 15)   # ID EXTERNO NOVO

    assert len(e) == 1
    assert e[0].id == antes, "deveria ser a mesma pessoa, nao uma nova"
    assert m.rastros.recosturas >= 1


# ---------------------------------------------------------------- esqueleto
def test_esqueleto_fica_em_pe_no_chao():
    m = motor(usar_plausibilidade=False)
    caixa = (280, 180, 360, 400)
    e = []
    for _ in range(6):
        e = m.atualizar([obs_alto(1, caixa), obs_pose("frontal"),
                         obs_pose("lateral")], 1 / 15)

    p = e[0]
    assert p.tem_esqueleto
    z = p.esqueleto[:, 2]
    assert abs(min(z[15], z[16])) < 0.02, "o pe tem que encostar no chao"
    assert z.max() > 1.0, "a cabeca tem que ficar acima de 1 m"
    assert {"frontal", "lateral"} <= p.visto_por


def test_esqueleto_sem_vista_de_pose():
    m = motor(usar_plausibilidade=False)
    e = []
    for _ in range(6):
        e = m.atualizar([obs_alto(1, (280, 180, 360, 400))], 1 / 15)
    assert not e[0].tem_esqueleto, "sem pose nao pode haver esqueleto"
    assert e[0].x is not None, "mas a POSICAO continua valendo"


# ---------------------------------------------------------------- honestidade
def test_duas_pessoas_marcam_associacao_nao_confiavel():
    """Com duas pessoas o sistema nao sabe qual pose pertence a qual corpo.
    Declarar isso e melhor que fingir que sabe."""
    m = motor(usar_plausibilidade=False)
    e = []
    for _ in range(6):
        e = m.atualizar([obs_alto(1, (240, 180, 300, 380)),
                         obs_alto(2, (380, 190, 440, 390)),
                         obs_pose("frontal")], 1 / 15)
    assert len(e) == 2
    assert all(not p.associacao_confiavel for p in e)
    assert all(not p.tem_esqueleto for p in e), \
        "sem saber de quem e a pose, nao se atribui esqueleto a ninguem"


def test_percorrido_distingue_quem_anda_de_quem_nao():
    """Unico sinal que mobilia nao falsifica."""
    m = motor(usar_plausibilidade=False)
    parada = (280, 180, 360, 400)
    for _ in range(15):
        e = m.atualizar([obs_alto(1, parada)], 1 / 15)
    assert e[0].percorrido < 0.05, "parada nao pode acumular percurso"

    m2 = motor(usar_plausibilidade=False)
    for i in range(15):
        cx = 240 + i * 6
        e2 = m2.atualizar([obs_alto(1, (cx, 180, cx + 80, 400))], 1 / 15)
    assert e2[0].percorrido > 0.15, f"andou pouco: {e2[0].percorrido:.3f} m"


def test_homografia_segue_a_resolucao_real_da_camera():
    """Pedir nao e receber, e a geometria tem que seguir o que veio.

    Em 10/08 o sistema pediu 1280x720 e a homografia foi reescalada para
    isso. Se a camera responder 640x480, cada pixel passa a valer o dobro e a
    posicao sai com o dobro do erro — sem nenhum sintoma alem do numero.
    """
    H = homografia_sintetica()
    m = SpatialEngine(H, resolucao_calibracao=(640, 480),
                      resolucao_captura=(1280, 720),
                      usar_plausibilidade=False)
    assert not np.allclose(m.H, H), "deveria ter reescalado para 1280x720"

    mudou = m.ajustar_para_resolucao(640, 480)
    assert mudou
    assert np.allclose(m.H, H), "voltando a resolucao de calibracao, H = H"


def test_reajuste_nao_acumula_escala():
    """Recalcula do original. Compor escala sobre escala erraria a cada vez."""
    H = homografia_sintetica()
    m = SpatialEngine(H, resolucao_calibracao=(640, 480),
                      resolucao_captura=(640, 480),
                      usar_plausibilidade=False)
    for _ in range(5):
        m.ajustar_para_resolucao(1280, 720)
    esperado = SpatialEngine(H, resolucao_calibracao=(640, 480),
                             resolucao_captura=(1280, 720),
                             usar_plausibilidade=False).H
    assert np.allclose(m.H, esperado), "a escala acumulou"


def test_filtro_se_abstem_quando_o_modelo_nao_ajusta():
    """Medido em 10/08: k=0,149 com dispersao de 48% recusou 358 de 650
    deteccoes de uma pessoa real, e o rastro durou 3 s em 60.

    Nao era o limiar. Era o modelo. Um filtro que nao consegue ajustar o
    proprio modelo deve se ABSTER, nao recusar.
    """
    from percepcao.chao import FiltroDePlausibilidade

    f = FiltroDePlausibilidade(homografia_sintetica(), minimo_amostras=10)

    # razoes espalhadas: a altura aparente nao segue a distancia ao horizonte
    import random
    random.seed(7)
    for _ in range(40):
        h = random.uniform(40, 260)
        f.observar((300, 400 - h, 380, 400), percorrido_m=2.0)

    assert f.k is not None, "deveria ter aprendido algo"
    assert f.desistiu, f"dispersao alta e ele nao se absteve: {f.diagnostico()}"
    assert not f.pronto
    assert "ABSTIDO" in f.diagnostico()

    ok, _ = f.plausivel((300, 100, 380, 400))
    assert ok, "abstido tem que ACEITAR, nao recusar"


def test_filtro_continua_ativo_quando_o_modelo_ajusta():
    """O contraste: com razoes consistentes ele julga normalmente."""
    from percepcao.chao import FiltroDePlausibilidade

    f = FiltroDePlausibilidade(homografia_sintetica(), minimo_amostras=10)

    # caixas coerentes com o modelo: altura proporcional a distancia ao horizonte
    for y2 in range(360, 440, 2):
        d = y2 - f.v_horizonte(340)
        h = 0.5 * d
        f.observar((300, y2 - h, 380, y2), percorrido_m=2.0)

    assert f.pronto, f"deveria estar ativo: {f.diagnostico()}"
    assert not f.desistiu
    assert "ativo" in f.diagnostico()

    y2 = 400
    d = y2 - f.v_horizonte(340)
    alta, _ = f.plausivel((300, y2 - 5 * d, 380, y2))
    assert not alta, "caixa absurdamente alta deveria ser recusada"


def esqueleto_em_pe():
    """Pessoa em pe, relativa ao quadril, convencao COCO 17."""
    j = np.zeros((17, 3))
    j[[11, 12]] = [[-0.10, 0, 0.00], [0.10, 0, 0.00]]      # quadris
    j[[5, 6]] = [[-0.18, 0, 0.50], [0.18, 0, 0.50]]        # ombros
    j[[13, 14]] = [[-0.10, 0, -0.45], [0.10, 0, -0.45]]    # joelhos
    j[[15, 16]] = [[-0.10, 0, -0.90], [0.10, 0, -0.90]]    # tornozelos
    j[0] = [0, 0, 0.72]                                     # nariz
    return j


def inclinar(j, theta):
    """Como o MediaPipe entrega quando a lente esta inclinada."""
    c, s = np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return (Rx @ j.T).T


def desvio_da_vertical(j):
    tronco = j[[5, 6]].mean(axis=0) - j[[11, 12]].mean(axis=0)
    return np.degrees(np.arctan2(np.hypot(tronco[0], tronco[1]), tronco[2]))


def test_esqueleto_fica_em_pe_com_a_inclinacao_corrigida():
    """REGRESSAO DE 10/08. O SpatialEngine chamava `para_o_mundo`, que faz
    tudo o que `ancorar_no_chao` faz MENOS desfazer a inclinacao da lente.
    Resultado medido: tronco 42 graus fora da vertical — o boneco deitado.
    """
    from percepcao.pose3d import EstimadorDeInclinacao, ancorar_no_chao

    verdade = np.radians(-42)
    visto = inclinar(esqueleto_em_pe(), -verdade)

    est = EstimadorDeInclinacao()
    for _ in range(30):
        est.observar(visto, velocidade_ms=0.8, visivel=np.ones(17))

    assert est.confiavel
    assert abs(np.degrees(est.valor) - (-42)) < 2, np.degrees(est.valor)

    torto = ancorar_no_chao(visto, 2.0, 1.0, 0.0, inclinacao_rad=0.0)
    assert desvio_da_vertical(torto) > 30, "o defeito deveria aparecer sem correcao"

    reto = ancorar_no_chao(visto, 2.0, 1.0, 0.0, inclinacao_rad=est.valor)
    assert desvio_da_vertical(reto) < 3, f"{desvio_da_vertical(reto):.1f} graus"
    assert abs(min(reto[15, 2], reto[16, 2])) < 1e-9, "o pe tem que tocar z=0"


def test_motor_aprende_a_inclinacao_de_quem_anda():
    """O estimador so aceita amostra de quem esta ANDANDO: parado, a pessoa
    pode estar curvada, e ai o tronco nao serve de referencia vertical."""
    from src.visao.observacao import Observacao

    verdade = np.radians(-35)
    visto = inclinar(esqueleto_em_pe(), -verdade)

    m = motor(usar_plausibilidade=False)
    for i in range(40):
        cx = 240 + i * 8                       # andando de verdade
        obs = [obs_alto(1, (cx, 180, cx + 80, 400)),
               Observacao(camera_id="f", papel="frontal", t_mono=i * 0.05,
                          juntas_3d=visto, conf_2d=np.ones(17))]
        m.atualizar(obs, 1 / 20)

    assert m.inclinacao.confiavel, "nao aprendeu com ninguem andando"
    assert abs(np.degrees(m.inclinacao.valor) - (-35)) < 5, \
        f"{np.degrees(m.inclinacao.valor):.1f} graus"


def test_pessoa_sem_esqueleto_ainda_e_uma_pessoa():
    """Com duas pessoas a associacao deixa de ser confiavel e nenhum esqueleto
    e montado. Isso nao pode significar 'nao ha ninguem'.

    Em 10/08 a janela ficou vazia enquanto o sistema seguia dois rastros.
    """
    m = motor(usar_plausibilidade=False)
    for i in range(12):
        a, b = 200 + i * 6, 420 + i * 6
        m.atualizar([obs_alto(1, (a, 180, a + 70, 400)),
                     obs_alto(2, (b, 190, b + 70, 405))], 1 / 15)
    estados = m.atualizar([obs_alto(1, (272, 180, 342, 400)),
                           obs_alto(2, (492, 190, 562, 405))], 1 / 15)

    assert len(estados) == 2, f"esperava 2 pessoas, veio {len(estados)}"
    assert all(e.esqueleto is None for e in estados)
    assert all(not e.associacao_confiavel for e in estados), \
        "com duas pessoas a associacao tem que se declarar duvidosa"
    for e in estados:
        assert e.x is not None and e.y is not None, "posicao continua valendo"


def pessoa_mediapipe():
    """Eixos do MediaPipe world: x direita, y BAIXO, z profundidade."""
    j = np.zeros((17, 3))
    j[[11, 12]] = [[-0.10, 0, 0], [0.10, 0, 0]]
    j[[5, 6]] = [[-0.20, -0.50, 0], [0.20, -0.50, 0]]
    j[[13, 14]] = [[-0.10, 0.45, 0], [0.10, 0.45, 0]]
    j[[15, 16]] = [[-0.10, 0.90, 0], [0.10, 0.90, 0]]
    j[0] = [0, -0.72, 0]
    return j


def de_lado(j):
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    return (np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ j.T).T


def test_fusao_reconstroi_a_pessoa_com_entrada_limpa():
    """A matematica da fusao nao era o problema em 10/08 — provado aqui."""
    from percepcao.fusao import fundir

    f = pessoa_mediapipe()
    j, v = fundir(f, de_lado(f))

    assert v.all()
    assert abs(np.ptp(j[:, 2]) - 1.62) < 0.02, f"altura {np.ptp(j[:, 2]):.2f}"
    assert abs(np.ptp(j[:, 0]) - 0.40) < 0.02, f"largura {np.ptp(j[:, 0]):.2f}"


def test_junta_que_ninguem_viu_nao_entra_no_esqueleto():
    """O MediaPipe SEMPRE devolve as 17 juntas, mesmo as que estao fora do
    quadro. A webcam de notebook pega do peito para cima e entrega tornozelos
    extrapolados — numeros com cara de medida e sem relacao com o corpo.

        Um esqueleto sem pernas e honesto. Um com pernas inventadas e mentira
        com aparencia de dado.
    """
    from percepcao.fusao import fundir

    f = pessoa_mediapipe()
    sem_pernas = np.ones(17, bool)
    sem_pernas[[13, 14, 15, 16]] = False

    _, v = fundir(f, de_lado(f), vis_frontal=sem_pernas,
                  vis_lateral=sem_pernas)
    assert v.sum() == 13
    assert not v[[13, 14, 15, 16]].any(), "perna invisivel virou junta valida"

    # se UMA das vistas ve, a junta continua valendo
    _, v2 = fundir(f, de_lado(f), vis_frontal=sem_pernas)
    assert v2.all(), "a lateral viu as pernas; elas nao podiam sumir"


def test_estado_carrega_quais_juntas_foram_medidas():
    """Quem desenha precisa saber o que e medida e o que e extrapolacao."""
    from percepcao.fusao import Fusor

    fus = Fusor()
    f = pessoa_mediapipe()
    sem_pernas = np.ones(17, bool)
    sem_pernas[[13, 14, 15, 16]] = False

    fus.ver_frontal(f, 10.0, sem_pernas)
    fus.ver_lateral(de_lado(f), 10.0, sem_pernas)
    juntas, vis = fus.esqueleto(10.1)

    assert juntas is not None and vis is not None
    assert vis.sum() == 13

    # fora da validade, o material vence e nao ha esqueleto
    assert fus.esqueleto(20.0) == (None, None)


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


# ============================ A CAIXA NA BORDA NAO TEM PE (19/08, pelo video)
#
# De 28 s em diante o Eduardo andou ate a beirada do campo da camera do teto.
# Ele continuava aparecendo — mas os PES saiam do quadro. A caixa deixa de
# terminar no pe e passa a terminar na BORDA DA IMAGEM, que e uma linha de
# pixels fixa; a homografia converte linha fixa em posicao fixa em metros.
#
# O boneco travou num ponto, o mapa de calor virou uma mancha unica, e a
# velocidade ficou em 0,00 a 0,06 m/s enquanto ele atravessava a sala.
#
#     Uma caixa cortada pela borda nao mede a pessoa: mede onde a imagem
#     acabou. E a borda nao se move.
#
# Pior que o erro: ele nao parece erro. Nada falha, o rastro sobrevive, e o
# funil marcava 90% de medidas — todas dizendo a mesma coisa errada.

from percepcao.chao import EstimadorDePe, para_metros    # noqa: E402


def test_caixa_encostada_na_borda_e_reconhecida():
    c = EstimadorDePe._cortada
    quadro = (640, 480)
    assert c((300, 200, 340, 479), quadro), "encostou embaixo"
    assert c((0, 200, 40, 400), quadro), "encostou na esquerda"
    assert c((600, 200, 639, 400), quadro), "encostou na direita"
    assert c((300, 0, 340, 400), quadro), "encostou em cima"
    assert not c((300, 200, 340, 400), quadro), "esta inteira"


def test_sem_tornozelo_e_com_caixa_cortada_nao_ha_pe():
    """A resposta honesta e nao ter resposta. O Kalman passa a prever SABENDO
    que preve, e a acao cai para `desconhecida` em vez de inventar."""
    e = EstimadorDePe()
    e.desvio[7] = np.array([0.0, -90.0])          # ja aprendeu o desvio
    pe, motivo = e.estimar(7, (10, 100, 90, 479), None, None, quadro=(640, 480))
    assert pe is None and "borda" in motivo


def test_com_tornozelo_a_borda_nao_importa():
    """Tornozelo visto e medida de verdade, venha a caixa cortada ou nao."""
    e = EstimadorDePe()
    j = np.zeros((17, 2)); j[15] = [40, 300]; j[16] = [50, 300]
    conf = np.zeros(17); conf[15] = conf[16] = 0.9
    pe, origem = e.estimar(7, (10, 100, 90, 479), j, conf, quadro=(640, 480))
    assert origem == "tornozelo" and pe == (45, 300)


def test_sem_saber_o_tamanho_do_quadro_nada_muda():
    """Quem nao passa `quadro` continua com o comportamento antigo."""
    e = EstimadorDePe()
    pe, origem = e.estimar(7, (10, 100, 90, 479), None, None)
    assert pe is not None and origem == "caixa"


def test_a_posicao_congela_quando_a_caixa_encosta_na_borda():
    """A prova do defeito: sem a guarda, andar para a beirada vira ficar parado.

    A pessoa se move 200 px na horizontal enquanto sai do quadro por baixo. A
    base da caixa fica travada na borda, e a posicao em metros para de mudar
    no eixo que a borda fixa.
    """
    H = homografia_sintetica()
    e = EstimadorDePe()
    e.desvio[1] = np.array([0.0, -20.0])
    ys = []
    for k in range(6):
        x = 200 + k * 40
        caixa = (x, 300, x + 60, 479)             # sempre colada embaixo
        pe, _ = e.estimar(1, caixa, None, None)   # SEM a guarda
        ys.append(para_metros(H, *pe)[1])
    assert max(ys) - min(ys) < 0.02, (
        "o teste nao reproduziu o congelamento; reveja a homografia sintetica")

    # com a guarda, nenhuma dessas viraria medida
    for k in range(6):
        x = 200 + k * 40
        pe, _ = e.estimar(1, (x, 300, x + 60, 479), None, None,
                          quadro=(640, 480))
        assert pe is None
