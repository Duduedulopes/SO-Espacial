"""A camera deduzida do chao que a trena mediu, provada contra verdade conhecida.

    Verdade conhecida nao e uma aproximacao melhor do real: e a unica
    situacao em que um erro pode ser MEDIDO em vez de estimado.

Monta-se uma camera SINTETICA com K, R e t escolhidos a mao, projeta-se o
plano do chao para obter a homografia que ela produziria, e exige-se que
`camera_da_homografia` devolva de volta os numeros de onde se partiu.

Depois monta-se um mapa de profundidade dessa mesma cena — com estante — e
exige-se que a nuvem volte em metros, no lugar certo, mesmo com a rede
errando a escala de proposito.

O QUE ISTO SUBSTITUI

Um dia inteiro de DUSt3R, que precisa de sobreposicao entre as vistas. As
tres cameras deste arranjo quase nao veem as mesmas superficies — o metodo
nao falhou, foi usado fora da hipotese dele.
"""
import math

import numpy as np
import pytest

from src.mundo.profundidade import (DISCORDANCIA_MAXIMA, camera_da_homografia,
                                    intrinseca_da_homografia, nuvem_do_alto,
                                    profundidade_do_chao)

LARG, ALT = 640, 480


def _camera_de_teste(focal=520.0, altura=2.50, inclinacao_graus=52.0,
                     giro_graus=18.0, olhando=(0.9, 0.7)):
    """Uma camera de teto plausivel. Devolve (K, R, t, posicao)."""
    K = np.array([[focal, 0, LARG / 2], [0, focal, ALT / 2], [0, 0, 1.0]])

    # a camera olha para baixo, inclinada, e girada em torno do proprio eixo
    tilt = math.radians(180.0 - inclinacao_graus)
    ct, st = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    g = math.radians(giro_graus)
    cg, sg = math.cos(g), math.sin(g)
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1.0]])
    R = Rx @ Rz                              # mundo -> camera

    C = np.array([olhando[0], olhando[1] - altura / math.tan(math.radians(
        inclinacao_graus)), altura])
    t = -R @ C
    return K, R, t, C


def _homografia_de(K, R, t):
    """A homografia pixel -> metro que esta camera produz sobre o chao z=0."""
    G = K @ np.column_stack([R[:, 0], R[:, 1], t])     # metro -> pixel
    return np.linalg.inv(G / G[2, 2])


# ------------------------------------------------------------ a focal
def test_a_focal_volta_do_plano_metrico():
    """O teste central: a trena mediu o chao, e o chao devolve a lente."""
    for focal in (380.0, 520.0, 700.0, 1100.0):
        K, R, t, _C = _camera_de_teste(focal=focal)
        achada, discordancia = intrinseca_da_homografia(
            _homografia_de(K, R, t), LARG, ALT)
        assert achada is not None, f"nao achou focal para f={focal}"
        assert achada[0, 0] == pytest.approx(focal, rel=0.02)
        assert discordancia < 0.02, "as duas estimativas deveriam concordar"


def test_as_duas_estimativas_concordam_quando_as_hipoteses_valem():
    """Perpendicularidade e mesmo comprimento sao regras diferentes.

    Elas so dao o mesmo numero se pixel quadrado e centro otico no meio
    forem verdade. Quando concordam, concordam sobre alguma coisa.
    """
    K, R, t, _C = _camera_de_teste()
    _achada, d = intrinseca_da_homografia(_homografia_de(K, R, t), LARG, ALT)
    assert d < DISCORDANCIA_MAXIMA / 5


def test_centro_optico_fora_do_lugar_e_denunciado():
    """A discordancia existe para isto: dizer quando a hipotese nao vale."""
    K, R, t, _C = _camera_de_teste()
    K[0, 2] += 140.0                       # centro optico bem fora do meio
    K[1, 2] -= 90.0
    _achada, d = intrinseca_da_homografia(_homografia_de(K, R, t), LARG, ALT)
    assert d > DISCORDANCIA_MAXIMA, f"discordancia de so {d:.3f}"


def test_homografia_degenerada_e_recusada():
    assert intrinseca_da_homografia(np.zeros((3, 3)), LARG, ALT)[0] is None
    assert intrinseca_da_homografia(np.eye(3), LARG, ALT)[0] is None


# ------------------------------------------------------- a pose e a altura
def test_a_altura_da_camera_volta_e_da_para_conferir_com_a_trena():
    """O numero que paga o metodo: um valor verificavel com uma fita metrica.

        Um metodo que produz, de graca, um numero que da para conferir com a
        trena vale mais que um metodo mais preciso que nao produz nenhum.
    """
    for altura in (2.20, 2.50, 3.10):
        K, R, t, C = _camera_de_teste(altura=altura)
        cam = camera_da_homografia(_homografia_de(K, R, t), LARG, ALT)
        assert cam is not None
        assert cam.altura_m == pytest.approx(altura, abs=0.06), (
            f"disse {cam.altura_m:.2f} m onde eram {altura:.2f}")


def test_a_posicao_inteira_da_camera_volta():
    K, R, t, C = _camera_de_teste()
    cam = camera_da_homografia(_homografia_de(K, R, t), LARG, ALT)
    assert cam.posicao == pytest.approx(C, abs=0.08)


def test_a_rotacao_volta_e_e_ortonormal():
    K, R, t, _C = _camera_de_teste()
    cam = camera_da_homografia(_homografia_de(K, R, t), LARG, ALT)
    assert cam.R @ cam.R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(cam.R) == pytest.approx(1.0, abs=1e-9)
    assert float(np.abs(cam.R - R).max()) < 0.05


def test_a_camera_nao_pode_sair_enterrada_no_chao():
    """A decomposicao aceita, com o mesmo erro algebrico, uma camera debaixo
    do piso olhando para cima. Escolher o sinal e parte de resolver."""
    K, R, t, _C = _camera_de_teste()
    cam = camera_da_homografia(_homografia_de(K, R, t), LARG, ALT)
    assert cam.altura_m > 0
    assert cam.t[2] > 0, "a cena tem que estar na frente da camera"


def test_sistema_canhoto_nao_enterra_a_camera_no_chao():
    """MEDIDO NA HOMOGRAFIA REAL, 19/08: a camera saiu a 2,73 m ABAIXO do piso.

    O (0,0) e os eixos do chao vieram da ordem em que os quatro cantos foram
    clicados na calibracao. Nada obriga essa ordem a produzir um sistema
    destro com z para cima — e a do Eduardo nao produz.

        Um resultado com o modulo certo e o sinal errado nao e um erro de
        conta: e uma convencao que ninguem declarou.

    Aqui o mesmo cenario e resolvido com o y invertido, que e o que a
    calibracao dele faz na pratica. A altura tem que sair positiva nos dois.
    """
    K, R, t, C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    canhota = np.diag([1.0, -1.0, 1.0]) @ H

    for nome, homografia, esperado_y in (("destro", H, C[1]),
                                         ("canhoto", canhota, -C[1])):
        cam = camera_da_homografia(homografia, LARG, ALT)
        assert cam is not None, f"nao resolveu o sistema {nome}"
        assert cam.altura_m > 0, f"camera enterrada no sistema {nome}"
        assert cam.altura_m == pytest.approx(C[2], abs=0.06)
        assert cam.posicao[1] == pytest.approx(esperado_y, abs=0.08)


def test_a_trena_no_teto_determina_a_focal():
    """O melhor resultado de 19/08, e veio de uma fita metrica.

    A focal deduzida pos a camera do Eduardo a 2,73 m; a trena disse 2,23.
    Mas a relacao vale nos dois sentidos: se a focal errada produz a altura
    errada, a altura CERTA determina a focal certa.

        Uma grandeza dificil de medir se resolve por outra facil de medir,
        quando as duas estao presas pela mesma geometria.
    """
    from src.mundo.profundidade import focal_pela_altura

    for focal, altura in ((430.0, 2.10), (551.0, 2.23), (700.0, 3.00)):
        K, R, t, C = _camera_de_teste(focal=focal, altura=altura)
        H = _homografia_de(K, R, t)
        achada = focal_pela_altura(H, LARG, ALT, altura)
        assert achada is not None
        assert achada[0, 0] == pytest.approx(focal, rel=0.02)
        cam = camera_da_homografia(H, LARG, ALT, K=achada)
        assert cam.altura_m == pytest.approx(altura, abs=0.01)


def test_altura_impossivel_devolve_None_em_vez_da_mais_proxima():
    """Nao ha resposta, e inventar a mais proxima esconderia que a medida ou
    a homografia estao erradas."""
    from src.mundo.profundidade import focal_pela_altura
    K, R, t, _C = _camera_de_teste(altura=2.5)
    H = _homografia_de(K, R, t)
    assert focal_pela_altura(H, LARG, ALT, 0.02) is None
    assert focal_pela_altura(H, LARG, ALT, 400.0) is None
    assert focal_pela_altura(H, LARG, ALT, 0.0) is None
    assert focal_pela_altura(H, LARG, ALT, None) is None


def test_a_altura_NAO_e_monotona_na_focal():
    """Eu supus que fosse, e estava errado. Este teste tranca o fato.

    A curva sobe, satura e DESCE — entao cada altura tem duas focais, e uma
    busca binaria sobre a faixa inteira pode cair em qualquer uma. Medido na
    homografia real:

        f=200 -> 0,93 m    f=800  -> 2,81 m    f=1500 -> 3,29 m  (pico)
        f=500 -> 2,08 m    f=1200 -> 3,23 m    f=3000 -> 2,66 m
    """
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    alturas = []
    for f in (300.0, 600.0, 1000.0, 2000.0, 3500.0):
        Kf = np.array([[f, 0, LARG / 2], [0, f, ALT / 2], [0, 0, 1.0]])
        alturas.append(camera_da_homografia(H, LARG, ALT, K=Kf).altura_m)
    assert alturas[0] < alturas[1], "o ramo de baixo tem que subir"
    assert alturas[-1] < max(alturas), "a curva tem que virar; se nao virar, "\
                                       "a busca podia ser mais simples"


def test_a_raiz_escolhida_e_a_de_campo_de_visao_plausivel():
    """Das duas focais possiveis, a grande descreve uma teleobjetiva.

        Duas solucoes matematicas nao sao ambiguidade quando uma delas
        descreve um aparelho que nao existe.
    """
    from src.mundo.profundidade import focal_pela_altura

    K, R, t, C = _camera_de_teste(focal=551.0, altura=2.23)
    H = _homografia_de(K, R, t)
    achada = focal_pela_altura(H, LARG, ALT, 2.23)
    diagonal = 2 * math.degrees(math.atan(
        math.hypot(LARG, ALT) / 2 / achada[0, 0]))
    assert 55.0 < diagonal < 100.0, f"{diagonal:.0f} graus nao e webcam"


def test_a_intrinseca_medida_vence_a_deduzida(tmp_path, monkeypatch):
    """`calibracao/intrinseca.py` existe desde o comeco e nunca foi rodado.

        Deduzir sob hipotese o que se pode medir em dez minutos e escolher
        carregar a hipotese para sempre.
    """
    import json

    from src.mundo import profundidade as P

    medida = tmp_path / "intrinseca.json"
    medida.write_text(json.dumps({
        "resolucao": [LARG, ALT],
        "K": [[444.0, 0, 300.0], [0, 444.0, 250.0], [0, 0, 1.0]]}))
    K = P.intrinseca_medida(medida)
    assert K[0, 0] == pytest.approx(444.0)

    # e reescalada quando a resolucao de uso e outra
    K2 = P.intrinseca_medida(medida, largura_px=LARG // 2, altura_px=ALT // 2)
    assert K2[0, 0] == pytest.approx(222.0)
    assert K2[0, 2] == pytest.approx(150.0)
    assert K2[1, 2] == pytest.approx(125.0)


def test_intrinseca_ausente_ou_quebrada_nao_derruba(tmp_path):
    from src.mundo.profundidade import intrinseca_medida
    assert intrinseca_medida(tmp_path / "nao_existe.json") is None
    ruim = tmp_path / "ruim.json"
    ruim.write_text("{isto nao e json")
    assert intrinseca_medida(ruim) is None


def test_camera_absurda_e_marcada_como_nao_confiavel():
    K, R, t, _C = _camera_de_teste()
    cam = camera_da_homografia(_homografia_de(K, R, t), LARG, ALT)
    assert cam.confiavel
    cam.posicao = np.array([0.0, 0.0, 14.0])          # 14 m de pe direito
    assert not cam.confiavel


# ------------------------------------------------- o chao previsto
def test_o_chao_previsto_bate_com_a_homografia():
    """Duas rotas ate o mesmo numero: a geometria da camera, e a homografia.

    Se elas discordassem, uma das duas estaria errada — e como a homografia
    veio da trena, seria a camera.
    """
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    cam = camera_da_homografia(H, LARG, ALT)
    prev, (us, vs) = profundidade_do_chao(cam, LARG, ALT, passo=40)

    for i, v in enumerate(vs):
        for j, u in enumerate(us):
            if not np.isfinite(prev[i, j]):
                continue
            p = H @ np.array([u, v, 1.0])
            chao = np.array([p[0] / p[2], p[1] / p[2], 0.0])
            esperado = float((R @ chao + t)[2])
            assert prev[i, j] == pytest.approx(esperado, rel=0.02)


# --------------------------------------------------------------- a nuvem
def _cena_com_estante(cam, altura_estante=1.90, centro=(0.9, 1.4),
                      largura=0.92, profundidade_m=0.30):
    """O mapa de profundidade que esta camera veria de um chao com estante.

    Verdade conhecida: sabemos onde a estante esta e quanto ela mede, entao
    da para exigir que ela volte de la.
    """
    prev, (us, vs) = profundidade_do_chao(cam, LARG, ALT, passo=1)
    uu, vv = np.meshgrid(us, vs)
    pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1)
    raios = pixels @ np.linalg.inv(cam.K).T
    dirs = raios @ cam.R
    C = cam.posicao

    # onde o raio encosta no TOPO da estante (plano z = altura)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_topo = (altura_estante - C[2]) / dirs[:, 2]
    bate = C + dirs * s_topo[:, None]
    dentro = ((np.abs(bate[:, 0] - centro[0]) <= largura / 2)
              & (np.abs(bate[:, 1] - centro[1]) <= profundidade_m / 2)
              & (s_topo > 0))

    z_topo = (np.asarray(bate) @ cam.R.T + cam.t)[:, 2]
    z = prev.ravel().copy()
    z[dentro] = z_topo[dentro]
    return z.reshape(ALT, LARG)


def test_a_nuvem_volta_em_metros_no_mundo_da_homografia():
    """De ponta a ponta, com a rede errando a escala em 12% de proposito."""
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    cam = camera_da_homografia(H, LARG, ALT)
    profundidade = _cena_com_estante(cam) * 1.12        # o erro da rede

    n = nuvem_do_alto(profundidade, H, tamanho_original=(LARG, ALT))
    assert n is not None and n.pronta, (
        f"residuo {n.residuo_chao_m * 100:.1f} cm" if n else "nada")
    assert n.escala == pytest.approx(1.12, rel=0.03), "nao desfez o erro"
    assert n.residuo_chao_m < 0.03, f"{n.residuo_chao_m * 100:.1f} cm de chao torto"


def test_a_estante_volta_com_a_altura_certa():
    """A regua independente: 1,90 m de trena tem que sair da nuvem."""
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    cam = camera_da_homografia(H, LARG, ALT)
    n = nuvem_do_alto(_cena_com_estante(cam) * 0.93, H,
                      tamanho_original=(LARG, ALT))
    assert n is not None
    alto = n.pontos[n.pontos[:, 2] > 1.0]
    assert len(alto) > 50, "a estante sumiu da nuvem"
    assert float(np.median(alto[:, 2])) == pytest.approx(1.90, abs=0.10)


def test_a_estante_volta_no_lugar_certo():
    """Pelos EXTREMOS, e nao pela mediana. A primeira versao deste teste
    falhou por 16 cm e estava errada — o codigo, nao.

    A mediana de um retangulo amostrado EM PERSPECTIVA nao cai no centro
    dele: pixels perto da camera cobrem menos chao, entao ha mais pontos
    daquele lado e a mediana e puxada para la. Medido: mediana em x = 0,74
    para um retangulo cujo centro e 0,90 — enquanto os extremos saem 0,441 e
    1,358 contra a verdade 0,44 e 1,36, ou seja, exatos ao milimetro.

        Amostragem nao uniforme move a mediana e nao move os extremos. Medir
        o centro pela mediana e medir a camera, nao o objeto.

    E a mesma razao pela qual `achar_estante` usa `minAreaRect` sobre o
    contorno, e nao a media dos pontos.
    """
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    cam = camera_da_homografia(H, LARG, ALT)
    n = nuvem_do_alto(_cena_com_estante(cam, centro=(0.9, 1.4)), H,
                      tamanho_original=(LARG, ALT))
    alto = n.pontos[n.pontos[:, 2] > 1.5]

    for eixo, centro, meia in ((0, 0.9, 0.46), (1, 1.4, 0.15)):
        baixo, cima = alto[:, eixo].min(), alto[:, eixo].max()
        assert (baixo + cima) / 2 == pytest.approx(centro, abs=0.03)
        assert cima - baixo == pytest.approx(2 * meia, abs=0.05)


def test_a_escala_e_decidida_pelo_CHAO_e_nao_pela_estante():
    """Mediana e nao media: a minoria alta nao pode mover o metro.

        Um estimador que a minoria consegue mover nao esta medindo a
        maioria.
    """
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    cam = camera_da_homografia(H, LARG, ALT)
    sem = nuvem_do_alto(_cena_com_estante(cam, largura=0.01), H,
                        tamanho_original=(LARG, ALT))
    com = nuvem_do_alto(_cena_com_estante(cam, largura=1.6,
                                          profundidade_m=1.6), H,
                        tamanho_original=(LARG, ALT))
    assert com.escala == pytest.approx(sem.escala, rel=0.02)


def test_o_residuo_do_chao_denuncia_uma_camera_errada():
    """A nota do conjunto. Se ela for ruim, o desenho nao presta ainda que
    fique bonito."""
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    boa = camera_da_homografia(H, LARG, ALT)
    profundidade = _cena_com_estante(boa)

    torta = camera_da_homografia(H, LARG, ALT)
    torta.posicao = torta.posicao + np.array([0.0, 0.0, 0.9])   # 90 cm acima

    n = nuvem_do_alto(profundidade, H, tamanho_original=(LARG, ALT),
                      camera=torta)
    assert n is None or n.residuo_chao_m > 0.05 or not n.pronta


def test_profundidade_vazia_e_recusada():
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    assert nuvem_do_alto(np.zeros((0, 0)), H) is None
    assert nuvem_do_alto(np.full((ALT, LARG), np.nan), H,
                         tamanho_original=(LARG, ALT)) is None


def test_a_nuvem_nao_e_recortada_alem_do_impossivel():
    """A licao de 18/08, que custou tres correcoes: nao encolher a vista.

    So sai o que nao pode existir — ponto enterrado no chao alem do ruido,
    ou mais alto que o pe direito.
    """
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    cam = camera_da_homografia(H, LARG, ALT)
    n = nuvem_do_alto(_cena_com_estante(cam), H, tamanho_original=(LARG, ALT),
                      passo=4)
    x0, x1 = n.pontos[:, 0].min(), n.pontos[:, 0].max()
    y0, y1 = n.pontos[:, 1].min(), n.pontos[:, 1].max()
    assert (x1 - x0) > 2.0 and (y1 - y0) > 2.0, (
        f"a nuvem cobre so {x1 - x0:.1f} x {y1 - y0:.1f} m")


def test_a_resolucao_do_mapa_pode_ser_outra():
    """A rede devolve na resolucao dela; a homografia foi medida na da camera.

    Reescalar a matriz da homografia ja foi fonte de erro neste projeto. Aqui
    reamostra-se a profundidade, que e o lado sem consequencia.
    """
    import cv2
    K, R, t, _C = _camera_de_teste()
    H = _homografia_de(K, R, t)
    cam = camera_da_homografia(H, LARG, ALT)
    cheia = _cena_com_estante(cam)
    menor = cv2.resize(cheia, (LARG // 2, ALT // 2), interpolation=cv2.INTER_NEAREST)

    a = nuvem_do_alto(cheia, H, tamanho_original=(LARG, ALT))
    b = nuvem_do_alto(menor, H, tamanho_original=(LARG, ALT))
    assert b is not None
    assert b.escala == pytest.approx(a.escala, rel=0.03)
    assert b.residuo_chao_m < 0.05
