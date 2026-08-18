"""O levantamento provado contra verdade conhecida.

COMO SE TESTA UM SOLUCIONADOR DE POSE SEM CAMERA

Ao contrario, e essa e a graca: inventa-se uma camera com pose CONHECIDA,
projetam-se os pontos do gabarito por ela, e entrega-se so os pixels ao
solucionador. Se ele nao devolver a pose que foi usada para gerar os pixels,
ele esta errado — e nao ha discussao possivel sobre isso.

    Verdade conhecida nao e uma aproximacao melhor do real: e a unica
    situacao em que um erro pode ser MEDIDO em vez de estimado.

E o teste ao contrario tambem: quando os pontos NAO descrevem a estante, o
residuo tem que subir e a pose tem que ser recusada. Um solucionador que so
acerta nao prova nada — otimizador nenhum se recusa a responder.
"""
import math

import numpy as np
import pytest

from src.mundo.ambiente import Gabarito
from src.mundo.levantamento import (ANGULO_MINIMO_GRAUS, Levantamento,
                                    NuvemDePontos, PoseDaCamera,
                                    intrinseca_estimada, nuvem_de,
                                    pontos_do_gabarito, resolver_pose,
                                    triangular)

GAB = Gabarito.de_arquivo("loja/estante.json")
MODELO = pontos_do_gabarito(GAB)
TAM = (640, 480)


def _camera_em(posicao, alvo=(0.0, 0.0, 0.95), tamanho=TAM):
    """Uma camera de pose CONHECIDA, olhando para um alvo. A verdade."""
    c = np.array(posicao, dtype=float)
    frente = np.array(alvo, dtype=float) - c
    frente /= np.linalg.norm(frente)
    direita = np.cross(frente, np.array([0.0, 0.0, 1.0]))
    direita /= np.linalg.norm(direita)
    baixo = np.cross(frente, direita)
    r = np.vstack([direita, baixo, frente])        # mundo -> camera
    import cv2
    rvec = cv2.Rodrigues(r)[0]
    tvec = (-r @ c).reshape(3, 1)
    return PoseDaCamera(papel="teste", rvec=rvec, tvec=tvec,
                        k=intrinseca_estimada(*tamanho))


# --------------------------------------------------------------- o gabarito 3D
def test_o_modelo_tem_os_pes_e_as_cinco_bandejas():
    assert len([n for n in MODELO if n.startswith("pe_")]) == 4
    for pid, _ in GAB.prateleiras:
        assert len([n for n in MODELO if n.startswith(f"{pid}_")]) == 4
    assert len(MODELO) == 4 + 4 * len(GAB.prateleiras)


def test_o_modelo_usa_as_medidas_de_trena():
    p = np.array(list(MODELO.values()))
    assert p[:, 0].max() - p[:, 0].min() == pytest.approx(GAB.largura)
    assert p[:, 1].max() - p[:, 1].min() == pytest.approx(GAB.profundidade)
    assert p[:, 2].max() == pytest.approx(GAB.altura)


def test_os_pes_estao_no_chao():
    for nome, (_, _, z) in MODELO.items():
        if nome.startswith("pe_"):
            assert z == 0.0


def test_o_modelo_nao_e_plano():
    """PnP precisa de pontos fora de um plano. Se fosse plano, a pose seria
    ambigua — duas solucoes espelhadas, e nenhuma forma de escolher."""
    p = np.array(list(MODELO.values()))
    assert np.linalg.matrix_rank(p - p.mean(axis=0)) == 3


# --------------------------------------------------------------- a pose volta
def test_a_pose_conhecida_e_recuperada():
    """O teste central: projetar por uma pose e recuperar a MESMA pose."""
    verdade = _camera_em((2.2, -1.4, 2.35))
    nomes = list(MODELO)
    pixels = verdade.projetar([MODELO[n] for n in nomes])
    marcados = dict(zip(nomes, map(tuple, pixels)))

    achada = resolver_pose("alto", marcados, MODELO, TAM)
    assert achada is not None
    assert achada.posicao == pytest.approx(verdade.posicao, abs=0.02)
    assert achada.residuo_px < 1.0
    assert achada.confiavel


def test_a_altura_da_camera_bate_com_a_fita_metrica():
    verdade = _camera_em((1.8, -1.2, 2.40))
    nomes = list(MODELO)
    marcados = dict(zip(nomes, map(tuple, verdade.projetar(
        [MODELO[n] for n in nomes]))))
    assert resolver_pose("alto", marcados, MODELO, TAM).altura == \
        pytest.approx(2.40, abs=0.05)


def test_funciona_de_varios_lugares():
    """Teto, lateral, frontal — as tres posicoes reais do arranjo."""
    for posicao in ((0.2, -2.0, 1.5), (2.6, 0.1, 1.2), (-1.0, -1.8, 2.3)):
        verdade = _camera_em(posicao)
        nomes = list(MODELO)
        marcados = dict(zip(nomes, map(tuple, verdade.projetar(
            [MODELO[n] for n in nomes]))))
        achada = resolver_pose("x", marcados, MODELO, TAM)
        assert achada is not None, f"nao resolveu de {posicao}"
        assert achada.posicao == pytest.approx(verdade.posicao, abs=0.05)


def test_meia_duzia_de_pontos_ja_resolve():
    """Nem sempre as 24 quinas aparecem. Seis tem que bastar."""
    verdade = _camera_em((2.2, -1.4, 2.35))
    nomes = list(MODELO)[:8]
    marcados = dict(zip(nomes, map(tuple, verdade.projetar(
        [MODELO[n] for n in nomes]))))
    achada = resolver_pose("alto", marcados, MODELO, TAM)
    assert achada is not None
    assert achada.posicao == pytest.approx(verdade.posicao, abs=0.10)


# --------------------------------------------------------------- e recusa
def test_poucos_pontos_nao_viram_pose():
    verdade = _camera_em((2.2, -1.4, 2.35))
    nomes = list(MODELO)[:4]
    marcados = dict(zip(nomes, map(tuple, verdade.projetar(
        [MODELO[n] for n in nomes]))))
    assert resolver_pose("alto", marcados, MODELO, TAM) is None


def test_pontos_embaralhados_levantam_o_residuo():
    """A prova de que o residuo tem dentes.

    Os mesmos pixels, trocados de nome. O PnP vai responder — otimizador
    sempre responde — e o residuo tem que denunciar.
    """
    verdade = _camera_em((2.2, -1.4, 2.35))
    nomes = list(MODELO)
    pixels = list(map(tuple, verdade.projetar([MODELO[n] for n in nomes])))
    embaralhados = dict(zip(nomes, pixels[7:] + pixels[:7]))

    achada = resolver_pose("alto", embaralhados, MODELO, TAM)
    assert achada is None or not achada.confiavel, \
        "aceitou uma pose de pontos trocados"


def test_nome_que_nao_existe_no_modelo_e_ignorado():
    verdade = _camera_em((2.2, -1.4, 2.35))
    nomes = list(MODELO)
    marcados = dict(zip(nomes, map(tuple, verdade.projetar(
        [MODELO[n] for n in nomes]))))
    marcados["a_geladeira"] = (10.0, 10.0)
    achada = resolver_pose("alto", marcados, MODELO, TAM)
    assert achada.pontos_usados == len(MODELO)


# --------------------------------------------------------------- a nuvem
def _duas_cameras():
    a = _camera_em((2.4, -1.6, 2.30));  a.papel = "alto"
    b = _camera_em((-1.6, -1.0, 1.30)); b.papel = "lateral"
    return a, b


def test_triangular_devolve_o_ponto_certo():
    a, b = _duas_cameras()
    verdadeiro = np.array([0.20, 0.05, 1.35])
    r = triangular(a, a.projetar([verdadeiro])[0], b, b.projetar([verdadeiro])[0])
    assert r is not None
    ponto, erro, angulo = r
    assert ponto == pytest.approx(verdadeiro, abs=0.01)
    assert erro < 0.01
    assert angulo > ANGULO_MINIMO_GRAUS


def test_cameras_quase_alinhadas_sao_recusadas():
    """Duas vistas quase iguais nao medem profundidade: opinam sobre ela."""
    a = _camera_em((2.40, -1.60, 2.30)); a.papel = "a"
    b = _camera_em((2.42, -1.61, 2.30)); b.papel = "b"
    p = np.array([0.2, 0.05, 1.35])
    assert triangular(a, a.projetar([p])[0], b, b.projetar([p])[0]) is None


def test_a_nuvem_reconstroi_a_estante():
    """A prova de ponta a ponta: duas cameras posadas devolvem o gabarito."""
    a, b = _duas_cameras()
    correspondencias = [
        {"alto": tuple(a.projetar([p])[0]), "lateral": tuple(b.projetar([p])[0])}
        for p in MODELO.values()]
    nuvem = nuvem_de({"alto": a, "lateral": b}, correspondencias)

    assert len(nuvem) >= len(MODELO) - 2
    baixo, alto = nuvem.caixa()
    assert alto[2] == pytest.approx(GAB.altura, abs=0.05)
    assert alto[0] - baixo[0] == pytest.approx(GAB.largura, abs=0.05)


def test_a_nuvem_guarda_de_onde_cada_ponto_veio():
    a, b = _duas_cameras()
    p = (0.2, 0.05, 1.35)
    nuvem = nuvem_de({"alto": a, "lateral": b},
                     [{"alto": tuple(a.projetar([p])[0]),
                       "lateral": tuple(b.projetar([p])[0])}])
    assert nuvem.vistos_por[0] == ("alto", "lateral")
    assert nuvem.angulos[0] > ANGULO_MINIMO_GRAUS


def test_ponto_visto_por_uma_camera_so_nao_entra():
    """Triangular exige duas. Uma vista sozinha nao tem profundidade."""
    a, b = _duas_cameras()
    nuvem = nuvem_de({"alto": a, "lateral": b},
                     [{"alto": (320.0, 240.0)}])
    assert len(nuvem) == 0


def test_firmes_separa_o_bem_medido():
    n = NuvemDePontos()
    n.somar((0, 0, 1), 0.01, 30, ("a", "b"))
    n.somar((1, 1, 1), 0.40, 30, ("a", "b"))
    assert len(n.firmes(0.05)) == 1


# --------------------------------------------------------------- o resultado
def test_uma_camera_situada_nao_e_levantamento():
    a, _ = _duas_cameras()
    nomes = list(MODELO)
    marcados = dict(zip(nomes, map(tuple, a.projetar(
        [MODELO[n] for n in nomes]))))
    lev = Levantamento(poses={"alto": resolver_pose("alto", marcados, MODELO, TAM)})
    assert lev.cameras_situadas == ("alto",)
    assert not lev.pronto, "monologo nao e fusao"


def test_duas_situadas_ja_e_levantamento():
    a, b = _duas_cameras()
    poses = {}
    for pose in (a, b):
        nomes = list(MODELO)
        marcados = dict(zip(nomes, map(tuple, pose.projetar(
            [MODELO[n] for n in nomes]))))
        poses[pose.papel] = resolver_pose(pose.papel, marcados, MODELO, TAM)
    lev = Levantamento(poses=poses)
    assert len(lev.cameras_situadas) == 2
    assert lev.pronto


def test_o_levantamento_vira_arquivo(tmp_path):
    import json
    a, b = _duas_cameras()
    poses = {}
    for pose in (a, b):
        nomes = list(MODELO)
        marcados = dict(zip(nomes, map(tuple, pose.projetar(
            [MODELO[n] for n in nomes]))))
        poses[pose.papel] = resolver_pose(pose.papel, marcados, MODELO, TAM)

    lev = Levantamento(poses=poses, prateleiras_por_camera={"alto": ["p1", "p2"]},
                       medido_em="2026-08-18T10:00:00")
    destino = tmp_path / "lev.json"
    lev.gravar(destino)

    d = json.loads(destino.read_text(encoding="utf-8"))
    assert d["cameras"]["alto"]["confiavel"] is True
    assert d["cameras"]["alto"]["altura_m"] == pytest.approx(2.30, abs=0.05)
    assert d["prateleiras_por_camera"]["alto"] == ["p1", "p2"]
    assert "estimada" in " ".join(d["_nota"]).lower()


def test_camera_que_nao_resolveu_aparece_como_nula():
    lev = Levantamento(poses={"frontal": None})
    assert lev.como_dicionario()["cameras"]["frontal"] is None
    assert lev.cameras_situadas == ()


# --------------------------------------------------------------- a intrinseca
def test_a_intrinseca_estimada_tem_a_forma_certa():
    k = intrinseca_estimada(640, 480, 60.0)
    assert k[0, 2] == 320 and k[1, 2] == 240
    assert k[0, 0] == pytest.approx(640 / 2 / math.tan(math.radians(30)))
    assert k[0, 0] == k[1, 1], "pixel quadrado"


def test_campo_de_visao_maior_encurta_a_focal():
    assert (intrinseca_estimada(640, 480, 90.0)[0, 0]
            < intrinseca_estimada(640, 480, 45.0)[0, 0])


# ------------------------------------------- a estante posta no mundo pelos pes
#
# O TESTE QUE FECHA O PROBLEMA DE 18/08.
#
# candidatos_do_alto media a BANDEJA DE CIMA e a passava pela homografia do
# chao: plano errado, estante a 1,79 m na diagonal, fora da area calibrada.
#
# Aqui a verdade e conhecida de novo: monta-se uma estante numa posicao
# escolhida, uma camera de pose conhecida olhando para ela, e uma homografia
# que leva pixel do chao a metro. So as quinas das BANDEJAS sao marcadas — os
# pes ficam escondidos, como na sala de verdade. A funcao tem que devolver a
# posicao que foi usada para montar tudo.
#
#     Nao era preciso ver o ponto que interessa. Era preciso ver o bastante
#     para saber onde ele estaria.

def _mundo_com_estante(x, y, giro):
    """Modelo da estante posto em (x, y) girado, no referencial do mundo."""
    co, si = math.cos(giro), math.sin(giro)
    r = np.array([[co, -si], [si, co]])
    fora = {}
    for nome, (mx, my, mz) in MODELO.items():
        px, py = r @ np.array([mx, my])
        fora[nome] = (px + x, py + y, mz)
    return fora


def _homografia_de(pose, chao_z=0.0):
    """A homografia que leva pixel -> metro no chao, para ESTA camera.

    Sai da propria pose: quatro pontos conhecidos do chao, projetados, e
    cv2.findHomography no sentido inverso. E o mesmo objeto que
    `calibracao/homografia.py` produz clicando quatro cantos com trena.
    """
    import cv2
    mundo = np.array([[0.0, 0.0], [1.65, 0.0], [1.65, 1.32], [0.0, 1.32]])
    espaco = np.hstack([mundo, np.full((4, 1), chao_z)])
    pixels = pose.projetar(espaco)
    h, _ = cv2.findHomography(pixels.astype(np.float64), mundo)
    return h


def test_a_estante_volta_ao_lugar_pelos_pes_escondidos():
    verdade_x, verdade_y, verdade_giro = 1.10, 0.95, 0.6
    mundo = _mundo_com_estante(verdade_x, verdade_y, verdade_giro)

    camera = _camera_em((0.8, -1.5, 2.40), alvo=(verdade_x, verdade_y, 1.0))
    h = _homografia_de(camera)

    # SO as bandejas sao marcadas. Os pes ficam de fora, como na sala.
    visiveis = {n: tuple(camera.projetar([mundo[n]])[0])
                for n in mundo if not n.startswith("pe_")}
    pose = resolver_pose("alto", visiveis, MODELO, TAM)
    assert pose is not None and pose.confiavel

    from src.mundo.levantamento import estante_no_mundo
    achado = estante_no_mundo(pose, MODELO, h, GAB)
    assert achado is not None
    x, y, rumo = achado
    assert (x, y) == pytest.approx((verdade_x, verdade_y), abs=0.06)


def test_o_rumo_tambem_volta():
    from src.mundo.levantamento import estante_no_mundo
    for giro in (0.0, 0.5, -0.8):
        mundo = _mundo_com_estante(1.0, 0.9, giro)
        camera = _camera_em((0.7, -1.4, 2.35), alvo=(1.0, 0.9, 1.0))
        visiveis = {n: tuple(camera.projetar([mundo[n]])[0])
                    for n in mundo if not n.startswith("pe_")}
        pose = resolver_pose("alto", visiveis, MODELO, TAM)
        _, _, rumo = estante_no_mundo(pose, MODELO, _homografia_de(camera), GAB)
        # o rumo do mundo e o giro aplicado, a menos de meia volta de face
        erro = abs(math.atan2(math.sin(rumo - giro), math.cos(rumo - giro)))
        assert min(erro, abs(erro - math.pi)) < 0.15, f"giro {giro}: rumo {rumo}"


def test_medir_o_topo_pela_homografia_erra_MUITO_mais():
    """A prova de que o conserto era necessario.

    Compara os dois caminhos: passar a bandeja de cima pela homografia do
    chao (o que se fazia) contra deduzir os pes pela pose (o que se faz).
    """
    from src.mundo.levantamento import estante_no_mundo
    vx, vy = 1.10, 0.95
    mundo = _mundo_com_estante(vx, vy, 0.3)
    camera = _camera_em((0.8, -1.5, 2.40), alvo=(vx, vy, 1.0))
    h = _homografia_de(camera)

    visiveis = {n: tuple(camera.projetar([mundo[n]])[0])
                for n in mundo if not n.startswith("pe_")}
    pose = resolver_pose("alto", visiveis, MODELO, TAM)
    x, y, _ = estante_no_mundo(pose, MODELO, h, GAB)
    erro_novo = math.hypot(x - vx, y - vy)

    # o caminho antigo: a bandeja do topo lida como se fosse chao
    from percepcao.chao import para_metros
    topo = [n for n in mundo if n.startswith("p5_")]
    pontos = [para_metros(h, *camera.projetar([mundo[n]])[0]) for n in topo]
    cx, cy = np.mean([np.asarray(p).ravel()[:2] for p in pontos], axis=0)
    erro_antigo = math.hypot(cx - vx, cy - vy)

    assert erro_novo < 0.10, f"o caminho novo errou {erro_novo:.2f} m"
    assert erro_antigo > erro_novo * 3, (
        f"antigo {erro_antigo:.2f} m, novo {erro_novo:.2f} m — o teste nao "
        f"distingue os dois caminhos")


def test_sem_pose_nao_ha_estante():
    from src.mundo.levantamento import estante_no_mundo
    assert estante_no_mundo(None, MODELO, np.eye(3), GAB) is None
