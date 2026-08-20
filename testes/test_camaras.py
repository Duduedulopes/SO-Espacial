"""As tres cameras num sistema so, provadas contra verdade conhecida.

    todas as cameras deveriam ajudar a dizer aonde esta a pessoa e unir todas
    as informacoes para apenas 1 movimento   — Eduardo, 19/08

Monta-se um conjunto de cameras SINTETICAS com K, R e t escolhidos a mao,
projeta-se um ponto 3D conhecido nas tres, e exige-se que a triangulacao o
devolva. O erro de reprojecao — que e o numero que valida uma calibracao — e
medido contra zero, porque aqui nao ha erro nenhum a nao ser o da conta.

    Verdade conhecida nao e uma aproximacao melhor do real: e a unica
    situacao em que um erro pode ser MEDIDO em vez de estimado.
"""
import numpy as np
import pytest

from src.mundo.camaras import Camara, Camaras

LARG, ALT = 640, 480


def _camara(papel, focal, posicao, olhando=(0.9, 0.9, 0.0)):
    """Uma camera que olha para um ponto, com `up` para cima no mundo."""
    C = np.array(posicao, dtype=float)
    frente = np.array(olhando, dtype=float) - C
    frente /= np.linalg.norm(frente)
    cima = np.array([0.0, 0.0, 1.0])
    if abs(float(frente @ cima)) > 0.999:
        cima = np.array([0.0, 1.0, 0.0])
    direita = np.cross(frente, cima)
    direita /= np.linalg.norm(direita)
    baixo = np.cross(frente, direita)

    R = np.stack([direita, baixo, frente])       # mundo -> camera
    t = -R @ C
    K = np.array([[focal, 0, LARG / 2], [0, focal, ALT / 2], [0, 0, 1.0]])

    # a homografia do chao sai da propria projecao: P sobre z=0
    P = K @ np.column_stack([R, t])
    G = P[:, [0, 1, 3]]                          # metro(x,y) -> pixel
    H = np.linalg.inv(G / G[2, 2])                # pixel -> metro
    return Camara(papel=papel, K=K, R=R, t=t, tamanho=(LARG, ALT),
                  homografia=H)


def _tres():
    return Camaras({
        "alto": _camara("alto", 551.0, (0.9, 0.2, 2.4)),
        "frontal": _camara("frontal", 600.0, (0.9, -1.6, 1.5)),
        "lateral": _camara("lateral", 600.0, (-1.4, 1.0, 1.4)),
    })


def _pixel(cam, ponto):
    return tuple(cam.projetar([ponto])[0])


# ------------------------------------------------------------ o basico
def test_a_matriz_de_projecao_tem_a_forma_certa():
    """`P = K [R | t]`, 3x4."""
    c = _tres()["alto"]
    assert c.P.shape == (3, 4)
    assert c.P == pytest.approx(c.K @ np.column_stack([c.R, c.t]))


def test_a_posicao_da_camera_volta():
    """`C = -R^T t`. Se isto errar, tudo depois erra junto."""
    for papel, esperado in (("alto", (0.9, 0.2, 2.4)),
                            ("frontal", (0.9, -1.6, 1.5)),
                            ("lateral", (-1.4, 1.0, 1.4))):
        assert _tres()[papel].posicao == pytest.approx(esperado, abs=1e-9)


def test_projetar_e_a_homografia_concordam_sobre_o_chao():
    """Duas rotas ate o mesmo pixel: a projecao 3D e a homografia do plano.

    Elas TEM que concordar — a homografia e a projecao restrita a z=0. Se
    discordassem, uma das duas estaria descrevendo outra camera.
    """
    cam = _tres()["alto"]
    for x, y in [(0.5, 0.5), (1.2, 0.8), (0.2, 1.4)]:
        u, v = _pixel(cam, (x, y, 0.0))
        assert cam.no_chao(u, v) == pytest.approx((x, y), abs=1e-6)


def test_atras_da_camera_devolve_NaN_e_nao_um_pixel_qualquer():
    """Um ponto atras da lente projeta num pixel plausivel e errado se
    ninguem checar o sinal da profundidade."""
    cam = _tres()["alto"]
    assert not np.isfinite(cam.projetar([(0.9, 0.2, 9.0)])[0]).all()


# --------------------------------------------------------- triangulacao
@pytest.mark.parametrize("alvo", [
    (0.90, 0.90, 1.20),      # a mao alcancando a prateleira do meio
    (0.40, 1.10, 1.60),      # mais alto e para o lado
    (1.30, 0.50, 0.10),      # quase no chao
])
def test_o_ponto_3d_volta_das_tres_vistas(alvo):
    """O teste central. Verdade conhecida, ida e volta."""
    cams = _tres()
    vistas = {p: _pixel(cams[p], alvo) for p in cams.papeis}
    achado = cams.triangular(vistas)
    assert achado is not None
    assert achado == pytest.approx(alvo, abs=1e-6)


def test_duas_vistas_ja_bastam():
    """A terceira melhora; ela nao e obrigatoria."""
    cams = _tres()
    alvo = (0.9, 0.9, 1.2)
    vistas = {p: _pixel(cams[p], alvo) for p in ("alto", "lateral")}
    assert cams.triangular(vistas) == pytest.approx(alvo, abs=1e-6)


def test_uma_vista_so_nao_triangula():
    """Um raio nao determina um ponto. Dizer 'nao da' e a resposta."""
    cams = _tres()
    assert cams.triangular({"alto": (320, 240)}) is None


def test_a_terceira_vista_melhora_quando_ha_ruido():
    """Cada camera acrescenta duas equacoes, e o sistema so melhora.

        Dois raios no espaco quase nunca se encontram. Triangular nao e achar
        o cruzamento: e achar o ponto que menos desagrada aos dois.
    """
    cams = _tres()
    alvo = np.array([0.9, 0.9, 1.2])
    rng = np.random.default_rng(7)
    erro_duas, erro_tres = [], []
    for _ in range(60):
        vistas = {}
        for p in cams.papeis:
            u, v = _pixel(cams[p], alvo)
            vistas[p] = (u + rng.normal(0, 1.5), v + rng.normal(0, 1.5))
        duas = cams.triangular({k: vistas[k] for k in ("alto", "lateral")})
        tres = cams.triangular(vistas)
        erro_duas.append(np.linalg.norm(duas - alvo))
        erro_tres.append(np.linalg.norm(tres - alvo))
    assert np.median(erro_tres) < np.median(erro_duas)


# ---------------------------------------------------- erro de reprojecao
def test_reprojecao_de_uma_calibracao_perfeita_e_zero():
    """O numero que valida tudo. Aqui nao ha erro alem do da conta.

        Um sistema de calibracao que nao mede o proprio erro nao esta
        calibrado: esta configurado.
    """
    cams = _tres()
    alvo = (0.9, 0.9, 1.2)
    vistas = {p: _pixel(cams[p], alvo) for p in cams.papeis}
    erros = cams.erro_de_reprojecao(alvo, vistas)
    assert set(erros) == set(cams.papeis)
    assert max(erros.values()) < 1e-6


def test_reprojecao_denuncia_uma_camera_torta():
    """Uma camera fora do lugar aparece aqui, e so aqui."""
    cams = _tres()
    alvo = (0.9, 0.9, 1.2)
    vistas = {p: _pixel(cams[p], alvo) for p in cams.papeis}
    cams["lateral"].t = cams["lateral"].t + np.array([0.0, 0.0, 0.10])
    erros = cams.erro_de_reprojecao(alvo, vistas)
    assert erros["alto"] < 1e-6
    assert erros["lateral"] > 2.0, f"{erros['lateral']:.1f} px nao denuncia"


# ---------------------------------------------------- fusao no chao
def test_a_fusao_de_duas_cameras_bate_melhor_que_a_melhor():
    """Escolher a melhor devolve a melhor. Combinar devolve melhor que ela."""
    cams = _tres()
    alvo = (0.9, 0.9, 0.0)
    rng = np.random.default_rng(3)
    so_alto, fundido = [], []
    for _ in range(200):
        vistas = {}
        for p in ("alto", "lateral"):
            u, v = _pixel(cams[p], alvo)
            vistas[p] = (u + rng.normal(0, 3), v + rng.normal(0, 3))
        a = cams.no_chao({"alto": vistas["alto"]})
        f = cams.no_chao(vistas)
        so_alto.append(np.hypot(a[0] - alvo[0], a[1] - alvo[1]))
        fundido.append(np.hypot(f[0] - alvo[0], f[1] - alvo[1]))
    assert np.median(fundido) < np.median(so_alto)


def test_a_incerteza_declarada_cai_quando_entra_a_segunda_camera():
    """1/s^2 = soma dos 1/s_i^2. A conta tem que aparecer na saida."""
    cams = _tres()
    alvo = (0.9, 0.9, 0.0)
    vistas = {p: _pixel(cams[p], alvo) for p in ("alto", "lateral")}
    _x1, _y1, s1, n1 = cams.no_chao({"alto": vistas["alto"]})
    _x2, _y2, s2, n2 = cams.no_chao(vistas)
    assert n1 == 1 and n2 == 2
    assert s2 < s1


def test_com_TRES_o_disparate_e_expulso_pela_mediana():
    """MEDIDO EM 20/08: sem porteiro, uma camera 40 px fora piorou 11 cm.

    Media ponderada pelo inverso da variancia e otima sob uma hipotese que
    ninguem enuncia: a de que cada medida erra dentro do proprio sigma. Ela
    protege contra IMPRECISAO declarada, nao contra ERRO GROSSEIRO — que
    chega com sigma pequeno e voto grande.

        Ponderar por incerteza declarada supoe que quem se declara preciso
        esta certo.
    """
    cams = _tres()
    alvo = (0.9, 0.9, 0.0)
    boas = {p: _pixel(cams[p], alvo) for p in ("alto", "frontal")}
    u, v = _pixel(cams["lateral"], alvo)
    com_disparate = dict(boas, lateral=(u + 60, v + 60))

    x1, y1, _s, n1 = cams.no_chao(boas)
    x2, y2, _s, n2 = cams.no_chao(com_disparate)
    assert n1 == 2 and n2 == 2, "a disparatada devia ter sido expulsa"
    assert np.hypot(x2 - x1, y2 - y1) < 0.02


def test_com_DUAS_que_discordam_a_duvida_cresce_em_vez_de_escolher():
    """Nao ha arbitro com duas testemunhas que se contradizem.

    Descartar a "pior" seria escolher pelo sigma — justamente o numero em
    que nao se pode confiar quando ha disparate.

        Com duas testemunhas que se contradizem, a resposta nao e escolher
        uma: e registrar que a duvida cresceu.
    """
    cams = _tres()
    alvo = (0.9, 0.9, 0.0)
    u, v = _pixel(cams["lateral"], alvo)
    juntas = {"alto": _pixel(cams["alto"], alvo), "lateral": (u, v)}
    brigadas = dict(juntas, lateral=(u + 60, v + 60))

    _x, _y, s_ok, _n = cams.no_chao(juntas)
    _x, _y, s_briga, n = cams.no_chao(brigadas)
    assert n == 2, "nao pode descartar nenhuma das duas"
    assert s_briga > 3 * s_ok, (
        f"o sigma devia inflar: {s_ok * 100:.1f} -> {s_briga * 100:.1f} cm")


def test_a_incerteza_no_chao_cresce_com_a_distancia():
    """Uma camera que ve a sala inteira nao mede a sala inteira igual.

        Um sensor sem mapa da propria incerteza so pode ser usado inteiro ou
        descartado inteiro.
    """
    cam = _tres()["alto"]
    perto = cam.incerteza_no_chao(*_pixel(cam, (0.9, 0.3, 0.0)))
    longe = cam.incerteza_no_chao(*_pixel(cam, (0.9, 2.6, 0.0)))
    assert longe > perto


def test_sem_camera_nenhuma_a_fusao_diz_que_nao_sabe():
    assert Camaras({}).no_chao({"alto": (100, 100)}) is None
    assert _tres().no_chao({}) is None


# --------------------------------------------------------- o carregador
def test_carregar_monta_a_do_alto_do_projeto_de_verdade():
    """A camera do teto ja tem homografia desde 08/08."""
    cams = Camaras.carregar(altura_da_camera={"alto": 2.23})
    assert "alto" in cams
    c = cams["alto"]
    assert c.posicao[2] == pytest.approx(2.23, abs=0.02)
    assert "trena" in c.origem_da_focal


def test_papel_sem_calibracao_simplesmente_nao_entra():
    """Uma camera sem lugar no mundo nao ajuda a dizer onde alguem esta."""
    cams = Camaras.carregar(papeis=("alto", "frontal", "lateral"))
    assert "frontal" not in cams and "lateral" not in cams
    assert cams.com_chao == ["alto"]


def test_carregar_de_uma_pasta_vazia_nao_estoura(tmp_path):
    (tmp_path / "calibracao").mkdir()
    assert len(Camaras.carregar(tmp_path)) == 0


def test_o_arquivo_antigo_do_alto_continua_valendo():
    """rodar.py, mapear.py e o --mono leem de homografia.json."""
    from pathlib import Path

    from src.mundo.camaras import _ler_homografia
    calib = Path(__file__).resolve().parent.parent / "calibracao"
    H, tam = _ler_homografia(calib, "alto")
    assert H is not None and tam == (640, 480)


# ============ A DISTORCAO CHEGA ATE A CAMARA (defeito achado em 20/08)
#
# `carregar` recebia `dist` de `_ler_homografia`, que nunca teve essa
# informacao e devolvia None sempre. O `intrinseca-<papel>.json` GRAVA os
# coeficientes desde o primeiro dia — e ninguem lia.
#
# O resultado seria mudo: `sem_distorcao` viraria uma funcao que nao faz
# nada, o barril da C920 continuaria torcendo as bordas, e o erro apareceria
# como centimetros no chao sem nada apontando para a lente.
#
#     Um dado gravado que ninguem le e pior que um dado ausente: o ausente
#     aparece na primeira execucao.


def _calibracao_falsa(tmp_path, com_distorcao=True):
    import json
    calib = tmp_path / "calibracao"
    calib.mkdir(parents=True, exist_ok=True)
    cam = _camara("alto", 551.0, (0.9, 0.2, 2.4))
    (calib / "homografia-alto.json").write_text(json.dumps({
        "H": cam.homografia.tolist(), "resolucao": [LARG, ALT]}))
    corpo = {"resolucao": [LARG, ALT], "K": cam.K.tolist()}
    if com_distorcao:
        corpo["dist"] = [-0.12, 0.08, 0.001, -0.002, 0.0]
    (calib / "intrinseca-alto.json").write_text(json.dumps(corpo))
    return tmp_path


def test_a_distorcao_do_tabuleiro_chega_na_camara(tmp_path):
    cams = Camaras.carregar(_calibracao_falsa(tmp_path))
    c = cams["alto"]
    assert c.origem_da_focal == "tabuleiro"
    assert c.dist is not None, "a distorcao foi gravada e nao foi lida"
    assert len(c.dist) == 5


def test_sem_distorcao_desfaz_o_barril_de_verdade(tmp_path):
    """Se `dist` chegar mas nao for usada, esta funcao vira um `return x`."""
    cams = Camaras.carregar(_calibracao_falsa(tmp_path))
    c = cams["alto"]
    borda = [(20.0, 30.0)]
    assert np.linalg.norm(c.sem_distorcao(borda)[0] - borda[0]) > 1.0


def test_sem_tabuleiro_a_distorcao_e_None_e_nao_zero(tmp_path):
    """Nao saber a distorcao e diferente de saber que ela e zero."""
    cams = Camaras.carregar(_calibracao_falsa(tmp_path, com_distorcao=False))
    assert cams["alto"].dist is None
    ponto = [(20.0, 30.0)]
    assert cams["alto"].sem_distorcao(ponto)[0] == pytest.approx(ponto[0])


def test_a_homografia_grava_a_resolucao_REAL(tmp_path):
    """`cam.set(FRAME_WIDTH, 640)` e um PEDIDO; num stream http nao faz nada.

        Resolucao pedida nao e resolucao obtida, e gravar a pedida e
        documentar uma intencao como se fosse uma medida.
    """
    import inspect

    import calibracao.homografia as hg
    fonte = inspect.getsource(hg.main)
    assert "frame.shape[1]" in fonte, "nao le a resolucao entregue"
    assert '"resolucao": [640, 480],' not in fonte, "ainda grava cravado"
