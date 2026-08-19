"""O tamanho do comodo vem de quem mede posicao, nao de quem reconhece objeto.

    o ambiente precisa crescer e precisamos pensar como monitorar mais a area
                                                        — Eduardo, 19/08

O DEFEITO QUE ESTES TESTES TRANCAM

Ate 19/08 o piso do gemeo era a extensao da nuvem do DUSt3R — os pontos que a
rede conseguiu casar entre as tres vistas. Medido na corrida do dia anterior:

    quarto desenhado pela nuvem       2,1 m2
    chao que a camera do alto mede    8,4 m2

Quatro vezes menor, e nao por falta de camera nem de calibracao: por deixar o
instrumento errado responder a pergunta.

    A reconstrucao diz o que TEM na sala. Quem sabe o TAMANHO dela e o campo
    de visao de quem mede posicao.

E o retangulo de 1,65 x 1,32 da calibracao nunca foi o limite da medida — era
o alcance da TRENA. A homografia mapeia o plano do chao inteiro.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from estado.planta import Planta
from percepcao.chao import caixa_do_contorno, pegada_no_chao
from visual.cena3d import CHAO, FUNDO, Cena3D

RAIZ = Path(__file__).resolve().parent.parent


def _area(poligono):
    p = np.asarray(poligono, dtype=float)
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2.0)


def _dentro(poligono, ponto):
    import cv2
    return cv2.pointPolygonTest(
        np.asarray(poligono, dtype=np.float32), tuple(map(float, ponto)),
        False) >= 0


# ------------------------------------------------------- a pegada, em geral
def test_camera_a_pino_devolve_o_retangulo_que_ela_ve():
    """Sem perspectiva, a pegada e a propria imagem em escala. Verdade conhecida."""
    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])   # 1 px = 1 cm
    c = pegada_no_chao(H, 640, 480)
    assert c is not None
    assert _area(c) == pytest.approx(6.39 * 4.79, rel=0.01)
    assert caixa_do_contorno(c) == pytest.approx((0, 6.39, 0, 4.79), abs=0.01)


def test_a_pegada_e_um_quadrilatero_e_nao_catorze_pontos():
    """O fecho de uma grade traz vertices quase colineares. Sao ruido."""
    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])
    assert len(pegada_no_chao(H, 640, 480)) == 4


def test_perspectiva_faz_o_piso_deixar_de_ser_retangulo():
    """E por isso que a caixa nao serve para desenhar: ela infla o comodo."""
    H = np.array(json.loads((RAIZ / "calibracao" / "homografia.json")
                            .read_text(encoding="utf-8"))["H"])
    c = pegada_no_chao(H, 640, 480)
    x0, x1, y0, y1 = caixa_do_contorno(c)
    assert _area(c) < 0.6 * (x1 - x0) * (y1 - y0), (
        "a caixa deveria ser bem maior que o piso; se nao for, o piso e "
        "quase retangular e este teste perdeu o sentido")


# ------------------------------------------------------------- o horizonte
def test_o_que_esta_alem_do_horizonte_fica_de_fora():
    """Alem da linha do horizonte o plano cai ATRAS da camera.

    `w = h31*u + h32*v + 1` zera ali. Um pixel adiante volta com sinal
    trocado — chao fantasma do outro lado do mundo. Sem este corte, a pegada
    nao seria so grande demais: seria de outro lugar.
    """
    # horizonte em v = 250
    H = np.array([[0.005, 0, 0], [0, 0.005, 0], [0, -0.004, 1.0]])
    c = pegada_no_chao(H, 640, 480, cm_por_pixel_maximo=100.0)
    assert c is not None
    Hi = np.linalg.inv(H)
    for x, y in c:
        p = Hi @ np.array([x, y, 1.0])
        assert p[1] / p[2] < 250.0, "aceitou pixel alem do horizonte"


def test_quanto_mais_folga_de_resolucao_mais_perto_do_horizonte_se_chega():
    """A fronteira e continua: nao ha borda subita, ha uma escolha declarada."""
    H = np.array([[0.005, 0, 0], [0, 0.005, 0], [0, -0.004, 1.0]])
    areas = [_area(pegada_no_chao(H, 640, 480, cm_por_pixel_maximo=lim))
             for lim in (5.0, 20.0, 100.0)]
    assert areas[0] < areas[1] < areas[2]


def test_camera_que_nao_serve_devolve_None_em_vez_de_um_chao_ruim():
    """Um pixel valendo metros continua sendo chao e ja nao responde 'onde'."""
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])      # 1 px = 1 m
    assert pegada_no_chao(H, 640, 480, cm_por_pixel_maximo=5.0) is None


# ------------------------------------------------- a camera de verdade dele
def test_a_camera_do_alto_ja_media_quatro_vezes_o_quarto_desenhado():
    """O numero que motivou tudo isto. Medido em 19/08, sem hardware novo."""
    H = np.array(json.loads((RAIZ / "calibracao" / "homografia.json")
                            .read_text(encoding="utf-8"))["H"])
    assert _area(pegada_no_chao(H, 640, 480)) == pytest.approx(8.4, abs=0.3)


def test_o_retangulo_da_trena_cabe_inteiro_na_pegada():
    """A trena mediu 1,65 x 1,32. A camera mede muito mais que isso.

    Se um canto da calibracao caisse fora, seria sinal de que a pegada esta
    errada — nao de que o comodo e pequeno.
    """
    H = np.array(json.loads((RAIZ / "calibracao" / "homografia.json")
                            .read_text(encoding="utf-8"))["H"])
    c = pegada_no_chao(H, 640, 480)
    for canto in [(0, 0), (1.65, 0), (1.65, 1.32), (0, 1.32)]:
        assert _dentro(c, canto), f"canto da trena {canto} ficou fora do piso"


def test_a_estante_medida_ontem_cai_dentro_do_piso_novo():
    """O mundo cresceu; o que ja estava nele continua no lugar."""
    H = np.array(json.loads((RAIZ / "calibracao" / "homografia.json")
                            .read_text(encoding="utf-8"))["H"])
    c = pegada_no_chao(H, 640, 480)
    q = json.loads((RAIZ / "loja" / "quarto.json").read_text(encoding="utf-8"))
    e = q["moveis"][0]
    assert _dentro(c, (e["x"], e["y"])), "a estante ficou fora do piso"


def test_ninguem_que_a_camera_DETECTE_pode_cair_fora_do_piso():
    """A garantia que o contorno traz de graca, e que faltava.

    A posicao do boneco sai de `para_metros(H, pixel_do_pe)`. O piso sai da
    MESMA matriz aplicada a MESMA imagem. Entao todo pe detectavel esta, por
    construcao, dentro do piso desenhado — nao por sorte, por identidade.

        Com o piso vindo da nuvem isso nao valia: era outro instrumento
        respondendo, e o boneco andava para fora do comodo.

    A tolerancia e de um MILIMETRO, e ela existe por um motivo especifico: os
    pixels da borda da imagem caem exatamente SOBRE a aresta do poligono, e
    ali quem decide de que lado eles estao e o arredondamento do ponto
    flutuante. Medido: o pior sai por 0,0002 micrometro.

        Exigir desigualdade estrita de um ponto que esta na fronteira nao
        testa geometria: testa o ultimo bit da mantissa.
    """
    import cv2

    from percepcao.chao import para_metros
    H = np.array(json.loads((RAIZ / "calibracao" / "homografia.json")
                            .read_text(encoding="utf-8"))["H"])
    c = pegada_no_chao(H, 640, 480).astype(np.float32)
    for u in range(0, 640, 37):
        for v in range(0, 480, 41):
            fundura = cv2.pointPolygonTest(c, para_metros(H, u, v), True)
            assert fundura > -0.001, (
                f"o pixel ({u},{v}) da um pe {-fundura * 100:.1f} cm fora do "
                f"piso desenhado")


# --------------------------------------------------------------- a caixa
def test_a_caixa_contem_tudo_que_foi_medido():
    """Caixa que nao contem tudo recorta em silencio."""
    from ferramentas.mapear import _caixa

    class FalsoAmbiente:
        chao = (-2.0, 0.5, 0.0, 0.4)

    contorno = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
    assert _caixa(FalsoAmbiente(), contorno) == (-2.0, 3.0, 0.0, 2.0)


def test_caixa_de_contorno_vazio_e_None():
    assert caixa_do_contorno(None) is None
    assert caixa_do_contorno([]) is None


# ------------------------------------------------------------- o desenho
def _cena_com_triangulo():
    """Piso triangular: metade da caixa e chao, metade nao.

    Um triangulo e o formato mais simples em que 'desenhou o contorno' e
    'desenhou a caixa' dao imagens diferentes. Com um retangulo os dois
    coincidem e o teste passaria sem provar nada.
    """
    contorno = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    return Cena3D(480, 360, chao=(0.0, 4.0, 0.0, 4.0), contorno=contorno)


def _cor_em(cena, x, y):
    img = cena.desenhar([], titulo="")
    (p,), z = cena.cam.projetar([[x, y, 0.0]])
    assert z[0] > 0
    return img[int(p[1]), int(p[0])]


def test_o_chao_e_desenhado_no_formato_do_contorno():
    cena = _cena_com_triangulo()
    dentro = _cor_em(cena, 0.6, 0.6)          # dentro do triangulo
    assert tuple(int(v) for v in dentro) != FUNDO, "o piso nao foi desenhado"


def test_fora_do_contorno_nao_se_inventa_chao():
    """O canto oposto da caixa esta DENTRO do retangulo e FORA do piso."""
    cena = _cena_com_triangulo()
    fora = tuple(int(v) for v in _cor_em(cena, 3.4, 3.4))
    assert fora == FUNDO, (
        f"desenhou chao em (3.4, 3.4), que a camera nunca mediu: {fora}")


def test_sem_contorno_o_desenho_continua_como_antes():
    """Planta antiga nao quebra: cai no retangulo."""
    cena = Cena3D(480, 360, chao=(0.0, 4.0, 0.0, 4.0), contorno=None)
    assert tuple(int(v) for v in _cor_em(cena, 3.4, 3.4)) == CHAO


def test_contorno_degenerado_e_tratado_como_ausente():
    """Dois pontos nao fazem um piso."""
    cena = Cena3D(480, 360, chao=(0.0, 4.0, 0.0, 4.0),
                  contorno=[(0.0, 0.0), (1.0, 1.0)])
    assert cena.contorno is None


def test_a_grade_nao_passa_do_piso():
    """Risco de grade no vazio sugere chao onde a camera nao olhou."""
    cena = _cena_com_triangulo()
    img = cena.desenhar([], titulo="")
    # o vertice (4,4) da caixa fica fora do triangulo; a 1 m dele, idem
    for x, y in [(3.9, 3.9), (3.0, 3.0)]:
        (p,), z = cena.cam.projetar([[x, y, 0.0]])
        if z[0] <= 0:
            continue
        assert tuple(int(v) for v in img[int(p[1]), int(p[0])]) == FUNDO


# --------------------------------------------------------------- a planta
def test_a_planta_le_o_contorno_quando_existe(tmp_path):
    d = json.loads((RAIZ / "loja" / "quarto.json").read_text(encoding="utf-8"))
    d["contorno"] = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]]
    p = tmp_path / "q.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert Planta.carregar(p).contorno == ((0.0, 0.0), (3.0, 0.0),
                                           (3.0, 2.0), (0.0, 2.0))


def test_planta_sem_contorno_carrega_igual(tmp_path):
    d = json.loads((RAIZ / "loja" / "quarto.json").read_text(encoding="utf-8"))
    d.pop("contorno", None)
    p = tmp_path / "q.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert Planta.carregar(p).contorno == ()


def test_a_camera_virtual_se_afasta_quando_o_comodo_cresce():
    """Comodo de 4 m nao se enquadra de onde se enquadrava um de 1,5."""
    perto = Cena3D(480, 360, chao=(0.0, 1.5, 0.0, 1.5))
    longe = Cena3D(480, 360, chao=(-1.0, 3.3, -1.9, 2.2))
    assert longe.cam.dist > perto.cam.dist
    assert longe.cam.alvo[0] == pytest.approx(1.15)


# ------------------------------------------------------- o enquadramento
@pytest.mark.parametrize("chao", [
    (0.0, 1.5, 0.0, 1.5),            # a bancada de teste
    (-0.0, 1.458, 0.201, 1.666),     # o quarto de 18/08
    (-0.83, 3.24, -1.87, 2.11),      # o quarto de 19/08
    (-5.0, 9.0, -4.0, 7.0),          # uma loja de verdade
])
def test_o_comodo_inteiro_cabe_na_tela_em_qualquer_tamanho(chao):
    """A regra de bolso acertava o caso em que foi afinada. A conta acerta todos.

    Inclui o piso a 1,85 m: cortar a cabeca de quem esta no canto e o mesmo
    defeito que cortar o canto.
    """
    cena = Cena3D(480, 360, chao=chao)
    x0, x1, y0, y1 = chao
    cantos = [[x, y, z] for x in (x0, x1) for y in (y0, y1)
              for z in (0.0, 1.85)]
    p, z = cena.cam.projetar(cantos)
    assert (z > 0).all()
    assert p[:, 0].min() >= -1 and p[:, 0].max() <= 481
    assert p[:, 1].min() >= -1 and p[:, 1].max() <= 361


def test_o_enquadramento_e_o_mais_perto_que_cabe_e_nao_mais():
    """Longe demais transforma uma pessoa de 1,75 m num risco na tela.

    "Mais perto que cabe" e uma afirmacao sobre a MARGEM de 5%, que e o que a
    busca resolve — nao sobre a borda crua da tela. Testar contra a borda
    daria folga de 5% de graca e o teste passaria com a resposta errada.
    """
    cena = Cena3D(480, 360, chao=(-0.83, 3.24, -1.87, 2.11))
    x0, x1, y0, y1 = cena.chao
    cantos = [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0.0, 1.85)]
    mx, my = 480 * 0.05, 360 * 0.05

    p, _ = cena.cam.projetar(cantos)
    assert p[:, 0].min() >= mx - 1 and p[:, 1].min() >= my - 1

    cena.cam.dist *= 0.97                     # 3% mais perto
    p, _ = cena.cam.projetar(cantos)
    assert (p[:, 0].min() < mx or p[:, 0].max() > 480 - mx
            or p[:, 1].min() < my or p[:, 1].max() > 360 - my), (
        "daria para chegar 3% mais perto sem invadir a margem")


def test_o_enquadramento_segue_o_contorno_e_nao_a_caixa():
    """O piso e o que se enquadra. A caixa em volta dele nao existe na tela.

    Um contorno pequeno no meio de uma caixa grande e o caso limpo: se o
    enquadramento olhasse a caixa, os dois dariam a mesma distancia.
    """
    caixa = Cena3D(480, 360, chao=(0.0, 10.0, 0.0, 10.0))
    miudo = Cena3D(480, 360, chao=(0.0, 10.0, 0.0, 10.0),
                   contorno=[(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)])
    assert miudo.cam.dist < caixa.cam.dist / 2


def test_o_alvo_da_camera_e_o_centro_do_comodo_e_nao_a_origem():
    """Com o comodo indo a x negativo, mirar (0,0) joga metade para fora."""
    cena = Cena3D(480, 360, chao=(-0.83, 3.24, -1.87, 2.11))
    assert cena.cam.alvo[0] == pytest.approx((-0.83 + 3.24) / 2)
    assert cena.cam.alvo[1] == pytest.approx((-1.87 + 2.11) / 2)


def test_a_area_do_piso_cresceu_de_verdade():
    """A prova de ponta a ponta: o numero que o Eduardo vai ver na tela."""
    H = np.array(json.loads((RAIZ / "calibracao" / "homografia.json")
                            .read_text(encoding="utf-8"))["H"])
    antes = 1.458 * (1.666 - 0.201)          # o quarto.json de 18/08
    depois = _area(pegada_no_chao(H, 640, 480))
    assert depois / antes > 3.5, f"cresceu so {depois / antes:.1f}x"
    assert math.isclose(antes, 2.14, abs_tol=0.05)
