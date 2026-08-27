"""x, y sao o CENTRO, e o rumo chega ate o desenho.

Estes testes existem por causa de um defeito que nao dava erro nenhum.

`src/mundo/ambiente.py` media a estante e devolvia o CENTRO dela. O
`visual/cena3d.py` recebia o mesmo par e montava a caixa de (x, y) ate
(x+largura, y+profundidade) — tratando-o como CANTO. Para a estante de
0,92 x 0,30 isso e 46 cm de erro em x e 15 cm em y, e a tela nao reclamava:
desenhava uma estante perfeitamente retangular no lugar errado.

    Um erro que nao levanta excecao precisa de um teste que o levante.

E o `rumo_da_face`, que as cameras mediam e o `achar_ambiente.py` gravava no
JSON, era simplesmente ignorado por `Planta.carregar` — o campo existia no
arquivo e morria na leitura.
"""
import json
import math

import numpy as np
import pytest

from estado.planta import Planta
from visual.cena3d import Cena3D


# ------------------------------------------------------- o centro e o centro
def _planta(tmp_path, **extra):
    d = {"id": "t", "nome": "teste",
         "chao": {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2},
         "moveis": [dict({"id": "e1", "nome": "Estante", "tipo": "estante",
                          "x": 1.0, "y": 0.5, "largura": 0.92,
                          "profundidade": 0.30, "altura": 1.90}, **extra)]}
    p = tmp_path / "planta.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return Planta.carregar(p)


def test_movel_sem_rumo_continua_valendo(tmp_path):
    """Arquivo antigo, sem o campo, nao pode quebrar nem girar sozinho."""
    m = _planta(tmp_path).moveis[0]
    assert m.rumo_da_face == 0.0
    assert m.prateleiras == []


def test_rumo_do_arquivo_chega_ao_movel(tmp_path):
    """Era isto que se perdia: o campo existia no JSON e morria na leitura."""
    m = _planta(tmp_path, rumo_da_face=1.2345).moveis[0]
    assert m.rumo_da_face == pytest.approx(1.2345)


def test_prateleiras_do_arquivo_chegam_ao_movel(tmp_path):
    m = _planta(tmp_path, prateleiras=[{"id": "p1", "altura": 0.15},
                                       {"id": "p3", "altura": 0.95}]).moveis[0]
    assert m.prateleiras == [("p1", 0.15), ("p3", 0.95)]


def test_o_rumo_atravessa_ate_a_cena(tmp_path):
    """Ler o rumo e guardar sem repassar seria o mesmo defeito, mais fundo."""
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    _planta(tmp_path, rumo_da_face=0.7).aplicar_na_cena(cena)
    assert cena.moveis[0][6] == pytest.approx(0.7)


# ------------------------------------------------------- a caixa e girada
def _cantos_no_chao(x, y, larg, prof, rumo):
    """Os quatro pes que A CENA usa para desenhar — nao uma copia da conta.

    Reimplementar a geometria aqui provaria que eu sei somar, e nao que o
    programa desenha no lugar certo. `pes_do_movel` e a mesma funcao que
    `_movel` chama.
    """
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    cena.add_movel(x, y, larg, prof, 1.0, "m", rumo=rumo)
    return cena.pes_do_movel(cena.moveis[0])


def test_o_centro_dos_cantos_e_o_x_y_dado():
    """A prova do centro: a media dos quatro pes tem que voltar ao ponto dado.

    Com a convencao de canto, esta media daria (x + larg/2, y + prof/2) — e o
    movel apareceria deslocado meio corpo, que era o defeito de 14/08.
    """
    for rumo in (0.0, 0.5, -1.3, 3.0):
        pes = _cantos_no_chao(1.0, 0.5, 0.92, 0.30, rumo)
        assert pes.mean(axis=0) == pytest.approx([1.0, 0.5])


def test_sem_rumo_a_largura_fica_no_eixo_x():
    """Rumo zero tem que reproduzir o comportamento antigo, so que centrado."""
    pes = _cantos_no_chao(1.0, 0.5, 0.92, 0.30, 0.0)
    assert pes[:, 0].min() == pytest.approx(1.0 - 0.46)
    assert pes[:, 0].max() == pytest.approx(1.0 + 0.46)
    assert pes[:, 1].min() == pytest.approx(0.5 - 0.15)
    assert pes[:, 1].max() == pytest.approx(0.5 + 0.15)


def test_girado_noventa_graus_troca_os_lados():
    pes = _cantos_no_chao(1.0, 0.5, 0.92, 0.30, math.pi / 2)
    assert pes[:, 0].max() - pes[:, 0].min() == pytest.approx(0.30)
    assert pes[:, 1].max() - pes[:, 1].min() == pytest.approx(0.92)


def test_a_largura_corre_ao_longo_de_cos_sin():
    """A mesma convencao de `ambiente.relacao`. Se divergirem, o desenho
    contradiz a conta que decide quem esta na frente da estante."""
    rumo = 0.6
    pes = _cantos_no_chao(0.0, 0.0, 1.0, 0.2, rumo)
    ao_longo = np.array([math.cos(rumo), math.sin(rumo)])
    projecao = pes @ ao_longo
    assert projecao.max() - projecao.min() == pytest.approx(1.0)


def test_o_movel_entra_na_chave_do_cache():
    """Mover a estante sem mudar a contagem deixava o desenho velho em cache."""
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    cena.add_movel(1.0, 0.5, 0.9, 0.3, 1.9, "e")
    antes = cena._chave_camera()
    cena.moveis[0] = (1.4, 0.5, 0.9, 0.3, 1.9, "e", 0.0, ())
    assert cena._chave_camera() != antes


# ------------------------------------------------------- desenhar de verdade
#
# ESTES TESTES EXISTEM PORQUE OS DE CIMA NAO BASTARAM.
#
# `pes_do_movel` foi testado por todos os angulos e passou. O `_movel`, que
# CHAMA `pes_do_movel` e desenha, ficou com uma variavel orfa (`rotulo`) da
# refatoracao — e os 570 testes seguiram verdes, porque nenhum deles chegava
# a desenhar um movel. O erro so apareceu com as tres cameras ligadas, na
# primeira vez que o quarto tinha uma estante dentro.
#
#     Testar a funcao que faz a conta e testar a conta. Enquanto ninguem
#     chamar quem usa a conta, o caminho ate ela continua sem prova.
#
# Desenhar num quadro de 320x240 custa milissegundos. Era o que faltava.

def test_desenhar_uma_cena_com_movel_nao_estoura():
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    cena.add_movel(1.0, 0.5, 0.92, 0.30, 1.90, "Estante", rumo=0.9)
    img = cena.desenhar([], titulo="teste")
    assert img.shape == (240, 320, 3)


def test_desenhar_movel_sem_rotulo_tambem_nao_estoura():
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    cena.add_movel(1.0, 0.5, 0.92, 0.30, 1.90)
    assert cena.desenhar([]) is not None


def test_desenhar_a_planta_inteira_do_arquivo(tmp_path):
    """O caminho completo: JSON -> Planta -> Cena -> imagem."""
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    _planta(tmp_path, rumo_da_face=2.55).aplicar_na_cena(cena)
    assert cena.desenhar([]) is not None


def test_desenhar_o_quarto_real():
    """A planta que o `rodar.py` carrega por padrao. Foi ela que estourou."""
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    Planta.carregar("loja/quarto.json").aplicar_na_cena(cena)
    assert cena.desenhar([]) is not None


def test_desenhar_a_bancada():
    cena = Cena3D(320, 240, chao=(-1.5, 3.0, -1.5, 3.0))
    Planta.carregar("loja/bancada.json").aplicar_na_cena(cena)
    assert cena.desenhar([]) is not None


# ------------------------------------------------------- a loja ficticia
def test_bancada_migrada_cabe_no_chao():
    """A conversao canto->centro nao pode ter jogado movel para fora do piso."""
    planta = Planta.carregar("loja/bancada.json")
    x0, x1, y0, y1 = planta.chao
    for m in planta.moveis:
        assert x0 <= m.x <= x1, f"{m.id} saiu do chao em x"
        assert y0 <= m.y <= y1, f"{m.id} saiu do chao em y"


# ------------------------------------- a camera virtual olha a loja de frente
#
# "a estante aparecia na frente do gemeo digital e na verdade e ao contrario,
#  eu estou na frente e ela esta atras"                     — Eduardo, 20/08
#
# O modelo estava certo: o motor punha a pessoa em (+0,29, +0,95), do lado da
# face (cos +0,94). Errado estava o ponto de vista — azimute -60 fixo, escrito
# antes de existir movel nenhum.
def _olho(cena):
    return np.asarray(cena.cam.posicao[:2], dtype=float)


def _normal(rumo):
    return np.array([-math.sin(rumo), math.cos(rumo)])


@pytest.mark.parametrize("rumo", [0.0, 0.54, -0.56, 1.9, 3.0, -2.4])
def test_a_camera_nasce_do_lado_da_face(rumo):
    """Para qualquer rumo, a camera fica do lado de quem pega produto."""
    cena = Cena3D(320, 240, chao=(-0.83, 3.24, -1.86, 2.11))
    cena.add_movel(0.68, 0.65, 0.92, 0.30, 1.90, "estante", rumo=rumo)
    assert cena.olhar_pela_face()
    para_o_olho = _olho(cena) - np.array([0.68, 0.65])
    para_o_olho /= np.linalg.norm(para_o_olho)
    assert float(para_o_olho @ _normal(rumo)) > 0.5, "olhando o fundo"


def test_o_caso_do_eduardo_a_pessoa_fica_na_frente_da_estante():
    """A queixa dele, virada numa conta.

    Ele em (+0,29, +0,95), a estante em (+0,68, +0,65) com a face em +31
    graus. Ele TEM que estar mais perto da camera virtual do que ela.
    """
    cena = Cena3D(960, 620, chao=(-0.83, 3.24, -1.86, 2.11))
    cena.add_movel(0.68, 0.65, 0.92, 0.30, 1.90, "estante",
                   rumo=math.radians(31))
    cena.olhar_pela_face()
    olho = _olho(cena)
    dele = float(np.linalg.norm(olho - np.array([0.29, 0.95])))
    dela = float(np.linalg.norm(olho - np.array([0.68, 0.65])))
    assert dele < dela, f"a pessoa a {dele:.2f} m, a estante a {dela:.2f} m"


def test_com_o_azimute_de_antes_a_queixa_se_reproduz():
    """A guarda: se alguem devolver o -60 fixo, este teste cai.

    Sem ele, nada impede o numero de voltar — e o defeito nao levanta excecao,
    so desenha errado.
    """
    cena = Cena3D(960, 620, chao=(-0.83, 3.24, -1.86, 2.11))
    cena.add_movel(0.68, 0.65, 0.92, 0.30, 1.90, "estante",
                   rumo=math.radians(31))
    cena.cam.azimute = np.deg2rad(-60)          # como era ate 20/08
    olho = _olho(cena)
    assert (np.linalg.norm(olho - np.array([0.29, 0.95]))
            > np.linalg.norm(olho - np.array([0.68, 0.65])))


def test_a_maior_face_e_que_manda():
    """Com dois moveis, quem escolhe o angulo e a estante, nao o balcao."""
    cena = Cena3D(320, 240, chao=(0.0, 4.0, 0.0, 4.0))
    cena.add_movel(3.0, 3.0, 1.60, 0.50, 0.90, "balcao", rumo=0.0)
    cena.add_movel(1.0, 1.0, 0.92, 0.30, 1.90, "estante", rumo=math.pi / 2)
    cena.olhar_pela_face()
    para_o_olho = _olho(cena) - np.array([1.0, 1.0])
    para_o_olho /= np.linalg.norm(para_o_olho)
    assert float(para_o_olho @ _normal(math.pi / 2)) > 0.5


def test_sem_movel_nao_inventa_angulo():
    cena = Cena3D(320, 240, chao=(0.0, 2.0, 0.0, 2.0))
    antes = cena.cam.azimute
    assert cena.olhar_pela_face() is False
    assert cena.cam.azimute == antes


def test_a_planta_ja_aponta_a_camera(tmp_path):
    """`aplicar_na_cena` faz as duas coisas: poe os moveis e vira a camera."""
    p = _planta(tmp_path, x=0.68, y=0.65,
                rumo_da_face=math.radians(31))
    cena = Cena3D(320, 240, chao=p.chao)
    p.aplicar_na_cena(cena)
    para_o_olho = _olho(cena) - np.array([0.68, 0.65])
    para_o_olho /= np.linalg.norm(para_o_olho)
    assert float(para_o_olho @ _normal(math.radians(31))) > 0.5
