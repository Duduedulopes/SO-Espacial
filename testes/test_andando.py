"""A pessoa andando calibra a camera, provado contra verdade conhecida.

    nao vou imprimir nada, a gente perdeu dias tentando fazer com que a
    camera fizesse esse trabalho                     — Eduardo, 20/08

Monta-se uma camera SINTETICA com K, R e t escolhidos a mao. Simula-se uma
caminhada: a pessoa passeia pelo chao, e a cada instante o pe (z=0) e a
cabeca (z=1,80) sao projetados nessa camera. Exige-se que `resolver` devolva
os numeros de onde se partiu.

    Verdade conhecida nao e uma aproximacao melhor do real: e a unica
    situacao em que um erro pode ser MEDIDO em vez de estimado.
"""
import numpy as np
import pytest

from src.mundo.andando import (ERRO_MAXIMO_PX, MINIMO_DE_PARES, Coleta,
                               diagnostico, homografia_da_pose,
                               pares_do_instante, resolver)

LARG, ALT = 640, 480
ESTATURA = 1.80


def _camera(focal=600.0, posicao=(-1.4, 1.0, 1.6), olhando=(0.9, 0.9, 0.9)):
    C = np.array(posicao, dtype=float)
    frente = np.array(olhando, dtype=float) - C
    frente /= np.linalg.norm(frente)
    cima = np.array([0.0, 0.0, 1.0])
    direita = np.cross(frente, cima)
    direita /= np.linalg.norm(direita)
    baixo = np.cross(frente, direita)
    R = np.stack([direita, baixo, frente])
    t = -R @ C
    K = np.array([[focal, 0, LARG / 2], [0, focal, ALT / 2], [0, 0, 1.0]])
    return K, R, t


def _projetar(K, R, t, ponto):
    p = R @ np.asarray(ponto, dtype=float) + t
    if p[2] <= 1e-9:
        return None
    q = K @ p
    u, v = q[0] / q[2], q[1] / q[2]
    if not (0 <= u < LARG and 0 <= v < ALT):
        return None
    return float(u), float(v)


def _caminhada(K, R, t, passos=90, ruido_px=0.0, semente=5,
               area=(0.2, 1.8, 0.2, 1.8)):
    """A pessoa passeia pelo chao. Devolve uma Coleta pronta."""
    rng = np.random.default_rng(semente)
    x0, x1, y0, y1 = area
    c = Coleta()
    for k in range(passos):
        x = x0 + (x1 - x0) * (0.5 + 0.5 * np.sin(k * 0.21))
        y = y0 + (y1 - y0) * (0.5 + 0.5 * np.cos(k * 0.13))
        pixeis = {}
        pe = _projetar(K, R, t, (x, y, 0.0))
        cab = _projetar(K, R, t, (x, y, ESTATURA))
        if pe is None and cab is None:
            continue
        def sujar(p):
            if p is None or ruido_px <= 0:
                return p
            return (p[0] + rng.normal(0, ruido_px),
                    p[1] + rng.normal(0, ruido_px))
        pixeis["lateral"] = {"pe": sujar(pe), "cabeca": sujar(cab)}
        c.quadros += 1
        for _papel, pares in pares_do_instante((x, y), ESTATURA,
                                               pixeis).items():
            for mundo, pixel in pares:
                c.juntar(mundo, pixel)
    return c


# ------------------------------------------------------- o teste central
def test_a_caminhada_devolve_a_camera_de_onde_ela_veio():
    """Sem ruido, os numeros tem que voltar."""
    K, R, t = _camera(focal=600.0)
    achado = resolver(_caminhada(K, R, t), (LARG, ALT))
    assert achado is not None
    K2, _d, R2, t2, erro = achado
    assert K2[0, 0] == pytest.approx(600.0, rel=0.03)
    assert erro < 0.5, f"{erro:.2f} px"
    assert float(np.abs(R2 - R).max()) < 0.05
    assert t2 == pytest.approx(t, abs=0.10)


@pytest.mark.parametrize("focal", [420.0, 520.0, 600.0])
def test_focais_diferentes_voltam(focal):
    K, R, t = _camera(focal=focal)
    achado = resolver(_caminhada(K, R, t), (LARG, ALT))
    assert achado is not None
    assert achado[0][0, 0] == pytest.approx(focal, rel=0.05)


def test_focal_longa_demais_perde_a_cabeca_e_e_recusada():
    """Nao e defeito: e a guarda funcionando.

    Com 850 px num quadro de 640, o campo cai para 41 graus e a cabeca sai
    da imagem. Sobram so os pes — todos em z=0, coplanares — e a focal fica
    indeterminada. `resolver` recusa, que e a resposta certa.
    """
    K, R, t = _camera(focal=850.0)
    c = _caminhada(K, R, t)
    assert c.alturas == 1, "o teste nao reproduziu a perda da cabeca"
    assert resolver(c, (LARG, ALT)) is None


def test_o_erro_de_reprojecao_nao_desce_abaixo_do_ruido():
    """O piso do instrumento. Medido: erro = ruido x raiz(2).

        Um limiar copiado de outro instrumento mede o outro instrumento.

    A regra "< 2 px" vale para tabuleiro, cujo canto e sub-pixel. Aqui o
    canto e um tornozelo que oscila 3 px, e exigir 2 reprovaria uma
    calibracao perfeita.
    """
    K, R, t = _camera()
    for ruido, esperado in ((1.0, 1.41), (3.0, 4.24)):
        r = resolver(_caminhada(K, R, t, passos=200, ruido_px=ruido),
                     (LARG, ALT))
        assert r is not None
        assert r[4] == pytest.approx(esperado, rel=0.25)
    assert ERRO_MAXIMO_PX > 4.24, "o limiar esta abaixo do piso do ruido"


def test_a_posicao_da_camera_volta():
    """`C = -R^T t`. E o numero que da para conferir com a trena."""
    K, R, t = _camera(posicao=(-1.4, 1.0, 1.6))
    _K, _d, R2, t2, _e = resolver(_caminhada(K, R, t), (LARG, ALT))
    assert (-R2.T @ t2) == pytest.approx((-1.4, 1.0, 1.6), abs=0.10)


def test_aguenta_ruido_de_deteccao():
    """O tornozelo do detector oscila alguns pixels. Medido: cerca de 3."""
    K, R, t = _camera()
    achado = resolver(_caminhada(K, R, t, passos=200, ruido_px=3.0),
                      (LARG, ALT))
    assert achado is not None
    K2, _d, _R, _t, erro = achado
    assert K2[0, 0] == pytest.approx(600.0, rel=0.12)
    assert erro < ERRO_MAXIMO_PX


# --------------------------------------------- o que NAO pode ser aceito
def test_so_o_chao_nao_calibra():
    """Pontos coplanares nao determinam a focal.

    A cabeca e o que tira o problema da degenerescencia: ela poe metade dos
    pontos num segundo plano. Sem ela, a solucao existe e nao significa nada.
    """
    K, R, t = _camera()
    c = _caminhada(K, R, t)
    so_chao = Coleta(quadros=c.quadros)
    for m, p in zip(c.mundo, c.pixel):
        if m[2] == 0.0:
            so_chao.juntar(m, p)
    assert so_chao.alturas == 1
    assert resolver(so_chao, (LARG, ALT)) is None


def test_poucos_pares_sao_recusados():
    """Seis pontos calibram e nao merecem confianca."""
    K, R, t = _camera()
    c = _caminhada(K, R, t, passos=8)
    assert len(c) < MINIMO_DE_PARES
    assert resolver(c, (LARG, ALT)) is None


def test_andar_em_cima_de_um_ponto_so_e_denunciado():
    """Cem pares identicos valem um par."""
    K, R, t = _camera()
    parado = _caminhada(K, R, t, passos=120, area=(0.9, 0.95, 0.9, 0.95))
    linhas, bom = diagnostico(parado, resolver(parado, (LARG, ALT)),
                              (LARG, ALT))
    assert not bom
    assert any("area MAIOR" in l for l in linhas)


def test_coleta_vazia_nao_estoura():
    assert resolver(Coleta(), (LARG, ALT)) is None
    linhas, bom = diagnostico(Coleta(), None, (LARG, ALT))
    assert not bom and any("NAO RESOLVEU" in l for l in linhas)


# ------------------------------------------------------ a saida encaixa
def test_a_homografia_da_pose_bate_com_o_chao():
    """A camera nova entra no mesmo formato das que ja existem.

    Se a homografia derivada da pose nao levasse pixel de volta a metro, a
    camera calibrada andando nao poderia conversar com a do teto — e a fusao
    somaria mundos diferentes.
    """
    K, R, t = _camera()
    H = homografia_da_pose(K, R, t)
    assert H is not None
    for x, y in [(0.6, 0.6), (1.2, 1.4), (0.3, 1.7)]:
        uv = _projetar(K, R, t, (x, y, 0.0))
        if uv is None:
            continue
        p = H @ np.array([uv[0], uv[1], 1.0])
        assert (p[0] / p[2], p[1] / p[2]) == pytest.approx((x, y), abs=1e-6)


def test_camera_enterrada_no_chao_e_recusada():
    """A solucao espelhada nao descreve aparelho nenhum."""
    K, R, t = _camera(posicao=(-1.4, 1.0, 1.6))
    c = _caminhada(K, R, t)
    espelhada = Coleta(quadros=c.quadros)
    for (x, y, z), p in zip(c.mundo, c.pixel):
        espelhada.juntar((x, y, -z), p)          # cabeca DEBAIXO do chao
    r = resolver(espelhada, (LARG, ALT))
    assert r is None or float((-r[2].T @ r[3])[2]) > 0


def test_o_diagnostico_traz_o_erro_de_reprojecao():
    """Tres numeros decidem se vale gravar, e nenhum e opiniao."""
    K, R, t = _camera()
    c = _caminhada(K, R, t)
    linhas, bom = diagnostico(c, resolver(c, (LARG, ALT)), (LARG, ALT))
    assert bom
    assert any("ERRO DE REPROJECAO" in l for l in linhas)
    assert any("cobriu" in l for l in linhas)


def test_pares_do_instante_sem_cabeca_ainda_da_o_pe():
    """Nenhuma vista ve tudo sempre. Meio par e melhor que par nenhum."""
    fora = pares_do_instante((1.0, 0.5), ESTATURA,
                             {"lateral": {"pe": (10, 20), "cabeca": None}})
    assert len(fora["lateral"]) == 1
    assert fora["lateral"][0][0] == (1.0, 0.5, 0.0)


def test_pares_do_instante_sem_estatura_nao_inventa_a_cabeca():
    fora = pares_do_instante((1.0, 0.5), None,
                             {"lateral": {"pe": (10, 20), "cabeca": (10, 5)}})
    assert len(fora["lateral"]) == 1


# ===== O AJUSTE ROBUSTO (20/08, medido sobre a caminhada real do Eduardo)
#
# `calibrateCamera` e minimos quadrados puro — nao tem RANSAC. Sobre os dados
# reais, so 16% dos pares fechavam dentro de 5 cm e 51% dentro de 20. Nao ha
# separacao limpa entre bom e ruim: e ruido continuo com cauda longa, e um
# passo chegou a 74 cm.
#
#     Numa media, um ponto ruim dilui. Num ajuste, ele arrasta — e no
#     quadrado do erro, ele arrasta pelo quadrado.


def test_uma_cauda_longa_de_erros_nao_arrasta_o_ajuste():
    """Um decimo dos pontos com erro grosseiro. Sem robustez, a focal foge."""
    K, R, t = _camera(focal=600.0)
    c = _caminhada(K, R, t, passos=250, ruido_px=1.0)
    rng = np.random.default_rng(11)
    sujo = Coleta(quadros=c.quadros)
    for k, (m, p) in enumerate(zip(c.mundo, c.pixel)):
        if k % 10 == 0:                        # 10% de disparate
            p = (p[0] + rng.normal(0, 120), p[1] + rng.normal(0, 120))
        sujo.juntar(m, p)

    achado = resolver(sujo, (LARG, ALT))
    assert achado is not None
    assert achado[0][0, 0] == pytest.approx(600.0, rel=0.15), (
        f"a focal fugiu para {achado[0][0, 0]:.0f}")


def test_dados_limpos_nao_sao_prejudicados_pela_robustez():
    """Expulsar quem nao ha nao pode custar precisao."""
    K, R, t = _camera(focal=600.0)
    achado = resolver(_caminhada(K, R, t), (LARG, ALT))
    assert achado[0][0, 0] == pytest.approx(600.0, rel=0.03)
    assert achado[4] < 0.5


def test_cinco_alturas_valem_mais_que_duas_quase_juntas():
    """ERRO MEU: ombro e nariz ficam a 17 cm um do outro.

    Dois planos a 17 cm, com a camera a 1,5 m, sao praticamente um plano so —
    e a focal fica indeterminada. Ela saiu 166 px numa corrida e 37 359 px
    noutra.

        Duas medidas quase no mesmo lugar nao sao duas medidas. Sao uma, com
        um numero a mais para dar confianca.
    """
    from ferramentas.calibrar_andando import PONTOS
    alturas = sorted(f for _i, f in PONTOS.values())
    assert len(alturas) >= 5
    assert alturas[-1] - alturas[0] > 0.85, "as alturas nao se espalham"

    # e as duas que eu usava antes, sozinhas, sao quase o mesmo plano
    assert PONTOS["nariz"][1] - PONTOS["ombro"][1] < 0.11


def test_so_ombro_e_nariz_dao_uma_focal_sem_sentido():
    """A prova de que o teste acima mede algo real."""
    K, R, t = _camera(focal=600.0)
    c = _caminhada(K, R, t, passos=250, ruido_px=2.0)
    E = ESTATURA
    quase_plano = Coleta(quadros=c.quadros)
    for (x, y, z), p in zip(c.mundo, c.pixel):
        # reposiciona os dois planos a 17 cm um do outro
        novo = 0.82 * E if z == 0.0 else 0.925 * E
        quase_plano.juntar((x, y, novo), p)
    achado = resolver(quase_plano, (LARG, ALT))
    if achado is not None:
        fugiu = not (0.5 < achado[0][0, 0] / 600.0 < 2.0)
        assert fugiu or achado[4] > ERRO_MAXIMO_PX, (
            "dois planos a 17 cm nao deveriam determinar a focal")
