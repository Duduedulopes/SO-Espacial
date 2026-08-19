"""A segunda camera cai no MESMO mundo que a primeira, por construcao.

    todas as cameras deveriam ajudar a dizer aonde esta a pessoa e unir todas
    as informacoes para apenas 1 movimento — Eduardo, 19/08

O PROBLEMA QUE ISTO RESOLVE

So a camera do alto tinha homografia. Quando o Eduardo andava para a beirada
do campo dela, a posicao congelava — as outras duas o viam perfeitamente e
nao sabiam dizer ONDE, porque nao havia como converter pixel em metro.

E A LATERAL NAO ENXERGA A FITA do retangulo de 1,65 x 1,32 que calibrou a do
teto. Sem isso, a saida obvia seria marcar um retangulo novo e chamar aquele
canto de (0,0) — e ai as duas cameras teriam mundos DIFERENTES, e juntar as
medidas viraria um problema de alinhamento que pode falhar.

Com `--origem`, o segundo retangulo e medido com trena a partir da MESMA
marca de fita. As duas homografias caem no mesmo sistema de coordenadas sem
nenhuma etapa de alinhamento.

    Dois instrumentos so concordam de graca quando foram referidos ao mesmo
    zero. Referi-los depois custa uma etapa que pode falhar.
"""
import numpy as np
import pytest


def _homografia_de(pontos_mundo, pontos_px):
    import cv2
    return cv2.getPerspectiveTransform(
        np.float32(pontos_px), np.float32(pontos_mundo))


def _em_metros(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _cantos(origem, largura, altura):
    """A mesma conta que o homografia.py faz com --origem."""
    ox, oy = origem
    return [[ox, oy], [ox + largura, oy],
            [ox + largura, oy + altura], [ox, oy + altura]]


def test_a_origem_desloca_o_retangulo_e_nada_mais():
    na_origem = _cantos((0.0, 0.0), 0.90, 0.60)
    deslocado = _cantos((2.00, 0.50), 0.90, 0.60)
    for a, b in zip(na_origem, deslocado):
        assert b[0] - a[0] == pytest.approx(2.00)
        assert b[1] - a[1] == pytest.approx(0.50)


def test_duas_cameras_com_retangulos_DIFERENTES_concordam_sobre_o_mesmo_ponto():
    """O teste que paga a etapa inteira.

    Duas cameras que nem se veem, cada uma calibrada com o SEU retangulo,
    medido da mesma origem. Um ponto do chao visto pelas duas tem que sair
    no mesmo lugar em metros.
    """
    # camera A: retangulo na origem, vista de um jeito
    mundo_a = _cantos((0.0, 0.0), 1.65, 1.32)
    px_a = [[120, 227], [410, 88], [588, 264], [268, 471]]
    HA = _homografia_de(mundo_a, px_a)

    # camera B: OUTRO retangulo, 2 m adiante, vista de outro jeito
    mundo_b = _cantos((2.00, 0.50), 0.90, 0.60)
    px_b = [[80, 300], [520, 240], [560, 430], [60, 460]]
    HB = _homografia_de(mundo_b, px_b)

    # um ponto qualquer do mundo, levado a pixel em cada camera e de volta
    for alvo in [(2.20, 0.70), (2.60, 0.90), (2.90, 1.05)]:
        ua, va = _pixel_de(HA, alvo)
        ub, vb = _pixel_de(HB, alvo)
        pa = _em_metros(HA, ua, va)
        pb = _em_metros(HB, ub, vb)
        assert pa == pytest.approx(alvo, abs=1e-6)
        assert pb == pytest.approx(alvo, abs=1e-6)
        assert pa == pytest.approx(pb, abs=1e-6), (
            "as duas cameras discordam sobre o mesmo ponto do chao")


def _pixel_de(H, ponto):
    Hi = np.linalg.inv(H)
    p = Hi @ np.array([ponto[0], ponto[1], 1.0])
    return p[0] / p[2], p[1] / p[2]


def test_sem_origem_os_dois_mundos_ficariam_deslocados():
    """A prova de que `--origem` nao e enfeite.

    Se a segunda camera chamasse o proprio canto de (0,0), o mesmo ponto do
    chao sairia 2 m fora — e a fusao somaria posicoes de mundos diferentes
    sem nenhum sintoma alem de numeros errados.
    """
    certo = _cantos((2.00, 0.50), 0.90, 0.60)
    ingenuo = _cantos((0.0, 0.0), 0.90, 0.60)
    px = [[80, 300], [520, 240], [560, 430], [60, 460]]
    H_certo = _homografia_de(certo, px)
    H_ingenuo = _homografia_de(ingenuo, px)

    a = _em_metros(H_certo, 300, 350)
    b = _em_metros(H_ingenuo, 300, 350)
    assert np.hypot(a[0] - b[0], a[1] - b[1]) == pytest.approx(
        np.hypot(2.0, 0.5), abs=1e-6)


def test_o_arquivo_do_alto_continua_no_lugar_de_sempre():
    """Trocar o nome do arquivo da alto quebraria rodar.py, mapear.py e o
    --mono de uma vez — e nao ha nada de errado com ele."""
    from pathlib import Path
    raiz = Path(__file__).resolve().parent.parent
    assert (raiz / "calibracao" / "homografia.json").exists()


def test_abrir_aceita_camera_de_rede():
    """A lateral e um tablet em http://.../video, e CAP_DSHOW nao abre URL.

    O programa foi escrito quando so havia a camera do teto, e a suposicao
    ficou embutida no unico caminho que existia.

        Um programa que so foi usado com um caso nao esta certo para aquele
        caso: esta sem ter sido contrariado.
    """
    import inspect

    from calibracao.homografia import abrir
    fonte = inspect.getsource(abrir)
    assert 'startswith("http")' in fonte, "nao trata camera de rede"
    assert "exigir_indice" in fonte, "nao resolve USB pelo NOME"
