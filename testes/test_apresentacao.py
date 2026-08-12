"""A tela e pura: da para montar tudo sem hardware e conferir o que apareceu."""
import numpy as np
import pytest

from visual.apresentacao import ALERTA, DESTAQUE, Apresentacao


@pytest.fixture
def tela():
    return Apresentacao(1200, 700)


def _tem_cor(img, cor, minimo=12, tolerancia=70):
    """Procura a cor POR PROXIMIDADE, nao por igualdade.

    O texto e desenhado com LINE_AA: quase todo pixel de uma letra e uma
    mistura da tinta com o fundo, e a cor exata quase nao aparece. Um teste que
    exige igualdade estaria medindo o antisserrilhado, nao a decisao de cor.
    """
    d = np.linalg.norm(img.astype(np.int16) - np.array(cor, np.int16), axis=2)
    return int((d < tolerancia).sum()) >= minimo


def _cena(tela):
    cw, ch = tela.tamanho_da_cena
    return np.full((ch, cw, 3), (200, 200, 200), np.uint8)


def test_a_cena_cabe_no_lugar_reservado(tela):
    """Quem desenha o gemeo precisa do tamanho ANTES de desenhar."""
    cw, ch = tela.tamanho_da_cena
    assert 0 < cw < tela.largura and 0 < ch < tela.altura
    img = tela.desenhar(_cena(tela))
    assert img.shape == (700, 1200, 3)


def test_sala_vazia_nao_finge_ninguem(tela):
    img = tela.desenhar(_cena(tela))
    assert img is not None


def test_camera_que_nao_chegou_fica_vermelha(tela):
    """Janela que some e ambigua; cartao vazio e uma acusacao."""
    faltando = tela.desenhar(_cena(tela), (), [("lateral", "alcance", None)])
    veio = tela.desenhar(
        _cena(tela), (),
        [("lateral", "alcance", np.zeros((60, 80, 3), np.uint8))])
    assert _tem_cor(faltando, ALERTA)
    assert not _tem_cor(veio, ALERTA)


def test_palpite_sem_certeza_e_marcado(tela):
    """Um p3 firme e um p3 chutado nao podem ter a mesma cara."""
    firme = tela.desenhar(_cena(tela), [dict(id=1, prateleira="p3", firme=True,
                                             quantas=1, fonte_bracos="frontal",
                                             fonte_escala="alto",
                                             fonte_rumo="alto")])
    duvida = tela.desenhar(_cena(tela), [dict(id=1, prateleira="p3", firme=False,
                                              quantas=1, fonte_bracos="frontal",
                                              fonte_escala="alto",
                                              fonte_rumo="alto")])
    assert not _tem_cor(firme, ALERTA)
    assert _tem_cor(duvida, ALERTA)


def test_leitura_sem_procedencia_fica_vermelha(tela):
    """O motivo de esta tela existir: o erro tem que ter onde ser visto."""
    img = tela.desenhar(_cena(tela), [dict(id=1, prateleira="p3", firme=True,
                                           quantas=1,
                                           bracos="dir LEVANTADO",
                                           fonte_bracos="",
                                           altura_mao="1.40 m",
                                           fonte_escala="alto",
                                           rumo="+10 graus",
                                           fonte_rumo="alto")])
    assert _tem_cor(img, ALERTA), (
        "braco LIDO sem dizer qual camera leu e defeito, e fica vermelho")


def test_contagem_zero_nao_ganha_destaque(tela):
    """Zero unidade nao pode ter o mesmo peso visual que uma venda."""
    zero = tela.desenhar(_cena(tela), [dict(id=1, quantas=0,
                                            fonte_bracos="f", fonte_escala="a",
                                            fonte_rumo="a")])
    uma = tela.desenhar(_cena(tela), [dict(id=1, quantas=1,
                                           fonte_bracos="f", fonte_escala="a",
                                           fonte_rumo="a")])
    assert _tem_cor(uma, DESTAQUE) and not np.array_equal(zero, uma)


def test_fonte_sem_valor_nao_e_procedencia(tela):
    """MEDIDO 12/08:  `mao   -   camera do alto` — de onde veio o nada?"""
    img = tela.desenhar(_cena(tela), [dict(id=1, quantas=0, bracos="ao lado",
                                           fonte_bracos="frontal",
                                           altura_mao="-", fonte_escala="alto",
                                           rumo="+10 graus", fonte_rumo="alto")])
    assert not _tem_cor(img, ALERTA), "sem valor nao e defeito, e silencio"
