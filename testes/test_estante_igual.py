"""A estante do ambiente virtual e a estante da trena. Sempre.

    a ideia era passar as medidas da prateleira e o sistema identificar e
    reproduzir ela em um ambiente virtual (...) vc ja poderia fazer exatamente
    ela como um objeto do nosso ambiente virtual MAIS PRECISA FICAR IGUAL!
                                                    — Eduardo, 18/08

O DEFEITO, e ele foi de projeto e nao de digitacao.

`reconhecer()` devolvia as dimensoes MEDIDAS pela camera, e o
`achar_ambiente.py` as gravava no `quarto.json`. Em 18/08 escreveu uma
estante de 1,01 x 0,23 m — quando ela mede 0,92 x 0,30 e isso esta na trena
desde 11/08.

Pior: a nota que a propria ferramenta escreve dentro do arquivo dizia

    "As dimensoes vem do gabarito de trena (loja/estante.json); o que as
     cameras acrescentam e ONDE ele esta e para onde a face olha."

Dizia e nao fazia.

    Documentacao que descreve a intencao em vez do codigo e pior que
    nenhuma: ela faz o leitor parar de conferir.

A DIVISAO DE TRABALHO, que estes testes travam:

    trena    largura, profundidade, altura, as 5 prateleiras
    cameras  x, y, rumo_da_face — e mais nada
"""
import math

import numpy as np
import pytest

from estado.planta import Planta
from src.mundo.ambiente import Gabarito, VistaDoAlto, reconhecer
from visual.cena3d import Cena3D

GAB = Gabarito.de_arquivo("loja/estante.json")


def _alto(maior=0.92, menor=0.30, cx=0.8, cy=0.6, ang=0.0):
    return VistaDoAlto(centro=(cx, cy), lado_maior=maior, lado_menor=menor,
                       angulo=ang)


# ---------------------------------------- a camera nao redefine o objeto
def test_a_trena_manda_nas_dimensoes():
    """Camera vendo 0,92 x 0,30 certinho: as dimensoes sao as da trena."""
    a = reconhecer(GAB, do_alto=_alto())
    assert a.largura == GAB.largura
    assert a.profundidade == GAB.profundidade
    assert a.altura == GAB.altura


def test_camera_torta_nao_deforma_a_estante():
    """O caso de 18/08: a camera viu 1,01 x 0,23 por extrapolacao.

    Antes, ISTO virava a estante do ambiente virtual. Agora vira apenas um
    reconhecimento — a estante continua com as medidas dela.
    """
    a = reconhecer(GAB, do_alto=_alto(maior=1.01, menor=0.23))
    assert a is not None, "deixou de reconhecer, e isso e outro problema"
    assert a.largura == pytest.approx(0.92)
    assert a.profundidade == pytest.approx(0.30)


def test_nenhuma_leitura_de_camera_sobrevive_nas_dimensoes():
    """Varre a faixa de tolerancia inteira: a saida nao pode variar."""
    vistos = set()
    for maior in (0.75, 0.85, 0.92, 1.05, 1.14):
        for menor in (0.24, 0.27, 0.30, 0.34, 0.37):
            a = reconhecer(GAB, do_alto=_alto(maior=maior, menor=menor))
            if a is not None:
                vistos.add((a.largura, a.profundidade))
    assert len(vistos) >= 1
    assert vistos == {(GAB.largura, GAB.profundidade)}, \
        "alguma leitura de camera vazou para as dimensoes"


def test_as_cinco_prateleiras_vem_inteiras_do_gabarito():
    a = reconhecer(GAB, do_alto=_alto())
    assert [round(h, 2) for _, h in a.prateleiras] == [0.15, 0.55, 0.95,
                                                       1.35, 1.90]


def test_o_que_a_camera_DA_e_posicao_e_rumo():
    """A outra metade da regra: isto SIM tem que vir da camera."""
    a = reconhecer(GAB, do_alto=_alto(cx=1.20, cy=0.40))
    assert (a.x, a.y) == pytest.approx((1.20, 0.40))

    b = reconhecer(GAB, do_alto=_alto(ang=0.5))
    c = reconhecer(GAB, do_alto=_alto(ang=1.1))
    assert b.rumo_da_face != c.rumo_da_face, "o rumo parou de vir da camera"


def test_estante_de_lado_para_a_camera_tambem_sai_com_a_trena():
    """Hipotese B: o lado maior visto e a profundidade."""
    a = reconhecer(GAB, do_alto=_alto(maior=0.30, menor=0.92, ang=0.0))
    assert a.largura == GAB.largura
    assert a.profundidade == GAB.profundidade


# ---------------------------------------- e o desenho mostra as cinco
def _cena_com_estante(prateleiras):
    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    cena.add_movel(1.0, 0.6, 0.92, 0.30, 1.90, "Estante", rumo=0.4,
                   prateleiras=prateleiras)
    return cena


def test_as_alturas_chegam_ate_a_cena():
    cena = _cena_com_estante(GAB.prateleiras)
    assert cena.moveis[0][7] == (0.15, 0.55, 0.95, 1.35, 1.90)


def test_a_estante_e_desenhada_como_estante_e_nao_como_bloco():
    """Um bloco macico e um armario. A pergunta do sistema e DE QUAL das cinco.

    A prova: a estante com prateleiras tem que produzir uma imagem DIFERENTE
    da mesma caixa sem elas.
    """
    com = _cena_com_estante(GAB.prateleiras).desenhar([])
    sem = _cena_com_estante(()).desenhar([])
    assert not np.array_equal(com, sem), "desenhou o bloco de sempre"


def test_desenhar_a_estante_com_as_cinco_nao_estoura():
    assert _cena_com_estante(GAB.prateleiras).desenhar([]) is not None


def test_as_alturas_desenhadas_sao_as_medidas_e_nao_espacadas_por_conta():
    """Espacar cinco niveis em 1,90/5 seria bonito e mentiroso.

    Os vaos reais sao 0,40 / 0,40 / 0,40 / 0,55 — o ultimo e maior. Se algum
    dia alguem trocar as alturas por uma divisao regular, isto acusa.
    """
    vaos = np.diff([h for _, h in GAB.prateleiras])
    assert not np.allclose(vaos, vaos[0]), "as alturas viraram regulares"
    cena = _cena_com_estante(GAB.prateleiras)
    assert np.allclose(cena.moveis[0][7], [h for _, h in GAB.prateleiras])


def test_o_caminho_inteiro_do_json_ate_o_desenho(tmp_path):
    """estante.json -> quarto.json -> Planta -> Cena -> imagem com 5 niveis."""
    import json
    d = {"id": "t", "nome": "t",
         "chao": {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2},
         "moveis": [{"id": "estante-aco", "nome": "Estante", "tipo": "estante",
                     "x": 1.0, "y": 0.6,
                     "largura": GAB.largura, "profundidade": GAB.profundidade,
                     "altura": GAB.altura, "rumo_da_face": 0.4,
                     "prateleiras": [{"id": i, "altura": h}
                                     for i, h in GAB.prateleiras]}]}
    p = tmp_path / "q.json"
    p.write_text(json.dumps(d), encoding="utf-8")

    cena = Cena3D(320, 240, chao=(0, 2, 0, 2))
    Planta.carregar(p).aplicar_na_cena(cena)
    assert cena.moveis[0][7] == (0.15, 0.55, 0.95, 1.35, 1.90)
    assert cena.desenhar([]) is not None
