"""O ambiente e reconhecido antes da pessoa, e o gabarito e o filtro."""

import pytest

from src.mundo.ambiente import (Gabarito, MemoriaDoAmbiente,
                                VistaDeFrente, VistaDoAlto, reconhecer)

GAB = Gabarito(id="estante-aco-teste", largura=0.92, profundidade=0.30,
               altura=1.90,
               prateleiras=[("p1", 0.15), ("p2", 0.55), ("p3", 0.95),
                            ("p4", 1.35), ("p5", 1.90)])


def _alto(cx=1.0, cy=1.10, maior=0.92, menor=0.30, ang=0.0):
    return VistaDoAlto(centro=(cx, cy), lado_maior=maior, lado_menor=menor,
                       angulo=ang)


def test_o_gabarito_vem_do_arquivo():
    g = Gabarito.de_arquivo("loja/estante.json")
    assert g.largura == 0.92 and g.profundidade == 0.30 and g.altura == 1.90
    assert len(g.prateleiras) == 5
    assert ("p3", 0.95) in g.prateleiras


def test_sem_a_camera_do_alto_nao_ha_posicao():
    """Forma sem homografia nao e lugar."""
    assert reconhecer(GAB, da_frente=VistaDeFrente([0.15, 0.55, 0.95])) is None


def test_o_retangulo_do_tamanho_certo_e_aceito():
    a = reconhecer(GAB, do_alto=_alto())
    assert a is not None
    assert a.largura == pytest.approx(0.92)
    assert a.profundidade == pytest.approx(0.30)
    assert a.cameras == ("alto",)
    assert not a.confiavel, "uma camera so arrisca, nao confia"


def test_a_mesa_nao_passa_pelo_gabarito():
    """1,40 x 0,80 nao e a estante — e o filtro que dispensa rede neural."""
    assert reconhecer(GAB, do_alto=_alto(maior=1.40, menor=0.80)) is None


def test_erro_de_calibracao_ainda_reconhece():
    """15% de erro nas duas dimensoes: e a estante, com folga."""
    a = reconhecer(GAB, do_alto=_alto(maior=1.06, menor=0.26))
    assert a is not None


def test_estante_de_lado_para_a_camera():
    """Se o lado MAIOR visto e a profundidade, a face gira 90 graus."""
    a = reconhecer(GAB, do_alto=_alto(maior=0.92, menor=0.30, ang=0.0))
    b = reconhecer(GAB, do_alto=_alto(maior=0.30, menor=0.92, ang=0.0))
    assert a is not None and b is not None
    assert abs(a.rumo_da_face - b.rumo_da_face) > 1.0, (
        "as duas hipoteses nao podem dar a mesma face"


    )


def test_as_alturas_confirmam_e_dao_confianca():
    a = reconhecer(GAB, do_alto=_alto(),
                   da_frente=VistaDeFrente([0.16, 0.54, 0.96, 1.34]),
                   da_lateral=VistaDeFrente([0.15, 0.95, 1.90]))
    assert set(a.cameras) == {"alto", "frontal", "lateral"}
    assert a.confiavel
    assert a.alturas_conferidas >= 4


def test_estante_cheia_ainda_e_reconhecida():
    """Produto tapa prateleira. Exigir as cinco seria rejeitar o uso normal."""
    a = reconhecer(GAB, do_alto=_alto(),
                   da_frente=VistaDeFrente([0.95, 1.35]))
    assert "frontal" in a.cameras


def test_linhas_que_nao_sao_prateleiras_nao_confirmam():
    a = reconhecer(GAB, do_alto=_alto(),
                   da_frente=VistaDeFrente([0.42, 0.71, 1.12]))
    assert a.cameras == ("alto",)


# ------------------------------------------------------- a relacao
def _ambiente():
    return reconhecer(GAB, do_alto=_alto(cx=1.0, cy=1.10),
                      da_frente=VistaDeFrente([0.15, 0.55, 0.95, 1.35, 1.90]))


def test_a_face_olha_para_dentro_da_area():
    """Atras da estante ha parede: a face util e a voltada para a origem."""
    a = _ambiente()
    adiante, _ = a.relacao(1.0, 0.60)      # alguem entre a origem e a estante
    assert adiante > 0, "quem esta na area util tem que estar 'a frente'"


def test_posicao_absoluta_vira_relacao():
    a = _ambiente()
    adiante, lateral = a.relacao(1.0, 0.60)
    assert adiante == pytest.approx(0.50, abs=0.02)
    assert abs(lateral) < 0.02


def test_de_qual_prateleira_exige_as_duas_condicoes():
    a = _ambiente()
    assert a.de_qual_prateleira(1.0, 0.60, 0.95) == "p3"
    assert a.de_qual_prateleira(0.0, 0.05, 0.95) is None, (
        "braco a 0,95 do outro lado da sala nao e pegar da p3")


def test_altura_no_meio_do_vao_nao_responde():
    a = _ambiente()
    assert a.prateleira_na_altura(0.75) is None


# ------------------------------------------------------- memoria
def test_a_memoria_guarda_e_nao_perde_com_falha():
    m = MemoriaDoAmbiente()
    assert not m.pronto
    m.registrar(_ambiente())
    assert m.pronto
    m.registrar(None)                       # um quadro ruim nao apaga o movel
    assert m.pronto


def test_reconhecimento_melhor_substitui():
    m = MemoriaDoAmbiente()
    m.registrar(reconhecer(GAB, do_alto=_alto()))
    assert len(m.atual.cameras) == 1
    m.registrar(_ambiente())
    assert len(m.atual.cameras) == 2
