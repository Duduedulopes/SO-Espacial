"""A estante e achada pelos gestos, nao declarada. Estes testes provam isso."""
import math

import numpy as np
import pytest

from src.acao.achar_estante import (
    AMOSTRAS_MINIMAS, Alcance, LocalizadorDeEstante, prateleira_alcancada,
    prateleira_por_altura)

# As cinco alturas medidas com trena em 11/08 (loja/estante.json).
PRATELEIRAS = [("p1", 0.15), ("p2", 0.55), ("p3", 0.95), ("p4", 1.35),
               ("p5", 1.90)]


def _gestos_de_frente(n=40, face_y=1.30, largura=0.92, ruido=0.0, semente=7):
    """Pessoas em pe diante de uma estante em y=face_y, olhando para +y."""
    r = np.random.default_rng(semente)
    fora = []
    for _ in range(n):
        x = 1.00 + r.uniform(-largura / 2, largura / 2)
        y = face_y - 0.50 + r.uniform(-0.12, 0.12)      # a um braco da face
        rumo = 0.0 + r.normal(0, ruido)                 # (-sin,cos) = (0,1)
        fora.append(Alcance(x=x + r.normal(0, ruido),
                            y=y + r.normal(0, ruido), rumo=rumo,
                            altura_mao=0.95))
    return fora


def test_poucos_gestos_nao_arriscam_resposta():
    loc = LocalizadorDeEstante()
    for a in _gestos_de_frente(AMOSTRAS_MINIMAS - 1):
        loc.observar(a)
    assert loc.resolver() is None, "meia duzia de gestos descreve o acaso"


def test_os_gestos_desenham_a_estante():
    loc = LocalizadorDeEstante()
    for a in _gestos_de_frente(60, face_y=1.30, ruido=0.02):
        loc.observar(a)
    e = loc.resolver()
    assert e is not None
    assert e.y == pytest.approx(1.30, abs=0.10), "achou a face onde ela esta"
    assert e.x == pytest.approx(1.00, abs=0.10)
    assert 0.6 <= e.largura <= 1.3, f"largura fora do esperado: {e.largura}"


def test_a_face_olha_para_quem_alcanca():
    """Saber QUAL das duas faces se usa e o que a normal resolve."""
    loc = LocalizadorDeEstante()
    for a in _gestos_de_frente(60, ruido=0.02):
        loc.observar(a)
    e = loc.resolver()
    # as pessoas estao em y menor que a face, entao a normal aponta para -y
    assert e.normal[1] < 0, f"normal apontou para o lado errado: {e.normal}"


def test_nuvem_redonda_nao_e_estante():
    """Gente alcancando coisas espalhadas nao vira uma face."""
    r = np.random.default_rng(3)
    loc = LocalizadorDeEstante()
    for _ in range(80):
        loc.observar(Alcance(x=r.uniform(0, 1.6), y=r.uniform(0, 1.3),
                             rumo=r.uniform(-math.pi, math.pi),
                             altura_mao=0.9))
    assert loc.resolver() is None, "sem plano, nao ha estante — e diz None"


def test_sem_rumo_nao_e_voto():
    loc = LocalizadorDeEstante()
    for a in _gestos_de_frente(30):
        a.rumo = None
        loc.observar(a)
    assert loc.amostras == 0 and loc.resolver() is None


def test_a_estante_pode_mudar_de_lugar():
    """A janela esquece o lugar antigo no ritmo em que o novo e usado."""
    loc = LocalizadorDeEstante(memoria=60)
    for a in _gestos_de_frente(60, face_y=1.30, ruido=0.02):
        loc.observar(a)
    assert loc.resolver().y == pytest.approx(1.30, abs=0.12)

    for a in _gestos_de_frente(60, face_y=0.55, ruido=0.02, semente=11):
        loc.observar(a)
    assert loc.resolver().y == pytest.approx(0.55, abs=0.12), (
        "movida a estante, os gestos novos a encontram no lugar novo")


def test_quem_esta_atras_da_estante_nao_alcanca():
    loc = LocalizadorDeEstante()
    for a in _gestos_de_frente(60, face_y=1.30, ruido=0.02):
        loc.observar(a)
    e = loc.resolver()
    assert e.de_frente(1.00, 0.85), "na frente, a meio metro: alcanca"
    assert not e.de_frente(1.00, 1.75), "do outro lado da estante: nao alcanca"
    assert not e.de_frente(1.00, 0.10), "longe demais: nao alcanca"


# ---------------------------------------------------------------- altura
def test_cada_altura_cai_na_prateleira_certa():
    for pid, alt in PRATELEIRAS:
        assert prateleira_por_altura(alt, PRATELEIRAS) == pid


def test_erro_medido_ainda_acerta_a_prateleira():
    """+-8 cm e o pior caso do gabarito; o vao e de 40 cm."""
    assert prateleira_por_altura(0.95 + 0.08, PRATELEIRAS) == "p3"
    assert prateleira_por_altura(1.35 - 0.08, PRATELEIRAS) == "p4"


def test_altura_no_meio_do_vao_nao_responde():
    """0,75 esta a 20 cm de p2 e de p3. Chutar seria pior que abster."""
    assert prateleira_por_altura(0.75, PRATELEIRAS) is None


def test_altura_ausente_nao_inventa():
    assert prateleira_por_altura(None, PRATELEIRAS) is None


def test_a_resposta_completa_exige_estar_de_frente():
    loc = LocalizadorDeEstante()
    for a in _gestos_de_frente(60, face_y=1.30, ruido=0.02):
        loc.observar(a)
    e = loc.resolver()
    assert prateleira_alcancada(e, PRATELEIRAS, 1.0, 0.85, 0.95) == "p3"
    assert prateleira_alcancada(e, PRATELEIRAS, 0.1, 0.10, 0.95) is None, (
        "braco a 0,95 m do outro lado da sala nao e pegar da p3")


def test_sem_estante_nao_ha_prateleira():
    assert prateleira_alcancada(None, PRATELEIRAS, 1.0, 0.85, 0.95) is None
