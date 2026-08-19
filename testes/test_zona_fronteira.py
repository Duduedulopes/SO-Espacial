"""A fronteira da zona nao e uma linha, e tratar como linha conta ruido.

MEDIDO EM 19/08, numa corrida de 45 s com o Eduardo SENTADO, parado:

    PERSON_ENTERED_ZONE      15
    PERSON_LEFT_ZONE         15

Quinze entradas e quinze saidas de quem nao saiu do lugar. A posicao tem um
ou dois centimetros de ruido; quem esta em cima da borda atravessa a linha
varias vezes por segundo sem se mexer.

    Um limiar unico sobre um sinal com ruido nao decide: ele conta o ruido.

Numa loja isso viraria quinze registros de um cliente que nao fez nada — e a
contagem de visitas e justamente o produto deste modulo.

E havia um segundo defeito, mais fundo: `gemeo._detectar_zonas` refazia o
teste geometrico em vez de perguntar a zona quem estava dentro. Duas
respostas para a mesma pergunta, e o fluxo de eventos lia a que ninguem
tinha filtrado.
"""
import random

import pytest

from estado.ocupacao import Zona


def _tremendo(zona, dt=1 / 15, segundos=20.0, borda=None, amplitude=0.02,
              semente=4, rid=1):
    """Alguem parado EM CIMA da borda, com ruido de medida. Devolve visitas."""
    rnd = random.Random(semente)
    x_borda = zona.x1 if borda is None else borda
    y = (zona.y0 + zona.y1) / 2
    for _ in range(int(segundos / dt)):
        zona.atualizar({rid: (x_borda + rnd.gauss(0, amplitude), y)}, dt)
    return zona.visitas


# --------------------------------------------------------------- o defeito
def test_parado_na_borda_nao_gera_uma_visita_por_quadro():
    """O teste que a versao anterior falhava, e feio."""
    z = Zona("Saida", 1.0, 1.45, 0.4, 0.99)
    assert _tremendo(z) <= 1


def test_sem_histerese_o_defeito_aparece():
    """A prova de que o teste acima mede algo.

    Reproduz o comportamento antigo — limiar unico, sem tempo — e exige que
    ele conte dezenas de visitas. Se alguem tirar a banda morta achando que e
    enfeite, e este teste que denuncia.
    """
    z = Zona("Saida", 1.0, 1.45, 0.4, 0.99, margem_m=0.0, confirmar_s=0.0)
    assert _tremendo(z) > 20


# ------------------------------------------------ e continua funcionando
def test_quem_entra_de_verdade_e_contado():
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0)
    for _ in range(20):
        z.atualizar({1: (0.2, 0.5)}, 0.1)       # fora, com folga
    for _ in range(20):
        z.atualizar({1: (1.5, 0.5)}, 0.1)       # dentro, no meio
    assert z.visitas == 1 and 1 in z.dentro


def test_quem_sai_de_verdade_sai():
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0)
    for _ in range(20):
        z.atualizar({1: (1.5, 0.5)}, 0.1)
    for _ in range(20):
        z.atualizar({1: (0.2, 0.5)}, 0.1)
    assert 1 not in z.dentro and z.visitas == 1


def test_o_tempo_dentro_continua_sendo_contado():
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0)
    for _ in range(50):
        z.atualizar({1: (1.5, 0.5)}, 0.1)
    assert z.tempo_total == pytest.approx(5.0, abs=0.5)


def test_duas_pessoas_sao_independentes():
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0)
    for _ in range(20):
        z.atualizar({1: (1.5, 0.5), 2: (0.1, 0.5)}, 0.1)
    assert z.dentro == {1}


# --------------------------------------------- o preco, medido e declarado
def test_a_entrada_atrasa_o_tempo_de_confirmacao_e_nao_mais():
    """Contar gente numa loja tolera 0,4 s. Quinze eventos falsos, nao."""
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0, confirmar_s=0.4)
    dt = 0.1
    for k in range(1, 12):
        z.atualizar({1: (1.5, 0.5)}, dt)
        if 1 in z.dentro:
            assert 0.4 <= k * dt <= 0.6, f"confirmou em {k * dt:.1f} s"
            return
    pytest.fail("nunca confirmou a entrada")


def test_o_tempo_da_espera_e_creditado_e_nao_perdido():
    """A pessoa ESTAVA la durante a confirmacao; so o programa nao decidira.

    Descartar esse tempo tiraria 0,4 s de CADA visita, sempre no mesmo
    sentido. Vies nao sai na media, e o tempo de permanencia e metade do que
    este modulo existe para medir.

        Atrasar a decisao e diferente de perder a medida. So a primeira e de
        graca.
    """
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0, confirmar_s=0.4)
    for _ in range(10):
        z.atualizar({1: (1.5, 0.5)}, 0.1)
    assert z.tempo_total == pytest.approx(1.0, abs=1e-9), (
        f"{z.tempo_total:.2f} s para 10 quadros de 0,1 s dentro")


def test_espera_que_nao_vira_visita_nao_credita_tempo():
    """O outro lado: quem passou raspando nao ganha 0,4 s de permanencia."""
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0, confirmar_s=0.4)
    for _ in range(20):
        z.atualizar({1: (0.5, 0.5)}, 0.1)
    z.atualizar({1: (1.5, 0.5)}, 0.1)
    z.atualizar({1: (0.5, 0.5)}, 0.1)
    assert z.tempo_total == 0.0


def test_salto_grande_e_breve_nao_conta_visita():
    """Ruido pequeno se resolve no espaco; ruido grande, no tempo.

    Um id reciclado ou um quadro em que o pe foi estimado pela caixa poe a
    pessoa 30 cm adiante de uma vez — atravessando a banda morta inteira. So
    o tempo pega isso.
    """
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0)
    for _ in range(20):
        z.atualizar({1: (0.5, 0.5)}, 0.1)       # fora, estavel
    z.atualizar({1: (1.5, 0.5)}, 0.1)           # um quadro dentro
    z.atualizar({1: (0.5, 0.5)}, 0.1)           # e voltou
    assert z.visitas == 0 and 1 not in z.dentro


# ---------------------------------------------------- os casos degenerados
def test_a_margem_nao_pode_engolir_uma_zona_pequena():
    """Porta de 0,45 m com margem de 0,25 encolheria para nada.

    Ninguem entraria nunca, e o defeito se manifestaria como SILENCIO — que
    e o mais caro de achar, porque nao ha o que investigar.
    """
    z = Zona("Porta", 0.0, 0.45, 0.0, 0.30, margem_m=0.25)
    assert z.margem_m <= 0.4 * 0.30
    for _ in range(20):
        z.atualizar({1: (0.225, 0.15)}, 0.1)    # bem no centro
    assert 1 in z.dentro, "ninguem consegue entrar nesta zona"


def test_rastro_que_morre_sai_na_hora():
    """Nao ha o que confirmar sobre alguem que o sistema deixou de ver."""
    z = Zona("Saida", 1.0, 2.0, 0.0, 1.0)
    for _ in range(20):
        z.atualizar({1: (1.5, 0.5)}, 0.1)
    assert 1 in z.dentro
    z.atualizar({}, 0.1)
    assert 1 not in z.dentro


def test_zona_de_area_zero_nao_estoura():
    z = Zona("Risco", 1.0, 1.0, 0.0, 0.0)
    z.atualizar({1: (1.0, 0.0)}, 0.1)
    assert z.margem_m == 0.0


# ------------------------------------ a duplicacao que causava a tempestade
def test_os_eventos_leem_a_zona_e_nao_refazem_a_conta():
    """`_detectar_zonas` refazia o teste cru e ignorava a histerese.

        Duas respostas para a mesma pergunta nao sao redundancia: uma delas
        vai ser lida por engano.
    """
    from src.espacial.estado import EstadoDePessoa
    from src.eventos.motor import EventEngine, Tipo
    from src.gemeo.gemeo import DigitalTwin

    class PlantaFalsa:
        def __init__(self, zonas):
            self.zonas = zonas

        def novo_mapa_de_calor(self, meia_vida_s=90.0):
            from estado.ocupacao import MapaDeCalor
            return MapaDeCalor(0, 3, 0, 3)

    z = Zona("Saida", 1.0, 1.45, 0.4, 0.99)
    z.id = "saida"
    eventos = EventEngine()
    gemeo = DigitalTwin(PlantaFalsa([z]), eventos=eventos)

    rnd = random.Random(4)
    for _ in range(300):
        x = 1.45 + rnd.gauss(0, 0.02)           # em cima da borda
        gemeo.atualizar([EstadoDePessoa(id=1, x=x, y=0.7)], dt=1 / 15)

    contagem = eventos.resumo()
    entradas = contagem.get(str(Tipo.PERSON_ENTERED_ZONE), 0)
    saidas = contagem.get(str(Tipo.PERSON_LEFT_ZONE), 0)
    assert entradas + saidas <= 2, (
        f"{entradas} entradas e {saidas} saidas de quem nao saiu do lugar")
