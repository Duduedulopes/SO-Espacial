"""Leitura ERRADA e leitura INUTIL nao sao a mesma coisa.

O gabarito da estante, 11/08, com as tres cameras funcionando:

    verdade   0,15   0,55   0,95   1,35   1,90
    lido      0,40   0,95   0,97   1,13   1,41

O bloco do VIES disse a coisa certa e parou cedo: o viés varia 0,89 m entre
prateleiras, logo nao ha constante a subtrair. Verdade — e insuficiente.

Ajustando uma reta, `lido = 0,51 x verdade + 0,48`, com ponto fixo em 0,96 m,
que cai em cima do quadril. A ancora esta certa; o que encolhe pela metade e o
deslocamento do pulso EM RELACAO ao quadril, para cima e para baixo, com a
mesma inclinacao dos dois lados.

Para a etapa D isso muda a pergunta. Nao precisamos do centimetro, precisamos
saber QUAL PRATELEIRA — e funcao monotona e repetivel e invertivel.

    Um instrumento errado e util quando o erro e estavel. Um instrumento
    instavel nao serve nem quando acerta a media.

O que o boletim tem que responder, entao, e a MARGEM: vao entre prateleiras
vizinhas, ja comprimido, dividido pela dispersao tipica.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ferramentas.conferir_altura import _reta_medida        # noqa: E402


def prat(verdade, centro, disp=0.04, n=12):
    """Uma prateleira cujas leituras ficam em torno de `centro`."""
    # Espalhamento total de 2x `disp` para que o IQR — que e o que o boletim
    # chama de dispersao — saia valendo `disp`. Gerar pontos "com dispersao X"
    # e medi-los como 4X faria o teste medir o gerador.
    passo = (2 * disp) / (n - 1) if n > 1 else 0.0
    lidas = [centro + passo * (i - (n - 1) / 2) for i in range(n)]
    return {"nome": f"p{verdade}", "verdade": verdade, "lidas": lidas,
            "sem_pessoa": 0, "sem_braco": 0}


def texto(resultados):
    return "\n".join(_reta_medida(resultados))


# ------------------------------------------------------------ o caso de 11/08
def test_encontra_a_compressao_e_o_ponto_fixo_no_quadril():
    medido = [prat(0.15, 0.40), prat(0.55, 0.95), prat(0.95, 0.97),
              prat(1.35, 1.13), prat(1.90, 1.41)]
    saida = texto(medido)

    assert "0.50" in saida, "a inclinacao medida foi 0,51 — metade"
    assert "sai COMPRIMIDO" in saida
    assert "ponto fixo (onde acerta sozinho): 0.96 m" in saida, (
        "o ponto fixo cai no quadril, e e isso que diz onde esta o defeito")


def test_compressao_limpa_e_reconhecida_como_invertivel():
    """Mesma compressao de 51%, sem o ponto fora da curva.

    Aqui a reta descreve tudo, os vaos comprimidos continuam grandes perto da
    dispersao, e o veredicto tem que ser: esta errado E serve.
    """
    medido = [prat(v, 0.96 + 0.51 * (v - 0.96), disp=0.04)
              for v in (0.15, 0.55, 0.95, 1.35, 1.90)]
    saida = texto(medido)

    assert "MARGEM" in saida
    assert "SERVE" in saida
    assert "ESTAVEL entre sessoes" in saida, "uma reta medida uma vez e coincidencia"


def test_r2_baixo_proibe_inverter():
    """Sem reta, inverter aplicaria um modelo a dados que nao sao o modelo."""
    medido = [prat(0.15, 0.90), prat(0.55, 0.40), prat(0.95, 1.30),
              prat(1.35, 0.50), prat(1.90, 1.10)]
    saida = texto(medido)

    assert "NAO e uma reta" in saida
    assert "Nao inverta" in saida


def test_margem_apertada_nao_e_vendida_como_sucesso():
    """Compressao tao forte que os vaos somem dentro do ruido."""
    medido = [prat(v, 0.96 + 0.03 * (v - 0.96), disp=0.10)
              for v in (0.15, 0.55, 0.95, 1.35, 1.90)]
    saida = texto(medido)

    assert "apertada" in saida
    assert "SERVE" not in saida


def test_leitura_perfeita_nao_reclama_de_compressao():
    medido = [prat(v, v, disp=0.03) for v in (0.15, 0.55, 0.95, 1.35, 1.90)]
    saida = texto(medido)

    assert "COMPRIMIDO" not in saida
    assert "MARGEM" in saida


# ------------------------------------------------------ recusa por falta de base
@pytest.mark.parametrize("quantas", [0, 1, 2])
def test_poucas_prateleiras_nao_viram_reta(quantas):
    """Dois pontos definem uma reta perfeita e nao provam nada.

    R2 de 1,0 com dois pontos e aritmetica, nao evidencia — e sairia com a
    mesma cara de um ajuste bem sustentado.
    """
    medido = [prat(0.15 + 0.4 * i, 0.5 + 0.3 * i) for i in range(quantas)]
    assert _reta_medida(medido) == []


def test_prateleira_com_menos_de_cinco_amostras_fica_de_fora():
    medido = [prat(0.15, 0.40), prat(0.55, 0.95), prat(0.95, 0.97),
              prat(1.35, 1.13, n=3)]
    saida = texto(medido)
    assert saida, "tres prateleiras validas ainda dao reta"

    medido[0]["lidas"] = medido[0]["lidas"][:2]
    assert _reta_medida(medido) == [], "sobraram duas: nao ha reta a afirmar"


def test_todas_na_mesma_altura_nao_produz_reta():
    """Sem variacao em x nao ha inclinacao a estimar — e dividir por zero."""
    medido = [prat(0.95, 0.90), prat(0.95, 0.95), prat(0.95, 1.00)]
    assert _reta_medida(medido) == []
