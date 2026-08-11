"""
EscalaVertical — a camera do ALTO mede a estatura de quem passa, em metros.

A FRASE DO EDUARDO QUE REORGANIZOU O PROBLEMA, 11/08

    nenhuma delas vai captar 100% de tudo, as 3 ja existem ao mesmo tempo
    para uma complementar a outra

Eu estava tentando fazer UMA camera responder tudo, e por isso a altura da mao
em metros parecia impossivel: a frontal e a lateral ficam sobre a mesa e nunca
verao um pe. Medido em 11/08: tornozelos visiveis em 0% dos quadros nas duas.

Mas a camera do alto ve os pes — e roda `yolo11n-pose.pt`, entao ja entrega
tornozelo. O que faltava nao era ver o pe: era converter o que ela ve em
metros.

    alto      ve os pes e o chao        -> ONDE, e QUANTO A PESSOA MEDE
    frontal   ombros 100%, pulsos 79%   -> geometria relativa ao quadril
    lateral   ombros e quadris 100%     -> reserva quando a frontal perde

Cada uma responde o que enxerga. Nenhuma precisa enxergar tudo.

A GEOMETRIA, E ELA JA ESTAVA NO PROJETO

O `FiltroDePlausibilidade` calcula, desde o bloco 1:

    razao = altura_da_caixa_px / (v_pe - v_horizonte)

E ha um resultado classico de visao de uma vista so: para um objeto apoiado no
plano do chao, visto por uma camera a uma altura Hc,

    razao = altura_do_objeto / Hc

O numero que o filtro usava so para RECUSAR MOVEL e, multiplicado por uma
constante, a estatura da pessoa em metros. Ele estava ali desde o comeco.

    O dado que faltava ja estava sendo calculado para outra finalidade.

A CALIBRACAO: UM NUMERO, UMA VEZ

O fator nao e medido com trena na parede. Basta uma pessoa de estatura
conhecida aparecer uma vez:

    fator = estatura_conhecida / razao_observada

Depois disso, `estatura = razao x fator` vale para qualquer um.

O FATOR NAO E A ALTURA DA CAMERA, E EU ERREI AO CHAMA-LO ASSIM

MEDIDO EM 11/08: com o Eduardo a 1,80 m, a razao ficou em 0,343 e o fator em
5,25 — e a camera do alto nao esta a cinco metros do chao. A ferramenta chegou
a avisar "altura improvavel", porque eu tinha escrito a suspeita errada.

A relacao `razao = altura / altura_da_camera` vale para camera SEM inclinacao,
onde as verticais do mundo continuam verticais na imagem. A camera do alto
olha o chao quase de cima: uma pessoa em pe aparece ENCURTADA, e a razao
medida fica em menos da metade do que o modelo simples previa. O fator
absorveu a inclinacao junto.

    Como fator de conversao, 5,25 esta certo. Como altura da camera, e ficcao.

A dispersao de 5% e o que diz que a relacao serve: ela e estavel pelo chao
inteiro, que e a propriedade projetiva de que precisamos. O significado fisico
do numero nao e usado em lugar nenhum.

    Uma constante empirica nao precisa ter nome fisico. Precisa ser estavel, e
    precisa ser medida do mesmo jeito que sera usada.

O LIMITE QUE ISSO IMPOE, DECLARADO

Sem inclinacao, `razao` e proporcional a altura e a conversao vale para
qualquer estatura. COM inclinacao forte, a proporcionalidade e aproximada e o
erro cresce conforme a pessoa se afasta da estatura de calibracao.

Calibrado a 1,80 m, o fator e exato para 1,80 e bom perto disso. Para uma
crianca de 1,10 m ele vai errar — quanto, so medindo. Enquanto o sistema for
usado com adultos, nao incomoda; quando deixar de ser, isto tem que ser
remedido, nao ajustado no chute.

O QUE ISTO NAO RESOLVE

A altura do QUADRIL ainda sai de uma proporcao (0,53 da estatura). Mas a
diferenca em relacao ao que havia antes e grande:

    antes    proporcao aplicada sobre o TRONCO estimado    ~8 cm de erro
    agora    proporcao aplicada sobre a estatura MEDIDA    ~3 cm de erro

Continua sendo modelo, e continua saindo marcado. O que mudou e sobre o que o
modelo e aplicado.

LIMITE DECLARADO

Vale para pessoa EM PE. Quem agacha tem a caixa menor e seria medido como
baixo — por isso a estatura so e amostrada quando a postura permite, e a
mediana por pessoa protege o resto.
"""

from collections import deque

import numpy as np

# Fracao da estatura em que fica o quadril (trocanter maior). Medida
# antropometrica classica, estavel entre adultos dentro de cerca de 3%.
QUADRIL_POR_ESTATURA = 0.53

# Faixa de estatura aceita como humana. Fora disso e reconstrucao ruim, e o
# numero nao deve virar referencia de nada.
ESTATURA_MIN = 0.80
ESTATURA_MAX = 2.20


class EscalaVertical:
    """Converte a razao geometrica da camera do alto em metros.

    Guarda uma mediana POR PESSOA: a razao de um unico quadro carrega o ruido
    da caixa do detector, que treme alguns pixels a cada inferencia. A mediana
    de algumas dezenas de quadros de UMA pessoa e estavel, e some junto com o
    rastro dela.
    """

    def __init__(self, fator=None, memoria=90):
        self.fator = fator
        self.memoria = memoria
        self._razoes = {}

    # ------------------------------------------------------------ calibracao
    @staticmethod
    def calibrar(estatura_conhecida_m, razao_observada):
        """Devolve o FATOR de conversao. Uma pessoa medida, uma vez.

        Nao ha trena na parede nem angulo a medir: quem sabe a propria altura
        aparece na cena, e a razao observada fecha a conta.
        """
        if razao_observada is None or razao_observada <= 1e-6:
            raise ValueError("razao invalida — a pessoa estava visivel?")
        if not ESTATURA_MIN <= estatura_conhecida_m <= ESTATURA_MAX:
            raise ValueError(
                f"estatura fora da faixa humana: {estatura_conhecida_m}")
        return estatura_conhecida_m / razao_observada

    @property
    def calibrada(self):
        return bool(self.fator)

    # ------------------------------------------------------------ medicao
    def observar(self, pessoa_id, razao, em_pe=True):
        """Alimenta a razao daquele quadro. Devolve a estatura, ou None.

        `em_pe` existe porque a caixa de quem agacha e menor de verdade, e uma
        amostra dali diria que a pessoa mede 1,10 m. O sinal vem do
        classificador, que ja decidiu a postura por outro caminho — a coxa.
        """
        if not self.calibrada or razao is None or not em_pe:
            return self.estatura(pessoa_id)

        alta = razao * self.fator
        if ESTATURA_MIN <= alta <= ESTATURA_MAX:
            self._razoes.setdefault(
                pessoa_id, deque(maxlen=self.memoria)).append(alta)
        return self.estatura(pessoa_id)

    def estatura(self, pessoa_id):
        h = self._razoes.get(pessoa_id)
        return float(np.median(h)) if h else None

    def altura_do_quadril(self, pessoa_id):
        """Onde fica o quadril desta pessoa, em metros acima do chao."""
        e = self.estatura(pessoa_id)
        return None if e is None else e * QUADRIL_POR_ESTATURA

    def esquecer(self, vivos):
        for pid in list(self._razoes):
            if pid not in vivos:
                del self._razoes[pid]

    # ------------------------------------------------------------ diagnostico
    @property
    def diagnostico(self):
        if not self.calibrada:
            return ("escala NAO CALIBRADA — altura da mao sai estimada pelo "
                    "tronco. Rode ferramentas/calibrar_escala.py")
        medidos = {p: round(self.estatura(p), 2) for p in self._razoes}
        return (f"escala fator {self.fator:.2f}   "
                f"estaturas {medidos or '-'}")
