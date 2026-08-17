"""Onde fica a estante? Pergunte aos braços.

    cria um sistema ou um metodo que identifique aonde essa estante esta... a
    posicao dela nao muda, mais podera mudar, entao nao de a ela um ponto fixo
                                                        — Eduardo, 13/08

A ESTANTE NAO E ACHADA PELA IMAGEM. E ACHADA PELO GESTO.

Reconhecer uma estante de aco num video e um problema difícil: iluminação,
oclusão, segmentação, e um modelo novo para treinar. E seria resolver o problema
errado — o sistema não precisa saber que aquilo é uma estante. Precisa saber
ONDE as pessoas alcançam.

E isso ele já mede. Toda vez que alguém estica o braço, está apontando para uma
prateleira. Cada alcance é um voto:

    "há uma prateleira ali, naquela direção, naquela altura"

Cem gestos desenham a estante sozinhos. Nenhuma trena, nenhum clique, nenhum
número escrito à mão — e se a estante mudar de lugar, os gestos novos a
encontram no lugar novo sem que ninguém precise editar arquivo.

    O móvel que interessa não é o que está na sala. É o que as pessoas usam.

COMO A CONTA FUNCIONA

De cada alcance vêm três coisas que o sistema já produz:

    (x, y)        onde a pessoa estava no chão      — homografia
    rumo          para onde o corpo apontava        — câmera do alto
    altura_mao    a que altura a mão chegou         — escala vertical

Quem alcança está de frente para o que alcança. Então o ponto de contato fica
adiante da pessoa, na direção do rumo, a um braço de distância. Projetando cada
alcance assim, obtém-se uma nuvem de pontos de contato — e essa nuvem tem uma
forma muito particular: ela é PLANA, porque a face de uma estante é plana.

Ajustar uma reta a essa nuvem devolve, de uma vez:

    a posição da face      (a mediana ao longo da reta)
    a orientação da face   (a direção da reta)
    a largura útil         (o espalhamento ao longo dela)
    e a confiança          (o quanto a nuvem é realmente plana)

A ÚLTIMA LINHA É A MAIS IMPORTANTE. Se os pontos não formarem um plano, não há
estante ali — há gente alcançando coisas diferentes em lugares diferentes, ou
ruído. O localizador diz `None` nesse caso, e dizer "não sei" é o que separa
este método de um que sempre devolve uma estante em algum lugar.

POR QUE ISSO RESOLVE A PRATELEIRA TAMBEM

Com a face localizada, a altura da mão deixa de ser um número solto e passa a
ser uma altura CONTRA UMA ESTANTE CONHECIDA — e as cinco alturas dela já estão
medidas com trena em `loja/estante.json`. A pergunta "qual prateleira?" vira uma
comparação, que é o tipo de pergunta que sobrevive ao ruído:

    Um bit sobrevive ao ruído que destrói um ângulo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Quanto o ponto de contato fica adiante do centro do corpo, em metros.
#
# Não é o comprimento do braço (0,60 m num adulto): é o quanto a MÃO avança
# além do eixo do corpo quando alguém pega algo numa prateleira. Ninguém
# encosta o peito na estante nem estica o braço até travar o cotovelo — para
# em algum lugar confortável no meio. Medido em 11/08 a partir dos gestos do
# gabarito, a distância típica ficou perto de meio metro.
#
# O valor não precisa ser exato, e essa é a graça: um erro constante desloca a
# face inteira sem torcê-la, e a ORIENTAÇÃO — que é o que decide de qual lado
# a pessoa está — não muda nada.
BRACO_ADIANTE = 0.50

# Mínimo de gestos para arriscar uma resposta. Abaixo disso a reta ajustada
# descreve o acaso, não a estante.
AMOSTRAS_MINIMAS = 12

# Quão plana a nuvem precisa ser. É a razão entre o espalhamento
# perpendicular à face e o espalhamento ao longo dela: 0 é um plano perfeito,
# 1 é uma nuvem redonda, que não é face de coisa nenhuma.
ACHATAMENTO_MAXIMO = 0.55


@dataclass
class Alcance:
    """Um gesto de pegar, com tudo que o sistema soube dele."""
    x: float
    y: float
    rumo: float                     # radianos, no mundo
    altura_mao: float | None = None

    @property
    def contato(self):
        """Onde a mão provavelmente encostou, no chão."""
        # A mesma convenção de `boneco._girar`: frente = (-sin, cos).
        return (self.x - math.sin(self.rumo) * BRACO_ADIANTE,
                self.y + math.cos(self.rumo) * BRACO_ADIANTE)


@dataclass
class Estante:
    """A estante que os gestos desenharam."""
    x: float                        # centro da face, no chão
    y: float
    rumo_da_face: float             # radianos: para onde a face OLHA
    largura: float
    achatamento: float              # 0 = face perfeita; perto de 1 = nuvem
    amostras: int

    @property
    def firme(self):
        return (self.amostras >= AMOSTRAS_MINIMAS * 2
                and self.achatamento <= ACHATAMENTO_MAXIMO * 0.7)

    @property
    def normal(self):
        """Vetor unitário saindo da face, na direção de quem alcança."""
        return np.array([-math.sin(self.rumo_da_face),
                         math.cos(self.rumo_da_face)])

    def de_frente(self, x, y, folga=0.85):
        """Esta pessoa está na frente da estante, e perto o bastante?

        Duas condições, e as duas importam: estar do LADO CERTO (produto
        escalar positivo com a normal — quem está atrás da estante não alcança
        nada) e dentro do alcance útil.
        """
        d = np.array([x - self.x, y - self.y])
        adiante = float(d @ self.normal)
        lateral = abs(float(d @ np.array([math.cos(self.rumo_da_face),
                                          math.sin(self.rumo_da_face)])))
        return 0.0 <= adiante <= folga and lateral <= self.largura / 2 + 0.25


def _ajustar_reta(pontos):
    """Direção principal e espalhamentos de uma nuvem 2D.

    Feito por decomposição em valores singulares em vez de mínimos quadrados,
    porque a face pode ser vertical no plano do chão — e uma regressão de y
    sobre x explode exatamente nesse caso, que é o mais comum quando a estante
    está encostada numa parede lateral.
    """
    centro = pontos.mean(axis=0)
    _, s, vt = np.linalg.svd(pontos - centro, full_matrices=False)
    n = len(pontos)
    ao_longo, perpendicular = s / math.sqrt(max(1, n - 1))
    return centro, vt[0], float(ao_longo), float(perpendicular)


@dataclass
class LocalizadorDeEstante:
    """Acumula gestos e responde onde está a estante.

    Guarda uma janela dos últimos alcances, e não todos: se a estante for
    movida, os gestos antigos passam a mentir sobre o presente. Uma janela
    esquece o lugar antigo sozinha, no ritmo em que o novo é usado.
    """

    memoria: int = 400
    braco_adiante: float = BRACO_ADIANTE
    _alcances: list = field(default_factory=list)

    def observar(self, alcance):
        if alcance.rumo is None:
            return                      # sem rumo não há direção: não é voto
        self._alcances.append(alcance)
        del self._alcances[:-self.memoria]

    def esquecer_tudo(self):
        self._alcances.clear()

    @property
    def amostras(self):
        return len(self._alcances)

    def resolver(self):
        """A estante, ou None quando os gestos não desenham uma face."""
        if len(self._alcances) < AMOSTRAS_MINIMAS:
            return None

        pontos = np.array([a.contato for a in self._alcances], dtype=float)
        centro, direcao, ao_longo, perpendicular = _ajustar_reta(pontos)

        if ao_longo < 1e-6:
            return None
        achatamento = perpendicular / ao_longo
        if achatamento > ACHATAMENTO_MAXIMO:
            return None                 # nuvem redonda: não há face aqui

        # A normal aponta para o lado de onde as pessoas vieram — é o único
        # jeito de saber qual das duas faces é a que se usa.
        normal = np.array([-direcao[1], direcao[0]])
        corpos = np.array([[a.x, a.y] for a in self._alcances], dtype=float)
        if float(np.mean((corpos - centro) @ normal)) < 0:
            normal = -normal

        # rumo tal que (-sin, cos) == normal, a mesma convenção do resto
        rumo = math.atan2(-normal[0], normal[1])

        # Largura pelos percentis, não pelos extremos: um gesto perdido não
        # deve esticar a estante até ele.
        t = (pontos - centro) @ direcao
        largura = float(np.percentile(t, 95) - np.percentile(t, 5))

        return Estante(x=float(centro[0]), y=float(centro[1]),
                       rumo_da_face=rumo, largura=max(0.30, largura),
                       achatamento=float(achatamento),
                       amostras=len(self._alcances))


def prateleira_por_altura(altura_mao, prateleiras, tolerancia=0.15):
    """Qual prateleira essa mão alcançou. `prateleiras`: [(id, altura_m)].

    A tolerância existe porque a altura da mão é medida, não conhecida: o
    gabarito de 11/08 encontrou +-3 cm quando o pé aparece e +-8 cm quando a
    altura vem do tronco. Quinze centímetros cobrem o pior caso com folga.

    E TEM QUE SER MENOR QUE METADE DO VÃO. Com 20 cm — exatamente meio vão de
    40 — não sobra zona nenhuma: toda altura cai em alguma prateleira, e a
    função perde a capacidade de dizer "não sei". Uma mão a meio caminho entre
    duas prateleiras não está em nenhuma das duas, e responder ali é inventar.

        Um limiar que nunca recusa não está classificando: está arredondando.

        Quarenta centímetros de vão contra três a oito de erro: a prateleira
        é distinguível. O centímetro exato não é — e não precisa ser.
    """
    if altura_mao is None or not prateleiras:
        return None
    pid, alvo = min(prateleiras, key=lambda p: abs(p[1] - altura_mao))
    return pid if abs(alvo - altura_mao) < tolerancia else None


def prateleira_alcancada(estante, prateleiras, x, y, altura_mao):
    """A resposta completa: esta pessoa, neste lugar, pegou de qual prateleira.

    Exige as DUAS condições, e é isso que a torna diferente de olhar só para a
    altura: estar de frente para a estante, e a mão estar na faixa de alguma
    prateleira. Alguém com o braço a 0,95 m do outro lado da sala não está
    pegando da p3 — está coçando a cabeça.
    """
    if estante is None or not estante.de_frente(x, y):
        return None
    return prateleira_por_altura(altura_mao, prateleiras)
