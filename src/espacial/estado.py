"""
EstadoDePessoa — o que o SISTEMA concluiu, depois de filtrar e fundir.

A DIFERENCA QUE ESTA CLASSE MARCA

    Observacao      o que uma camera VIU     — hipotese, pode ser cadeira
    EstadoDePessoa  o que o sistema CONCLUIU — passou por todos os filtros

No sistema antigo os dois eram a mesma coisa: a caixa do YOLO virava posicao,
que virava desenho. Por isso a cadeira com roupas entrava no mapa de calor
antes de qualquer filtro ter chance de opinar.

CAMPOS QUE PARECEM DETALHE E NAO SAO

`incerteza` cresce quando o Kalman esta so prevendo. Quem consome precisa
saber a diferenca entre "ela esta ali" e "ela deveria estar ali".

`percorrido` e o unico sinal que mobilia nao consegue falsificar. Cadeira nao
anda. E o que impede o filtro de altura de aprender com o proprio erro — em
08/08 ele aprendeu com uma cadeira e passou a aceitar cadeiras.

`visto_por` diz quais vistas contribuiram. Um esqueleto montado so com a
frontal tem profundidade chutada; com frontal e lateral, medida.

`associacao_confiavel` e honestidade sobre o limite atual: com mais de uma
pessoa em cena, o sistema nao sabe qual pessoa da frontal corresponde a qual
do alto. Marcar isso e melhor que fingir que sabe.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EstadoDePessoa:
    id: int
    x: float
    y: float

    vx: float = 0.0
    vy: float = 0.0
    incerteza: float = 0.0
    rumo: float = 0.0

    esqueleto: np.ndarray | None = None      # (17,3) metros, no mundo
    # Quais juntas foram MEDIDAS. O MediaPipe sempre devolve as 17, mesmo as
    # que estao fora do quadro — desenhar essas e inventar com cara de dado.
    juntas_visiveis: np.ndarray | None = None

    # O QUE A PESSOA ESTA FAZENDO, no vocabulario fechado da arquitetura v3.
    # E isto que o desenho consome — nenhuma coordenada de junta atravessa
    # essa fronteira. Ver docs/ARQUITETURA-v3-ACAO.md.
    acao: object | None = None
    prevendo: int = 0                        # quadros sem medicao
    percorrido: float = 0.0
    quadros: int = 0
    visto_por: set = field(default_factory=set)
    associacao_confiavel: bool = True

    # DOIS INSTANTES, E ELES NAO SAO O MESMO.
    #
    #     t_mono    quando o SISTEMA concluiu
    #     t_medido  quando a CAMERA viu
    #
    # Entre um e outro passam o detector (130 ms medidos em 19/08), a fila e
    # o ciclo. Quem desenha precisa do segundo: a posicao entregue aqui e onde
    # a pessoa ESTAVA, e sem saber ha quanto tempo nao da para dizer onde ela
    # esta. A 1 m/s sao 15 cm de diferenca — um pe inteiro.
    #
    #     Uma medida sem a hora em que foi tirada so serve enquanto nada se
    #     move.
    t_mono: float = 0.0
    t_medido: float = 0.0

    @property
    def idade_s(self):
        """Ha quanto tempo a camera viu isto. Zero quando nao se sabe."""
        if not self.t_medido or not self.t_mono:
            return 0.0
        return max(0.0, self.t_mono - self.t_medido)

    @property
    def velocidade(self):
        return float(np.hypot(self.vx, self.vy))

    @property
    def parada(self):
        return self.velocidade < 0.15

    @property
    def tem_esqueleto(self):
        return self.esqueleto is not None

    def para_dicionario(self):
        return {
            "id": self.id,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "vx": round(self.vx, 3), "vy": round(self.vy, 3),
            "velocidade": round(self.velocidade, 3),
            "incerteza": round(self.incerteza, 3),
            "rumo": round(self.rumo, 3),
            "prevendo": self.prevendo,
            "percorrido": round(self.percorrido, 2),
            "quadros": self.quadros,
            "visto_por": sorted(self.visto_por),
            "tem_esqueleto": self.tem_esqueleto,
            "juntas_medidas": (int(self.juntas_visiveis.sum())
                               if self.juntas_visiveis is not None else 0),
            "acao": self.acao.para_dicionario() if self.acao else None,
            "associacao_confiavel": self.associacao_confiavel,
        }

    def __repr__(self):
        return (f"Pessoa#{self.id}({self.x:.2f},{self.y:.2f} "
                f"{self.velocidade:.2f}m/s "
                f"{'prevendo' if self.prevendo else 'medida'})")
