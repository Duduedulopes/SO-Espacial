"""
Observacao — o que UMA camera viu de UMA pessoa, num instante.

A DISTINCAO QUE ESTA ESTRUTURA IMPOE

    Observacao      o que uma camera VIU        (pode estar errado)
    EstadoDePessoa  o que o sistema CONCLUIU    (depois de filtrar e fundir)

O sistema antigo nao separava os dois. A caixa do YOLO virava posicao, que
virava desenho, tudo no mesmo laco. Consequencia pratica: quando a cadeira era
detectada como pessoa, ela ja tinha entrado no mapa de calor antes de qualquer
filtro ter chance de opinar.

Separando, a observacao e sempre uma HIPOTESE. Quem decide o que e verdade e o
SpatialEngine, com os filtros de plausibilidade, tornozelo e Kalman.

`id_externo` e o identificador do rastreador DAQUELA camera. Nao e a identidade
da pessoa — duas cameras dao numeros diferentes para o mesmo corpo. Amarrar as
duas coisas exige re-identificacao, que ainda nao existe.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observacao:
    camera_id: str
    papel: str
    t_mono: float

    caixa: tuple | None = None          # (x1, y1, x2, y2) em pixels
    id_externo: int = -1                # id do rastreador daquela camera
    confianca: float = 0.0

    juntas_2d: np.ndarray | None = None   # (17,2) pixels
    conf_2d: np.ndarray | None = None     # (17,)
    juntas_3d: np.ndarray | None = None   # (17,3) metros, rel. ao quadril

    extras: dict = field(default_factory=dict)

    @property
    def tem_pose(self):
        return self.juntas_3d is not None

    @property
    def tem_caixa(self):
        return self.caixa is not None

    def __repr__(self):
        p = []
        if self.tem_caixa:
            p.append(f"caixa#{self.id_externo}")
        if self.juntas_2d is not None:
            p.append("2d")
        if self.tem_pose:
            p.append("3d")
        return f"Obs({self.papel} {'+'.join(p) or 'vazia'})"
