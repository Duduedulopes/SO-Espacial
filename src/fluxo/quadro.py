"""
Frame e Instante — o vocabulario que atravessa o sistema.

DOIS RELOGIOS, E O MOTIVO

    t_mono   monotonico. Nunca anda para tras. So serve para medir intervalos.
    t_wall   relogio de parede, ISO 8601 COM FUSO. Serve para casar com o
             mundo — outro computador, o RFID, um registro humano.

Guardar so o de parede foi tentador e seria errado: ele pula com NTP e horario
de verao, e um pulo no meio de uma sessao estraga todos os intervalos sem que
ninguem perceba. Guardar so o monotonico tambem seria errado: ele nao tem
significado fora do processo.

O custo de guardar os dois e um float. O custo de descobrir tarde que faltava
um deles ja foi medido em horas.

`seq` existe para detectar buraco: se pular de 41 para 45, tres quadros se
perderam entre a fonte e aqui — coisa que fps medio nao revela.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


def agora_iso():
    """Relogio de parede com fuso. Sempre com fuso."""
    return datetime.now().astimezone().isoformat()


@dataclass(frozen=True)
class Frame:
    camera_id: str
    papel: str
    seq: int
    t_mono: float
    t_wall: str
    imagem: np.ndarray

    @property
    def largura(self):
        return self.imagem.shape[1]

    @property
    def altura(self):
        return self.imagem.shape[0]

    @property
    def brilho(self):
        """Amostrado de 8 em 8 pixels. A media completa de um quadro 720p custa
        ~2 ms; a amostrada custa 0,03 ms e da o mesmo diagnostico."""
        return float(self.imagem[::8, ::8].mean())

    def idade_ms(self, agora_mono):
        return (agora_mono - self.t_mono) * 1000.0

    def __repr__(self):
        return (f"Frame({self.camera_id}/{self.papel} #{self.seq} "
                f"{self.largura}x{self.altura})")


@dataclass
class Instante:
    """Quadros de fontes diferentes considerados simultaneos.

    `defasagem_ms` e a maior diferenca dentro do grupo. Nao e enfeite: quem
    consome precisa saber quanta simultaneidade esta assumindo. A 1,4 m/s,
    120 ms sao 17 cm — e a fusao de eixos assume que as vistas mostram o mesmo
    instante.
    """

    t_ref: float
    quadros: dict = field(default_factory=dict)      # papel -> Frame
    defasagem_ms: float = 0.0

    def __contains__(self, papel):
        return papel in self.quadros

    def get(self, papel):
        return self.quadros.get(papel)

    @property
    def papeis(self):
        return list(self.quadros)

    def __len__(self):
        return len(self.quadros)

    def __repr__(self):
        return (f"Instante({'+'.join(self.papeis)} "
                f"defasagem={self.defasagem_ms:.0f}ms)")
