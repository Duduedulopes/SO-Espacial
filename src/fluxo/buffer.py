"""
Fila limitada com descarte do mais antigo.

O PRINCIPIO

Para visao em tempo real, quadro velho nao tem valor. Se o consumidor esta
atrasado, o certo e ele pegar o PRESENTE, nao ir processando o passado com
atraso crescente.

Foi exatamente esse o defeito de 08/08: a camera entregava 30 quadros por
segundo, o processamento fazia 4, e os 26 restantes ficavam enfileirados no
driver. A imagem aparecia velha e o atraso CRESCIA sem parar — 688 ms medidos
num teste. Parecia lentidao; era fila.

POR QUE `maxlen=2` E NAO 1

Com 1, produtor e consumidor disputam o mesmo espaco: enquanto o consumidor le,
o produtor nao tem onde escrever, e a taxa efetiva cai. Com 2 ha folga de um
quadro — cerca de 30 ms a 30 fps, imperceptivel — e os dois trabalham sem se
esperar.

O QUE ISTO DA DE GRACA

`descartados`. Ate agora era impossivel responder "quantos quadros se
perderam?" porque ninguem contava. Aqui, todo descarte incrementa um contador.
"""

import threading
from collections import deque


class FrameBuffer:
    """Uma fila por fonte. Thread-safe. Descarta o antigo quando enche."""

    def __init__(self, maxlen=2, metricas=None):
        self._fila = deque(maxlen=maxlen)
        self._trava = threading.Lock()
        self._metricas = metricas
        self.maxlen = maxlen

    def colocar(self, frame):
        """Chamado pela thread da fonte. Nunca bloqueia.

        Bloquear aqui pararia a captura — e uma fonte parada e pior que um
        quadro perdido.
        """
        with self._trava:
            descartou = len(self._fila) == self._fila.maxlen
            self._fila.append(frame)
        if descartou and self._metricas is not None:
            self._metricas.registrar_descarte()
        return not descartou

    def pegar(self):
        """O MAIS RECENTE, jogando fora os anteriores.

        Nao e `popleft()`. Se ha dois quadros esperando, o antigo ja perdeu a
        validade — devolve-lo faria o sistema trabalhar no passado.
        """
        with self._trava:
            if not self._fila:
                return None
            velhos = len(self._fila) - 1
            frame = self._fila[-1]
            self._fila.clear()
        if velhos and self._metricas is not None:
            self._metricas.registrar_descarte(velhos)
        return frame

    def espiar(self):
        """Olha sem consumir. Usado pelo sincronizador para decidir o grupo."""
        with self._trava:
            return self._fila[-1] if self._fila else None

    def limpar(self):
        with self._trava:
            n = len(self._fila)
            self._fila.clear()
        if n and self._metricas is not None:
            self._metricas.registrar_descarte(n)

    def __len__(self):
        with self._trava:
            return len(self._fila)

    def vazio(self):
        return len(self) == 0
