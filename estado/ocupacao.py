"""
Ocupacao do espaco: mapa de calor e zonas.

E aqui que percepcao vira INFORMACAO DE NEGOCIO. Ate agora o sistema sabia
onde as pessoas estao. A partir daqui ele sabe onde elas FICAM — que e a
pergunta que o varejo paga para responder.

Duas peças:

    MapaDeCalor   acumula permanencia por metro quadrado, com esquecimento
    Zona          regiao nomeada; conta visitas e tempo de permanencia

Nenhuma depende de camera ou de desenho. Recebem metros e devolvem numeros.
"""

import time

import cv2
import numpy as np


class MapaDeCalor:
    """Acumula quanto tempo cada pedaco do chao ficou ocupado.

    POR QUE ESQUECER

    Sem decaimento, o mapa vira o acumulado desde que o programa subiu, e
    depois de uma hora tudo fica saturado — todo lugar parece igualmente
    movimentado. Com meia-vida, ele mostra o padrao RECENTE, que e o que
    interessa para decidir onde por um produto hoje.

    POR QUE BORRAR

    A posicao tem ruido de alguns centimetros e o passo humano nao e continuo.
    Somar num unico ponto produziria um mapa granulado. O borrao gaussiano
    espalha a contribuicao pela incerteza real da medida.
    """

    def __init__(self, xmin, xmax, ymin, ymax, px_por_m=60, meia_vida_s=90.0):
        self.ext = (xmin, xmax, ymin, ymax)
        self.ppm = px_por_m
        self.larg = max(8, int((xmax - xmin) * px_por_m))
        self.alt = max(8, int((ymax - ymin) * px_por_m))
        self.grade = np.zeros((self.alt, self.larg), dtype=np.float32)
        self.meia_vida = meia_vida_s
        self._t = time.monotonic()

    def _px(self, x, y):
        xmin, _, ymin, _ = self.ext
        return int((x - xmin) * self.ppm), int((y - ymin) * self.ppm)

    def acumular(self, x, y, segundos):
        gx, gy = self._px(x, y)
        if 0 <= gx < self.larg and 0 <= gy < self.alt:
            self.grade[gy, gx] += segundos

    def passo(self):
        """Aplica o esquecimento. Chame uma vez por quadro."""
        agora = time.monotonic()
        dt = agora - self._t
        self._t = agora
        if self.meia_vida > 0 and dt > 0:
            self.grade *= 0.5 ** (dt / self.meia_vida)

    def imagem(self, suavizar=21, gama=0.40):
        """Devolve (colorido BGR, alfa 0..1). Vazio onde ninguem passou.

        POR QUE NAO NORMALIZAR LINEARMENTE

        Uma pessoa parada 30 s num ponto acumula muito mais que um corredor
        percorrido dezenas de vezes de passagem. Dividindo pelo maximo, o
        corredor vira preto e o mapa perde justamente a informacao de fluxo.

        A correcao e comprimir a escala: valor^0.4 levanta os medios e baixos
        sem estourar o pico. E o mesmo motivo pelo qual grafico de audiencia
        ou de terremoto usa escala logaritmica.
        """
        g = cv2.GaussianBlur(self.grade, (suavizar | 1, suavizar | 1), 0)
        pico = float(g.max())
        if pico < 1e-6:
            return None, None

        norm = np.clip(g / pico, 0, 1) ** gama
        cor = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # transparente onde ninguem passou, para o chao continuar visivel
        alfa = np.clip(norm * 1.1, 0, 0.72)
        alfa[g < pico * 1e-3] = 0.0
        return cor, alfa


class Zona:
    """Regiao nomeada do chao. Conta quem entrou e por quanto tempo.

    O tempo e contado POR RASTRO, nao por deteccao — senao uma pessoa parada
    por 10 s contaria 300 vezes a 30 fps em vez de 10 segundos.
    """

    def __init__(self, nome, x0, x1, y0, y1):
        self.nome = nome
        self.x0, self.x1 = min(x0, x1), max(x0, x1)
        self.y0, self.y1 = min(y0, y1), max(y0, y1)
        self.dentro: set[int] = set()
        self.tempo: dict[int, float] = {}     # rastro -> segundos acumulados
        self.visitas = 0

    def contem(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def atualizar(self, rastros_pos, dt):
        """rastros_pos: dict rastro -> (x, y)."""
        agora = set()
        for rid, (x, y) in rastros_pos.items():
            if self.contem(x, y):
                agora.add(rid)
                self.tempo[rid] = self.tempo.get(rid, 0.0) + dt

        self.visitas += len(agora - self.dentro)
        self.dentro = agora

    @property
    def ocupacao(self):
        return len(self.dentro)

    @property
    def tempo_total(self):
        return sum(self.tempo.values())

    @property
    def tempo_medio(self):
        return self.tempo_total / len(self.tempo) if self.tempo else 0.0
