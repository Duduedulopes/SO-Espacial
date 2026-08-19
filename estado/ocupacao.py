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

    A FRONTEIRA NAO E UMA LINHA. MEDIDO EM 19/08.

    Numa corrida de 45 s com uma pessoa SENTADA, parada:

        PERSON_ENTERED_ZONE      15
        PERSON_LEFT_ZONE         15

    Quinze entradas e quinze saidas de quem nao saiu do lugar. A posicao tem
    um ou dois centimetros de ruido, e quem esta em cima da borda atravessa a
    linha varias vezes por segundo sem se mexer.

        Um limiar unico sobre um sinal com ruido nao decide: ele conta o
        ruido.

    Numa loja isso viraria quinze registros de um cliente que nao fez nada, e
    a metrica de visitas — que e o produto disto aqui — perderia o sentido.

    DUAS DEFESAS, PORQUE ELAS PEGAM COISAS DIFERENTES

    `margem_m` e a banda morta: para ENTRAR e preciso estar dentro com folga;
    para SAIR, fora com folga. Sao dois limiares em vez de um, e entre eles
    vale o estado anterior. Pega o tremor pequeno e continuo.

    `confirmar_s` e o tempo: a mudanca so conta se durar. Pega o salto grande
    e raro — um id reciclado, um quadro em que o pe foi estimado pela caixa —
    que atravessaria a banda morta inteira de uma vez.

        Ruido pequeno se resolve no espaco; ruido grande, no tempo. Uma
        defesa so deixa passar metade.

    O preco e declarado: entradas e saidas aparecem `confirmar_s` mais tarde.
    Para contar gente numa loja, 0,4 s de atraso nao custa nada; quinze
    eventos falsos custam a metrica inteira.
    """

    def __init__(self, nome, x0, x1, y0, y1, margem_m=0.10, confirmar_s=0.4):
        self.nome = nome
        self.x0, self.x1 = min(x0, x1), max(x0, x1)
        self.y0, self.y1 = min(y0, y1), max(y0, y1)
        self.confirmar_s = confirmar_s

        # A MARGEM NAO PODE ENGOLIR A ZONA.
        #
        # Uma porta de 0,45 m com 0,25 de margem encolheria para nada, e
        # ninguem conseguiria entrar nunca — um defeito que se manifesta como
        # silencio, que e o mais caro de achar. Teto de 40% do lado menor.
        lado_menor = min(self.x1 - self.x0, self.y1 - self.y0)
        self.margem_m = min(margem_m, 0.4 * lado_menor) if lado_menor > 0 else 0.0

        self.dentro: set[int] = set()
        self.tempo: dict[int, float] = {}     # rastro -> segundos acumulados
        self.visitas = 0
        self._mudando: dict[int, float] = {}  # rastro -> ha quanto tempo insiste

    def contem(self, x, y):
        """O teste geometrico cru, sem histerese. Para desenho e consulta.

        Quem quer saber quem ESTA na zona pergunta a `dentro`. Este metodo
        responde outra coisa: se um ponto cai no retangulo. Foi por confundir
        os dois que os eventos saiam sem filtro — ver `gemeo._detectar_zonas`.
        """
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def _quer_estar(self, x, y, estava):
        """A banda morta: a zona cresce para quem esta dentro, encolhe para
        quem esta fora. Sao dois limiares, e entre eles nada muda."""
        m = self.margem_m if estava else -self.margem_m
        return (self.x0 - m <= x <= self.x1 + m
                and self.y0 - m <= y <= self.y1 + m)

    def atualizar(self, rastros_pos, dt):
        """rastros_pos: dict rastro -> (x, y)."""
        for rid, (x, y) in rastros_pos.items():
            estava = rid in self.dentro
            quer = self._quer_estar(x, y, estava)

            if quer == estava:
                self._mudando.pop(rid, None)
            else:
                insistindo = self._mudando.get(rid, 0.0) + dt
                if insistindo >= self.confirmar_s:
                    self._mudando.pop(rid, None)
                    if quer:
                        self.dentro.add(rid)
                        self.visitas += 1
                        # O TEMPO DA ESPERA E CREDITADO, E ISSO NAO E DETALHE.
                        #
                        # A pessoa ESTAVA na zona durante a confirmacao; so o
                        # programa e que ainda nao tinha decidido. Descartar
                        # esse tempo tiraria `confirmar_s` de cada visita, de
                        # forma sistematica — um vies, e vies nao sai na media.
                        #
                        #     Atrasar a decisao e diferente de perder a
                        #     medida. So a primeira e de graca.
                        #
                        # `- dt` porque o dt deste quadro entra logo abaixo,
                        # pelo caminho normal.
                        self.tempo[rid] = (self.tempo.get(rid, 0.0)
                                           + max(0.0, insistindo - dt))
                    else:
                        self.dentro.discard(rid)
                else:
                    self._mudando[rid] = insistindo

            if rid in self.dentro:
                self.tempo[rid] = self.tempo.get(rid, 0.0) + dt

        # Rastro que morreu sai na hora, sem esperar confirmacao: nao ha o que
        # confirmar sobre alguem que o sistema deixou de ver.
        for rid in list(self.dentro):
            if rid not in rastros_pos:
                self.dentro.discard(rid)
                self._mudando.pop(rid, None)
        for rid in list(self._mudando):
            if rid not in rastros_pos:
                del self._mudando[rid]

    @property
    def ocupacao(self):
        return len(self.dentro)

    @property
    def tempo_total(self):
        return sum(self.tempo.values())

    @property
    def tempo_medio(self):
        return self.tempo_total / len(self.tempo) if self.tempo else 0.0
