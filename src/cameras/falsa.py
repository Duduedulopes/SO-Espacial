"""
Fonte falsa — a peca que torna o sistema testavel.

POR QUE ISTO E A CLASSE MAIS IMPORTANTE DO PACOTE

Todos os problemas de 07 e 08/08 foram descobertos rodando o sistema com
hardware, na sala, com a pessoa andando na frente da camera. Consequencias:

  - cada verificacao levava minutos
  - so era possivel testar o que o ambiente permitia
  - queda de camera so acontecia por acaso
  - nenhum teste podia rodar duas vezes igual

Com uma fonte falsa, um teste roda em milissegundos, em qualquer maquina, e
reproduz exatamente o mesmo caso — inclusive os que sao dificeis de provocar
de proposito: a camera que cai aos 3 s, a que entrega preto, a que trava.

    Se um defeito so aparece com hardware, ele so sera investigado quando
    houver hardware por perto. E ai ja e tarde.

Modos:
    "movimento"  um retangulo atravessando a cena — verifica o encanamento
    "preta"      quadros pretos — reproduz a C920 com exposicao travada
    "instavel"   entrega e para, em ciclo — exercita DEGRADADA e FALHA
    "lenta"      entrega abaixo do alvo — testa se atrasa as outras fontes
    "morta"      nunca entrega — testa o recuo exponencial
"""

import time

import numpy as np

from src.cameras.fonte import FonteDeVideo
from src.nucleo.erros import CameraNaoAbriu


class FonteFalsa(FonteDeVideo):
    tipo = "falsa"

    def __init__(self, papel, modo="movimento", fps=30, id=None,
                 largura=640, altura=480, falhar_ao_abrir=False,
                 cair_apos_s=None, **kw):
        super().__init__(id=id or f"falsa-{papel}", papel=papel,
                         largura=largura, altura=altura, fps_alvo=fps, **kw)
        self.modo = modo
        self.falhar_ao_abrir = falhar_ao_abrir
        self.cair_apos_s = cair_apos_s
        self._aberta = False
        self._t_abertura = 0.0
        self._n = 0

    def _abrir(self):
        if self.falhar_ao_abrir:
            raise CameraNaoAbriu("fonte falsa configurada para falhar",
                                 modo=self.modo)
        self._aberta = True
        self._t_abertura = time.monotonic()
        self._n = 0

    def _fechar(self):
        self._aberta = False

    def _ler_bruto(self):
        if not self._aberta:
            return None

        time.sleep(1.0 / max(1, self.fps_alvo))
        self._n += 1
        decorrido = time.monotonic() - self._t_abertura

        if self.cair_apos_s and decorrido > self.cair_apos_s:
            return None

        if self.modo == "morta":
            return None

        if self.modo == "preta":
            return np.zeros((self.altura, self.largura, 3), np.uint8)

        if self.modo == "instavel":
            # 2 s entregando, 2 s em silencio
            if int(decorrido) % 4 >= 2:
                return None

        if self.modo == "lenta":
            time.sleep(0.2)

        return self._quadro_com_movimento()

    def _quadro_com_movimento(self):
        """Fundo cinza, retangulo claro atravessando e um contador.

        O retangulo se move de forma deterministica, entao um teste pode
        conferir a POSICAO esperada — nao so que "chegou algum quadro".
        """
        q = np.full((self.altura, self.largura, 3), 60, np.uint8)
        x = int((self._n * 6) % max(1, self.largura - 80))
        y = self.altura // 2 - 40
        q[y:y + 80, x:x + 80] = 200
        q[4:4 + 6, 4:4 + min(self.largura - 8, self._n % 200)] = 255
        return q
