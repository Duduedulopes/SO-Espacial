"""A tela da banca. Uma janela, lida do fundo da sala.

    acho que a ideia da tela de apresentacao e otima
    acho que a tela de apresentacao e o mais "rapido" e acho que com ela ja
    vamos identificar alguns erros                      — Eduardo, 12/08

A SEGUNDA FRASE E QUE DEFINE ESTE ARQUIVO.

Uma tela que so mostra o resultado bonito e um enfeite: quando o boneco levanta
o braco sozinho, ela nao ajuda em nada — o erro aparece e nao tem onde ser
investigado, e a resposta so pode ser "estranho". Entao cada resposta desta
tela vem acompanhada de DE ONDE ELA VEIO.

    Campo que nao diz de onde veio nao pode ser depurado, so trocado.
                                            — corpo.py, sobre `fonte_rumo`

`fonte_braco_esq`, `fonte_braco_dir`, `fonte_rumo` e `fonte_escala` ja existiam
em `LeituraDoCorpo` desde 12/08 e nunca tinham chegado a uma tela. Aqui eles
ficam ao lado da resposta, no mesmo instante, para o gesto e a procedencia
serem lidos juntos. Se o braco subir sozinho, da para ver na hora QUAL camera
disse isso.

Isso serve a banca e serve a depuracao ao mesmo tempo, e nao por acidente: uma
demonstracao que mostra como sabe e mais convincente que uma que so afirma.

O LAYOUT

    +----------------------------------+---------------------+
    |                                  |  QUEM               |
    |                                  |  O QUE              |
    |          GEMEO 3D                |  QUANTAS            |
    |                                  |---------------------|
    |                                  |  procedencia        |
    +----------------------------------+---------------------+
    |   ALTO      |   FRONTAL    |   LATERAL                 |
    +----------------------------------+---------------------+

As tres cameras ficam na tela o tempo todo, lado a lado. Nao e enfeite: e a
unica maneira de olhar e VER a complementaridade, em vez de ouvir alguem
afirmar que ela existe.
"""
from __future__ import annotations

import cv2
import numpy as np

FUNDO = (247, 246, 246)
CARTAO = (255, 255, 255)
BORDA = (228, 226, 228)
TINTA = (58, 52, 58)
FRACO = (150, 145, 150)
DESTAQUE = (122, 42, 108)          # o mesmo roxo do boneco
ALERTA = (44, 44, 190)             # BGR: vermelho

FONTE = cv2.FONT_HERSHEY_SIMPLEX
FAIXA_CAMERAS = 190
COLUNA = 470
MARGEM = 16


def _texto(img, txt, xy, escala=0.5, cor=TINTA, grossura=1):
    cv2.putText(img, txt, xy, FONTE, escala, cor, grossura, cv2.LINE_AA)


def _cartao(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), CARTAO, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDA, 1, cv2.LINE_AA)


def _encaixar(img, w, h):
    """Redimensiona preservando a proporcao e centraliza na caixa."""
    fundo = np.full((h, w, 3), FUNDO, np.uint8)
    if img is None or img.size == 0:
        return fundo
    ih, iw = img.shape[:2]
    e = min(w / iw, h / ih)
    novo = cv2.resize(img, (max(1, int(iw * e)), max(1, int(ih * e))))
    nh, nw = novo.shape[:2]
    y, x = (h - nh) // 2, (w - nw) // 2
    fundo[y:y + nh, x:x + nw] = novo
    return fundo


class Apresentacao:
    """Compoe a tela. Nao le camera, nao decide nada — so desenha.

    Recebe imagens e numeros ja apurados e devolve UM quadro. Por ser pura, ela
    tem teste: da para montar a tela inteira sem hardware nenhum e conferir que
    a resposta certa foi parar no lugar certo.
    """

    def __init__(self, largura=1600, altura=900):
        self.largura = largura
        self.altura = altura

    @property
    def tamanho_da_cena(self):
        """Onde o gemeo 3D vai. Quem desenha a cena precisa saber ANTES."""
        return (self.largura - COLUNA - MARGEM * 3,
                self.altura - FAIXA_CAMERAS - MARGEM * 3 - 44)

    # ------------------------------------------------------------------ partes
    def _cabecalho(self, img, rodape):
        _texto(img, "SO ESPACIAL", (MARGEM + 4, 30), 0.72, DESTAQUE, 2)
        _texto(img, "gemeo digital  |  3 cameras", (200, 30), 0.5, FRACO)
        if rodape:
            largura_txt = cv2.getTextSize(rodape, FONTE, 0.45, 1)[0][0]
            _texto(img, rodape, (self.largura - largura_txt - MARGEM, 30),
                   0.45, FRACO)

    def _respostas(self, img, x, y, w, h, pessoas):
        _cartao(img, x, y, w, h)

        if not pessoas:
            _texto(img, "ninguem em cena", (x + 22, y + 58), 0.62, FRACO)
            _texto(img, "as tres cameras estao abertas e esperando.",
                   (x + 22, y + 88), 0.44, FRACO)
            return

        # AS TRES PERGUNTAS, ESCRITAS. Uma coluna de numeros sem titulo faz
        # quem assiste adivinhar o que esta olhando — e numa banca a plateia
        # nao pergunta, so deixa de entender.
        _texto(img, "QUEM", (x + 22, y + 16), 0.4, FRACO)
        _texto(img, "O QUE", (x + 240, y + 16), 0.4, FRACO)
        _texto(img, "QUANTAS", (x + w - 96, y + 16), 0.4, FRACO)
        cv2.line(img, (x + 16, y + 24), (x + w - 16, y + 24), BORDA, 1)

        linha = y + 30
        for p in pessoas[:3]:
            linha = self._uma_pessoa(img, x, linha, w, p)

    def _uma_pessoa(self, img, x, y, w, p):
        """QUEM / O QUE / QUANTAS de uma pessoa, com a procedencia embaixo."""
        _texto(img, f"#{p['id']}", (x + 22, y + 36), 0.95, DESTAQUE, 2)
        _texto(img, p.get("postura", ""), (x + 90, y + 36), 0.5, TINTA)
        if p.get("locomocao"):
            _texto(img, p["locomocao"], (x + 90, y + 58), 0.44, FRACO)

        # O QUE: a prateleira. Sem palpite firme, o texto DIZ que nao e firme —
        # um "p3" com a mesma cara quando o sistema tem certeza e quando nao
        # tem seria a pior coisa que esta tela poderia fazer.
        prat, firme = p.get("prateleira"), p.get("firme", False)
        if prat:
            cor = DESTAQUE if firme else FRACO
            _texto(img, prat.upper(), (x + 240, y + 40), 1.0, cor, 2)
            if not firme:
                _texto(img, "sem certeza", (x + 240, y + 62), 0.4, ALERTA)
        else:
            _texto(img, "-", (x + 240, y + 40), 1.0, FRACO, 2)

        n = p.get("quantas", 0)
        _texto(img, str(n), (x + w - 78, y + 40), 1.0,
               DESTAQUE if n else FRACO, 2)
        _texto(img, "un", (x + w - 44, y + 62), 0.4, FRACO)

        # PROCEDENCIA. A parte que existe para o erro ter onde ser visto.
        alto = y + 86
        for rotulo, valor, fonte in (
                ("bracos", p.get("bracos", ""), p.get("fonte_bracos", "")),
                ("mao", p.get("altura_mao", ""), p.get("fonte_escala", "")),
                ("rumo", p.get("rumo", ""), p.get("fonte_rumo", ""))):
            vazio = str(valor).strip() in ("", "-", "?")
            _texto(img, rotulo, (x + 22, alto), 0.4, FRACO)
            _texto(img, str(valor), (x + 100, alto), 0.44,
                   FRACO if vazio else TINTA)

            # FONTE SEM VALOR NAO E PROCEDENCIA, E RUIDO.
            #
            # MEDIDO EM 12/08, na primeira corrida:
            #
            #     mao   -            camera do alto
            #
            # Nao ha altura de mao nenhuma, e ao lado do nada esta escrito de
            # onde o nada veio. `fonte_escala` fica preenchido enquanto a
            # camera do alto responde a ESCALA — o que e verdade e nao tem
            # relacao com haver ou nao uma mao lida naquele quadro.
            #
            # Uma coluna que explica respostas inexistentes ensina a plateia a
            # ignorar a coluna, e ela existe justamente para ser lida.
            if vazio:
                _texto(img, "nao respondeu", (x + 300, alto), 0.4, FRACO)
            else:
                # Sem fonte declarada o valor veio de lugar nenhum que se
                # possa apontar. Isso e um defeito, e ele fica vermelho.
                _texto(img, fonte or "sem fonte", (x + 300, alto), 0.4,
                       FRACO if fonte else ALERTA)
            alto += 20

        cv2.line(img, (x + 16, alto + 8), (x + w - 16, alto + 8), BORDA, 1)
        return alto + 22

    def _cameras(self, img, y, vistas):
        """As tres, sempre no mesmo lugar, com o que cada uma entrega."""
        n = max(1, len(vistas))
        largura = (self.largura - MARGEM * (n + 1)) // n
        alto = FAIXA_CAMERAS - 30
        for i, (papel, entrega, quadro) in enumerate(vistas):
            x = MARGEM + i * (largura + MARGEM)
            _cartao(img, x, y, largura, FAIXA_CAMERAS)
            img[y + 26:y + 26 + alto, x + 6:x + 6 + largura - 12] = _encaixar(
                quadro, largura - 12, alto)
            # Camera que nao chegou fica escrita em vermelho, no lugar dela.
            # Uma janela que some e ambigua; um cartao vazio e uma acusacao.
            viva = quadro is not None
            _texto(img, papel.upper(), (x + 10, y + 18), 0.46,
                   DESTAQUE if viva else ALERTA, 1)
            _texto(img, entrega if viva else "nao chegou",
                   (x + 90, y + 18), 0.4, FRACO if viva else ALERTA)

    # ------------------------------------------------------------------ tela
    def desenhar(self, cena, pessoas=(), vistas=(), rodape=""):
        img = np.full((self.altura, self.largura, 3), FUNDO, np.uint8)
        self._cabecalho(img, rodape)

        cw, ch = self.tamanho_da_cena
        topo = 44
        _cartao(img, MARGEM, topo, cw, ch)
        img[topo + 1:topo + ch, MARGEM + 1:MARGEM + cw] = _encaixar(
            cena, cw - 1, ch - 1)

        self._respostas(img, MARGEM * 2 + cw, topo, COLUNA, ch, list(pessoas))
        self._cameras(img, topo + ch + MARGEM, list(vistas))
        return img
