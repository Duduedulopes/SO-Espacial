"""
Fonte de video com descarte de quadros velhos.

O PROBLEMA QUE ISTO RESOLVE

A camera entrega 30 quadros por segundo. Se o processamento faz 4, os 26
restantes NAO somem — o driver e o OpenCV os enfileiram. Quando voce chama
read(), recebe o mais antigo da fila, nao o mais recente.

Resultado: a imagem fica atrasada e anda aos trancos, e o atraso CRESCE com o
tempo. Parece lentidao, mas e fila.

A SOLUCAO

Uma thread separada le a camera o tempo todo, na velocidade dela, e guarda
somente o ULTIMO quadro. Quem processa sempre pega o presente, e os quadros
que nao deram tempo de processar sao descartados de proposito.

Isto nao acelera nada. Mas troca "imagem velha e travada" por "imagem atual,
com menos quadros" — que e muito melhor de olhar e de usar.
"""

import threading
import time

import cv2


def _brilho(cap, descartar=4):
    """Brilho medio do quadro atual. Descarta alguns: a camera leva tempo
    para reagir a uma mudanca de exposicao."""
    q = None
    for _ in range(descartar):
        ok, q = cap.read()
        if not ok:
            return 0.0
        time.sleep(0.03)
    return float(q[::8, ::8].mean()) if q is not None else 0.0


def aquecer(cap, segundos=2.0, minimo=8):
    """Espera o sensor acordar antes de julgar a imagem.

    ERRO QUE ISTO CORRIGE (08/08)

    A C920 entrega quadros PRETOS nos primeiros ~1,5 s depois de aberta — o
    sensor esta ligando, e isso e normal. O codigo media o brilho no primeiro
    quadro, via zero, e saia disparando dezenas de `set()` de exposicao.

    Resultado: a camera travava de vez, e passava a devolver preto em TODAS as
    configuracoes, inclusive exposicao -1, que numa camera sa estouraria de
    claro. Um diagnostico apressado virou a causa do defeito.

        Antes de concluir que algo esta quebrado, confira se ja terminou de
        comecar.

    Devolve o brilho apos o aquecimento.
    """
    t0 = time.monotonic()
    b = 0.0
    while time.monotonic() - t0 < segundos:
        ok, q = cap.read()
        if ok and q is not None:
            b = float(q[::8, ::8].mean())
            if b >= minimo:
                return b
        time.sleep(0.05)
    return b


def garantir_imagem_visivel(cap, minimo=30, verboso=True):
    """Insiste ate a imagem ficar visivel — e devolve o que funcionou.

    POR QUE ISTO EXISTE

    Mandar `AUTO_EXPOSURE = 0.75` deveria bastar. Nao basta: o valor que
    significa "automatico" varia entre drivers e versoes do OpenCV (0.75, 1, 3,
    -1 aparecem por ai), e alguns drivers simplesmente ignoram o pedido.

    A C920 guarda a exposicao manual no proprio hardware. Depois de rodar com
    -6, ela fica preta e continua preta entre execucoes.

        Nao confie no `set()`. Meça o resultado.

    Estrategia: tenta os valores de "automatico" conhecidos, mede o brilho, e
    se nada resolver varre a exposicao manual do escuro para o claro ate a
    imagem aparecer.
    """
    # PRIMEIRO deixa acordar. Sem isto, o codigo confunde "ainda ligando" com
    # "quebrada" e estraga a camera tentando consertar.
    b = aquecer(cap)
    if b >= minimo:
        if verboso:
            print(f"    ok apos aquecer (brilho {b:.0f})")
        return ("ja-ok", None, b)

    tentativas = [("apos aquecer", b)]

    for v in (0.75, 1, 3, -1):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
        b = _brilho(cap)
        tentativas.append((f"auto={v}", b))
        if b >= minimo:
            if verboso:
                print(f"    exposicao automatica ok (auto={v}, brilho {b:.0f})")
            return ("auto", v, b)

    # O automatico nao pegou. Varre o manual, do mais escuro para o mais claro.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    for e in (-6, -5, -4, -3, -2, -1):
        cap.set(cv2.CAP_PROP_EXPOSURE, e)
        b = _brilho(cap)
        tentativas.append((f"manual={e}", b))
        if b >= minimo:
            if verboso:
                print(f"    exposicao manual {e} (brilho {b:.0f})")
            return ("manual", e, b)

    if verboso:
        print("    NAO CONSEGUI CLAREAR. tentativas:")
        for nome, b in tentativas:
            print(f"      {nome:14} brilho {b:5.1f}")
        if all(b <= 0.5 for _, b in tentativas):
            print("    Brilho ZERO em TODAS as configuracoes, inclusive as mais")
            print("    claras. Isso nao e exposicao — a camera nao esta")
            print("    entregando imagem. Quase sempre e handle preso de uma")
            print("    execucao anterior.")
            print("    -> feche este programa, desconecte o cabo USB por 10s,")
            print("       reconecte, e rode UMA vez so.")
    return (None, None, tentativas[-1][1] if tentativas else 0.0)


def configurar_exposicao(cap, exposicao=None):
    """Define a exposicao — e VOLTA AO AUTOMATICO quando nao ha valor.

    ARMADILHA DESCOBERTA EM 08/08

    A C920 guarda a exposicao manual NO PROPRIO DRIVER. Depois de rodar uma vez
    com -6, ela fica assim: fechar o programa nao desfaz, reiniciar o Python
    nao desfaz. Só desconectar o cabo, ou mandar explicitamente voltar ao
    automatico.

    A versao anterior tratava `None` como "nao mexer" — e nao mexer deixava a
    configuracao velha valendo. Resultado: imagem preta, zero deteccoes, e
    horas procurando erro no detector.

        Nao mexer nao e o mesmo que voltar ao padrao.

    Vale para qualquer estado que persista fora do seu processo.
    """
    if exposicao is None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 0.75 = automatico (DSHOW)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, exposicao)


class CameraAoVivo:
    def __init__(self, indice=0, largura=640, altura=480, fps=30,
                 exposicao=None, backend=cv2.CAP_DSHOW):
        # O BACKEND E PARTE DA IDENTIDADE DA CAMERA.
        #
        # DSHOW e MSMF enumeram os dispositivos em ordens DIFERENTES. Um indice
        # vindo da lista do DirectShow so significa alguma coisa se aberto com
        # DirectShow. Abrir com outro backend aponta para outra camera — e foi
        # isso que trocou as vistas de lugar varias vezes em 08/08.
        self.cap = cv2.VideoCapture(indice, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Pede ao driver o menor buffer possivel. Nem toda camera obedece —
        # por isso a thread existe mesmo assim.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # IMAGEM VERDE: quando o backend MSMF entrega o quadro em formato de
        # cor cru (YUV, NV12), o OpenCV interpreta como BGR e a imagem sai
        # esverdeada. Pedir a conversao explicita resolve.
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        configurar_exposicao(self.cap, exposicao)

        if not self.cap.isOpened():
            raise SystemExit(
                f"nao consegui abrir a camera {indice}.\n"
                "  - outro programa pode estar usando (Iriun, app Camera, Teams)\n"
                "  - processo python.exe travado segurando o dispositivo\n"
                "  - o indice pode ter mudado: python captura/identificar.py")

        # Confere o que a camera REALMENTE aceitou. Pedir nao e receber, e
        # descobrir isso tarde custou horas em 07/08.
        self.largura = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.altura = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.indice = indice

        self._quadro = None
        self._t = 0.0
        self._n = 0
        self._trava = threading.Lock()
        self._rodando = True
        self._thread = threading.Thread(target=self._laco, daemon=True)
        self._thread.start()

        # espera o primeiro quadro
        t0 = time.monotonic()
        while self._quadro is None and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        if self._quadro is None:
            self.fechar()
            raise SystemExit(
                f"camera {indice} abriu mas nao entregou nenhum quadro em 5s.\n"
                "  Costuma ser disputa de banda USB quando ha varias cameras.\n"
                "  Tente resolucao menor, ou outra porta USB.")

        print(f"  camera {indice}: {self.largura}x{self.altura} ok")

    def _laco(self):
        while self._rodando:
            ok, q = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._trava:
                self._quadro = q
                self._t = time.monotonic()
                self._n += 1

    def ler(self):
        """Devolve (quadro, instante). Sempre o mais recente.

        Nao copia: entrega a referencia e zera o slot. A thread de captura
        cria um array novo a cada read() do OpenCV, entao ninguem escreve por
        cima do que voce esta usando. Copiar um quadro de 720p custa ~2,7 MB
        por chamada, e a 10 fps isso e memoria jogada fora sem motivo.
        """
        with self._trava:
            if self._quadro is None:
                return None, 0.0
            q, t = self._quadro, self._t
            return q, t

    @property
    def brilho(self):
        with self._trava:
            if self._quadro is None:
                return 0.0
            return float(self._quadro[::8, ::8].mean())

    @property
    def quadros_capturados(self):
        return self._n

    def fechar(self):
        self._rodando = False
        self._thread.join(timeout=1.0)
        self.cap.release()
