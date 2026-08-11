"""
Camera USB local, identificada por NOME.

REGRAS APRENDIDAS EM 07 e 08/08, TODAS COM CUSTO MEDIDO

1. SOMENTE DirectShow.
   A enumeracao por nome vem do DSHOW. Abrir por MSMF aponta para outro
   dispositivo. Um fallback entre backends ja trocou cameras de lugar e
   produziu cinco sintomas que pareciam cinco bugs diferentes.

2. AQUECER antes de julgar.
   A C920 entrega quadros pretos nos primeiros ~1,5 s — o sensor esta ligando.
   Medir o brilho no primeiro quadro, ver zero e sair disparando `set()` de
   exposicao TRAVA a camera de vez. O diagnostico apressado virava a causa
   do defeito.

3. VOLTAR ao automatico explicitamente.
   A camera guarda exposicao, brilho e ganho NO HARDWARE. Fechar o programa
   nao desfaz; reiniciar o Windows nao desfaz. "Nao mexer" deixa a
   configuracao velha valendo — e uma exposicao -6 herdada deixa a imagem
   preta e ninguem e detectado.

4. MJPG antes da resolucao.
   Sem compressao, 1280x720 sem compressao a 30 fps pede 55 MB/s, acima do que
   o USB 2.0 entrega. A camera responde baixando a taxa em silencio.

5. Abrir e caro. Abra uma vez.
   Abrir e fechar a mesma camera tres vezes seguidas travava a C920 ate o cabo
   ser desconectado.
"""

import time

import cv2

from src.cameras.dispositivos import exigir_indice
from src.cameras.fonte import FonteDeVideo
from src.nucleo.erros import CameraImagemInvalida, CameraNaoAbriu, CameraSemImagem


class UsbCameraSource(FonteDeVideo):
    tipo = "usb"

    def __init__(self, nome, papel, exposicao=None, brilho_minimo=8, **kw):
        super().__init__(id=nome, papel=papel, **kw)
        self.nome = nome
        self.exposicao = exposicao
        self.brilho_minimo = brilho_minimo
        self._cap = None
        self._indice = None

    # ------------------------------------------------------------ abertura
    def _abrir(self):
        self._indice = exigir_indice(self.nome)

        cap = cv2.VideoCapture(self._indice, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise CameraNaoAbriu(f"'{self.nome}' nao abriu",
                                 indice=self._indice, backend="DSHOW")

        # ordem importa: codec antes da resolucao (regra 4)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.altura)
        cap.set(cv2.CAP_PROP_FPS, self.fps_alvo)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)      # evita a imagem esverdeada

        self._configurar_exposicao(cap)

        brilho = self._aquecer(cap)               # regra 2
        if brilho < self.brilho_minimo:
            brilho = self._insistir_por_imagem(cap)

        if brilho < self.brilho_minimo:
            cap.release()
            raise CameraImagemInvalida(
                f"'{self.nome}' abriu mas a imagem esta preta",
                indice=self._indice, brilho=round(brilho, 1))

        # pedir nao e receber: registre o que a camera de fato aceitou
        self.largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self.log.info("aberta", indice=self._indice,
                      resolucao=f"{self.largura}x{self.altura}",
                      brilho=round(brilho, 1))

        # PASSAR RASPANDO NAO E PASSAR BEM.
        #
        # MEDIDO EM 10/08: o tablet abriu com brilho 11,2 — acima do minimo de
        # 8, entao ficou ONLINE e entregou 462 quadros. O MediaPipe achou zero
        # poses em todos, porque 11 de 255 e preto com ruido. A camera virtual
        # do Windows entrega video para o painel de Configuracoes por um
        # caminho e preto para o DirectShow por outro.
        #
        # O limiar existe para separar "sem imagem" de "com imagem". Nao
        # separa "com imagem" de "com imagem UTIL". Entre os dois ha uma faixa
        # onde o sistema funciona no papel e nao ve nada — e essa faixa
        # precisava de voz.
        if brilho < self.brilho_minimo * 4:
            self.log.aviso(
                "imagem quase preta — a camera abriu mas pode nao estar "
                "entregando video de verdade",
                brilho=round(brilho, 1), minimo=self.brilho_minimo,
                dica="cena real costuma ficar entre 40 e 120")

    def _configurar_exposicao(self, cap):
        """None => automatico EXPLICITO. Regra 3."""
        if self.exposicao is None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposicao)

    def _aquecer(self, cap, segundos=2.0):
        """Le e descarta ate a imagem aparecer. Regra 2."""
        t0 = time.monotonic()
        b = 0.0
        while time.monotonic() - t0 < segundos:
            ok, q = cap.read()
            if ok and q is not None and q.size:
                b = float(q[::8, ::8].mean())
                if b >= self.brilho_minimo:
                    return b
            time.sleep(0.05)
        return b

    def _insistir_por_imagem(self, cap):
        """Varre modos de exposicao ate a imagem aparecer.

        Absorve o antigo `reparar.py`. Fica aqui, e nao num script separado,
        porque e parte da abertura — nao uma ferramenta que o usuario lembra
        de rodar quando algo da errado.
        """
        self.log.aviso("imagem escura, tentando recuperar")
        for v in (0.75, 1, 3, -1):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
            b = self._aquecer(cap, 0.8)
            if b >= self.brilho_minimo:
                self.log.info("recuperada no automatico", auto=v,
                              brilho=round(b, 1))
                return b

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        for e in (-6, -5, -4, -3, -2, -1):
            cap.set(cv2.CAP_PROP_EXPOSURE, e)
            b = self._aquecer(cap, 0.5)
            if b >= self.brilho_minimo:
                self.log.info("recuperada no manual", exposicao=e,
                              brilho=round(b, 1))
                return b
        return 0.0

    def _recuperar(self):
        """Antes de reabrir apos falha: da tempo ao driver.

        Reabrir imediatamente costuma pegar o dispositivo ainda preso pelo
        handle anterior — e foi o que travou a C920 varias vezes.
        """
        self.log.info("aguardando o driver liberar")
        time.sleep(1.0)
        return True

    # ------------------------------------------------------------ leitura
    def _ler_bruto(self):
        if self._cap is None:
            raise CameraSemImagem("camera nao esta aberta", nome=self.nome)
        ok, q = self._cap.read()
        return q if ok else None

    def _fechar(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
