"""
Camera remota por URL — celular, camera IP, RTSP.

POR QUE UMA CLASSE SEPARADA, E NAO "mais uma webcam"

O celular vinha sendo tratado como webcam local, porque o Iriun instala um
driver que o disfarca de webcam. Isso ESCONDE do sistema que a fonte e remota.

                        USB                  remota
    identidade          nome do dispositivo  URL
    queda               rara                 comum (rede)
    latencia            ~1 quadro            variavel, precisa ser medida
    recuperacao         mexer no driver      reabrir a conexao
    ocupa indice        sim                  nao

Quando o sistema nao sabe que a fonte e remota, ele nao tem como tratar queda
de rede, nem reconectar, nem medir latencia. E o driver virtual ainda ocupa um
indice no DirectShow — o que ja trocou a identidade de outras cameras.

ESCOLHA DO PROTOCOLO

    MJPEG/HTTP   200-500 ms   trivial: VideoCapture le a URL direto   <- inicio
    RTSP         0,5-2 s      tambem nativo; mais latencia em LAN
    WebRTC       100-500 ms   melhor latencia, exige servidor (MediaMTX)

Comecamos por MJPEG porque o VideoCapture le nativamente e a fonte passa a ser
EXPLICITAMENTE remota — que e o ganho que importa agora. Trocar de protocolo
mexe so nesta classe.

TIMEOUT E ESSENCIAL

Sem timeout, `VideoCapture.read()` numa conexao morta bloqueia a thread da
fonte para sempre. A thread e por fonte, entao so aquela camera congela — mas
ela congela para sempre, e o estado nunca chega a FALHA.
"""

import ipaddress
import os
import socket
import time
from urllib.parse import urlparse

import cv2

from src.cameras.fonte import FonteDeVideo
from src.nucleo.erros import ConexaoPerdida, TempoEsgotado


class RemoteCameraSource(FonteDeVideo):
    tipo = "remota"

    def __init__(self, url, papel, timeout_ms=4000, **kw):
        super().__init__(id=url, papel=papel, **kw)
        self.url = url
        self.timeout_ms = timeout_ms
        self._cap = None
        self._t_ultimo_ok = 0.0

        p = urlparse(url)
        self.protocolo = (p.scheme or "?").lower()
        self.host = p.hostname or "?"

    # ------------------------------------------------------------ abertura
    def _abrir(self):
        # O FFMPEG do OpenCV le estas variaveis na ABERTURA. Sem elas, uma
        # conexao morta bloqueia a leitura indefinidamente.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rw_timeout;{self.timeout_ms * 1000}"
            f"|stimeout;{self.timeout_ms * 1000}"
            "|max_delay;500000"
            "|fflags;nobuffer"          # nao acumular: preferimos o presente
        )

        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise ConexaoPerdida(f"nao consegui abrir {self.url}",
                                 protocolo=self.protocolo, host=self.host,
                                 **self._diagnosticar_rede())

        # GUARDAR O HANDLE. Sim, esta linha ja faltou.
        #
        # MEDIDO EM 11/08: a lateral ficou 58 s em `conectando`, com 0 quadros
        # e 0 falhas, logo depois de `achar_ip.py` ter lido video daquela mesma
        # URL com brilho 47,5. A conexao ABRIA — e o objeto era descartado ao
        # fim deste metodo, porque `cap` era local. `_ler_bruto` entao batia em
        # `self._cap is None` e levantava "nao conectada" para sempre.
        #
        # O estado nunca chegava a FALHA porque nenhuma LEITURA acontecia: a
        # camera ficava eternamente em CONECTANDO, que e o estado mais mudo de
        # todos. Uma falha barulhenta teria sido encontrada em 10/08.
        #
        #     O `usb.py` faz `self._cap = cap` na mesma posicao. Duas
        #     implementacoes do mesmo contrato, e so uma o cumpria.
        self._cap = cap
        self._t_ultimo_ok = time.monotonic()

        # Pedir nao e receber, aqui tambem: quem responde e o servidor do
        # tablet, e ele decide a resolucao. Registrar o que de fato chegou e o
        # que impede a homografia de ser reescalada para um tamanho que a
        # fonte nunca entregou — defeito silencioso corrigido em 10/08 para as
        # cameras USB e que valeria igual para esta.
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if largura and altura:
            self.largura, self.altura = largura, altura

        self.log.info("aberta", url=self.url, protocolo=self.protocolo,
                      resolucao=f"{self.largura}x{self.altura}")

    def _diagnosticar_rede(self):
        """Quando falha, dizer O QUE HA DE ERRADO, nao so QUE deu errado.

        MEDIDO EM 10/08: a lateral falhou por meia hora com a mensagem
        `nao consegui abrir http://192.168.1.2:8080/video`. Verdadeira e
        inutil. O tablet estava em 192.168.1.2, na rede de casa; o PC estava
        em 172.20.10.14, no Acesso Pessoal do iPhone. Duas redes sem rota
        entre elas — e a mensagem nao dava nenhuma pista disso.

            Erro que so repete o pedido nao ajuda. Erro util compara o que se
            pediu com o que se tem.

        NUNCA LEVANTA EXCECAO. Na primeira versao deste metodo eu deixei um
        `getaddrinfo` sem protecao e o diagnostico derrubou a abertura da
        camera — troquei um erro ruim por um erro pior. Diagnostico e canal
        lateral, mesma regra do publicador.
        """
        try:
            alvo = ipaddress.ip_address(socket.gethostbyname(self.host))
        except Exception:
            return {"dica": "nao consegui resolver o host"}

        # Qual IP local o sistema USARIA para falar com esse alvo. Um socket
        # UDP "conectado" nao envia nada: so consulta a tabela de rotas.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0.3)
                s.connect((str(alvo), 9))
                meu = ipaddress.ip_address(s.getsockname()[0])
        except Exception:
            return {"dica": "sem rota conhecida ate o host"}

        if meu.packed[:3] == alvo.packed[:3]:
            return {"meu_ip": str(meu),
                    "dica": "mesma faixa; o servidor pode estar parado"}
        return {"meu_ip": str(meu),
                "dica": (f"REDES DIFERENTES: voce esta em {meu} e o host em "
                         f"{alvo}. Sem rota entre elas.")}

    # ------------------------------------------------------------ leitura
    def _ler_bruto(self):
        if self._cap is None:
            raise ConexaoPerdida("nao conectada", url=self.url)

        t0 = time.monotonic()
        ok, q = self._cap.read()
        if ok and q is not None and q.size:
            # A latencia de uma fonte remota varia com a rede e precisa ser
            # medida, nao suposta. Numa fonte USB seria constante.
            self.metricas.registrar_latencia((time.monotonic() - t0) * 1000)
            self._t_ultimo_ok = time.monotonic()
            return q

        if time.monotonic() - self._t_ultimo_ok > self.timeout_ms / 1000.0:
            raise ConexaoPerdida("stream parou de responder", url=self.url)

        # RESPIRA ANTES DE TENTAR DE NOVO.
        #
        # MEDIDO EM 10/08: o painel mostrou `falhas 13272` numa camera que
        # nunca entregou um quadro — mais de 300 leituras falhas por segundo.
        # Sem esta pausa, `read()` numa conexao morta volta na hora e a thread
        # roda em laco fechado, queimando um nucleo inteiro.
        #
        # A conta e cruel: o detector, que disputa a mesma CPU, subiu de 68
        # para 76 ms durante a execucao. Uma camera QUE NAO FUNCIONA estava
        # deixando o resto do sistema mais lento.
        #
        #     Falha rapida sem pausa nao e resiliencia, e desperdicio ativo.
        #
        # A reconexao ja tem recuo exponencial. Faltava o recuo da LEITURA.
        time.sleep(0.05)
        return None

    def _recuperar(self):
        """Reconexao de rede: so esperar. Nao ha driver para consertar."""
        return True

    def _fechar(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
