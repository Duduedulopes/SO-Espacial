"""
FonteDeVideo — contrato comum a toda origem de imagem.

O QUE MUDA EM RELACAO AO QUE HAVIA

A classe antiga era "uma webcam local aberta por indice". Tudo o mais foi
sendo empurrado para dentro dela. Consequencias medidas em 07 e 08/08:

  - o celular era tratado como webcam USB, escondendo que a fonte e remota
  - nao havia reconexao: camera caida devolvia o ULTIMO quadro para sempre,
    e o sistema processava imagem congelada sem perceber
  - nao havia contagem de descarte
  - o backend fazia parte da identidade e ninguem tinha modelado isso

Aqui a fonte tem ESTADO, e o estado e observavel de fora.

A MAQUINA DE ESTADOS, E POR QUE DEGRADADA EXISTE

    DESCONECTADA --iniciar()--> CONECTANDO --entregou quadro--> ONLINE
                                     ^                            |
                                     |                  silencio > 2 s
                        recuo exponencial                         v
                                     |                        DEGRADADA
                                     |                            |
                                  FALHA <---- silencio > 10 s ----+

DEGRADADA e o estado que faltava. Sem ele, uma camera que para de entregar
continua devolvendo o ultimo quadro — o sistema nao trava, nao acusa erro, e
processa uma imagem parada como se fosse o presente.

    Pior que falhar: mentir.

Em DEGRADADA, `ler()` devolve None. Quem consome sabe que nao ha dado.
"""

import threading
import time
from abc import ABC, abstractmethod
from enum import Enum

from src.fluxo.buffer import FrameBuffer
from src.fluxo.quadro import Frame, agora_iso
from src.nucleo.log import Log
from src.nucleo.metricas import MetricasDeFonte


class Estado(str, Enum):
    DESCONECTADA = "desconectada"
    CONECTANDO = "conectando"
    ONLINE = "online"
    DEGRADADA = "degradada"
    FALHA = "falha"
    PARADA = "parada"


class FonteDeVideo(ABC):
    """Base de toda fonte. A subclasse implementa apenas _abrir/_ler/_fechar."""

    tipo = "abstrata"

    def __init__(self, id, papel, largura=1280, altura=720, fps_alvo=30,
                 buffer_maxlen=2, silencio_degradada=2.0, silencio_falha=10.0,
                 recuo_max=30.0, ao_mudar_estado=None):
        self.id = id
        self.papel = papel
        self.largura = largura
        self.altura = altura
        self.fps_alvo = fps_alvo

        self.metricas = MetricasDeFonte()
        self.buffer = FrameBuffer(buffer_maxlen, self.metricas)
        self.log = Log(f"fonte.{papel}")

        self._estado = Estado.DESCONECTADA
        self.ultimo_erro = None
        self._seq = 0
        self._ja_conectou = False

        self._silencio_degradada = silencio_degradada
        self._silencio_falha = silencio_falha
        self._recuo_max = recuo_max
        self._recuo = 1.0
        self._proxima_tentativa = 0.0

        self._ao_mudar = ao_mudar_estado          # callback para o EventEngine
        self._thread = None
        self._rodando = False

    # ------------------------------------------------------------ subclasse
    @abstractmethod
    def _abrir(self):
        """Abre o dispositivo. Levanta ErroDeCamera/ErroDeStream se falhar."""

    @abstractmethod
    def _ler_bruto(self):
        """Devolve ndarray BGR ou None. Nao levanta em falha comum de leitura."""

    @abstractmethod
    def _fechar(self):
        """Libera o recurso. Deve ser seguro chamar mesmo sem ter aberto."""

    def _recuperar(self):
        """Tentativa de conserto antes de reabrir. Padrao: nada."""
        return False

    # ------------------------------------------------------------ estado
    @property
    def estado(self):
        return self._estado

    def _mudar(self, novo, **dados):
        if novo == self._estado:
            return
        antigo, self._estado = self._estado, novo
        self.log.info(f"{antigo.value} -> {novo.value}", **dados)
        if self._ao_mudar:
            try:
                self._ao_mudar(self, antigo, novo)
            except Exception as e:
                # Callback de evento nao pode derrubar a fonte.
                self.log.aviso("callback de estado falhou", erro=str(e))

    @property
    def disponivel(self):
        return self._estado == Estado.ONLINE

    # ------------------------------------------------------------ ciclo
    def iniciar(self):
        if self._rodando:
            return
        self._rodando = True
        self._thread = threading.Thread(target=self._laco, daemon=True,
                                        name=f"fonte-{self.papel}")
        self._thread.start()

    def parar(self):
        self._rodando = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._fechar_seguro()
        self._mudar(Estado.PARADA)

    def _fechar_seguro(self):
        try:
            self._fechar()
        except Exception as e:
            self.log.debug("erro ao fechar", erro=str(e))

    # ------------------------------------------------------------ laco
    def _laco(self):
        """Uma thread por fonte. Captura e reconexao vivem aqui.

        A fonte NUNCA levanta excecao para fora da thread: uma camera com
        problema nao pode derrubar o processo. Ela muda de estado e registra.
        """
        while self._rodando:
            agora = time.monotonic()

            if self._estado in (Estado.DESCONECTADA, Estado.FALHA):
                if agora < self._proxima_tentativa:
                    time.sleep(0.1)
                    continue
                self._tentar_conectar()
                continue

            quadro = None
            try:
                quadro = self._ler_bruto()
            except Exception as e:
                self.metricas.falhas_leitura += 1
                self.ultimo_erro = str(e)
                self.log.debug("falha de leitura", erro=str(e))

            if quadro is not None and getattr(quadro, "size", 0) > 0:
                self._publicar(quadro, agora)
            else:
                self.metricas.falhas_leitura += 1
                self._avaliar_silencio(agora)
                time.sleep(0.005)

        self._fechar_seguro()

    def _publicar(self, imagem, agora):
        self._seq += 1
        f = Frame(camera_id=self.id, papel=self.papel, seq=self._seq,
                  t_mono=agora, t_wall=agora_iso(), imagem=imagem)
        self.buffer.colocar(f)
        self.metricas.registrar_quadro(agora, brilho=f.brilho)
        if self._estado in (Estado.CONECTANDO, Estado.DEGRADADA):
            self._recuo = 1.0
            self._mudar(Estado.ONLINE, fps=round(self.metricas.fps, 1))

    def _avaliar_silencio(self, agora):
        s = self.metricas.silencio_s(agora)
        if s > self._silencio_falha:
            self.log.aviso("sem imagem ha muito tempo", segundos=round(s, 1))
            self._cair()
        elif s > self._silencio_degradada and self._estado == Estado.ONLINE:
            self._mudar(Estado.DEGRADADA, silencio_s=round(s, 1))

    def _cair(self):
        self._fechar_seguro()
        self._mudar(Estado.FALHA, erro=self.ultimo_erro)
        self._agendar_retentativa()

    def _agendar_retentativa(self):
        """Recuo exponencial: 1, 2, 4, 8... ate o teto.

        Sem recuo, uma camera ausente e martelada dezenas de vezes por segundo
        — gasta CPU e polui o log ate ele ficar inutil.
        """
        self._proxima_tentativa = time.monotonic() + self._recuo
        self.log.info("nova tentativa agendada", em_s=round(self._recuo, 1))
        self._recuo = min(self._recuo * 2, self._recuo_max)

    def _tentar_conectar(self):
        # `reconexoes` conta RE-conexoes, nao a primeira. Um contador que soma
        # a conexao inicial diz "reconexoes: 1" numa camera que nunca caiu — e
        # metrica que mente e pior que metrica ausente, porque parece confiavel.
        primeira = not self._ja_conectou

        self._mudar(Estado.CONECTANDO)
        self._fechar_seguro()
        try:
            if not primeira:
                self._recuperar()
            self._abrir()
            self.metricas.ultimo_quadro_em = time.monotonic()
            self._ja_conectou = True
            if not primeira:
                self.metricas.reconexoes += 1
                self.log.info("reconectada", vez=self.metricas.reconexoes)
        except Exception as e:
            self.ultimo_erro = str(e)
            self.log.erro("falha ao conectar", erro=str(e))
            self._mudar(Estado.FALHA, erro=str(e))
            self._agendar_retentativa()

    # ------------------------------------------------------------ consumo
    def ler(self):
        """O quadro mais recente, ou None.

        Devolve None quando NAO ONLINE. Nunca entrega imagem velha fingindo
        ser atual — foi esse o defeito que motivou o estado DEGRADADA.
        """
        if self._estado != Estado.ONLINE:
            return None
        return self.buffer.pegar()

    def resumo(self):
        return {
            "id": self.id, "papel": self.papel, "tipo": self.tipo,
            "estado": self._estado.value,
            "resolucao": f"{self.largura}x{self.altura}",
            "erro": self.ultimo_erro,
            **self.metricas.resumo(),
        }

    def __repr__(self):
        return f"{type(self).__name__}({self.papel}:{self.id} {self._estado.value})"
