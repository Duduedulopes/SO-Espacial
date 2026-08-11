"""
Trabalhador — uma etapa de visao, com thread propria.

O PROBLEMA MEDIDO EM 08/08

    yolo 84 ms  +  frontal 26 ms  +  lateral 33 ms  =  143 ms   ->  7 fps

Os tres rodavam em SEQUENCIA no mesmo laco. As cameras capturavam em paralelo,
mas o consumo era serializado — e uma camera lenta atrasava todas.

Em paralelo, o custo passa a ser o do MAIS LENTO: ~84 ms, cerca de 11 fps.

POR QUE THREAD DEDICADA, E NAO ThreadPoolExecutor

YOLO e MediaPipe guardam ESTADO entre chamadas — o rastreador precisa do
quadro anterior, e o `detect_for_video` do MediaPipe exige carimbo de tempo
estritamente crescente. Com um pool, a mesma tarefa pode cair em threads
diferentes, e bibliotecas nativas nem sempre gostam disso.

Thread dedicada garante que cada modelo seja sempre tocado pela mesma thread.
Custa uma thread por trabalhador — tres, no caso.

POR QUE O GIL NAO ESTRAGA A FESTA

Tanto o YOLO (PyTorch) quanto o MediaPipe (TFLite/C++) liberam o GIL durante a
inferencia. O trabalho pesado acontece em codigo nativo, e ha paralelismo real.
Se um dia houver um trabalhador puramente Python, ele NAO vai paralelizar — e
isso precisa ser lembrado antes de escrever um.
"""

import threading
import time
from abc import ABC, abstractmethod

from src.nucleo.log import Log
from src.nucleo.metricas import MetricasDeTrabalhador


class Trabalhador(ABC):
    """Recebe um Frame, devolve Observacoes. Sincrono, sem thread.

    A thread e responsabilidade do VisionEngine. Assim o trabalhador continua
    testavel chamando `processar()` direto, sem concorrencia no meio.
    """

    nome = "trabalhador"

    def __init__(self, papel, a_cada_n=1):
        self.papel = papel
        self.a_cada_n = max(1, a_cada_n)
        self.metricas = MetricasDeTrabalhador()
        self.log = Log(f"visao.{self.nome}.{papel}")
        self._n = 0

    def deve_processar(self, frame):
        """Frame skipping por trabalhador.

        Deteccao pode rodar a cada 2 quadros enquanto a pose roda em todos, ou
        o contrario. E configuravel porque o gargalo muda com a maquina — no
        PC atual e o YOLO, com 84 ms contra 26 da pose.
        """
        self._n += 1
        return self._n % self.a_cada_n == 0

    @abstractmethod
    def _processar(self, frame):
        """list[Observacao]. Pode levantar; o motor trata."""

    def processar(self, frame):
        t0 = time.perf_counter()
        try:
            obs = self._processar(frame) or []
        except Exception as e:
            self.metricas.falhas += 1
            self.log.erro("falha na inferencia", exc=e, seq=frame.seq)
            return []
        ms = (time.perf_counter() - t0) * 1000
        self.metricas.registrar(ms, len(obs))
        return obs

    def iniciar(self):
        """Carga do modelo. Separada do construtor de proposito: quem monta o
        sistema pode registrar tudo antes de pagar segundos de carregamento."""

    def parar(self):
        pass


class _Executor:
    """Thread dedicada que serve UM trabalhador.

    Protocolo simples: `submeter(frame)` acorda a thread, `colher()` espera o
    resultado. O motor faz submeter em todos e depois colher em todos — e e
    esse intervalo entre os dois que produz o paralelismo.
    """

    def __init__(self, trabalhador):
        self.t = trabalhador
        self._entrada = None
        self._saida = []
        self._ha_trabalho = threading.Event()
        self._pronto = threading.Event()
        self._pronto.set()
        self._rodando = True
        self._thread = threading.Thread(
            target=self._laco, daemon=True, name=f"visao-{trabalhador.papel}")
        self._thread.start()

    def _laco(self):
        while self._rodando:
            if not self._ha_trabalho.wait(timeout=0.2):
                continue
            self._ha_trabalho.clear()
            frame = self._entrada
            self._saida = self.t.processar(frame) if frame is not None else []
            self._pronto.set()

    def submeter(self, frame):
        self._pronto.clear()
        self._entrada = frame
        self._saida = []
        self._ha_trabalho.set()

    def colher(self, timeout=5.0):
        """Espera o resultado.

        Se estourar, devolve vazio e SEGUE. Um trabalhador travado nao pode
        parar o sistema — perde-se aquela vista, nao a sessao.
        """
        if not self._pronto.wait(timeout):
            self.t.log.aviso("trabalhador nao respondeu a tempo",
                             timeout_s=timeout)
            return []
        return self._saida

    def parar(self):
        self._rodando = False
        self._ha_trabalho.set()
        self._thread.join(timeout=2.0)
        self.t.parar()
