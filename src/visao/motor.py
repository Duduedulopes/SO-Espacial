"""
VisionEngine — despacha um Instante para os trabalhadores, em paralelo.

O CICLO

    submeter em TODOS  ->  cada thread trabalha  ->  colher em TODOS

E o intervalo entre submeter e colher que produz o ganho. Se o motor
submetesse e colhesse um de cada vez, seria o laco sequencial de antes com
mais cerimonia.

    sequencial:   84 + 26 + 33 = 143 ms
    paralelo:     max(84, 26, 33) = 84 ms

O QUE O MOTOR NAO FAZ

Nao interpreta o que foi visto. Devolve Observacoes — hipoteses. Quem decide o
que e verdade e o SpatialEngine.

Nao desenha, nao sabe o que e metro, nao conhece a homografia.

TOLERANCIA A FALHA

Trabalhador que levanta excecao devolve lista vazia e conta a falha.
Trabalhador que trava e abandonado apos o timeout. Em nenhum dos casos o
sistema para — perde-se uma vista, nao a sessao.
"""

import time

from src.nucleo.log import Log
from src.visao.trabalhador import _Executor


class VisionEngine:
    def __init__(self, timeout_s=5.0):
        self.executores = {}          # papel -> _Executor
        self.timeout = timeout_s
        self.log = Log("visao")
        self._ultimo_ms = 0.0
        self.instantes = 0
        self.pulados = 0

    def registrar(self, trabalhador):
        if trabalhador.papel in self.executores:
            raise ValueError(f"papel '{trabalhador.papel}' ja tem trabalhador")
        trabalhador.iniciar()
        self.executores[trabalhador.papel] = _Executor(trabalhador)
        self.log.info("trabalhador registrado", papel=trabalhador.papel,
                      tipo=trabalhador.nome, a_cada_n=trabalhador.a_cada_n)
        return trabalhador

    def processar(self, instante):
        """Devolve list[Observacao] de todas as vistas do instante."""
        t0 = time.perf_counter()

        # --- fase 1: submeter em todos, sem esperar ---
        submetidos = []
        for papel, ex in self.executores.items():
            frame = instante.get(papel)
            if frame is None:
                continue
            if not ex.t.deve_processar(frame):
                self.pulados += 1
                continue
            ex.submeter(frame)
            submetidos.append(ex)

        # --- fase 2: colher ---
        observacoes = []
        for ex in submetidos:
            observacoes.extend(ex.colher(self.timeout))

        self._ultimo_ms = (time.perf_counter() - t0) * 1000
        self.instantes += 1
        return observacoes

    @property
    def ultimo_ms(self):
        return self._ultimo_ms

    def parar(self):
        for ex in self.executores.values():
            ex.parar()
        self.log.info("parado")

    def resumo(self):
        return {p: ex.t.metricas.resumo() for p, ex in self.executores.items()}

    def painel(self):
        linhas = []
        for papel, ex in self.executores.items():
            m = ex.t.metricas
            linhas.append(
                f"{papel:9} {ex.t.nome:12} {m.ms_medio:6.1f}ms "
                f"(pior {m.ms_pior:5.0f}) quadros{m.quadros:6d} "
                f"saidas{m.saidas:6d} falhas{m.falhas:4d}")
        return linhas

    def diagnostico_paralelismo(self):
        """Compara o tempo real com a soma dos trabalhadores.

        Se `paralelo` ficar perto de `soma`, o paralelismo NAO esta
        acontecendo — provavel trabalhador em Python puro segurando o GIL.
        Vale medir, nao supor.
        """
        soma = sum(ex.t.metricas.ms_medio for ex in self.executores.values())
        pior = max((ex.t.metricas.ms_medio for ex in self.executores.values()),
                   default=0.0)
        return {
            "soma_sequencial_ms": round(soma, 1),
            "pior_trabalhador_ms": round(pior, 1),
            "real_ms": round(self._ultimo_ms, 1),
            "ganho": round(soma / self._ultimo_ms, 2) if self._ultimo_ms else 0,
        }
