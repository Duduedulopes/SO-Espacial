"""
EventEngine — fatos consumados, no passado, com carimbo de tempo.

A DISTINCAO QUE DECIDE O DESENHO

    evento   "a pessoa 3 entrou na zona frente-A as 21:04:12"   -> fato
    comando  "acenda a luz da zona frente-A"                    -> ordem

Aqui so existem FATOS. Nenhum evento manda alguem fazer nada, e nenhum
componente age porque emitiu um evento. Quem quiser agir se inscreve.

Essa separacao e o que permite, depois: gravar a sessao e reproduzi-la,
alimentar automacoes sem tocar no nucleo, e mandar tudo para um painel web
sem que o nucleo saiba que existe painel.

REGRA DE SEGURANCA

Assinante que levanta excecao NAO derruba quem emitiu. Um painel quebrado nao
pode parar o rastreamento — mesma regra que o Publicador aprendeu em 08/08,
quando um arquivo aberto no editor quebrou a sessao inteira.
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.nucleo.log import Log


# Tipos previstos. Constantes em vez de strings soltas: erro de digitacao
# vira NameError na hora, nao evento que nunca dispara.
class Tipo:
    CAMERA_CONNECTED = "CAMERA_CONNECTED"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    CAMERA_DEGRADED = "CAMERA_DEGRADED"
    CAMERA_RECONNECTED = "CAMERA_RECONNECTED"
    CAMERA_ERROR = "CAMERA_ERROR"

    TRACK_STARTED = "TRACK_STARTED"
    TRACK_LOST = "TRACK_LOST"
    TRACK_REIDENTIFIED = "TRACK_REIDENTIFIED"

    PERSON_ENTERED_ZONE = "PERSON_ENTERED_ZONE"
    PERSON_LEFT_ZONE = "PERSON_LEFT_ZONE"

    # Arquitetura v3: o que a pessoa esta FAZENDO, no vocabulario fechado.
    # Emitidos na MUDANCA, nunca por quadro — mesma regra das zonas.
    LOCOMOCAO_MUDOU = "LOCOMOCAO_MUDOU"
    POSTURA_MUDOU = "POSTURA_MUDOU"

    # Etapa B. Carrega a ALTURA DA MAO em metros quando ela foi medida — e
    # esse campo que, na etapa D, vira "a mao entrou na prateleira do produto
    # X". Sem ele o evento diria que o braco subiu e nao para onde.
    BRACO_MUDOU = "BRACO_MUDOU"

    OBJECT_DETECTED = "OBJECT_DETECTED"
    OBJECT_MOVED = "OBJECT_MOVED"

    SYSTEM_STARTED = "SYSTEM_STARTED"
    SYSTEM_DEGRADED = "SYSTEM_DEGRADED"
    FRAME_DROPPED_BURST = "FRAME_DROPPED_BURST"


@dataclass
class Evento:
    tipo: str
    t_wall: str
    t_mono: float
    dados: dict = field(default_factory=dict)

    def __repr__(self):
        d = " ".join(f"{k}={v}" for k, v in list(self.dados.items())[:3])
        return f"{self.t_wall[11:19]} {self.tipo} {d}"


class EventEngine:
    def __init__(self, memoria=500, arquivo=None):
        """`arquivo`: caminho .jsonl onde gravar TUDO o que aconteceu.

        POR QUE O ARQUIVO IMPORTA MAIS DO QUE PARECE

        Em 10/08 o sistema anunciou 17 mudancas de locomocao em 46 s e eu
        passei tres rodadas ajustando limiares — sem ter como saber se 17
        estava certo. Nao havia com o que comparar. Ajustar contra um numero
        que nao se consegue avaliar e adivinhar com trabalho, exatamente o que
        o cronometro tinha evitado em 08/08.

            Sem registro do que aconteceu, nao ha como julgar o que o sistema
            disse que aconteceu.

        A memoria em `deque` responde "o que esta acontecendo agora" e tem
        teto. O arquivo responde "o que aconteceu" e so cresce. Sao perguntas
        diferentes — a mesma separacao entre `estado_atual.json` e historico
        que o Publicador ja declarava e nunca ganhou a segunda metade.
        """
        self._assinantes = {}          # tipo | "*" -> [callback]
        self.historico = deque(maxlen=memoria)
        self.contagem = {}
        self.log = Log("eventos")

        self.arquivo = Path(arquivo) if arquivo else None
        self._saida = None
        self.falhas_de_escrita = 0
        if self.arquivo:
            try:
                self.arquivo.parent.mkdir(parents=True, exist_ok=True)
                self._saida = self.arquivo.open("a", encoding="utf-8")
            except OSError as e:
                self.log.aviso("nao consegui abrir o historico",
                               arquivo=str(self.arquivo), erro=str(e))

    def assinar(self, tipo, callback):
        self._assinantes.setdefault(tipo, []).append(callback)
        return callback

    def emitir(self, tipo, dados=None):
        e = Evento(tipo=tipo, t_wall=datetime.now().astimezone().isoformat(),
                   t_mono=time.monotonic(), dados=dados or {})
        self.historico.append(e)
        self.contagem[tipo] = self.contagem.get(tipo, 0) + 1
        self._gravar(e)

        for cb in self._assinantes.get(tipo, []) + self._assinantes.get("*", []):
            try:
                cb(e)
            except Exception as exc:
                # Assinante quebrado nao derruba quem emitiu.
                self.log.aviso("assinante falhou", tipo=tipo, erro=str(exc))
        return e

    def _gravar(self, e):
        """Uma linha por evento. Gravar NUNCA pode derrubar quem emitiu —
        mesma regra do publicador, aprendida em 08/08 com um arquivo aberto
        no editor."""
        if self._saida is None:
            return
        try:
            self._saida.write(json.dumps(
                {"t": e.t_wall, "tipo": e.tipo, **e.dados},
                ensure_ascii=False) + "\n")
            self._saida.flush()
        except Exception as exc:
            self.falhas_de_escrita += 1
            if self.falhas_de_escrita == 1:
                self.log.aviso("historico falhou", erro=str(exc))
            self._saida = None

    def fechar(self):
        if self._saida is not None:
            try:
                self._saida.close()
            finally:
                self._saida = None

    def ultimos(self, n=10, tipo=None):
        h = [e for e in self.historico if tipo is None or e.tipo == tipo]
        return list(h)[-n:]

    def resumo(self):
        return dict(sorted(self.contagem.items(), key=lambda kv: -kv[1]))
