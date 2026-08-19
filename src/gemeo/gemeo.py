"""
DigitalTwin — o dono unico da verdade sobre o ambiente.

O PRINCIPIO

    Estado do mundo vive num lugar so.

No sistema antigo, a posicao de uma pessoa existia em quatro lugares ao mesmo
tempo: no rastro do Kalman, na variavel do laco, no dicionario do mapa de calor
e no arquivo publicado. Quando divergiam — e divergiam — nao havia como saber
qual estava certa.

Aqui ha um objeto. Quem quiser saber onde alguem esta pergunta a ele.

O QUE ELE E, E O QUE NAO E

E: o modelo estruturado do ambiente num instante — pessoas, zonas, cameras,
   ocupacao, com carimbo de tempo.

NAO e: uma imagem. O desenho e uma das saidas possiveis, nao o gemeo. Foi por
   confundir os dois que o sistema ficou "visualmente bonito, sem arquitetura
   confiavel por tras".

EVENTOS SAO CONSEQUENCIA, NAO ENFEITE

Entrar numa zona, perder um rastro, reencontrar uma identidade — o gemeo
detecta a MUDANCA comparando com o estado anterior e emite. Ninguem precisa
lembrar de chamar `emitir()` no lugar certo do laco.
"""

import sys
import time
from collections import deque
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.eventos.motor import Tipo                          # noqa: E402
from src.nucleo.log import Log                              # noqa: E402


class DigitalTwin:
    def __init__(self, planta, eventos=None, meia_vida_calor=90.0,
                 memoria_trilha=90):
        self.planta = planta
        self.eventos = eventos
        self.log = Log("gemeo")

        self.pessoas = {}          # id -> EstadoDePessoa
        self.cameras = {}          # papel -> resumo
        self.zonas = planta.zonas
        self.calor = planta.novo_mapa_de_calor(meia_vida_s=meia_vida_calor)

        # POR ONDE PASSOU, e nao so onde esta.
        #
        # O mapa de calor responde "onde as pessoas FICAM" em agregado. Ele
        # nao responde "por onde ESTA pessoa veio", que e a pergunta de quem
        # olha a cena e quer entender um percurso. Sao coisas diferentes e a
        # segunda estava faltando: a janela 3D pedia `historico` e recebia
        # lista vazia.
        self.trilhas = {}                  # id -> deque[(x, y)]
        self._memoria_trilha = memoria_trilha

        self.t_mono = 0.0
        self.quadros = 0
        self._ids_anteriores = set()
        self._zonas_anteriores = {}    # id_pessoa -> set(id_zona)

    # ------------------------------------------------------------ ciclo
    def atualizar(self, estados, metricas_cameras=None, dt=1 / 30,
                  acoes=None):
        self.t_mono = time.monotonic()
        self.quadros += 1

        agora = {e.id: e for e in estados}
        self._detectar_entradas_e_saidas(agora)
        self.pessoas = agora

        if metricas_cameras is not None:
            self.cameras = metricas_cameras

        # O calor so acumula o que foi MEDIDO. Enquanto o Kalman preve, nada
        # entra — senao registrariamos permanencia num lugar onde ninguem foi
        # visto, so estimado.
        self.calor.passo()
        posicoes = {}
        for p in estados:
            if not p.prevendo:
                self.calor.acumular(p.x, p.y, dt)
            posicoes[p.id] = (p.x, p.y)

        for p in estados:
            trilha = self.trilhas.get(p.id)
            if trilha is None:
                trilha = self.trilhas[p.id] = deque(maxlen=self._memoria_trilha)
            trilha.append((p.x, p.y))

        for z in self.zonas:
            z.atualizar(posicoes, dt)

        self._detectar_zonas(agora)
        self._detectar_acoes(agora, acoes or {})

    def _detectar_entradas_e_saidas(self, agora):
        ids = set(agora)

        for i in ids - self._ids_anteriores:
            p = agora[i]
            self._emitir(Tipo.TRACK_STARTED,
                         {"pessoa": i, "x": round(p.x, 2), "y": round(p.y, 2)})

        for i in self._ids_anteriores - ids:
            self._emitir(Tipo.TRACK_LOST, {"pessoa": i})
            self._zonas_anteriores.pop(i, None)
            self.trilhas.pop(i, None)

        self._ids_anteriores = ids

    def _detectar_zonas(self, agora):
        """Entrada e saida por comparacao com o quadro anterior.

        A zona ja sabe quem esta dentro; o gemeo e que percebe a MUDANCA.

        E ATE 19/08 ELE NAO PERGUNTAVA A ELA. Este metodo refazia o teste
        geometrico com `z.contem(p.x, p.y)` — a linha crua, sem a histerese
        que a zona aplica. Duas respostas para a mesma pergunta, e os eventos
        liam a que ninguem tinha filtrado. Resultado numa corrida de 45 s com
        uma pessoa sentada e parada:

            PERSON_ENTERED_ZONE      15
            PERSON_LEFT_ZONE         15

        O docstring acima ja dizia "a zona ja sabe quem esta dentro". Estava
        certo e nao estava sendo obedecido.

            Duas respostas para a mesma pergunta nao sao redundancia: uma
            delas vai ser lida por engano.
        """
        for pid, p in agora.items():
            dentro = {getattr(z, "id", z.nome) for z in self.zonas
                      if pid in z.dentro}
            antes = self._zonas_anteriores.get(pid, set())

            for z in dentro - antes:
                self._emitir(Tipo.PERSON_ENTERED_ZONE,
                             {"pessoa": pid, "zona": z})
            for z in antes - dentro:
                self._emitir(Tipo.PERSON_LEFT_ZONE,
                             {"pessoa": pid, "zona": z})

            self._zonas_anteriores[pid] = dentro

    def _detectar_acoes(self, agora, acoes):
        """Emite so a MUDANCA, nunca o estado por quadro.

        A 10 fps, uma pessoa andando 30 segundos geraria 300 eventos dizendo
        a mesma coisa. O `Estavel` do classificador ja segurou o ruido; aqui
        so passa quem de fato mudou de estado.
        """
        for pid, (acao, mudancas) in acoes.items():
            if pid not in agora:
                continue
            if mudancas.get("locomocao"):
                self._emitir(Tipo.LOCOMOCAO_MUDOU, {
                    "pessoa": pid, "estado": acao.locomocao,
                    "confianca": round(acao.confianca, 2),
                    "motivo": acao.motivo})
            if mudancas.get("postura"):
                self._emitir(Tipo.POSTURA_MUDOU, {
                    "pessoa": pid, "estado": acao.postura,
                    "proporcao": round(acao.razao_altura, 2)})

            for lado, chave, estado, altura in (
                    ("esquerdo", "braco_esquerdo", acao.braco_esquerdo,
                     acao.altura_mao_esq),
                    ("direito", "braco_direito", acao.braco_direito,
                     acao.altura_mao_dir)):
                if not mudancas.get(chave):
                    continue
                dados = {"pessoa": pid, "lado": lado, "estado": estado}
                # A altura so entra quando existe. Publicar `null` convida
                # quem le a tratar ausencia como zero — e zero metro do chao
                # e uma afirmacao, nao uma falta de resposta.
                if altura is not None:
                    dados["altura_m"] = round(altura, 3)
                self._emitir(Tipo.BRACO_MUDOU, dados)

    def _emitir(self, tipo, dados):
        if self.eventos is not None:
            self.eventos.emitir(tipo, dados)

    # ------------------------------------------------------------ leitura
    def instantaneo(self):
        """Serializavel. E o que vai para JSON, WebSocket ou motor de jogo.

        Nao inclui imagem nenhuma de proposito: o gemeo e o ESTADO, e quem
        quiser desenhar consome isto.
        """
        return {
            "loja": {"id": self.planta.id, "nome": self.planta.nome},
            "t": round(self.t_mono, 3),
            "quadros": self.quadros,
            "pessoas": [p.para_dicionario() for p in self.pessoas.values()],
            "zonas": [{
                "id": getattr(z, "id", z.nome),
                "nome": z.nome,
                "ocupacao": z.ocupacao,
                "visitas": z.visitas,
                "tempo_total_s": round(z.tempo_total, 1),
                "tempo_medio_s": round(z.tempo_medio, 1),
            } for z in self.zonas],
            "cameras": self.cameras,
        }

    def resumo(self):
        return {
            "pessoas": len(self.pessoas),
            "quadros": self.quadros,
            "zonas_ocupadas": sum(1 for z in self.zonas if z.ocupacao),
        }
