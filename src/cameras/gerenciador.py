"""
GerenciadorDeCameras — ciclo de vida e supervisao.

O REQUISITO QUE ELE ATENDE

    "Se a camera do celular cair, as cameras USB e notebook devem continuar
     funcionando."

Isso so e possivel se ninguem tratar o conjunto de cameras como um bloco. Aqui
cada fonte e independente: tem sua thread, seu estado, seu recuo. O
gerenciador observa e conta, nao conduz.

NAO E UM ORQUESTRADOR

Ele nao processa quadro, nao desenha, nao decide o que fazer com a imagem. Se
um dia precisar saber o que e uma pessoa, algo esta no lugar errado.

A supervisao e passiva: as fontes ja se reconectam sozinhas na propria thread.
O gerenciador so observa transicoes para emitir evento e registrar.
"""

import time

from src.cameras.fonte import Estado
from src.nucleo.log import Log


class GerenciadorDeCameras:
    def __init__(self, ao_evento=None):
        self.fontes = {}                  # papel -> FonteDeVideo
        self.log = Log("cameras")
        self._ao_evento = ao_evento
        self._t0 = time.monotonic()

    # ------------------------------------------------------------ registro
    def registrar(self, fonte):
        if fonte.papel in self.fontes:
            raise ValueError(f"papel '{fonte.papel}' ja registrado")
        fonte._ao_mudar = self._mudou_estado
        self.fontes[fonte.papel] = fonte
        self.log.info("registrada", papel=fonte.papel, id=fonte.id,
                      tipo=fonte.tipo)
        return fonte

    def _mudou_estado(self, fonte, antigo, novo):
        """Traduz transicao de estado em evento de dominio.

        DESCONECTAR EXIGE TER CONECTADO ANTES.

        Em 10/08 o painel mostrou quatro CAMERA_DISCONNECTED do tablet ANTES
        do primeiro CAMERA_CONNECTED. Ler aquilo e impossivel: nao se cai de
        um lugar onde nunca se esteve. O que havia eram quatro tentativas de
        abertura falhadas, com a espera crescendo entre elas.

        As duas situacoes pedem acoes opostas e nao podem dividir o mesmo nome:

            nunca subiu   -> conferir cabo, nome, driver, permissao
            caiu depois   -> conferir rede, energia, contencao pelo outro app

        Mesma classe de defeito de 08/08, quando `reconexoes` contava 1 numa
        camera que nunca tinha caido. Metrica que mente custa mais caro que
        metrica que falta: a que falta manda medir; a que mente manda procurar
        no lugar errado.
        """
        if not self._ao_evento:
            return

        if novo is Estado.ONLINE:
            tipo = ("CAMERA_RECONNECTED" if fonte.metricas.reconexoes
                    else "CAMERA_CONNECTED")
        elif novo is Estado.DEGRADADA:
            tipo = "CAMERA_DEGRADED"
        elif novo is Estado.FALHA:
            # `_ja_conectou` e a mesma marca que impede a contagem de
            # reconexoes de mentir. Aqui ela separa queda de estreia falhada.
            tipo = ("CAMERA_DISCONNECTED" if getattr(fonte, "_ja_conectou",
                                                     False)
                    else "CAMERA_ERROR")
        else:
            return

        # A ORDEM DOS CAMPOS E CONTEUDO, NAO ESTILO.
        #
        # O painel mostra os tres primeiros campos de cada evento. Com a ordem
        # antiga, tres CAMERA_ERROR do tablet apareceram assim:
        #
        #     CAMERA_ERROR camera=Galaxy Tab A7 Lite papel=lateral de=conectando
        #
        # Ou seja: falhou, e o motivo — o unico campo que dizia o que fazer a
        # respeito — ficou de fora. Em evento de falha, o erro vem primeiro.
        dados = {"camera": fonte.id, "papel": fonte.papel,
                 "de": antigo.value, "para": novo.value}
        if fonte.ultimo_erro:
            dados = {"erro": fonte.ultimo_erro, **dados}
        self._ao_evento(tipo, dados)

    # ------------------------------------------------------------ ciclo
    def iniciar(self):
        for f in self.fontes.values():
            f.iniciar()
        self.log.info("iniciadas", quantidade=len(self.fontes))

    def parar(self):
        for f in self.fontes.values():
            f.parar()
        self.log.info("paradas")

    def esperar_online(self, timeout=15.0, minimo=1):
        """Bloqueia ate ter `minimo` fontes online, ou estourar o tempo.

        Devolve a lista do que ficou online. NAO levanta se faltar: o sistema
        deve rodar com o que houver, e quem chamou decide se e suficiente.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            prontas = self.online()
            if len(prontas) >= minimo:
                return prontas
            time.sleep(0.2)
        return self.online()

    # ------------------------------------------------------------ consulta
    def online(self):
        return [f for f in self.fontes.values() if f.disponivel]

    def por_papel(self, papel):
        return self.fontes.get(papel)

    def buffers(self, apenas_online=True):
        """dict[papel -> FrameBuffer] para o sincronizador."""
        return {
            p: f.buffer for p, f in self.fontes.items()
            if not apenas_online or f.disponivel
        }

    def tem(self, papel):
        f = self.fontes.get(papel)
        return f is not None and f.disponivel

    # ------------------------------------------------------------ metricas
    def resumo(self):
        return {p: f.resumo() for p, f in self.fontes.items()}

    def painel(self):
        """Uma linha por camera, para terminal ou dashboard."""
        linhas = []
        for papel, f in self.fontes.items():
            m = f.metricas
            marca = {"online": "ON ", "degradada": "DEG", "falha": "OFF",
                     "conectando": "...", "desconectada": "---",
                     "parada": "STOP"}.get(f.estado.value, "?")
            linhas.append(
                f"{marca} {papel:9} {f.id[:26]:26} "
                f"{f.largura}x{f.altura:<5} "
                f"{m.fps:5.1f}fps  rec{m.recebidos:6d} desc{m.descartados:5d} "
                f"falhas {m.falhas_leitura:6d} rec.x{m.reconexoes}"
            )
        return linhas

    def total(self):
        r = self.resumo()
        return {
            "cameras": len(self.fontes),
            "online": len(self.online()),
            "recebidos": sum(v["recebidos"] for v in r.values()),
            "descartados": sum(v["descartados"] for v in r.values()),
            "reconexoes": sum(v["reconexoes"] for v in r.values()),
            "tempo_s": round(time.monotonic() - self._t0, 1),
        }
