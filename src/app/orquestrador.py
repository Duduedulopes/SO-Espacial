"""
Orquestrador — monta e conecta. Nao processa nada.

O CONTRASTE COM O QUE HAVIA

O `gemeo_multi.py` tinha 317 linhas e nove responsabilidades: abria camera,
rodava YOLO, rodava MediaPipe, convertia pixel em metro, filtrava Kalman,
contava zona, publicava JSON e desenhava. Acrescentar uma quarta camera
significava editar aquele laco.

Aqui o laco tem cinco linhas de trabalho:

    instante = sincronizador.montar(...)
    observacoes = visao.processar(instante)
    estados = espacial.atualizar(observacoes, dt)
    gemeo.atualizar(estados, ...)
    saidas...

Cada etapa e um objeto testado separadamente. Se algo estiver errado, o teste
diz qual.

FONTE FALSA: SO COM PEDIDO EXPLICITO

`--falsas` e a UNICA porta que instancia fonte sintetica. A montagem normal le
`config/cameras.json` e abre hardware. Nao ha caminho que caia em simulacao
por acidente.
"""

import json
import sys
import time
from collections import deque
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from estado.planta import Planta, Publicador                # noqa: E402
from percepcao.chao import carregar_homografia              # noqa: E402
from src.cameras.gerenciador import GerenciadorDeCameras    # noqa: E402
from src.cameras.remota import RemoteCameraSource           # noqa: E402
from src.cameras.usb import UsbCameraSource                 # noqa: E402
from src.espacial.motor import SpatialEngine                # noqa: E402
from src.eventos.motor import EventEngine, Tipo             # noqa: E402
from src.fluxo.sincronizador import Sincronizador           # noqa: E402
from src.gemeo.gemeo import DigitalTwin                     # noqa: E402
from src.nucleo.log import Log                              # noqa: E402
from src.visao.detector import DetectorDePessoas            # noqa: E402
from src.visao.motor import VisionEngine                    # noqa: E402
from src.visao.pose import EstimadorDePose                  # noqa: E402

CONFIG_CAMERAS = RAIZ / "config" / "cameras.json"


class Orquestrador:
    def __init__(self, planta="loja/bancada.json", captura=(1280, 720),
                 imgsz=320, conf=0.35, deteccao_a_cada=1, pose_a_cada=1,
                 lado_lateral="direita", tolerancia_ms=150,
                 meia_vida_calor=90.0, exposicao=None, com_pose=True,
                 usar_plausibilidade=True, salvar_quadros_s=0.0):
        self.log = Log("app")
        self.captura = captura
        self.com_pose = com_pose

        self.eventos = EventEngine(
            arquivo=RAIZ / "dados" / "eventos.jsonl")
        self.cameras = GerenciadorDeCameras(
            ao_evento=lambda t, d: self.eventos.emitir(t, d))

        self.planta = Planta.carregar(RAIZ / planta)
        H, meta = carregar_homografia()

        self.espacial = SpatialEngine(
            H,
            resolucao_calibracao=meta.get("resolucao", [640, 480]),
            resolucao_captura=captura,
            lado_lateral=lado_lateral,
            usar_plausibilidade=usar_plausibilidade)

        self.visao = VisionEngine()
        self.sincronizador = Sincronizador(tolerancia_ms=tolerancia_ms,
                                           papel_obrigatorio="alto")
        self.gemeo = DigitalTwin(self.planta, self.eventos,
                                 meia_vida_calor=meia_vida_calor)
        self.publicador = Publicador(RAIZ / "dados" / "estado_atual.json")

        self._imgsz = imgsz
        self._conf = conf
        self._det_a_cada = deteccao_a_cada
        self._pose_a_cada = pose_a_cada
        self._exposicao = exposicao

        self.quadros = 0
        self._t_anterior = 0.0
        self._captura_por_papel = {}

        # VER O QUE O PROGRAMA VE, nao o que o sistema operacional mostra.
        #
        # Em 10/08 a lateral entregou 0 poses em 462 quadros enquanto a
        # frontal — vista muito pior, camera de notebook — entregou 86%. Zero
        # absoluto com uma pessoa em cena nao e enquadramento ruim; e outra
        # coisa. E a imagem do painel de Configuracoes do Windows nao serve de
        # prova: ela vem de outro caminho.
        self._salvar_a_cada = salvar_quadros_s
        self._pasta_quadros = RAIZ / "dados" / "quadros"
        self._t_ultimo_salvo = 0.0

        # ONDE O TEMPO VAI.
        #
        # Em 10/08 o sistema girou a 3,4 fps enquanto a visao dizia custar
        # 48 ms — numeros que nao podem ser verdade ao mesmo tempo. Faltavam
        # 230 ms por ciclo e nao havia onde olhar; a unica saida era adivinhar.
        #
        # Adivinhar com trabalho ja custou duas rodadas de otimizacao no lugar
        # errado, em 08/08. Cada etapa agora tem cronometro, inclusive a espera.
        # TOTAIS, nao medias moveis. Uma media por etapa parece mais fina e
        # mente na soma: a espera acontece em TODA volta, as etapas so nas
        # produtivas. Somar medias de denominadores diferentes da um "ciclo"
        # que nunca existiu. Com totais, a soma fecha com o relogio de parede.
        self.tempos = {k: 0.0 for k in
                       ("sincronizar", "visao", "espacial", "gemeo",
                        "publicar", "esperando")}
        self.vazios = 0
        self._t_inicio = 0.0
        self._t_volta = 0.0

        # Media desde o inicio esconde o regime; janela sozinha esconde o
        # historico. O painel mostra as duas, e a diferenca entre elas E a
        # informacao: se divergem, houve um evento raro e caro.
        self._janela = deque(maxlen=90)

    # ------------------------------------------------------------ montagem
    def montar_cameras_reais(self):
        if not CONFIG_CAMERAS.exists():
            raise SystemExit(
                f"nao achei {CONFIG_CAMERAS}\n"
                "Rode antes: python captura/identificar.py --alto ... ")

        cfg = json.loads(CONFIG_CAMERAS.read_text(encoding="utf-8"))
        for papel, valor in cfg.items():
            fonte, papel_cfg = self._montar_uma(papel, valor)
            self.cameras.registrar(fonte)
        return self

    def _montar_uma(self, papel, valor):
        """Aceita duas formas, e a segunda existe por um motivo medido.

            "alto": "HD Pro Webcam C920"
            "alto": {"fonte": "HD Pro Webcam C920", "captura": "640x480"}

        RESOLUCAO E POR CAMERA, NAO DO SISTEMA.

        Em 10/08 as tres rodaram a 1280x720 porque `--captura` era global. A
        C920, em luz fraca, alongou a exposicao e caiu para 1,0 fps — e como
        ela e o papel obrigatorio, ditou 1 fps para o sistema inteiro. O
        tablet, virtual, entregava 30.

        O 720p da C920 ainda era desperdicio duplo: o YOLO reduz para 320 px
        antes de inferir, e a homografia foi calibrada em 640x480, entao a
        resolucao maior era reduzida na entrada e reescalada na geometria.

            Cada camera tem capacidade e tarefa proprias. Uma configuracao
            unica obriga a melhor a andar no passo da pior.
        """
        if isinstance(valor, dict):
            fonte_id = str(valor.get("fonte") or valor.get("id"))
            captura = valor.get("captura")
            exposicao = valor.get("exposicao", self._exposicao)
        else:
            fonte_id, captura, exposicao = str(valor), None, self._exposicao

        if captura:
            w, h = (int(v) for v in str(captura).lower().split("x"))
        else:
            w, h = self.captura
        self._captura_por_papel[papel] = (w, h)

        if fonte_id.startswith(("http://", "https://", "rtsp://")):
            return RemoteCameraSource(fonte_id, papel, largura=w, altura=h), papel
        return UsbCameraSource(fonte_id, papel, exposicao=exposicao,
                               largura=w, altura=h), papel

    def montar_cameras_falsas(self):
        """SO com pedido explicito. Nunca por caminho automatico."""
        from src.cameras.falsa import FonteFalsa
        self.log.aviso("FONTES SINTETICAS — nao ha camera real nesta execucao")
        for papel in ("alto", "frontal", "lateral"):
            self.cameras.registrar(FonteFalsa(papel, fps=20))
        return self

    def montar_visao(self):
        self.visao.registrar(DetectorDePessoas(
            "alto", imgsz=self._imgsz, conf=self._conf,
            a_cada_n=self._det_a_cada))
        if self.com_pose:
            for papel in ("frontal", "lateral"):
                if papel in self.cameras.fontes:
                    self.visao.registrar(
                        EstimadorDePose(papel, a_cada_n=self._pose_a_cada))
        return self

    # ------------------------------------------------------------ ciclo
    def iniciar(self, espera_s=20):
        self.cameras.iniciar()
        prontas = self.cameras.esperar_online(timeout=espera_s, minimo=1)
        self.log.info("cameras prontas", quantidade=len(prontas),
                      papeis=[f.papel for f in prontas])

        if not self.cameras.tem("alto"):
            self.log.aviso("sem camera 'alto': nao havera posicao no chao")
        else:
            self._casar_homografia_com_a_camera()

        self.eventos.emitir(Tipo.SYSTEM_STARTED,
                            {"cameras": len(self.cameras.fontes)})
        self._t_anterior = time.monotonic()
        self._t_inicio = self._t_anterior
        return self

    def _casar_homografia_com_a_camera(self):
        """A geometria segue a resolucao REAL, nao a pedida.

        `pedir nao e receber` ja valia para o log da abertura da camera. Aqui
        a mesma verdade vira consequencia: se a C920 responder 640x480 a um
        pedido de 1280x720, cada pixel passa a valer o dobro e a posicao no
        chao sai com o dobro do erro — sem sintoma nenhum alem do numero.
        """
        fonte = self.cameras.por_papel("alto")
        if fonte is None or not fonte.largura or not fonte.altura:
            return
        if self.espacial.ajustar_para_resolucao(fonte.largura, fonte.altura):
            self.log.info("homografia casada com a camera",
                          resolucao=f"{fonte.largura}x{fonte.altura}")

    def passo(self):
        """Um ciclo. Devolve o instantaneo do gemeo, ou None se nada chegou.

        E aqui que a arquitetura se paga: cinco chamadas, cada uma para um
        objeto testado sozinho.
        """
        # Tempo desde o fim do ciclo anterior. E aqui que mora a espera:
        # se o laco gira devagar e nenhuma etapa e cara, o custo esta FORA
        # das etapas — no chamador, ou esperando quadro chegar.
        t = time.perf_counter()
        if self._t_volta:
            self.tempos["esperando"] += t - self._t_volta

        instante = self.sincronizador.montar(self.cameras.buffers())
        self.tempos["sincronizar"] += time.perf_counter() - t
        if instante is None:
            self.vazios += 1
            self._t_volta = time.perf_counter()
            return None

        agora = time.monotonic()
        dt = max(1e-3, min(0.5, agora - self._t_anterior))
        self._t_anterior = agora
        self.quadros += 1
        self._janela.append(agora)

        t = time.perf_counter()
        observacoes = self.visao.processar(instante)
        self.tempos["visao"] += time.perf_counter() - t

        t = time.perf_counter()
        estados = self.espacial.atualizar(observacoes, dt)
        self.tempos["espacial"] += time.perf_counter() - t

        t = time.perf_counter()
        self.gemeo.atualizar(estados, self.cameras.resumo(), dt,
                             acoes=self.espacial.acoes)
        self.tempos["gemeo"] += time.perf_counter() - t

        # O gemeo e o dono da verdade; o publicador so a leva para fora.
        t = time.perf_counter()
        self.publicador.publicar_estado(self.gemeo.instantaneo(), agora)
        self.tempos["publicar"] += time.perf_counter() - t

        if self._salvar_a_cada:
            self._salvar(instante, agora, com_pessoa=bool(estados))

        self._t_volta = time.perf_counter()
        return instante

    def _salvar(self, instante, agora, com_pessoa):
        """Grava um quadro de cada camera, marcando se havia pessoa rastreada.

        O nome do arquivo carrega o que importa na hora de olhar depois:
        quando foi, de qual vista, e se o sistema tinha alguem no chao naquele
        instante. Sem isso sobra uma pasta de imagens sem contexto.
        """
        if agora - self._t_ultimo_salvo < self._salvar_a_cada:
            return
        self._t_ultimo_salvo = agora
        try:
            import cv2
            self._pasta_quadros.mkdir(parents=True, exist_ok=True)
            marca = "com-pessoa" if com_pessoa else "sem-pessoa"
            for papel, q in instante.quadros.items():
                nome = f"{papel}-{self.quadros:05d}-{marca}.jpg"
                cv2.imwrite(str(self._pasta_quadros / nome), q.imagem)
        except Exception as e:
            # canal lateral: nunca derruba o laco
            self.log.aviso("nao consegui salvar quadro", erro=str(e))
            self._salvar_a_cada = 0.0

    @property
    def fps_regime(self):
        """fps das ultimas ~90 entregas — o que o sistema faz AGORA.

        Em 10/08 a media desde o inicio disse 4,0 fps num sistema que estava
        em 16: uma unica inferencia de estreia de 15,2 s carregou toda a media.
        Uma amostra rara e cara nao pode definir o numero pelo qual se decide
        o que otimizar.
        """
        if len(self._janela) < 2:
            return 0.0
        span = self._janela[-1] - self._janela[0]
        return (len(self._janela) - 1) / span if span > 0 else 0.0

    @property
    def fps(self):
        """Quadros por segundo REAIS, medidos no relogio de parede.

        Nao e a soma das etapas nem a taxa das cameras. E o que o sistema
        entrega. Quando este numero discorda das etapas, ha custo escondido —
        e o painel diz onde.
        """
        if not self._t_inicio:
            return 0.0
        decorrido = time.monotonic() - self._t_inicio
        return self.quadros / decorrido if decorrido > 0 else 0.0

    def parar(self):
        self.cameras.parar()
        self.visao.parar()
        self.eventos.fechar()
        self.log.info("parado", quadros=self.quadros)

    # ------------------------------------------------------------ status
    def painel(self):
        linhas = ["CAMERAS"]
        linhas += ["  " + l for l in self.cameras.painel()]
        linhas += ["", "VISAO"]
        linhas += ["  " + l for l in self.visao.painel()]

        d = self.visao.diagnostico_paralelismo()
        linhas += ["", f"  sequencial seria {d['soma_sequencial_ms']:.0f} ms, "
                       f"real {d['real_ms']:.0f} ms, ganho {d['ganho']}x"]

        linhas += ["", "ONDE VAI O TEMPO   (ms por quadro entregue)"]
        n = max(1, self.quadros)
        medido = sum(self.tempos.values())
        decorrido = max(1e-6, time.monotonic() - (self._t_inicio or 0))

        for nome, seg in sorted(self.tempos.items(), key=lambda kv: -kv[1]):
            ms = seg * 1000 / n
            barra = "#" * int(round(30 * seg / max(decorrido, 1e-6)))
            linhas.append(f"  {nome:12} {ms:7.1f} ms  "
                          f"{100 * seg / decorrido:5.1f}%  {barra}")

        # O que sobra e o que nao esta sendo cronometrado: o desenho, o painel,
        # o terminal. Se esta linha crescer, o custo esta FORA do orquestrador.
        fora = decorrido - medido
        linhas.append(f"  {'nao medido':12} {fora * 1000 / n:7.1f} ms  "
                      f"{100 * fora / decorrido:5.1f}%   <- fora do passo()")
        linhas.append(f"  {'CICLO REAL':12} {decorrido * 1000 / n:7.1f} ms  "
                      f"-> {self.fps:.1f} fps desde o inicio")
        linhas.append(f"  {'REGIME':12} {1000 / max(self.fps_regime, 1e-6):7.1f} ms"
                      f"  -> {self.fps_regime:.1f} fps agora"
                      f"   (voltas sem quadro {self.vazios})")

        e = self.espacial.resumo()
        f = e["funil"]
        obs = max(1, f["observadas"])
        linhas += ["", "ESPACIAL — FUNIL DA DETECCAO ATE A MEDIDA",
                   f"  observadas   {f['observadas']:6}",
                   f"  sem id       {f['sem_id']:6}  "
                   f"({100 * f['sem_id'] / obs:4.1f}%)",
                   f"  plausibilid. {f['plausibilidade']:6}  "
                   f"({100 * f['plausibilidade'] / obs:4.1f}%)  descartadas",
                   f"  tornozelo    {f['tornozelo']:6}  "
                   f"({100 * f['tornozelo'] / obs:4.1f}%)  descartadas",
                   f"  MEDIDAS      {f['medidas']:6}  "
                   f"({100 * f['medidas'] / obs:4.1f}%)  sobreviveram",
                   "",
                   f"  rastros {e['rastros']}  recosturas {e['recosturas']}",
                   f"  altura[{e['altura']}]  vistas[{e['vistas']}]",
                   f"  inclinacao {e['inclinacao']}"]

        acoes = self.espacial.acoes
        if acoes:
            linhas += ["", "ACAO  (vocabulario fechado, v3)"]
            for pid, (a, _, _) in sorted(acoes.items()):
                linhas.append(
                    f"  #{pid}  {a.locomocao:18} {a.postura:12} "
                    f"conf {a.confianca:4.0%}   {a.velocidade_ms:.2f} m/s  "
                    f"{a.giro_graus_s:+5.0f} graus/s   [{a.motivo}]")

        g = self.gemeo.resumo()
        linhas += ["", "GEMEO",
                   f"  pessoas {g['pessoas']}  quadros {g['quadros']}  "
                   f"zonas ocupadas {g['zonas_ocupadas']}"]

        ult = self.eventos.ultimos(6)
        if ult:
            linhas += ["", "EVENTOS"]
            linhas += [f"  {ev}" for ev in ult]
        return linhas
