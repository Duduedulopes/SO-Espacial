"""
Gemeo digital 3D — a cadeia inteira, do sensor ao mundo.

    camera (thread)  sempre o quadro mais recente, os velhos sao descartados
      -> YOLO         SO deteccao e rastreio. Nao calcula pose.
      -> MediaPipe    pose 2D e 3D, uma vez por pessoa
      -> homografia   pe em metros no chao            (bloco 1)
      -> Kalman       suaviza e preve na ausencia     (bloco 2)
      -> suavizador   tira o tremor das juntas
      -> Cena3D       desenho, calor e zonas

O QUE MUDOU EM 08/08, E POR QUE

1. UM modelo de pose, nao dois. Antes o yolo11n-pose calculava 17 pontos que
   quase nao usavamos e o MediaPipe recalculava 33 na mesma pessoa. Agora o
   YOLO so detecta e rastreia (yolo11n comum, mais barato), e o MediaPipe
   fornece tanto os tornozelos na imagem quanto a pose 3D.

2. Captura em thread. A camera entrega 30 q/s; se processamos 4, os outros 26
   ficavam na fila e a imagem aparecia velha e aos trancos. Agora os quadros
   nao processados sao descartados de proposito.

3. Entrada menor no YOLO (--imgsz 320). Menos pixels, menos tempo. Para
   detectar uma pessoa que ocupa boa parte do quadro, 320 basta.

4. Recorte ampliado antes do MediaPipe. Ele foi treinado com a pessoa grande
   no quadro; recorte pequeno rende landmarks ruins.

5. Suavizacao temporal das juntas. Cada quadro e estimado do zero, entao as
   juntas tremiam mesmo com a pessoa parada.

PRE-REQUISITO
    calibracao/homografia.json valido, camera na MESMA posicao.

Uso:
    python percepcao/gemeo3d.py --camera 0
    python percepcao/gemeo3d.py --camera 0 --imgsz 256 --deteccao-a-cada 2
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from captura.fonte import CameraAoVivo                    # noqa: E402
from estado.planta import Planta, Publicador              # noqa: E402
from estado.rastreio import GerenciadorDeRastros          # noqa: E402
from percepcao.chao import (                              # noqa: E402
    EstimadorDePe, FiltroDePlausibilidade, FiltroDeTornozelo,
    carregar_homografia, para_metros,
)
from percepcao.pose3d import (                            # noqa: E402
    EstimadorDeInclinacao, Pose3D, SuavizadorDeEsqueleto, ancorar_no_chao,
)
from visual.cena3d import Cena3D, Esqueleto               # noqa: E402


class Relogio:
    """Mede onde o tempo esta indo. Sem isso, otimizar e adivinhar."""

    def __init__(self, memoria=30):
        from collections import defaultdict, deque
        self.m = defaultdict(lambda: deque(maxlen=memoria))
        self._t = {}

    def inicio(self, nome):
        self._t[nome] = time.perf_counter()

    def fim(self, nome):
        if nome in self._t:
            self.m[nome].append((time.perf_counter() - self._t[nome]) * 1000)

    def resumo(self):
        return "  ".join(f"{k} {np.mean(v):.0f}" for k, v in self.m.items() if v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--modelo", type=str, default="yolo11n.pt",
                   help="deteccao apenas. NAO use o -pose: o MediaPipe ja faz pose.")
    p.add_argument("--imgsz", type=int, default=320,
                   help="entrada do YOLO. 320 e ~4x mais barato que 640.")
    p.add_argument("--exposicao", type=float, default=None,
                   help="exposicao manual em log2 s. PADRAO: automatica.\n"
                        "A exposicao -6 foi calibrada em 07/08 para arrancar "
                        "30 fps a 640x480. A 720p a camera esta limitada pela "
                        "banda do USB a ~10 fps de qualquer jeito, entao nao ha "
                        "fps a ganhar escurecendo — so imagem preta e nenhuma "
                        "deteccao.")
    p.add_argument("--captura", type=str, default="1280x720",
                   help="resolucao da camera. MEDIDO em 07/08: a 720p a C920 "
                        "entrega no maximo ~10 fps (limite do USB 2.0), mas nos "
                        "processamos ~7. Entao 720p dobra a resolucao da pessoa "
                        "SEM custar throughput — e resolucao e o que falta para "
                        "a pose sair decente. Use 640x480 em maquina fraca.")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--planta", type=str, default="loja/bancada.json")
    p.add_argument("--min-tornozelo", type=int, default=3)
    p.add_argument("--inclinacao", type=float, default=None)
    p.add_argument("--meia-vida", type=float, default=90.0)
    p.add_argument("--deteccao-a-cada", type=int, default=1,
                   help="rodar o YOLO a cada N quadros. 2 quase dobra o fps; "
                        "entre deteccoes o Kalman segura o rastro.")
    p.add_argument("--calor-hz", type=float, default=4.0,
                   help="atualizacoes por segundo do mapa de calor. Ele muda "
                        "devagar; 4 e visualmente identico a 30 e custa 1/8.")
    p.add_argument("--sem-plausibilidade", action="store_true",
                   help="desliga o filtro de altura esperada, para comparar")
    p.add_argument("--sem-3d", action="store_true",
                   help="pula o MediaPipe. Fica muito mais rapido, sem esqueleto.")
    args = p.parse_args()

    H, meta = carregar_homografia()
    planta = Planta.carregar(RAIZ / args.planta)
    print(f"planta: {planta.nome}  ({len(planta.moveis)} moveis, "
          f"{len(planta.zonas)} zonas)")

    from ultralytics import YOLO
    print(f"carregando {args.modelo} ...")
    yolo = YOLO(args.modelo)
    pose3d = None if args.sem_3d else Pose3D()

    cap_w, cap_h = (int(v) for v in args.captura.lower().split("x"))
    cam = CameraAoVivo(args.camera, cap_w, cap_h, 30, exposicao=args.exposicao)

    # A HOMOGRAFIA FOI CALIBRADA NUMA RESOLUCAO. Se a captura mudar, os pixels
    # mudam de escala e a calibracao deixa de valer. Em vez de exigir
    # recalibracao, corrigimos a matriz: basta compor com a mudanca de escala.
    calib_w, calib_h = meta.get("resolucao", [640, 480])
    if (cap_w, cap_h) != (calib_w, calib_h):
        sx, sy = calib_w / cap_w, calib_h / cap_h
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])
        H = H @ S
        print(f"homografia reescalada de {calib_w}x{calib_h} "
              f"para {cap_w}x{cap_h}")

    cena = Cena3D(960, 620, chao=planta.chao, calor_hz=args.calor_hz)
    planta.aplicar_na_cena(cena)
    calor = planta.novo_mapa_de_calor(meia_vida_s=args.meia_vida)
    zonas = planta.zonas
    publicador = Publicador(RAIZ / "dados" / "estado_atual.json")

    estimador = EstimadorDePe()
    filtro = FiltroDeTornozelo(minimo=args.min_tornozelo)
    plausibilidade = FiltroDePlausibilidade(H)   # depois do reescalonamento
    print(f"filtro de altura: {plausibilidade.diagnostico()}")
    gerente = GerenciadorDeRastros()
    suave = SuavizadorDeEsqueleto()
    auto_incl = EstimadorDeInclinacao()
    relogio = Relogio()

    manual = args.inclinacao is not None
    inclinacao = np.deg2rad(args.inclinacao) if manual else 0.0

    ultimo_esq: dict[int, tuple] = {}
    rumos: dict[int, float] = {}
    ultimas_caixas: dict[int, tuple] = {}
    t_ant = time.monotonic()
    i = 0

    print()
    print("  A inclinacao e medida sozinha: ande de um lado para o outro.")
    print("  setas giram a camera   +/- zoom   C limpa   ESC sai")
    print()

    while True:
        frame, t_frame = cam.ler()
        if frame is None:
            break
        i += 1

        agora = time.monotonic()
        dt = max(1e-3, min(0.5, agora - t_ant))
        t_ant = agora

        # ---------- deteccao e rastreio ----------
        if i % max(1, args.deteccao_a_cada) == 0 or not ultimas_caixas:
            relogio.inicio("yolo")
            r = yolo.track(frame, persist=True, conf=args.conf, classes=[0],
                           imgsz=args.imgsz, verbose=False)[0]
            relogio.fim("yolo")

            ultimas_caixas = {}
            if r.boxes is not None and len(r.boxes):
                for k in range(len(r.boxes)):
                    b = r.boxes[k]
                    tid = int(b.id[0]) if b.id is not None else -1
                    if tid >= 0:
                        ultimas_caixas[tid] = tuple(int(v) for v in b.xyxy[0])

        medidas = []
        caixa_de = {}

        for tid, (x1, y1, x2, y2) in ultimas_caixas.items():
            # PRIMEIRO a geometria, que e gratis. So depois o MediaPipe, que
            # custa ~30 ms. Nao adianta estimar a pose de uma cadeira.
            if not args.sem_plausibilidade:
                ok_geo, motivo_geo = plausibilidade.plausivel((x1, y1, x2, y2))
                if not ok_geo:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 190), 1)
                    cv2.putText(frame, motivo_geo, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 220), 1)
                    continue

            kp_xy = kp_conf = juntas_rel = None

            if pose3d is not None:
                relogio.inicio("pose")
                juntas_rel, vis, px2d = pose3d.estimar(frame, (x1, y1, x2, y2))
                relogio.fim("pose")
                if px2d is not None:
                    kp_xy = px2d
                    kp_conf = vis.astype(float)

            pe, origem = estimador.estimar(tid, (x1, y1, x2, y2), kp_xy, kp_conf)
            if not filtro.ver(tid, origem):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 190), 1)
                continue

            mx, my = para_metros(H, *pe)
            medidas.append((tid, mx, my))
            caixa_de[tid] = (x1, y1, x2, y2)
            if juntas_rel is not None:
                ultimo_esq[tid] = (juntas_rel, vis)

            # A amostragem acontece depois, com a distancia percorrida em maos.
            # Ver o laco dos rastros abaixo.

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
            cv2.circle(frame, pe, 5,
                       (0, 255, 0) if origem == "tornozelo" else (0, 165, 255), -1)

        vivos = set(ultimas_caixas)
        estimador.esquecer(vivos)
        filtro.esquecer(vivos)

        rastros = gerente.atualizar(medidas, dt)

        # ---------- ocupacao ----------
        calor.passo()
        posicoes = {}
        for meu, rr in rastros.items():
            if not rr.coasting:
                calor.acumular(*rr.pos, dt)
            posicoes[meu] = rr.pos
        for z in zonas:
            z.atualizar(posicoes, dt)
        publicador.publicar(planta, rastros, agora)

        # ---------- esqueletos ----------
        esqueletos = []
        for meu, rr in rastros.items():
            fx, fy = rr.pos
            vx, vy = rr.kf.vel
            if np.hypot(vx, vy) > 0.15:
                rumos[meu] = float(np.arctan2(vy, vx))
            rumo = rumos.get(meu, -np.pi / 2)

            # SO APRENDE COM QUEM ANDOU. Mobilia nunca alcanca o limiar, entao
            # nunca envenena o modelo de altura. Foi o erro de 08/08: a cadeira
            # virava amostra e ensinava o filtro a aceitar cadeiras.
            ext_c = [e for e in rr.ids_externos if e in caixa_de]
            if ext_c:
                plausibilidade.observar(caixa_de[ext_c[-1]], rr.percorrido)

            ext = [e for e in rr.ids_externos if e in ultimo_esq]
            if not ext:
                continue
            juntas_rel, vis = ultimo_esq[ext[-1]]

            if not manual:
                auto_incl.observar(juntas_rel, rr.kf.velocidade, vis)
                if auto_incl.confiavel:
                    inclinacao = auto_incl.valor

            juntas = ancorar_no_chao(juntas_rel, fx, fy, rumo, inclinacao)
            juntas = suave.suavizar(meu, juntas)

            esqueletos.append(Esqueleto(id=meu, juntas=juntas, visivel=vis,
                                        prevendo=rr.coasting,
                                        historico=rr.historico))

        suave.esquecer(set(rastros))
        for e in list(ultimo_esq):
            if e not in vivos:
                del ultimo_esq[e]

        # BRILHO NO TITULO. Em 08/08 a imagem ficou preta por uma exposicao
        # herdada e ninguem detectou nada — o sintoma era "0 pessoas", que
        # manda procurar no lugar errado. Agora a causa aparece na tela.
        brilho = float(frame[::8, ::8].mean())
        if brilho < 35:
            cv2.putText(frame, "IMAGEM ESCURA DEMAIS PARA DETECTAR",
                        (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        atraso = (time.monotonic() - t_frame) * 1000
        titulo = (f"q{i}  {1/dt:4.1f} fps  atraso {atraso:3.0f}ms  brilho {brilho:3.0f}  "
                  f"{len(rastros)} pessoa(s)  rec {gerente.recosturas}  "
                  f"incl {np.rad2deg(inclinacao):+.0f}g  "
                  f"altura[{plausibilidade.diagnostico()}]  [{relogio.resumo()}]")

        relogio.inicio("desenho")
        cv2.imshow("gemeo 3D", cena.desenhar(esqueletos, titulo,
                                             calor=calor, zonas=zonas))
        cv2.imshow("camera", frame)
        relogio.fim("desenho")

        k = cv2.waitKeyEx(1) & 0xFFFFFF
        if k in (ord("c"), ord("C")):
            for rr in rastros.values():
                rr.historico.clear()
        elif k in (ord("a"), ord("A")):
            manual = False
            print("automatico")
        elif k in (ord("z"), ord("Z")):
            manual = True
            inclinacao -= np.deg2rad(2)
        elif k in (ord("x"), ord("X")):
            manual = True
            inclinacao += np.deg2rad(2)
        elif not cena.tecla(k):
            break

    print()
    print(f"quadros capturados pela camera : {cam.quadros_capturados}")
    print(f"quadros processados            : {i}")
    print(f"descartados (de proposito)     : {cam.quadros_capturados - i}")
    print(f"tempos medios (ms)             : {relogio.resumo()}")

    cam.fechar()
    if pose3d:
        pose3d.fechar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
