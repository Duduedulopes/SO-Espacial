"""
Gemeo digital com tres cameras — cada uma no que ela e boa.

    camera do ALTO    YOLO + homografia  ->  ONDE a pessoa esta   (2-5 cm)
    camera de FRENTE  MediaPipe          ->  largura e altura
    camera de LADO    MediaPipe          ->  profundidade

Sem calibracao multi-camera. Sem tabuleiro. Sem extrinsecas.

POR QUE ISTO FUNCIONA

O MediaPipe foi treinado com vistas frontais. A camera do teto e uma vista
fora da distribuicao de treino — ele adivinha, e nenhuma suavizacao conserta
isso. Era a causa do esqueleto torto, e nao dava para ver medindo o codigo.

Cada camera tem um eixo que ela NAO mede, so estima: a profundidade. De
frente, a profundidade e o frente-tras da pessoa. De lado, e o
esquerda-direita. O eixo fraco de uma e o eixo forte da outra.

Pegamos de cada vista so o que ela mede de verdade. Nao e triangulacao
rigorosa — e fusao por competencia. Custa zero calibracao.

Medido em simulacao (ruido so na profundidade):
    so frontal  32,9 cm  |  so lateral  25,0 cm  |  FUNDIDAS  1,3 cm

Uso:
    python percepcao/gemeo_multi.py --alto 0 --frontal 1 --lateral 2
    python percepcao/gemeo_multi.py --alto 0 --frontal 1        (sem lateral)
    python percepcao/gemeo_multi.py --alto 0 --lado esquerda

A camera do alto e a UNICA que precisa de homografia calibrada.
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
from percepcao.fusao import Fusor, para_o_mundo           # noqa: E402
from percepcao.pose3d import Pose3D, SuavizadorDeEsqueleto  # noqa: E402
from visual.cena3d import Cena3D, Esqueleto               # noqa: E402


class Relogio:
    def __init__(self, memoria=30):
        from collections import defaultdict, deque
        self.m = defaultdict(lambda: deque(maxlen=memoria))
        self._t = {}

    def inicio(self, k):
        self._t[k] = time.perf_counter()

    def fim(self, k):
        if k in self._t:
            self.m[k].append((time.perf_counter() - self._t[k]) * 1000)

    def resumo(self):
        return " ".join(f"{k}{np.mean(v):.0f}" for k, v in self.m.items() if v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alto", type=int, default=None,
                   help="sobrescreve o indice de config/cameras.json")
    p.add_argument("--frontal", type=int, default=None)
    p.add_argument("--lateral", type=int, default=None)
    p.add_argument("--lado", choices=("direita", "esquerda"), default="direita",
                   help="de que lado da pessoa a camera lateral esta")
    p.add_argument("--modelo", type=str, default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--captura", type=str, default="1280x720")
    p.add_argument("--planta", type=str, default="loja/bancada.json")
    p.add_argument("--calor-hz", type=float, default=4.0)
    p.add_argument("--meia-vida", type=float, default=90.0)
    p.add_argument("--pose-a-cada", type=int, default=1)
    args = p.parse_args()

    # OS INDICES VEM DO ARQUIVO, nao da linha de comando.
    #
    # Indices de camera no Windows mudam sozinhos — ao reconectar um cabo, ao
    # instalar um driver virtual, ao reiniciar. Em 08/08 isso fez as janelas
    # "alto" e "lateral" mostrarem a MESMA camera, e uma sessao inteira foi
    # perdida procurando erro no lugar errado.
    from captura.identificar import carregar as carregar_cameras
    cams = carregar_cameras()
    if args.alto is not None:
        cams["alto"] = args.alto
    if args.frontal is not None:
        cams["frontal"] = args.frontal
    if args.lateral is not None:
        cams["lateral"] = args.lateral

    if "alto" not in cams:
        raise SystemExit("nenhuma camera com papel 'alto'. "
                         "Rode: python captura/identificar.py")

    H, meta = carregar_homografia()
    planta = Planta.carregar(RAIZ / args.planta)
    cap_w, cap_h = (int(v) for v in args.captura.lower().split("x"))

    calib_w, calib_h = meta.get("resolucao", [640, 480])
    if (cap_w, cap_h) != (calib_w, calib_h):
        S = np.array([[calib_w / cap_w, 0, 0], [0, calib_h / cap_h, 0], [0, 0, 1.0]])
        H = H @ S
        print(f"homografia reescalada {calib_w}x{calib_h} -> {cap_w}x{cap_h}")

    from ultralytics import YOLO
    print(f"carregando {args.modelo} ...")
    yolo = YOLO(args.modelo)

    # Uma instancia de MediaPipe POR CAMERA. Elas guardam estado temporal
    # entre quadros; compartilhar uma so faria as vistas se contaminarem.
    idx_alto = cams["alto"]
    idx_f = cams.get("frontal")
    idx_l = cams.get("lateral")

    pose_frontal = Pose3D() if idx_f is not None else None
    pose_lateral = Pose3D() if idx_l is not None else None

    print(f"  alto    : indice {idx_alto}")
    print(f"  frontal : indice {idx_f}")
    print(f"  lateral : indice {idx_l}  (a {args.lado} da pessoa)")

    # ABRIR A MAIS EXIGENTE PRIMEIRO. Se a camera de maior demanda entrar por
    # ultimo, as outras ja reservaram banda e ela falha sem motivo aparente.
    cam_alto = CameraAoVivo(idx_alto, cap_w, cap_h, 30)
    cam_f = CameraAoVivo(idx_f, cap_w, cap_h, 30) if idx_f is not None else None
    cam_l = CameraAoVivo(idx_l, cap_w, cap_h, 30) if idx_l is not None else None

    # CONFERE QUE NAO SAO A MESMA CAMERA. Foi exatamente isto que passou
    # despercebido em 08/08: duas janelas mostrando o mesmo fluxo.
    from captura.identificar import parecidas
    fontes = [("alto", cam_alto), ("frontal", cam_f), ("lateral", cam_l)]
    fontes = [(n, c) for n, c in fontes if c is not None]
    for a in range(len(fontes)):
        for b in range(a + 1, len(fontes)):
            qa, _ = fontes[a][1].ler()
            qb, _ = fontes[b][1].ler()
            if qa is not None and qb is not None and parecidas(qa, qb):
                print(f"\n  ATENCAO: '{fontes[a][0]}' e '{fontes[b][0]}' estao "
                      f"mostrando a MESMA imagem.")
                print("  Rode: python captura/identificar.py\n")

    cena = Cena3D(960, 620, chao=planta.chao, calor_hz=args.calor_hz)
    planta.aplicar_na_cena(cena)
    calor = planta.novo_mapa_de_calor(meia_vida_s=args.meia_vida)
    publicador = Publicador(RAIZ / "dados" / "estado_atual.json")

    estimador = EstimadorDePe()
    filtro = FiltroDeTornozelo()
    plausibilidade = FiltroDePlausibilidade(H)
    gerente = GerenciadorDeRastros()
    suave = SuavizadorDeEsqueleto()
    relogio = Relogio()

    # Um fusor por rastro. Com varias pessoas isto vira o problema de saber
    # quem e quem entre as vistas — que exigiria re-identificacao. Por ora,
    # com UMA pessoa em cena, o unico fusor recebe tudo.
    fusor = Fusor(lado_lateral=args.lado)

    rumos: dict[int, float] = {}
    t_ant = time.monotonic()
    i = 0

    print("\nsetas giram a camera virtual   +/- zoom   C limpa   ESC sai\n")

    while True:
        frame, t_frame = cam_alto.ler()
        if frame is None:
            break
        i += 1
        agora = time.monotonic()
        dt = max(1e-3, min(0.5, agora - t_ant))
        t_ant = agora

        # ---------- ONDE: camera do alto ----------
        relogio.inicio("yolo")
        r = yolo.track(frame, persist=True, conf=args.conf, classes=[0],
                       imgsz=args.imgsz, verbose=False)[0]
        relogio.fim("yolo")

        caixas, poses = r.boxes, r.keypoints
        n = len(caixas) if caixas is not None else 0
        medidas, caixa_de = [], {}

        for k in range(n):
            b = caixas[k]
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            tid = int(b.id[0]) if b.id is not None else -1

            ok_geo, motivo = plausibilidade.plausivel((x1, y1, x2, y2))
            if not ok_geo:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 190), 1)
                continue

            kp_xy = kp_conf = None
            if poses is not None and k < len(poses):
                kp_xy = poses[k].xy[0].cpu().numpy()
                kp_conf = (poses[k].conf[0].cpu().numpy()
                           if poses[k].conf is not None else np.ones(17))

            pe, origem = estimador.estimar(tid, (x1, y1, x2, y2), kp_xy, kp_conf)
            if not filtro.ver(tid, origem):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 190), 1)
                continue

            mx, my = para_metros(H, *pe)
            medidas.append((tid, mx, my))
            caixa_de[tid] = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
            cv2.circle(frame, pe, 5, (0, 255, 0), -1)

        vivos = {int(v) for v in caixas.id} if (caixas is not None and
                                                caixas.id is not None) else set()
        estimador.esquecer(vivos)
        filtro.esquecer(vivos)

        rastros = gerente.atualizar(medidas, dt)

        # ---------- COMO: cameras de frente e de lado ----------
        # Rodam no quadro INTEIRO, sem deteccao. Sao vistas dedicadas a UMA
        # pessoa; procurar de novo seria trabalho repetido.
        if i % max(1, args.pose_a_cada) == 0:
            if cam_f is not None:
                qf, tf = cam_f.ler()
                if qf is not None:
                    relogio.inicio("frontal")
                    j, _, _ = pose_frontal.estimar(qf)
                    relogio.fim("frontal")
                    fusor.ver_frontal(j, agora)
                    cv2.imshow("frontal", cv2.resize(qf, None, fx=0.35, fy=0.35))

            if cam_l is not None:
                ql, tl = cam_l.ler()
                if ql is not None:
                    relogio.inicio("lateral")
                    j, _, _ = pose_lateral.estimar(ql)
                    relogio.fim("lateral")
                    fusor.ver_lateral(j, agora)
                    cv2.imshow("lateral", cv2.resize(ql, None, fx=0.35, fy=0.35))

        juntas_pessoa = fusor.esqueleto(agora)

        # ---------- ocupacao ----------
        calor.passo()
        posicoes = {}
        for meu, rr in rastros.items():
            if not rr.coasting:
                calor.acumular(*rr.pos, dt)
            posicoes[meu] = rr.pos
            ext = [e for e in rr.ids_externos if e in caixa_de]
            if ext:
                plausibilidade.observar(caixa_de[ext[-1]], rr.percorrido)
        for z in planta.zonas:
            z.atualizar(posicoes, dt)
        publicador.publicar(planta, rastros, agora)

        # ---------- esqueletos no mundo ----------
        esqueletos = []
        for meu, rr in rastros.items():
            vx, vy = rr.kf.vel
            if np.hypot(vx, vy) > 0.15:
                rumos[meu] = float(np.arctan2(vy, vx))
            rumo = rumos.get(meu, -np.pi / 2)

            if juntas_pessoa is None:
                continue
            j = para_o_mundo(juntas_pessoa, rr.pos[0], rr.pos[1], rumo)
            j = suave.suavizar(meu, j)
            esqueletos.append(Esqueleto(id=meu, juntas=j, prevendo=rr.coasting,
                                        historico=rr.historico))

        suave.esquecer(set(rastros))

        brilho = float(frame[::8, ::8].mean())
        atraso = (time.monotonic() - t_frame) * 1000
        titulo = (f"q{i} {1/dt:4.1f}fps atraso{atraso:3.0f}ms brilho{brilho:3.0f} "
                  f"{len(rastros)}p rec{gerente.recosturas} "
                  f"vistas[{fusor.diagnostico}] [{relogio.resumo()}]")

        cv2.imshow("gemeo 3D", cena.desenhar(esqueletos, titulo,
                                             calor=calor, zonas=planta.zonas))
        cv2.imshow("alto", cv2.resize(frame, None, fx=0.45, fy=0.45))

        k = cv2.waitKeyEx(1) & 0xFFFFFF
        if k in (ord("c"), ord("C")):
            for rr in rastros.values():
                rr.historico.clear()
        elif not cena.tecla(k):
            break

    for c in (cam_alto, cam_f, cam_l):
        if c:
            c.fechar()
    for p_ in (pose_frontal, pose_lateral):
        if p_:
            p_.fechar()
    cv2.destroyAllWindows()
    print(f"\ntempos medios (ms): {relogio.resumo()}")


if __name__ == "__main__":
    main()
