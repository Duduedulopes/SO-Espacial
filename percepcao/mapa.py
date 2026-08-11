"""
Mapa 2D ao vivo — vista de cima, sem esqueleto 3D.

Versao leve do `gemeo3d.py`: mesma cadeia de percepcao, sem MediaPipe e sem
cena 3D. Serve para duas coisas:

    - conferir rapidamente se a calibracao esta boa
    - rodar em maquina fraca, ou quando so interessa POSICAO

    camera -> YOLO-pose -> homografia -> Kalman -> Cena2D

REESCRITO EM 08/08. Antes este arquivo tinha 469 linhas e era PROGRAMA e
BIBLIOTECA ao mesmo tempo: o `gemeo3d.py` importava classes de dentro dele.
Isso significava que rodar o gemeo carregava o codigo de desenho do mapa, e
mexer num quebrava o outro.

O nucleo foi para `percepcao/chao.py`, o desenho para `visual/cena2d.py`, e
aqui sobrou so o programa.

Uso:
    python percepcao/mapa.py --camera 0
    python percepcao/mapa.py --camera 0 --area -1 2 -1 2 --registrar
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from captura.fonte import CameraAoVivo                    # noqa: E402
from estado.rastreio import GerenciadorDeRastros          # noqa: E402
from percepcao.chao import (                              # noqa: E402
    EstimadorDePe, FiltroDePlausibilidade, FiltroDeTornozelo,
    carregar_homografia, para_metros,
)
from visual.cena2d import Agente, Cena2D                  # noqa: E402

TORNOZELO_ESQ, TORNOZELO_DIR = 15, 16


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--modelo", type=str, default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--px-por-m", type=int, default=140)
    p.add_argument("--min-tornozelo", type=int, default=3)
    p.add_argument("--area", type=float, nargs=4,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"), default=None)
    p.add_argument("--exposicao", type=float, default=None,
                   help="exposicao manual. Padrao: automatica.")
    p.add_argument("--registrar", action="store_true")
    args = p.parse_args()

    H, meta = carregar_homografia()
    lm, am = meta["largura_m"], meta["altura_m"]

    if args.area is None:
        xmin, xmax = -lm / 2, lm * 1.5
        ymin, ymax = -am / 2, am * 1.5
    else:
        xmin, xmax, ymin, ymax = args.area

    from ultralytics import YOLO
    print(f"carregando {args.modelo} ...")
    yolo = YOLO(args.modelo)

    # exposicao automatica por padrao. A -6 fixa, herdada de 07/08, deixava a
    # imagem preta e ninguem era detectado.
    cam = CameraAoVivo(args.camera, 640, 480, 30, exposicao=args.exposicao)
    cena = Cena2D(xmin, xmax, ymin, ymax, px_por_m=args.px_por_m,
                  area_calibrada=(lm, am))

    estimador = EstimadorDePe()
    filtro = FiltroDeTornozelo(minimo=args.min_tornozelo)
    plausibilidade = FiltroDePlausibilidade(H)
    gerente = GerenciadorDeRastros()

    registro = None
    if args.registrar:
        destino = RAIZ / "dados" / f"mapa_{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"
        destino.parent.mkdir(parents=True, exist_ok=True)
        registro = destino.open("w", encoding="utf-8")
        print(f"registrando em {destino}")

    print(f"area calibrada : {lm:.2f} x {am:.2f} m")
    print(f"area navegavel : x [{xmin:.2f}, {xmax:.2f}]  y [{ymin:.2f}, {ymax:.2f}]")
    print("C limpa rastros   ESC sai")

    t_ant = time.monotonic()
    i = 0
    aceitos = rejeitados = 0

    while True:
        frame, _ = cam.ler()
        if frame is None:
            break
        i += 1
        agora = time.monotonic()
        dt = max(1e-3, min(0.5, agora - t_ant))
        t_ant = agora

        r = yolo.track(frame, persist=True, conf=args.conf, classes=[0],
                       imgsz=args.imgsz, verbose=False)[0]
        caixas, poses = r.boxes, r.keypoints
        n = len(caixas) if caixas is not None else 0

        medidas = []
        for k in range(n):
            b = caixas[k]
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            tid = int(b.id[0]) if b.id is not None else -1

            ok_geo, motivo = plausibilidade.plausivel((x1, y1, x2, y2))

            kp_xy = kp_conf = None
            if poses is not None and k < len(poses):
                kp_xy = poses[k].xy[0].cpu().numpy()
                kp_conf = (poses[k].conf[0].cpu().numpy()
                           if poses[k].conf is not None else np.ones(17))

            pe, origem = estimador.estimar(tid, (x1, y1, x2, y2), kp_xy, kp_conf)
            mx, my = para_metros(H, *pe)

            dentro = (xmin <= mx <= xmax) and (ymin <= my <= ymax)
            e_pessoa = filtro.ver(tid, origem)
            valido = dentro and e_pessoa and ok_geo

            if valido:
                aceitos += 1
                medidas.append((tid, mx, my))
                plausibilidade.observar((x1, y1, x2, y2))
            else:
                rejeitados += 1
                motivo = (motivo or ("fora da area" if not dentro
                                     else "sem tornozelo"))

            cor = (0, 220, 220) if valido else (60, 60, 190)
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2 if valido else 1)
            rot = f"ID {tid} ({mx:.2f},{my:.2f})m" if valido else f"X {motivo}"
            cv2.putText(frame, rot, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(frame, rot, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)
            cv2.circle(frame, pe, 5,
                       (0, 255, 0) if origem == "tornozelo" else (0, 165, 255), -1)

        vivos = {int(v) for v in caixas.id} if (caixas is not None and
                                                caixas.id is not None) else set()
        estimador.esquecer(vivos)
        filtro.esquecer(vivos)

        rastros = gerente.atualizar(medidas, dt)

        agentes = [
            Agente(id=meu, x=rr.pos[0], y=rr.pos[1],
                   vx=rr.kf.vel[0], vy=rr.kf.vel[1],
                   incerteza=rr.kf.incerteza, prevendo=rr.sem_medicao,
                   historico=rr.historico)
            for meu, rr in rastros.items()
        ]

        if registro is not None:
            for meu, rr in rastros.items():
                registro.write(json.dumps({
                    "i": i, "t": round(agora, 4), "rastro": meu,
                    "x_m": round(rr.pos[0], 4), "y_m": round(rr.pos[1], 4),
                    "prevendo": rr.sem_medicao,
                }) + "\n")

        titulo = (f"q{i}  {1/dt:4.1f} fps  {len(rastros)} pessoa(s)  "
                  f"rec {gerente.recosturas}  "
                  f"aceitos {aceitos} rejeitados {rejeitados}  "
                  f"altura[{plausibilidade.diagnostico()}]")

        cv2.imshow("mapa", cena.desenhar(agentes, titulo))
        cv2.imshow("camera", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k in (ord("c"), ord("C")):
            for rr in rastros.values():
                rr.historico.clear()

    cam.fechar()
    if registro:
        registro.close()
    cv2.destroyAllWindows()

    total = aceitos + rejeitados
    if total:
        print(f"\naceitos {aceitos} ({100*aceitos/total:.0f}%)  "
              f"rejeitados {rejeitados}")


if __name__ == "__main__":
    main()
