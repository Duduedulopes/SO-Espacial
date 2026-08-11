"""
Rastreamento + pose — blocos 3 e 5 juntos, e o experimento do ponto de chao.

Faz tres coisas ao mesmo tempo:

1. DETECTA pessoas com um modelo de pose (17 pontos do corpo)
2. RASTREIA cada uma com ByteTrack, dando um ID que persiste entre quadros
3. COMPARA dois jeitos de estimar a posicao no chao:

       VERMELHO  centro inferior da caixa   (heuristica)
       VERDE     ponto medio dos tornozelos (medicao)

   A linha branca entre eles e o desacordo. Se a heuristica fosse boa, os
   dois pontos coincidiriam sempre.

O EXPERIMENTO: fique PARADO no mesmo lugar e mude de postura — levante o
braco, incline-se, agache. A posicao no chao nao mudou. Veja qual dos dois
pontos concorda com a realidade.

Uso:
    python percepcao/rastrear.py --camera 0
    python percepcao/rastrear.py                      # ultima sessao gravada
    python percepcao/rastrear.py --camera 0 --registrar

Teclas:
    ESPACO  pausa
    ESC     sai
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
SESSOES = RAIZ / "dados" / "sessoes"

# Indices dos pontos do corpo no padrao COCO, usado pelo modelo de pose.
TORNOZELO_ESQ = 15
TORNOZELO_DIR = 16

# Cores distintas por ID, para o rastro ficar legivel quando houver varias pessoas.
PALETA = [
    (0, 255, 255), (255, 128, 0), (0, 255, 128), (255, 0, 255),
    (128, 255, 0), (0, 128, 255), (255, 255, 0), (128, 0, 255),
]


def abrir_fonte(args):
    if args.camera is not None:
        cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        return cam, f"camera {args.camera}"

    pasta = SESSOES / args.sessao if args.sessao else sorted(
        (p for p in SESSOES.iterdir() if p.is_dir()), key=lambda p: p.name
    )[-1]
    video = pasta / "video.mp4"
    if not video.exists():
        raise SystemExit(f"nao achei {video}")
    return cv2.VideoCapture(str(video)), pasta.name


def ponto_tornozelos(kp_xy, kp_conf, minimo=0.5):
    """Ponto no chao a partir dos tornozelos.

    Usa a media dos dois quando ambos sao confiaveis; se so um for, usa ele.
    Devolve None quando nenhum tornozelo foi visto — e devolver None e melhor
    que devolver um chute, porque uma posicao errada contamina o rastro inteiro.
    """
    validos = [
        kp_xy[i] for i in (TORNOZELO_ESQ, TORNOZELO_DIR)
        if kp_conf[i] >= minimo
    ]
    if not validos:
        return None
    p = np.mean(validos, axis=0)
    return int(p[0]), int(p[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessao", type=str, default=None)
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--modelo", type=str, default="yolo11n-pose.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--registrar", action="store_true",
                   help="grava os dois pontos por quadro, para medir o desacordo depois")
    args = p.parse_args()

    from ultralytics import YOLO

    print(f"carregando {args.modelo} ...")
    modelo = YOLO(args.modelo)

    cap, descricao = abrir_fonte(args)
    if not cap.isOpened():
        raise SystemExit("nao consegui abrir a fonte")

    print(f"fonte: {descricao}")
    print("ESPACO pausa, ESC sai")
    print()

    registro = None
    if args.registrar:
        destino = RAIZ / "dados" / f"pontos_{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"
        registro = destino.open("w", encoding="utf-8")
        print(f"registrando em {destino}")

    rastros: dict[int, list] = {}   # id -> ultimos pontos de chao, para o rastro
    i = 0
    soma_ms = 0.0
    desacordos = []
    pausado = False
    ids_vistos = set()

    while True:
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                print("fim do video")
                break

            t0 = time.perf_counter()
            # persist=True diz ao rastreador que este quadro e a continuacao
            # do anterior. Sem isso, cada chamada recomeca do zero e nenhum ID
            # sobrevive.
            r = modelo.track(frame, persist=True, conf=args.conf,
                             classes=[0], verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000
            soma_ms += ms
            i += 1

            vista = frame.copy()
            caixas = r.boxes
            poses = r.keypoints

            n = len(caixas) if caixas is not None else 0

            for k in range(n):
                cx = caixas[k]
                x1, y1, x2, y2 = (int(v) for v in cx.xyxy[0])

                tid = int(cx.id[0]) if cx.id is not None else -1
                ids_vistos.add(tid)
                cor = PALETA[tid % len(PALETA)] if tid >= 0 else (200, 200, 200)

                cv2.rectangle(vista, (x1, y1), (x2, y2), cor, 2)
                rotulo = f"ID {tid}" if tid >= 0 else "sem ID"
                cv2.putText(vista, rotulo, (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(vista, rotulo, (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

                # --- os dois candidatos a "posicao no chao" ---
                pe_caixa = ((x1 + x2) // 2, y2)

                pe_tornozelo = None
                if poses is not None and k < len(poses):
                    kp_xy = poses[k].xy[0].cpu().numpy()
                    kp_conf = poses[k].conf[0].cpu().numpy() if poses[k].conf is not None else np.ones(17)
                    pe_tornozelo = ponto_tornozelos(kp_xy, kp_conf)

                    # desenha os tornozelos vistos
                    for idx in (TORNOZELO_ESQ, TORNOZELO_DIR):
                        if kp_conf[idx] >= 0.5:
                            px, py = int(kp_xy[idx][0]), int(kp_xy[idx][1])
                            cv2.circle(vista, (px, py), 4, (0, 200, 0), -1)

                cv2.circle(vista, pe_caixa, 6, (0, 0, 255), -1)
                cv2.circle(vista, pe_caixa, 6, (255, 255, 255), 1)

                if pe_tornozelo is not None:
                    cv2.circle(vista, pe_tornozelo, 6, (0, 255, 0), -1)
                    cv2.circle(vista, pe_tornozelo, 6, (255, 255, 255), 1)
                    cv2.line(vista, pe_caixa, pe_tornozelo, (255, 255, 255), 1)

                    d = float(np.hypot(pe_caixa[0] - pe_tornozelo[0],
                                       pe_caixa[1] - pe_tornozelo[1]))
                    desacordos.append(d)
                    cv2.putText(vista, f"{d:.0f}px", (pe_caixa[0] + 10, pe_caixa[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(vista, f"{d:.0f}px", (pe_caixa[0] + 10, pe_caixa[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    # rastro do ponto confiavel
                    if tid >= 0:
                        rastros.setdefault(tid, []).append(pe_tornozelo)
                        rastros[tid] = rastros[tid][-60:]

                if registro is not None:
                    registro.write(json.dumps({
                        "i": i, "id": tid,
                        "caixa": [x1, y1, x2, y2],
                        "pe_caixa": list(pe_caixa),
                        "pe_tornozelo": list(pe_tornozelo) if pe_tornozelo else None,
                    }) + "\n")

            # rastros
            for tid, pts in rastros.items():
                if len(pts) > 1:
                    cv2.polylines(vista, [np.array(pts, dtype=np.int32)], False,
                                  PALETA[tid % len(PALETA)], 2)

            media_d = np.mean(desacordos[-90:]) if desacordos else 0.0
            info = f"q{i}  {ms:5.1f}ms  {n} pessoa(s)  desacordo medio {media_d:4.0f}px"
            cv2.putText(vista, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(vista, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(vista, "vermelho = base da caixa    verde = tornozelos",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(vista, "vermelho = base da caixa    verde = tornozelos",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            cv2.imshow("rastreamento + pose - ESPACO pausa, ESC sai", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        elif tecla == 32:
            pausado = not pausado

    cap.release()
    cv2.destroyAllWindows()
    if registro is not None:
        registro.close()

    if i:
        print()
        print(f"quadros              : {i}")
        print(f"tempo medio          : {soma_ms/i:.1f} ms  ({1000*i/soma_ms:.1f} fps)")
        print(f"IDs distintos criados: {len(ids_vistos - {-1})}")
        if desacordos:
            a = np.array(desacordos)
            print()
            print("desacordo entre base da caixa e tornozelos (px):")
            print(f"  mediana : {np.median(a):6.1f}")
            print(f"  media   : {a.mean():6.1f}")
            print(f"  p90     : {np.percentile(a, 90):6.1f}")
            print(f"  maximo  : {a.max():6.1f}")


if __name__ == "__main__":
    main()
