"""
Deteccao com YOLO — bloco 3, primeiro degrau.

Roda um detector pre-treinado sobre uma sessao gravada ou sobre a webcam.
Nao treina nada: o modelo ja vem sabendo 80 categorias, entre elas "person".

A FONTE E UMA INTERFACE. Este mesmo programa roda sobre video gravado
(--sessao) ou ao vivo (--camera). E a decisao de arquitetura que torna os
experimentos reproduziveis: voce grava uma vez e experimenta cem vezes sobre
o mesmo material, comparando resultados de forma justa.

Uso:
    python percepcao/detectar.py                    # ultima sessao gravada
    python percepcao/detectar.py --sessao 2026-08-07_180640
    python percepcao/detectar.py --camera 0         # ao vivo
    python percepcao/detectar.py --tudo             # todas as 80 classes

Teclas:
    ESPACO  pausa / continua
    ESC     sai
"""

import argparse
import time
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
SESSOES = RAIZ / "dados" / "sessoes"


def abrir_fonte(args):
    """Devolve (captura, descricao). A unica parte que sabe de onde vem o video."""
    if args.camera is not None:
        cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cam.set(cv2.CAP_PROP_EXPOSURE, -6)
        return cam, f"camera {args.camera}"

    if args.sessao:
        pasta = SESSOES / args.sessao
    else:
        pastas = sorted(p for p in SESSOES.iterdir() if p.is_dir())
        if not pastas:
            raise SystemExit("nenhuma sessao gravada em dados/sessoes/")
        pasta = pastas[-1]

    video = pasta / "video.mp4"
    if not video.exists():
        raise SystemExit(f"nao achei {video}")

    return cv2.VideoCapture(str(video)), pasta.name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessao", type=str, default=None, help="id da pasta em dados/sessoes/")
    p.add_argument("--camera", type=int, default=None, help="usa a webcam em vez de video")
    p.add_argument("--modelo", type=str, default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.35, help="confianca minima")
    p.add_argument("--tudo", action="store_true", help="detecta as 80 classes, nao so pessoas")
    args = p.parse_args()

    # importado aqui porque leva alguns segundos e nao vale a pena
    # pagar esse custo quando o programa vai falhar por outro motivo
    from ultralytics import YOLO

    print(f"carregando {args.modelo} ...")
    modelo = YOLO(args.modelo)  # baixa os pesos na primeira vez
    nomes = modelo.names

    # classe 0 do COCO e "person"
    classes = None if args.tudo else [0]

    cap, descricao = abrir_fonte(args)
    if not cap.isOpened():
        raise SystemExit("nao consegui abrir a fonte de video")

    print(f"fonte: {descricao}")
    print("ESPACO pausa, ESC sai")
    print()

    i = 0
    pausado = False
    soma_ms = 0.0
    quadros_com_deteccao = 0
    total_deteccoes = 0

    while True:
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                print("fim do video")
                break

            t0 = time.perf_counter()
            resultado = modelo(frame, conf=args.conf, classes=classes, verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000
            soma_ms += ms
            i += 1

            caixas = resultado.boxes
            n = len(caixas)
            total_deteccoes += n
            if n:
                quadros_com_deteccao += 1

            vista = frame.copy()

            for cx in caixas:
                x1, y1, x2, y2 = (int(v) for v in cx.xyxy[0])
                conf = float(cx.conf[0])
                cls = int(cx.cls[0])

                cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 255, 255), 2)

                rotulo = f"{nomes[cls]} {conf:.2f}"
                cv2.putText(vista, rotulo, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(vista, rotulo, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # PONTO DOS PES: centro inferior da caixa.
                #
                # E o unico ponto da caixa que fica sobre o chao — e portanto o
                # unico que a homografia mapeia corretamente para metros. A
                # cabeca nao esta no plano do chao; mapea-la daria uma posicao
                # deslocada para longe da camera.
                #
                # Nao e convencao arbitraria: e consequencia da geometria.
                pe = ((x1 + x2) // 2, y2)
                cv2.circle(vista, pe, 5, (0, 0, 255), -1)
                cv2.circle(vista, pe, 5, (255, 255, 255), 1)

            info = f"{i}  {ms:5.1f} ms  ({1000/ms:4.1f} fps)  {n} obj"
            cv2.putText(vista, info, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vista, info, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("deteccao - ESPACO pausa, ESC sai", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        elif tecla == 32:
            pausado = not pausado

    cap.release()
    cv2.destroyAllWindows()

    if i:
        print()
        print(f"quadros processados     : {i}")
        print(f"tempo medio por quadro  : {soma_ms/i:.1f} ms  ({1000*i/soma_ms:.1f} fps)")
        print(f"quadros com deteccao    : {quadros_com_deteccao} ({100*quadros_com_deteccao/i:.0f}%)")
        print(f"deteccoes por quadro    : {total_deteccoes/i:.2f}")


if __name__ == "__main__":
    main()
