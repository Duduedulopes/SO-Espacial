"""
Diagnostico da camera — substitui quatro scripts soltos.

Consolida `testar_camera.py`, `procurar_camera.py`, `capturar_fps.py` e
`testar_codec.py`, que existiam separados por acidente historico: cada um
nasceu para responder uma pergunta e ficou.

Uso:
    python captura/diagnostico.py                 varredura + medicao completa
    python captura/diagnostico.py --ver 0         so mostra a imagem
    python captura/diagnostico.py --camera 0      so mede a camera 0

O QUE JA APRENDEMOS AQUI (07/08, Logitech C920, USB 2.0)

    1280x720, exposicao automatica  ->  10,0 fps
     640x480, exposicao automatica  ->  15,8 fps
     640x480, exposicao MANUAL -6   ->  30,0 fps

Causa: em luz fraca a camera expoe o sensor por mais tempo e reduz a taxa em
degraus fixos (30, 15, 10, 7,5), sem avisar. `CAP_PROP_FPS` continua dizendo 30.

E o codec: `CAP_PROP_FOURCC` sempre volta YUY2 — o backend DSHOW ignora o
pedido de MJPG nas duas ordens de chamada.
"""

import argparse
import time

import cv2


def fourcc_legivel(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def procurar():
    print("procurando cameras...\n")
    achadas = []
    for nome, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
        for idx in range(4):
            cam = cv2.VideoCapture(idx, backend)
            ok, f = cam.read()
            if ok:
                h, w = f.shape[:2]
                print(f"  OK  {nome:6} indice={idx}  {w}x{h}")
                achadas.append((nome, backend, idx))
            cam.release()
    if not achadas:
        print("  nenhuma camera respondeu.\n"
              "  - feche o app Camera do Windows e a tela de Configuracoes\n"
              "  - Configuracoes > Privacidade > Camera > permitir apps da area de trabalho")
    return achadas


def medir(indice, backend=cv2.CAP_DSHOW, largura=640, altura=480,
          exposicao=None, n=120):
    cam = cv2.VideoCapture(indice, backend)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    cam.set(cv2.CAP_PROP_FPS, 30)
    if exposicao is not None:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cam.set(cv2.CAP_PROP_EXPOSURE, exposicao)

    if not cam.isOpened():
        print(f"  camera {indice}: nao abriu")
        return

    for _ in range(10):
        cam.read()

    t0 = time.monotonic()
    brilho = 0.0
    for _ in range(n):
        ok, f = cam.read()
        if ok:
            brilho += float(f.mean())
    dt = time.monotonic() - t0

    print(f"  {largura}x{altura}  exp={'auto' if exposicao is None else exposicao}"
          f"  codec={fourcc_legivel(cam.get(cv2.CAP_PROP_FOURCC))}"
          f"  {n/dt:5.1f} fps  brilho {brilho/n:5.1f}/255")
    cam.release()


def ver(indice, backend=cv2.CAP_DSHOW, exposicao=-6):
    cam = cv2.VideoCapture(indice, backend)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if exposicao is not None:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cam.set(cv2.CAP_PROP_EXPOSURE, exposicao)
    print("ESC sai")
    while True:
        ok, f = cam.read()
        if not ok:
            break
        cv2.putText(f, f"brilho {f.mean():.0f}/255", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(f"camera {indice} - ESC sai", f)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


def varias(indices, largura=1280, altura=720, segundos=20):
    """Duas cameras ao mesmo tempo — o teste que destrava a triangulacao.

    O RISCO REAL: duas cameras USB no mesmo controlador dividem a banda. A C920
    a 720p sem compressao ja consome quase tudo que o USB 2.0 tem. Se a segunda
    for USB tambem, uma das duas pode simplesmente parar de entregar quadros.

    A webcam embutida do notebook costuma estar num barramento interno
    separado, entao a combinacao C920 + embutida tende a funcionar.

    O que este teste mede:
      - as duas abrem juntas?
      - quantos quadros cada uma entrega por segundo?
      - a taxa cai em relacao a cada uma sozinha?
    """
    import numpy as np

    # ABRIR NAO PODE SER TUDO-OU-NADA.
    #
    # A versao anterior abortava quando uma camera falhava — e voce perdia a
    # informacao de quais funcionaram. Diagnostico que desiste no primeiro
    # problema nao diagnostica nada.
    #
    # A ordem tambem importa: a camera mais exigente (a USB de maior
    # resolucao) abre PRIMEIRO, enquanto ha banda sobrando. Se ela entrar por
    # ultimo, pode nao achar espaco e falhar sem motivo aparente.
    caps, abertas = [], []
    for idx in indices:
        c = None
        for nome_b, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
            c = cv2.VideoCapture(idx, backend)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = c.read()
            if ok:
                print(f"  camera {idx}: abriu com {nome_b}")
                break
            c.release()
            c = None

        if c is None:
            print(f"  camera {idx}: NAO ABRIU (com as outras ja abertas)")
            continue
        caps.append(c)
        abertas.append(idx)

    if not caps:
        print("\nnenhuma camera abriu.")
        return

    indices = abertas
    print(f"\n{len(caps)} de {len(indices)} funcionando. medindo {segundos}s... ESC sai\n")
    n = [0] * len(caps)
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        quadros = []
        for k, c in enumerate(caps):
            ok, f = c.read()
            if ok:
                n[k] += 1
                h, w = f.shape[:2]
                esc = 320 / h
                pequeno = cv2.resize(f, None, fx=esc, fy=esc)
                cv2.putText(pequeno,
                            f"cam {indices[k]}  {w}x{h}  brilho {f.mean():.0f}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                quadros.append(pequeno)
        if len(quadros) == len(caps):
            alt = min(q.shape[0] for q in quadros)
            cv2.imshow("cameras - ESC sai",
                       np.hstack([q[:alt] for q in quadros]))
        if cv2.waitKey(1) == 27:
            break

    dt = time.monotonic() - t0
    for c in caps:
        c.release()
    cv2.destroyAllWindows()

    print()
    for k, idx in enumerate(indices):
        print(f"camera {idx}: {n[k]/dt:5.1f} fps")

    if min(n) / dt < 4:
        print("\nAlguma ficou abaixo de 4 fps — provavel disputa de banda USB.")
        print("Tente: outra porta USB, resolucao menor (--res 640x480),")
        print("ou o iPhone por Wi-Fi (o Iriun por Wi-Fi nao usa banda USB).")
    else:
        print("\nTodas acima de 4 fps. Da para triangular.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--ver", type=int, default=None)
    p.add_argument("--juntas", type=int, nargs="+", metavar="IDX",
                   help="testa varias cameras funcionando ao mesmo tempo. "
                        "Ponha a mais exigente PRIMEIRO.")
    p.add_argument("--res", type=str, default="1280x720",
                   help="resolucao no teste conjunto. Use 640x480 se houver "
                        "disputa de banda.")
    args = p.parse_args()

    if args.juntas:
        w, h = (int(v) for v in args.res.lower().split("x"))
        return varias(args.juntas, largura=w, altura=h)
    if args.ver is not None:
        return ver(args.ver)

    # BUG corrigido: aqui estava `for _, _, i in ...` com dois `_`, entao o
    # segundo sobrescrevia o primeiro e a comparacao com "DSHOW" nunca era
    # verdadeira. So a camera 0 era medida. Nome repetido esconde erro.
    if args.camera is not None:
        indices = [args.camera]
    else:
        indices = [idx for nome, _backend, idx in procurar() if nome == "DSHOW"]
        indices = list(dict.fromkeys(indices)) or [0]

    for idx in dict.fromkeys(indices):
        print(f"\ncamera {idx}:")
        for larg, alt, exp in ((1280, 720, None), (640, 480, None),
                               (640, 480, -6), (640, 480, -4)):
            medir(idx, largura=larg, altura=alt, exposicao=exp)


if __name__ == "__main__":
    main()
