"""
Painel de cameras — a tela de diagnostico da Fase 3.

Sobe as fontes conforme `config/cameras.json` e mostra, ao vivo:

    estado, resolucao, fps, recebidos, descartados, falhas, reconexoes

E o que responde as perguntas que ate agora exigiam adivinhacao:
"quantos quadros se perderam?", "essa camera caiu?", "reconectou?".

Uso:
    python ferramentas/cameras.py                 painel no terminal
    python ferramentas/cameras.py --ver           + janela com as imagens
    python ferramentas/cameras.py --falsas        sem hardware, para testar

CONFIGURACAO — `config/cameras.json`

    {
      "alto":    "HD Pro Webcam C920",           nome  -> camera USB
      "frontal": "VGA camera",
      "lateral": "http://192.168.1.20:8080/video" URL  -> camera remota
    }

O tipo e deduzido do valor: se comeca com http/rtsp, e remota. Trocar o
celular de Iriun para um app que serve URL e so editar esta linha.
"""

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.cameras.falsa import FonteFalsa                     # noqa: E402
from src.cameras.gerenciador import GerenciadorDeCameras     # noqa: E402
from src.cameras.remota import RemoteCameraSource            # noqa: E402
from src.cameras.usb import UsbCameraSource                  # noqa: E402
from src.nucleo import log as logmod                         # noqa: E402
from src.nucleo.log import Log                               # noqa: E402

CONFIG = RAIZ / "config" / "cameras.json"


def montar_fontes(g, largura, altura, exposicao):
    if not CONFIG.exists():
        raise SystemExit(f"nao achei {CONFIG}\n"
                         "Rode antes: python ferramentas/abrir_camera.py")

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for papel, valor in cfg.items():
        v = str(valor)
        if v.startswith(("http://", "https://", "rtsp://")):
            g.registrar(RemoteCameraSource(v, papel,
                                           largura=largura, altura=altura))
        else:
            g.registrar(UsbCameraSource(v, papel, exposicao=exposicao,
                                        largura=largura, altura=altura))
    return g


def montar_falsas(g):
    g.registrar(FonteFalsa("alto", fps=30))
    g.registrar(FonteFalsa("frontal", fps=30))
    g.registrar(FonteFalsa("lateral", fps=30, cair_apos_s=8,
                           silencio_degradada=1.5, silencio_falha=4))
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ver", action="store_true", help="mostra as imagens")
    p.add_argument("--falsas", action="store_true", help="sem hardware")
    p.add_argument("--captura", default="1280x720")
    p.add_argument("--exposicao", type=float, default=None)
    p.add_argument("--segundos", type=float, default=0,
                   help="0 = ate ESC / Ctrl+C")
    p.add_argument("--log", default="INFO")
    args = p.parse_args()

    logmod.configurar(args.log)
    log = Log("painel")
    largura, altura = (int(v) for v in args.captura.lower().split("x"))

    eventos = []
    g = GerenciadorDeCameras(
        ao_evento=lambda t, d: eventos.append(
            (time.strftime("%H:%M:%S"), t, d.get("papel", ""))))

    if args.falsas:
        montar_falsas(g)
    else:
        montar_fontes(g, largura, altura, args.exposicao)

    g.iniciar()
    log.info("aguardando as cameras subirem...")
    g.esperar_online(timeout=20, minimo=1)

    cv2 = None
    if args.ver:
        import cv2 as _cv2
        cv2 = _cv2

    t0 = time.monotonic()
    try:
        while True:
            if args.segundos and time.monotonic() - t0 > args.segundos:
                break

            print("\033[H\033[J", end="")           # limpa a tela
            print("PAINEL DE CAMERAS      "
                  "ESC na janela ou Ctrl+C para sair\n")
            print("EST PAPEL     DISPOSITIVO                RESOLUCAO   FPS "
                  "    RECEBIDOS DESCART FALHAS RECON")
            for linha in g.painel():
                print("  " + linha)

            t = g.total()
            print(f"\n  {t['online']}/{t['cameras']} online   "
                  f"recebidos {t['recebidos']}   "
                  f"descartados {t['descartados']}   "
                  f"reconexoes {t['reconexoes']}   "
                  f"tempo {t['tempo_s']}s")

            if eventos:
                print("\n  EVENTOS")
                for h, tipo, papel in eventos[-8:]:
                    print(f"    {h}  {tipo:22} {papel}")

            if cv2 is not None:
                import numpy as np
                tiras = []
                for papel, f in g.fontes.items():
                    q = f.buffer.espiar()
                    if q is None:
                        img = np.full((180, 320, 3), 30, np.uint8)
                        cv2.putText(img, f"{papel}: {f.estado.value}", (10, 96),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (60, 60, 220), 1)
                    else:
                        img = cv2.resize(q.imagem, (320, 180))
                        cv2.putText(img, f"{papel} #{q.seq}", (8, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 220, 255), 1)
                    tiras.append(img)
                if tiras:
                    cv2.imshow("cameras", np.hstack(tiras))
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break

            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        g.parar()
        if cv2 is not None:
            cv2.destroyAllWindows()

    print("\nRESUMO FINAL")
    print(json.dumps(g.resumo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
