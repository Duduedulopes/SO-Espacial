"""
Reparo de camera — quando o driver guarda uma configuracao ruim.

O CASO QUE MOTIVOU ISTO (08/08)

A C920 passou a entregar brilho ZERO. Nao "escuro": zero absoluto em todos os
pixels. Subexposicao produz valores baixos COM ruido; zero absoluto e outra
coisa — e configuracao travada num extremo.

E o Windows reportava `Status: OK`. O dispositivo estava saudavel; a
configuracao e que estava ruim.

    A camera guarda brilho, contraste, ganho, exposicao e foco NO PROPRIO
    HARDWARE. Fechar o programa nao desfaz. Reiniciar o Python nao desfaz.
    Reiniciar o Windows nao desfaz.

Este programa devolve tudo ao padrao, medindo o efeito de cada passo.

Uso:
    python captura/reparar.py                    repara todas
    python captura/reparar.py --camera 0
    python captura/reparar.py --camera 0 --painel   abre o painel do driver
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from captura.dispositivos import listar  # noqa: E402

# Propriedades que persistem no hardware e podem escurecer a imagem.
# O valor e o "meio da escala" tipico do DirectShow: 0.5 em escala 0..1.
AJUSTES = [
    ("brilho", cv2.CAP_PROP_BRIGHTNESS, 0.5),
    ("contraste", cv2.CAP_PROP_CONTRAST, 0.5),
    ("saturacao", cv2.CAP_PROP_SATURATION, 0.5),
    ("ganho", cv2.CAP_PROP_GAIN, 0.5),
]


def brilho_atual(cap, n=5):
    q = None
    for _ in range(n):
        ok, q = cap.read()
        if not ok:
            return 0.0
        time.sleep(0.03)
    return float(q[::8, ::8].mean()) if q is not None else 0.0


def reparar(indice, nome="", painel=False, minimo=25):
    print(f"\n=== indice {indice}  {nome} ===")
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  nao abriu")
        return False

    b0 = brilho_atual(cap)
    print(f"  brilho inicial: {b0:.1f}")

    if b0 >= minimo:
        print("  ja esta boa, nada a fazer")
        cap.release()
        return True

    # 1) exposicao e balanco de branco automaticos
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    b = brilho_atual(cap)
    print(f"  apos exposicao automatica: {b:.1f}")

    # 2) propriedades de imagem ao meio da escala
    if b < minimo:
        for rotulo, prop, valor in AJUSTES:
            antes = cap.get(prop)
            cap.set(prop, valor)
            depois = cap.get(prop)
            print(f"    {rotulo:10} {antes:.3f} -> {depois:.3f}")
        b = brilho_atual(cap)
        print(f"  apos ajustar propriedades: {b:.1f}")

    # 3) exposicao manual, do escuro ao claro
    if b < minimo:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        for e in (-7, -6, -5, -4, -3, -2, -1):
            cap.set(cv2.CAP_PROP_EXPOSURE, e)
            b = brilho_atual(cap, 4)
            print(f"    exposicao {e:3d} -> brilho {b:5.1f}")
            if b >= minimo:
                break
        if b >= minimo:
            print("  clareou no manual. Voltando ao automatico...")
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            b = brilho_atual(cap)
            print(f"  automatico agora: {b:.1f}")

    # 4) painel do proprio driver — o botao "Padrao" dele resolve o que o
    #    OpenCV nao alcanca
    if b < minimo and painel:
        print("\n  abrindo o painel do driver...")
        print("  procure o botao PADRAO / DEFAULT e clique. Depois feche.")
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
        input("  ENTER quando terminar: ")
        b = brilho_atual(cap)
        print(f"  apos o painel: {b:.1f}")

    cap.release()

    if b >= minimo:
        print(f"  REPARADA (brilho {b:.1f})")
        return True

    print(f"  AINDA ESCURA (brilho {b:.1f})")
    print("  proximos passos, em ordem:")
    print("    1. rode com --painel e clique em PADRAO no painel do driver")
    print("    2. desconecte o cabo USB por 10s e reconecte")
    print("    3. confira se ha algo na frente da lente")
    print("    4. abra o app Camera do Windows e veja se aparece imagem")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--painel", action="store_true",
                   help="abre o painel de configuracao do proprio driver")
    args = p.parse_args()

    devs = listar()
    if args.camera is not None:
        nome = ""
        if devs:
            nome = dict(devs).get(args.camera, "")
        reparar(args.camera, nome, args.painel)
        return

    if not devs:
        print("sem lista de dispositivos (pip install pygrabber).")
        print("reparando indices 0 a 3...")
        devs = [(i, "") for i in range(4)]

    for idx, nome in devs:
        reparar(idx, nome, args.painel)


if __name__ == "__main__":
    main()
