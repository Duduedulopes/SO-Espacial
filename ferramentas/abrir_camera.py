"""Por que ESTA camera nao abre?

    python ferramentas/abrir_camera.py "EMEET PIXY"
    python ferramentas/abrir_camera.py 4              tambem aceita indice

DIAGNOSTICO EM VEZ DE PALPITE.

`CAMERA_ERROR` diz que a abertura falhou e nao diz por que. As causas possiveis
sao poucas e MUTUAMENTE EXCLUDENTES, entao vale tentar todas e ver qual passa:

    - o aplicativo do fabricante esta segurando o dispositivo
    - a camera nao aceita a resolucao pedida
    - o backend errado (DirectShow x Media Foundation)
    - falta de banda no barramento USB

Cada uma tem conserto diferente, e adivinhar qual e custa mais tempo do que
esta varredura inteira. Ela abre a camera com cada combinacao de backend e
resolucao, e imprime uma tabela do que funcionou.

    Uma tabela de 12 tentativas responde em 20 segundos o que uma hipotese
    responde em meia hora — quando acerta.
"""

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2                                              # noqa: E402

from src.cameras import dispositivos                    # noqa: E402

BACKENDS = (("DirectShow", cv2.CAP_DSHOW),
            ("MediaFoundation", cv2.CAP_MSMF),
            ("padrao", cv2.CAP_ANY))

RESOLUCOES = ((640, 480), (1280, 720), (1920, 1080), (None, None))


def _tentar(indice, backend, w, h):
    """Abre, pede a resolucao, le um quadro. Devolve (ok, o que veio)."""
    cap = cv2.VideoCapture(indice, backend)
    try:
        if not cap.isOpened():
            return False, "nao abriu"
        if w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Camera com PTZ demora a entregar o primeiro quadro: o motor ainda
        # esta assentando. Uma unica leitura reprovaria injustamente.
        for _ in range(6):
            ok, quadro = cap.read()
            if ok and quadro is not None:
                real = f"{quadro.shape[1]}x{quadro.shape[0]}"
                return True, real
            time.sleep(0.15)
        return False, "abriu, sem quadro"
    finally:
        cap.release()


def main():
    if len(sys.argv) < 2:
        dispositivos.imprimir()
        print("\nuso:  python ferramentas/abrir_camera.py \"EMEET PIXY\"")
        return

    alvo = sys.argv[1]
    if alvo.isdigit():
        indice, nome = int(alvo), f"indice {alvo}"
    else:
        indice = dispositivos.indice_de(alvo)
        nome = alvo
        if indice is None:
            print(f"'{alvo}' nao esta na lista do DirectShow:\n")
            dispositivos.imprimir()
            return

    print(f"\n  {nome}  ->  indice {indice}\n")
    print(f"  {'backend':18} {'pedido':12} resultado")
    print(f"  {'-'*18} {'-'*12} {'-'*22}")

    vitorias = []
    for rotulo, backend in BACKENDS:
        for w, h in RESOLUCOES:
            pedido = f"{w}x{h}" if w else "sem pedir"
            ok, msg = _tentar(indice, backend, w, h)
            print(f"  {rotulo:18} {pedido:12} {'OK  ' if ok else '--  '}{msg}")
            if ok:
                vitorias.append((rotulo, pedido, msg))

    print()
    if not vitorias:
        print("  NENHUMA COMBINACAO ABRIU.")
        print("  A causa mais provavel e outro programa segurando a camera —")
        print("  feche o aplicativo do fabricante (EMEET STUDIO, OBS, Teams,")
        print("  navegador com a camera ligada) e rode de novo.")
        print("  Se ja estiverem fechados, tente outra porta USB: tres cameras")
        print("  na mesma controladora nao cabem em resolucao alta.")
        return

    rotulo, pedido, real = vitorias[0]
    print(f"  FUNCIONA: backend {rotulo}, pedindo {pedido}, entrega {real}")
    print(f"  Ponha \"captura\": \"{real}\" no config/cameras.json.")


if __name__ == "__main__":
    main()
