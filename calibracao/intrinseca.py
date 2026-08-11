"""
Calibracao intrinseca — os parametros internos da lente.

POR QUE ISTO E NECESSARIO AGORA

Com UMA camera, a homografia bastava: ela mapeia o plano do chao, e so
perguntavamos coisas sobre o chao.

Para TRIANGULAR pontos fora do chao — cotovelo, ombro, cabeca — e preciso a
matriz de projecao completa de cada camera. E ha um resultado classico:

    H = K·[r1 r2 t]        (homografia de um plano)
    r3 = r1 × r2           (a terceira coluna sai do produto vetorial)
    P  = K·[r1 r2 r3 t]    (matriz de projecao completa)

Ou seja: sabendo K, a homografia do chao que voce ja sabe fazer entrega a
orientacao completa da camera. Este programa mede K.

O QUE E K

    fx, fy   distancia focal em pixels
    cx, cy   onde o eixo optico atravessa o sensor
    k1..k3   distorcao radial — o "barril" das lentes grandes angulares
    p1, p2   distorcao tangencial — sensor levemente torto

A C920 tem distorcao visivel nas bordas. Sem corrigir, um ponto na borda pode
errar dezenas de pixels — e no fim vira erro de centimetros no mundo.

COMO USAR

    1. Imprima um tabuleiro de xadrez e COLE numa superficie RIGIDA.
       Papel entortado e a fonte de erro mais comum aqui.
    2. python calibracao/intrinseca.py --camera 0
    3. Mostre o tabuleiro em varias posicoes e INCLINACOES.
       Aperte ESPACO para capturar quando ficar verde.
    4. Com 15 a 20 capturas variadas, aperte C para calcular.

VARIACAO E TUDO. Vinte fotos do tabuleiro de frente, no centro, ensinam menos
que oito fotos inclinadas cobrindo os cantos. A distorcao so aparece nas
bordas — se voce nunca puser o tabuleiro la, ela nao sera medida.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

SAIDA = RAIZ / "calibracao" / "intrinsecas"

CRITERIO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def cobertura(pontos, w, h, celulas=3):
    """Que fracao da imagem os tabuleiros ja cobriram.

    Serve para dizer ao usuario ONDE ainda falta por o tabuleiro. Sem isso,
    as pessoas capturam vinte vezes no centro e a distorcao das bordas fica
    sem medicao.
    """
    grade = np.zeros((celulas, celulas), dtype=bool)
    for p in pontos:
        for (x, y) in p.reshape(-1, 2):
            i = min(celulas - 1, int(y / h * celulas))
            j = min(celulas - 1, int(x / w * celulas))
            grade[i, j] = True
    return grade


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--nome", type=str, default=None,
                   help="nome do arquivo de saida. Padrao: cam<indice>")
    p.add_argument("--cantos", type=int, nargs=2, default=(9, 6),
                   metavar=("COLUNAS", "LINHAS"),
                   help="cantos INTERNOS do tabuleiro. Um tabuleiro 10x7 "
                        "quadrados tem 9x6 cantos internos.")
    p.add_argument("--quadrado-mm", type=float, default=25.0)
    p.add_argument("--captura", type=str, default="1280x720")
    args = p.parse_args()

    nx, ny = args.cantos
    cap_w, cap_h = (int(v) for v in args.captura.lower().split("x"))
    nome = args.nome or f"cam{args.camera}"

    # Pontos do tabuleiro no proprio referencial dele: z=0, espacamento real.
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= args.quadrado_mm / 1000.0        # em metros

    cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
    if not cam.isOpened():
        raise SystemExit(f"nao consegui abrir a camera {args.camera}")

    pts_obj, pts_img = [], []
    print(__doc__)
    print(f"tabuleiro {nx}x{ny} cantos internos, quadrado {args.quadrado_mm} mm")
    print("ESPACO captura   C calcula   D descarta a ultima   ESC sai\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        achou, cantos = cv2.findChessboardCorners(
            cinza, (nx, ny),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE)

        vista = frame.copy()
        if achou:
            cv2.drawChessboardCorners(vista, (nx, ny), cantos, achou)

        # mapa de cobertura, para o usuario saber onde falta
        grade = cobertura(pts_img, w, h)
        cel_w, cel_h = w // 3, h // 3
        for i in range(3):
            for j in range(3):
                cor = (0, 180, 0) if grade[i, j] else (60, 60, 60)
                cv2.rectangle(vista, (j * cel_w + 4, i * cel_h + 4),
                              ((j + 1) * cel_w - 4, (i + 1) * cel_h - 4), cor, 1)

        estado = "TABULEIRO OK - ESPACO captura" if achou else "procurando tabuleiro..."
        cor = (0, 220, 0) if achou else (0, 165, 255)
        cv2.putText(vista, f"{len(pts_img)} capturas   {estado}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vista, f"{len(pts_img)} capturas   {estado}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(vista, f"cobertura {grade.sum()}/9 regioes", (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(f"calibracao intrinseca - camera {args.camera}", vista)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        elif k == 32 and achou:
            finos = cv2.cornerSubPix(cinza, cantos, (11, 11), (-1, -1), CRITERIO)
            pts_obj.append(objp.copy())
            pts_img.append(finos)
            print(f"  captura {len(pts_img)}  (cobertura {cobertura(pts_img,w,h).sum()}/9)")

        elif k in (ord("d"), ord("D")) and pts_img:
            pts_obj.pop(); pts_img.pop()
            print(f"  descartada. restam {len(pts_img)}")

        elif k in (ord("c"), ord("C")):
            if len(pts_img) < 8:
                print(f"  poucas capturas ({len(pts_img)}). Junte pelo menos 8.")
                continue

            print("\ncalculando...")
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                pts_obj, pts_img, (w, h), None, None)

            # Erro de reprojecao por imagem: revela a foto ruim do conjunto.
            erros = []
            for i in range(len(pts_obj)):
                proj, _ = cv2.projectPoints(pts_obj[i], rvecs[i], tvecs[i], K, dist)
                erros.append(float(cv2.norm(pts_img[i], proj, cv2.NORM_L2) /
                                   len(proj)))

            print(f"\n  erro de reprojecao (RMS): {rms:.3f} px")
            print(f"  pior imagem             : {max(erros):.3f} px")
            print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
                  f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
            print(f"  distorcao: {np.round(dist.ravel(), 4).tolist()}")

            if rms < 0.5:
                print("\n  RMS abaixo de 0,5 px: calibracao boa.")
            elif rms < 1.0:
                print("\n  RMS entre 0,5 e 1 px: aceitavel, da para melhorar.")
            else:
                print("\n  RMS acima de 1 px: RUIM. Provaveis causas —")
                print("    tabuleiro entortado, poucas inclinacoes, ou imagem borrada.")

            SAIDA.mkdir(parents=True, exist_ok=True)
            destino = SAIDA / f"{nome}.json"
            destino.write_text(json.dumps({
                "camera": args.camera,
                "resolucao": [w, h],
                "K": K.tolist(),
                "dist": dist.ravel().tolist(),
                "rms_px": float(rms),
                "pior_imagem_px": float(max(erros)),
                "capturas": len(pts_img),
                "tabuleiro": {"cantos": [nx, ny], "quadrado_mm": args.quadrado_mm},
            }, indent=2), encoding="utf-8")
            print(f"\n  salvo em {destino}")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
