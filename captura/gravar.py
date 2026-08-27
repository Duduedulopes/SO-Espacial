"""
Gravador de sessoes — degrau 0.

Grava video da webcam junto com um carimbo de tempo por quadro, no formato
definido em docs/DATASET.md.

Este programa e deliberadamente burro: ele nao detecta, nao analisa, nao pensa.
So captura e escreve. Codigo que cria dado insubstituivel precisa ser chato.

Uso:
    python captura/gravar.py
    python captura/gravar.py --camera 0 --nota "prateleira, luz da tarde"

Para parar: ESC ou Q na janela de previa.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
SESSOES = RAIZ / "dados" / "sessoes"


def agora_iso() -> str:
    """Relogio de parede, ISO 8601, sempre com fuso.

    Timestamp sem fuso e uma divida que sempre vence.
    """
    return datetime.now().astimezone().isoformat()


def main() -> None:
    p = argparse.ArgumentParser(description="Grava uma sessao de captura.")
    p.add_argument("--camera", type=int, default=0, help="indice da camera")
    p.add_argument("--largura", type=int, default=640)
    p.add_argument("--altura", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--exposicao",
        type=float,
        default=-6,
        help="exposicao manual em log2 segundos (-6 = 1/64s). Menor = mais escuro e mais rapido.",
    )
    p.add_argument(
        "--ganho",
        type=float,
        default=None,
        help="amplifica o sinal do sensor. Clareia sem perder fps, mas amplifica ruido junto.",
    )
    p.add_argument(
        "--auto-exposicao",
        action="store_true",
        help="deixa a camera escolher. NAO recomendado: derruba o fps e muda o brilho sozinho.",
    )
    p.add_argument("--nota", type=str, default="", help="anotacao livre sobre a cena")
    p.add_argument("--local", type=str, default="", help="onde a camera esta")
    args = p.parse_args()

    # ---------- abrir a camera ----------
    cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise SystemExit(f"nao consegui abrir a camera {args.camera}")

    # MJPG antes da resolucao: sem isso muitas webcams USB limitam o fps.
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.altura)
    cam.set(cv2.CAP_PROP_FPS, args.fps)

    # Exposicao manual. Isto nao e detalhe: no automatico a C920 derruba a taxa
    # de quadros em luz fraca (30 -> 15 -> 10 -> 7,5) sem avisar, e o brilho da
    # cena muda sozinho ao longo da gravacao. Dataset com brilho instavel ensina
    # o modelo a associar coisas que nao tem relacao.
    if not args.auto_exposicao:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual, 0.75 = auto
        cam.set(cv2.CAP_PROP_EXPOSURE, args.exposicao)

    if args.ganho is not None:
        cam.set(cv2.CAP_PROP_GAIN, args.ganho)

    # O que a camera REALMENTE aceitou. Pedido != obtido.
    largura_real = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_real = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_real = float(cam.get(cv2.CAP_PROP_FPS))
    exposicao_real = float(cam.get(cv2.CAP_PROP_EXPOSURE))
    auto_exp_real = float(cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))

    # ---------- criar a pasta da sessao ----------
    sessao_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    pasta = SESSOES / sessao_id
    pasta.mkdir(parents=True, exist_ok=True)

    caminho_video = pasta / "video.mp4"
    escritor = cv2.VideoWriter(
        str(caminho_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_real if fps_real > 0 else args.fps,
        (largura_real, altura_real),
    )

    meta = {
        "sessao_id": sessao_id,
        "inicio_wall": agora_iso(),
        "camera": {
            "indice": args.camera,
            "backend": "DSHOW",
            "largura_pedida": args.largura,
            "altura_pedida": args.altura,
            "fps_pedido": args.fps,
            "largura_real": largura_real,
            "altura_real": altura_real,
            "fps_real": fps_real,
            "fourcc_pedido": "MJPG",
            "fourcc_real": "".join(
                chr((int(cam.get(cv2.CAP_PROP_FOURCC)) >> (8 * k)) & 0xFF)
                for k in range(4)
            ),
            "auto_exposicao": args.auto_exposicao,
            "exposicao_pedida": args.exposicao,
            "exposicao_real": exposicao_real,
            "auto_exposicao_real": auto_exp_real,
        },
        "cena": {"local": args.local, "descricao": args.nota},
        "notas": args.nota,
    }

    print(f"gravando em {pasta}")
    print(f"{largura_real}x{altura_real} @ {fps_real:.2f} fps")
    print("ESC ou Q para parar")
    print()
    print(">>> CLAQUETE: passe a tag conhecida no leitor 3x agora <<<")
    print()

    # ---------- laco de captura ----------
    i = 0
    t0 = time.monotonic()
    caminho_frames = pasta / "frames.jsonl"

    try:
        with caminho_frames.open("w", encoding="utf-8") as f_frames:
            while True:
                ok, frame = cam.read()

                # O carimbo vem IMEDIATAMENTE depois do read(), nunca antes
                # do laco nem depois do processamento.
                t_mono_ns = time.monotonic_ns()
                t_wall = agora_iso()

                if not ok:
                    print("falha ao ler quadro — encerrando")
                    break

                escritor.write(frame)
                f_frames.write(
                    json.dumps(
                        {"i": i, "t_mono_ns": t_mono_ns, "t_wall": t_wall},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Previa com indicador de gravacao.
                previa = frame.copy()
                decorrido = time.monotonic() - t0

                # Brilho medio, para voce ver na hora se a cena esta escura
                # demais. Abaixo de ~60 o video fica praticamente inutilizavel;
                # entre 90 e 140 e uma faixa saudavel.
                brilho = float(frame.mean())

                cv2.circle(previa, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    previa,
                    f"REC {decorrido:6.1f}s  q{i}  brilho {brilho:3.0f}",
                    (50, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("gravando - ESC para parar", previa)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q")):
                    break

                i += 1

    except KeyboardInterrupt:
        print("interrompido pelo teclado")

    finally:
        # ---------- fechar tudo com cuidado ----------
        duracao = time.monotonic() - t0
        cam.release()
        escritor.release()
        cv2.destroyAllWindows()

        meta["fim_wall"] = agora_iso()
        meta["quadros"] = i
        meta["duracao_s"] = round(duracao, 3)
        meta["fps_medido"] = round(i / duracao, 3) if duracao > 0 else 0.0

        (pasta / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print()
        print("CLAQUETE DE FIM: passe a tag 3x agora, antes de sair da cena")
        print()
        print(f"sessao   : {sessao_id}")
        print(f"quadros  : {i}")
        print(f"duracao  : {duracao:.1f}s")
        print(f"fps medido: {meta['fps_medido']}")
        print(f"pasta    : {pasta}")


if __name__ == "__main__":
    main()
