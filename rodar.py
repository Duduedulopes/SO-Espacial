"""
SO Espacial — ponto de entrada.

    python rodar.py                      camaras reais + cena 3D
    python rodar.py --sem-janela         so terminal, mais leve
    python rodar.py --deteccao-a-cada 2  quase dobra o fps
    python rodar.py --sem-pose           so posicao, sem esqueleto
    python rodar.py --falsas             SEM HARDWARE, fontes sinteticas

Substitui `percepcao/gemeo_multi.py`, que tinha 317 linhas e nove
responsabilidades. Aqui o laco tem cinco chamadas, e cada etapa e um objeto
com teste proprio.

`--falsas` e a UNICA porta para simulacao. A execucao normal abre hardware.
"""

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.app.orquestrador import Orquestrador       # noqa: E402
from src.nucleo import log as logmod                # noqa: E402


def main():
    p = argparse.ArgumentParser(description="SO Espacial — gemeo digital")
    p.add_argument("--planta", default="loja/bancada.json")
    p.add_argument("--captura", default="1280x720")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--deteccao-a-cada", type=int, default=1)
    p.add_argument("--pose-a-cada", type=int, default=1)
    p.add_argument("--lado", choices=("direita", "esquerda"), default="direita")
    p.add_argument("--tolerancia-ms", type=float, default=150)
    p.add_argument("--meia-vida", type=float, default=90.0)
    p.add_argument("--exposicao", type=float, default=None)
    p.add_argument("--sem-pose", action="store_true")
    p.add_argument("--salvar-quadros", type=float, default=0,
                   metavar="SEGUNDOS",
                   help="grava um quadro de cada camera a cada N segundos em "
                        "dados/quadros/ — para ver o que o PROGRAMA recebe")
    p.add_argument("--sem-plausibilidade", action="store_true",
                   help="desliga o filtro de altura — para COMPARAR, nao para "
                        "usar: sem ele, mobilia volta a virar pessoa")
    p.add_argument("--sem-janela", action="store_true")
    p.add_argument("--falsas", action="store_true",
                   help="fontes sinteticas — sem hardware")
    p.add_argument("--segundos", type=float, default=0)
    p.add_argument("--log", default="INFO")
    args = p.parse_args()

    logmod.configurar(args.log)
    w, h = (int(v) for v in args.captura.lower().split("x"))

    app = Orquestrador(
        planta=args.planta, captura=(w, h), imgsz=args.imgsz, conf=args.conf,
        deteccao_a_cada=args.deteccao_a_cada, pose_a_cada=args.pose_a_cada,
        lado_lateral=args.lado, tolerancia_ms=args.tolerancia_ms,
        meia_vida_calor=args.meia_vida, exposicao=args.exposicao,
        com_pose=not args.sem_pose,
        usar_plausibilidade=not args.sem_plausibilidade,
        salvar_quadros_s=args.salvar_quadros)

    if args.falsas:
        app.montar_cameras_falsas()
    else:
        app.montar_cameras_reais()

    app.montar_visao()
    app.iniciar()

    cena = None
    if not args.sem_janela:
        import cv2
        from visual.cena3d import Cena3D, Esqueleto
        cena = Cena3D(960, 620, chao=app.planta.chao, calor_hz=4.0)
        app.planta.aplicar_na_cena(cena)

    t0 = time.monotonic()
    ultimo_painel = 0.0
    try:
        while True:
            if args.segundos and time.monotonic() - t0 > args.segundos:
                break

            instante = app.passo()
            if instante is None:
                time.sleep(0.005)
                continue

            agora = time.monotonic()
            if agora - ultimo_painel > 0.5:
                ultimo_painel = agora
                print("\033[H\033[J", end="")
                print("\n".join(app.painel()))

            if cena is not None:
                esqueletos = [
                    Esqueleto(id=p.id, juntas=p.esqueleto,
                              visivel=p.juntas_visiveis,
                              prevendo=bool(p.prevendo),
                              historico=list(app.gemeo.trilhas.get(p.id, ())))
                    for p in app.gemeo.pessoas.values() if p.tem_esqueleto
                ]
                # Quem nao tem esqueleto vira PINO, nao desaparece. Com duas
                # pessoas em cena a associacao deixa de ser confiavel e nenhum
                # esqueleto e montado — e a janela ficava vazia enquanto o
                # sistema seguia dois rastros.
                marcadores = [(p.id, p.x, p.y, bool(p.prevendo))
                              for p in app.gemeo.pessoas.values()
                              if not p.tem_esqueleto]
                titulo = (f"q{app.quadros}  {len(app.gemeo.pessoas)}p  "
                          f"defasagem {instante.defasagem_ms:.0f}ms")
                cv2.imshow("gemeo 3D", cena.desenhar(
                    esqueletos, titulo, calor=app.gemeo.calor,
                    zonas=app.planta.zonas, marcadores=marcadores))

                q = instante.get("alto")
                if q is not None:
                    cv2.imshow("alto", cv2.resize(q.imagem, None,
                                                  fx=0.45, fy=0.45))

                k = cv2.waitKeyEx(1) & 0xFFFFFF
                if not cena.tecla(k):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        app.parar()
        if cena is not None:
            import cv2
            cv2.destroyAllWindows()

    print("\n" + "\n".join(app.painel()))
    print("\nEVENTOS POR TIPO")
    for tipo, n in app.eventos.resumo().items():
        print(f"  {tipo:24} {n}")


if __name__ == "__main__":
    main()
