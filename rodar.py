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

# Ciclos de respiracao por segundo. Nao ha mais passada aqui: as pernas
# pararam de andar em 12/08 (ver `src/gemeo/boneco.py`, FLUTUACAO).
RESPIROS_POR_SEGUNDO = 0.30

# COMO CADA CAMERA APARECE, E O QUE ELA ENTREGA.
#
#     nao sei se as 3 cameras estao trabalhando juntas para isso, pq quando a
#     simulacao comeca eu so vejo a camera superior    — Eduardo, 12/08
#
# As tres SEMPRE alimentaram a fusao — `motor.py` junta as vistas antes de
# decidir qualquer coisa. Mas so a de cima era exibida, e um sistema que nao
# mostra o que usa esta pedindo para nao ser acreditado. Uma afirmacao minha
# de que "elas trabalham juntas" nao vale nada; tres janelas abertas valem.
#
# Cada uma leva escrito o que E DELA na resposta final. Nao e legenda: e a
# unica maneira de olhar para a tela e ver a complementaridade acontecendo.
PAINEL_DAS_CAMERAS = (
    ("alto",    "ALTO — posicao no chao, rumo, estatura"),
    ("frontal", "FRONTAL — bracos, altura da mao"),
    ("lateral", "LATERAL — profundidade do alcance"),
)


def main():
    p = argparse.ArgumentParser(description="SO Espacial — gemeo digital")
    # O QUARTO REAL, e nao a loja ficticia. Trocado em 14/08.
    #
    # `bancada.json` descreve duas gondolas e um checkout que nao existem em
    # lugar nenhum. Enquanto ele foi o padrao, todo teste com camera mostrava
    # o gemeo andando dentro de uma loja inventada — e a estante de verdade,
    # que esta a um metro da pessoa, nao aparecia.
    #
    #     Um cenario de demonstracao que nao e o cenario medido nao ilustra o
    #     sistema: ilustra outro sistema.
    #
    # `--planta loja/bancada.json` continua disponivel para ensaiar uma loja
    # maior que o quarto.
    p.add_argument("--planta", default="loja/quarto.json")
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
    suavizador = None
    if not args.sem_janela:
        import cv2
        from visual.cena3d import Cena3D, Esqueleto
        from src.gemeo import boneco
        from src.gemeo.suave import Suavizador
        from src.acao.vocabulario import Locomocao
        cena = Cena3D(960, 620, chao=app.planta.chao, calor_hz=4.0,
                      contorno=app.planta.contorno)
        suavizador = Suavizador()
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
                # O BONECO E ANIMADO PELA DESCRICAO, NAO PELAS JUNTAS CRUAS.
                #
                # Ate 12/08 esta lista usava `p.esqueleto` — a reconstrucao do
                # MediaPipe, com todos os defeitos que dois dias de medicao
                # tinham isolado: braco comprimido, junta extrapolada fora do
                # quadro, ancora errada. O vocabulario fechado existia na
                # camada que LE e nao existia na camada que DESENHA.
                #
                #     Se "deitado" nao esta no vocabulario, o boneco nao
                #     consegue deitar.        — vocabulario.py, 10/08
                #
                # Agora `src/gemeo/boneco.py` MONTA as 17 juntas a partir da
                # acao e da antropometria. Nenhuma coordenada de junta
                # atravessa esta fronteira, que e exatamente o que a `Acao`
                # promete no proprio docstring dela.
                fase = (agora * RESPIROS_POR_SEGUNDO) % 1.0
                esqueletos = []
                for p in app.gemeo.pessoas.values():
                    item = app.espacial.acoes.get(p.id)
                    if item is None:
                        continue
                    leitura = app.espacial.leituras.get(p.id)
                    # O filtro fica AQUI, do lado de quem desenha, e nao no
                    # Kalman: a logica precisa da medida crua e rapida, o olho
                    # precisa dela lisa. Ver `src/gemeo/suave.py`.
                    x, y, rumo = suavizador.suavizar(
                        p.id, p.x, p.y, getattr(leitura, "rumo_corpo", None))
                    esqueletos.append(Esqueleto(
                        id=p.id,
                        juntas=boneco.montar(
                            estatura=app.espacial.escala.estatura(p.id),
                            x=x, y=y,
                            rumo=(rumo or 0.0),
                            postura=item[0].postura,
                            locomocao=item[0].locomocao,
                            braco_esq=item[0].braco_esquerdo,
                            braco_dir=item[0].braco_direito,
                            altura_mao_esq=item[0].altura_mao_esq,
                            altura_mao_dir=item[0].altura_mao_dir,
                            fase=fase),
                        visivel=None,
                        prevendo=bool(p.prevendo),
                        rumo=rumo,
                        andando=item[0].locomocao not in (
                            Locomocao.PARADO, Locomocao.DESCONHECIDA),
                        historico=list(app.gemeo.trilhas.get(p.id, ()))))
                suavizador.esquecer(set(app.gemeo.pessoas))

                # Quem ainda nao tem ACAO vira pino, nao desaparece. Antes o
                # criterio era `tem_esqueleto`; agora e ter descricao, que e o
                # que o desenho passou a consumir.
                com_acao = set(app.espacial.acoes)
                marcadores = [(p.id, p.x, p.y, bool(p.prevendo))
                              for p in app.gemeo.pessoas.values()
                              if p.id not in com_acao]
                # DE QUAL PRATELEIRA A MAO VEIO, NO TITULO DA CENA.
                #
                # O palpite tem que ficar ao lado do boneco, no mesmo instante
                # em que ele se mexe. Em painel separado, a correspondencia
                # entre o gesto e a resposta depende da memoria de quem olha —
                # foi a mesma razao pela qual o mosaico das cameras nasceu.
                palpites = [f"#{p} {v.prateleira}"
                            f"{'' if v.firme else '?'}"
                            for p, v in sorted(app.espacial.palpites.items())
                            if v]
                titulo = (f"q{app.quadros}  {len(app.gemeo.pessoas)}p  "
                          f"defasagem {instante.defasagem_ms:.0f}ms"
                          + ("   " + "  ".join(palpites) if palpites else ""))
                cv2.imshow("gemeo 3D", cena.desenhar(
                    esqueletos, titulo, calor=app.gemeo.calor,
                    zonas=app.planta.zonas, marcadores=marcadores))

                for papel, entrega in PAINEL_DAS_CAMERAS:
                    q = instante.get(papel)
                    if q is None:
                        continue
                    vista = cv2.resize(q.imagem, None, fx=0.45, fy=0.45)
                    cv2.putText(vista, entrega, (8, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(vista, entrega, (8, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow(papel, vista)

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
