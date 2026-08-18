"""A LEITURA DE AMBIENTE. Roda uma vez, antes do sistema.

    python ferramentas/levantar_ambiente.py            marca e mostra
    python ferramentas/levantar_ambiente.py --gravar   escreve o levantamento

O QUE ELA FAZ, E POR QUE NESTA ORDEM

    1. abre as tres cameras
    2. voce marca, em cada uma, os cantos da estante que aparecem
    3. cada camera resolve a PROPRIA POSE contra o gabarito de trena
    4. com duas ou mais poses, triangula a nuvem de pontos
    5. grava tudo em `loja/levantamento.json`

A ESTANTE MEDE AS CAMERAS, E NAO O CONTRARIO.

Em 18/08 o caminho inverso falhou de um jeito instrutivo: a camera do teto
enxerga a BANDEJA DE CIMA, a 1,90 m, e a homografia so vale para o chao. O
resultado foi uma estante de 1,01 x 0,23 m na diagonal.

    Para uma camera a 2,5 m, um ponto a 1,90 aparece quatro vezes mais longe
    do que esta. Nao era imprecisao: era o plano errado.

Aqui a estante entra pelo lado forte dela — as medidas de trena. Um corpo
rigido de dimensoes conhecidas, visto por uma camera, determina onde essa
camera esta. E com as cameras situadas, as tres passam a viver no mesmo
sistema de coordenadas, que e o que faltava para elas trabalharem juntas.

POR QUE MARCAR A MAO, E POR QUE ISSO NAO E O "PONTO FIXO" DE ANTES

    a posicao dela nao muda, mais podera mudar, entao nao de a ela um ponto
    fixo                                            — Eduardo, 13/08

Marcar nao e digitar coordenada. E o mesmo gesto que `calibracao/homografia.py`
ja pede — clicar quatro cantos no chao — e pela mesma razao: alguem precisa
dizer ao programa QUAL coisa na imagem e a referencia. O que sai do clique nao
e a resposta; e a pergunta bem posta. A resposta (onde a camera esta, onde a
estante esta, o que existe no ambiente) o programa calcula.

E se a estante mudar de lugar, roda-se de novo. Um minuto.

COMO MARCAR

Para cada camera aparece a imagem e o nome do ponto a marcar. Clique nele, ou
aperte ESPACO para pular o que nao aparece nessa vista. Seis pontos ja
resolvem; quanto mais, menor o residuo.

    p3_esq_frente  = canto ESQUERDO da FRENTE da bandeja 3 (0,95 m)
    pe_dir_fundo   = pe DIREITO do FUNDO, no chao

TECLAS
    ESPACO  pula este ponto (nao aparece nesta camera)
    z       desfaz o ultimo
    ENTER   termina esta camera
    ESC     cancela tudo
"""
import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2                                                   # noqa: E402
import numpy as np                                           # noqa: E402

from ferramentas.achar_ambiente import _quadro_estavel       # noqa: E402
from src.app.orquestrador import Orquestrador                # noqa: E402
from src.mundo.ambiente import Gabarito                      # noqa: E402
from src.mundo.levantamento import (Levantamento, nuvem_de,  # noqa: E402
                                    pontos_do_gabarito, resolver_pose)
from src.nucleo import log as logmod                          # noqa: E402

# A ordem em que os pontos sao pedidos. Comeca pelos pes porque sao os que
# ancoram a escala no chao, e segue de baixo para cima — a mesma ordem em que
# o olho percorre uma estante.
def _ordem(modelo):
    pes = [n for n in modelo if n.startswith("pe_")]
    resto = sorted((n for n in modelo if not n.startswith("pe_")),
                   key=lambda n: modelo[n][2])
    return pes + resto


def _marcar(imagem, papel, modelo):
    """Colhe {nome: (u, v)} clicando na imagem. Devolve None se cancelar."""
    marcados, ordem, i = {}, _ordem(modelo), 0
    janela = f"levantamento - {papel}"
    clique = {}

    def ao_clicar(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            clique["p"] = (float(x), float(y))

    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, ao_clicar)

    while True:
        tela = imagem.copy()
        for nome, (u, v) in marcados.items():
            cv2.circle(tela, (int(u), int(v)), 4, (60, 230, 60), -1)
            cv2.putText(tela, nome, (int(u) + 6, int(v) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 230, 60), 1)
        alvo = ordem[i] if i < len(ordem) else "-- pronto --"
        cv2.rectangle(tela, (0, 0), (tela.shape[1], 46), (24, 24, 28), -1)
        cv2.putText(tela, f"{papel}   marque: {alvo}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        cv2.putText(tela, f"{len(marcados)} marcados   "
                          f"ESPACO pula   z desfaz   ENTER termina   ESC sai",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 160), 1)
        cv2.imshow(janela, tela)

        if "p" in clique and i < len(ordem):
            marcados[ordem[i]] = clique.pop("p")
            i += 1
        clique.pop("p", None)

        t = cv2.waitKey(20) & 0xFF
        if t == 27:
            cv2.destroyWindow(janela)
            return None
        if t in (13, 10):
            break
        if t == 32 and i < len(ordem):
            i += 1
        if t == ord("z") and marcados:
            marcados.pop(ordem[i - 1] if i else next(reversed(marcados)), None)
            i = max(0, i - 1)
        if i >= len(ordem) and len(marcados):
            pass

    cv2.destroyWindow(janela)
    return marcados


def main():
    p = argparse.ArgumentParser(description="a leitura de ambiente")
    p.add_argument("--planta", default="loja/quarto.json")
    p.add_argument("--saida", default="loja/levantamento.json")
    p.add_argument("--captura", default="1280x720")
    p.add_argument("--fov", type=float, default=60.0,
                   help="campo de visao horizontal estimado, em graus")
    p.add_argument("--gravar", action="store_true")
    p.add_argument("--espera", type=float, default=30.0)
    p.add_argument("--log", default="WARNING")
    args = p.parse_args()

    logmod.configurar(args.log)
    w, h = (int(v) for v in args.captura.lower().split("x"))

    gab = Gabarito.de_arquivo("loja/estante.json")
    modelo = pontos_do_gabarito(gab)
    print(f"\n  GABARITO  {gab.largura:.2f} x {gab.profundidade:.2f} x "
          f"{gab.altura:.2f} m   {len(modelo)} pontos de referencia")
    print("  A estante mede as cameras. Marque os cantos que voce enxergar.\n")

    app = Orquestrador(planta=args.planta, captura=(w, h), com_pose=False)
    app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    poses, marcacoes = {}, {}
    try:
        for papel in ("alto", "frontal", "lateral"):
            print(f"  --- {papel} ---")
            quadro = _quadro_estavel(app, papel, limite_s=args.espera)
            if quadro is None:
                poses[papel] = None
                continue
            marcados = _marcar(quadro, papel, modelo)
            if marcados is None:
                print("\n  cancelado.\n")
                return
            marcacoes[papel] = marcados
            pose = resolver_pose(papel, marcados, modelo,
                                 (quadro.shape[1], quadro.shape[0]), args.fov)
            poses[papel] = pose
            if pose is None:
                print(f"  {len(marcados)} pontos — nao deu para resolver a pose.")
            else:
                x, y, z = pose.posicao
                print(f"  pose: ({x:+.2f}, {y:+.2f}, {z:+.2f}) m   "
                      f"residuo {pose.residuo_px:.1f} px   "
                      f"{'confiavel' if pose.confiavel else 'RUIM'}")
    finally:
        app.parar()

    lev = Levantamento(poses=poses,
                       medido_em=time.strftime("%Y-%m-%dT%H:%M:%S"))

    print("\n  CAMERAS SITUADAS:", ", ".join(lev.cameras_situadas) or "nenhuma")
    if not lev.pronto:
        print("\n  MENOS DE DUAS CAMERAS SITUADAS.")
        print("  Sem duas nao ha triangulacao, e sem triangulacao nao ha")
        print("  nuvem nem fusao — ha uma camera opinando sozinha.\n")
        return

    # As correspondencias saem das proprias marcacoes: o mesmo canto marcado
    # em duas cameras E um ponto visto por duas cameras.
    comuns = set.intersection(*(set(marcacoes[p]) for p in lev.cameras_situadas))
    correspondencias = [{p: marcacoes[p][nome] for p in lev.cameras_situadas}
                        for nome in sorted(comuns)]
    lev.nuvem = nuvem_de({p: poses[p] for p in lev.cameras_situadas},
                         correspondencias)

    print(f"  NUVEM: {len(lev.nuvem)} pontos de {len(comuns)} correspondencias")
    caixa = lev.nuvem.caixa()
    if caixa:
        b, a = caixa
        print(f"    x {b[0]:+.2f} a {a[0]:+.2f}    "
              f"y {b[1]:+.2f} a {a[1]:+.2f}    z {b[2]:+.2f} a {a[2]:+.2f} m")
        print(f"    altura reconstruida {a[2] - b[2]:.2f} m  "
              f"(trena diz {gab.altura:.2f})")

    if args.gravar:
        lev.gravar(args.saida)
        print(f"\n  gravado em {args.saida}\n")
    else:
        print("\n  (nao gravei — use --gravar)\n")


if __name__ == "__main__":
    main()
