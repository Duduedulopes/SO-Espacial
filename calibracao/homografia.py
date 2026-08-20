"""
Calibracao de homografia — bloco 1 do plano de estudo.

Estabelece a correspondencia entre pixels da imagem e metros no chao.

COMO USAR

1. Marque 4 pontos no chao formando um retangulo. Fita crepe serve.
   Meca com trena. Quanto maior o retangulo, melhor a precisao.

2. Posicione a camera olhando o chao EM ANGULO, como ficaria no teto
   de uma loja. De frente ou de cima a homografia perde a graca.

3. Rode:
       python calibracao/homografia.py --largura-m 1.5 --altura-m 1.0

4. Clique nos 4 cantos NESTA ORDEM:
       1. superior esquerdo
       2. superior direito
       3. inferior direito
       4. inferior esquerdo
   (olhando o retangulo de cima, como no mapa que voce quer produzir)

5. Depois disso, clique em qualquer ponto do chao e ele imprime a
   coordenada em metros. Ponha o pe num lugar conhecido e confira.

TECLAS
    r  recomeca a marcacao
    v  alterna a vista de cima (bird's eye)
    s  salva a homografia em calibracao/homografia.json
    ESC sai

ATENCAO: so pontos SOBRE O CHAO sao mapeados corretamente. A cabeca de
uma pessoa nao esta no chao; os pes estao. E por isso que rastreadores
usam o centro inferior da caixa delimitadora como posicao.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "calibracao" / "homografia.json"

ROTULOS = ["sup. esquerdo", "sup. direito", "inf. direito", "inf. esquerdo"]

pontos_img: list[tuple[int, int]] = []
pontos_ordenados: list[tuple[int, int]] = []
cliques_teste: list[tuple[int, int]] = []

LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ao_clicar(evento, x, y, flags, params):
    if evento != cv2.EVENT_LBUTTONDOWN:
        return
    if len(pontos_img) < 4:
        pontos_img.append((x, y))
    else:
        cliques_teste.append((x, y))


def ordenar_cantos(pts):
    """Reordena 4 pontos clicados numa volta coerente, sem cruzar.

    O QUE IMPORTA DE VERDADE para a homografia e que os 4 pontos sejam
    percorridos em volta do quadrilatero, sem pular de um canto para o oposto.
    Se a ordem cruzar, o retangulo vira uma gravata borboleta e a transformacao
    sai sem sentido.

    Metodo: ordenar pelo angulo em torno do centro. Funciona para QUALQUER
    rotacao — inclusive piso com ladrilho na diagonal, que quebrou a versao
    anterior baseada em "menor x+y e o canto superior esquerdo".

    Depois giramos a lista para comecar sempre pelo ponto mais acima e a
    esquerda, so para haver uma convencao estavel entre calibracoes.
    """
    p = np.array(pts, dtype=np.float64)

    centro = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - centro[1], p[:, 0] - centro[0])
    p = p[np.argsort(ang)]

    # area pelo teorema do cadarco: se for ~0, os pontos sao colineares
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 100:  # menos de 100 px^2 nao e um quadrilatero utilizavel
        return None

    inicio = int(np.argmin(p[:, 0] + p[:, 1]))
    p = np.roll(p, -inicio, axis=0)

    return [(int(a), int(b)) for a, b in p]


def pixel_para_metro(H, x, y) -> tuple[float, float]:
    """Aplica a homografia a um ponto.

    O vetor vai como (x, y, 1) — coordenadas homogeneas. No fim dividimos
    por w, e e essa divisao que produz o efeito de perspectiva.
    """
    v = H @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def abrir(papel=None, indice=None):
    """Abre a camera do papel, seja ela USB ou de REDE. Devolve (cap, fonte).

    DOIS MOTIVOS PARA ISTO NAO SER UM `VideoCapture(indice)`.

    1. INDICE MUDA SOZINHO. E a licao que o projeto ja pagou com o
       DirectShow, e esta escrita no requirements.txt. Com tres cameras
       ligadas, calibrar a errada nao da erro: da uma homografia plausivel
       de outro ponto de vista.

    2. A LATERAL NAO E USB. Ela e um tablet em `http://.../video`, e
       `CAP_DSHOW` nao abre URL. O programa foi escrito quando so havia a
       camera do teto, e a suposicao ficou embutida no unico caminho.

           Um programa que so foi usado com um caso nao esta certo para
           aquele caso: esta sem ter sido contrariado.
    """
    import json as _json
    import sys as _sys

    if indice is not None:
        return cv2.VideoCapture(indice, cv2.CAP_DSHOW), f"indice {indice}"

    if str(RAIZ) not in _sys.path:
        _sys.path.insert(0, str(RAIZ))
    config = RAIZ / "config" / "cameras.json"
    if not config.exists():
        raise SystemExit(f"\n  nao achei {config}\n")
    d = _json.loads(config.read_text(encoding="utf-8"))
    if papel not in d:
        raise SystemExit(f"\n  nao ha papel '{papel}' em config/cameras.json. "
                         f"Ha: {', '.join(d)}\n")
    fonte = d[papel]["fonte"]

    if str(fonte).startswith("http"):
        return cv2.VideoCapture(fonte), fonte

    from src.cameras.dispositivos import exigir_indice
    return cv2.VideoCapture(exigir_indice(fonte), cv2.CAP_DSHOW), fonte


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--papel", default="alto",
                   help="qual camera, pelo NOME em config/cameras.json")
    p.add_argument("--camera", type=int, default=None,
                   help="indice cru, se voce souber o que esta fazendo")
    p.add_argument("--largura-m", type=float, required=True, help="lado horizontal do retangulo, em metros")
    p.add_argument("--altura-m", type=float, required=True, help="lado vertical do retangulo, em metros")
    p.add_argument("--origem", type=float, nargs=2, default=(0.0, 0.0),
                   metavar=("X", "Y"),
                   help="onde o PRIMEIRO canto do retangulo esta no mundo, em "
                        "metros, medido com trena a partir da origem da fita. "
                        "Padrao 0 0 — a propria origem.")
    p.add_argument("--px-por-m", type=int, default=200, help="escala da vista de cima")
    p.add_argument(
        "--vista-escala",
        type=float,
        default=1.0,
        help="quantas vezes a area do retangulo a vista de cima deve cobrir. "
        "Ex.: 6 mostra uma regiao 6x maior, centrada no retangulo.",
    )
    args = p.parse_args()

    # Cantos do retangulo no MUNDO, em metros, na mesma ordem dos cliques.
    #
    # O RETANGULO NAO PRECISA ESTAR NA ORIGEM, E PARA A SEGUNDA CAMERA ELE
    # NAO PODE ESTAR.
    #
    # A camera lateral nao enxerga a fita do retangulo de 1,65 x 1,32 que
    # calibrou a do teto. Sem `--origem` a unica saida seria marcar um
    # retangulo novo e chamar aquele canto de (0,0) — e ai as duas cameras
    # teriam mundos DIFERENTES, e a fusao viraria um problema de alinhamento.
    #
    # Com a origem declarada, o segundo retangulo e medido com trena a
    # partir da MESMA marca de fita, e as duas homografias caem no mesmo
    # sistema de coordenadas por construcao. Nao ha o que alinhar depois.
    #
    #     Dois instrumentos so concordam de graca quando foram referidos ao
    #     mesmo zero. Referi-los depois custa uma etapa que pode falhar.
    ox, oy = args.origem
    pontos_mundo = np.array(
        [
            [ox, oy],
            [ox + args.largura_m, oy],
            [ox + args.largura_m, oy + args.altura_m],
            [ox, oy + args.altura_m],
        ],
        dtype=np.float32,
    )

    cam, fonte = abrir(args.papel, args.camera)
    print(f"\n  calibrando '{fonte}'")
    print(f"  retangulo de {args.largura_m:.2f} x {args.altura_m:.2f} m, "
          f"primeiro canto em ({ox:+.2f}, {oy:+.2f})")
    print(f"  vai salvar em calibracao/homografia-{args.papel}.json\n")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE, -6)

    if not cam.isOpened():
        raise SystemExit("nao consegui abrir a camera")

    janela = "calibracao - clique os 4 cantos"
    cv2.namedWindow(janela)
    cv2.setMouseCallback(janela, ao_clicar)

    H = None
    ver_de_cima = False
    # A RESOLUCAO REAL, E NAO A PEDIDA.
    #
    # `cam.set(FRAME_WIDTH, 640)` e um PEDIDO; a camera responde o que quiser,
    # e num stream http ele nao faz nada — o tablet manda o que ele decidiu.
    # Ate 20/08 o arquivo gravava `[640, 480]` cravado.
    #
    # Se o aparelho entregasse 1280x720, a homografia estaria certa nos pixels
    # reais e o arquivo diria outra coisa — e tudo que reescala depois
    # (`motor._ajustar_escala`, `intrinseca_medida`) erraria por um fator de
    # dois, sem sintoma nenhum alem de numeros errados.
    #
    #     Resolucao pedida nao e resolucao obtida, e gravar a pedida e
    #     documentar uma intencao como se fosse uma medida.
    resolucao = None

    print(__doc__)

    while True:
        ok, frame = cam.read()
        if ok and frame is not None and resolucao is None:
            resolucao = [int(frame.shape[1]), int(frame.shape[0])]
            print(f"  a camera entregou {resolucao[0]}x{resolucao[1]}")
        if not ok:
            break

        vista = frame.copy()

        # desenha os pontos ja marcados
        for i, (x, y) in enumerate(pontos_img):
            cv2.circle(vista, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(vista, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # liga os pontos
        if len(pontos_img) >= 2:
            cv2.polylines(vista, [np.array(pontos_img)], len(pontos_img) == 4,
                          (0, 255, 255), 2)

        # calcula a homografia assim que houver 4 pontos
        if len(pontos_img) == 4 and H is None:
            ordenados = ordenar_cantos(pontos_img)
            if ordenados is None:
                print("\npontos degenerados (colineares?). Aperte r e refaca.")
                pontos_img.clear()
            else:
                pontos_ordenados = ordenados
                H, _ = cv2.findHomography(
                    np.array(pontos_ordenados, dtype=np.float32), pontos_mundo
                )
                print("\nordem detectada automaticamente:")
                for r, pt in zip(ROTULOS, pontos_ordenados):
                    print(f"  {r:15} {pt}")
                print("\nhomografia calculada:")
                print(H)
                print("\nagora clique em qualquer ponto do chao para ver a coordenada")

        # instrucao ou leitura
        if len(pontos_img) < 4:
            texto = f"clique os 4 cantos (ordem livre) - faltam {4 - len(pontos_img)}"
        else:
            texto = f"pontos de teste: {len(cliques_teste)}   [d] apaga  [l] salva"

        # desenha TODOS os pontos de teste, cada um com sua letra e leitura
        if H is not None:
            for k, (tx, ty) in enumerate(cliques_teste):
                mx, my = pixel_para_metro(H, tx, ty)
                cv2.circle(vista, (tx, ty), 7, (0, 0, 255), -1)
                cv2.circle(vista, (tx, ty), 7, (255, 255, 255), 1)
                rotulo = f"{LETRAS[k % 26]} {mx:.2f},{my:.2f}"
                cv2.putText(vista, rotulo, (tx + 11, ty + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(vista, rotulo, (tx + 11, ty + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(vista, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow(janela, vista)

        # vista de cima: o chao "desentortado"
        if ver_de_cima and H is not None:
            k = args.vista_escala
            ppm = args.px_por_m

            # Regiao do mundo que a vista vai cobrir, em metros.
            # Com k=1 e exatamente o retangulo; com k>1 ela cresce ao redor,
            # mantendo o retangulo no centro.
            largura_vista = args.largura_m * k
            altura_vista = args.altura_m * k
            x0 = args.largura_m * (1 - k) / 2
            y0 = args.altura_m * (1 - k) / 2

            # metro -> pixel da vista:  (wx - x0) * ppm
            mundo_para_vista = np.array(
                [[ppm, 0, -x0 * ppm], [0, ppm, -y0 * ppm], [0, 0, 1]],
                dtype=np.float64,
            )

            tamanho = (int(largura_vista * ppm), int(altura_vista * ppm))
            topo = cv2.warpPerspective(frame, mundo_para_vista @ H, tamanho)

            # grade de 10 cm, para dar referencia visual
            passo = int(0.10 * ppm)
            if passo > 4:
                for gx in range(0, tamanho[0], passo):
                    cv2.line(topo, (gx, 0), (gx, tamanho[1]), (60, 60, 60), 1)
                for gy in range(0, tamanho[1], passo):
                    cv2.line(topo, (0, gy), (tamanho[0], gy), (60, 60, 60), 1)

            # contorno do retangulo calibrado, em amarelo
            cantos = np.array(
                [
                    [(-x0) * ppm, (-y0) * ppm],
                    [(args.largura_m - x0) * ppm, (-y0) * ppm],
                    [(args.largura_m - x0) * ppm, (args.altura_m - y0) * ppm],
                    [(-x0) * ppm, (args.altura_m - y0) * ppm],
                ],
                dtype=np.int32,
            )
            cv2.polylines(topo, [cantos], True, (0, 255, 255), 2)

            cv2.imshow("vista de cima", topo)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            break
        elif tecla == ord("r"):
            pontos_img.clear()
            cliques_teste.clear()
            H = None
            print("marcacao reiniciada")
        elif tecla == ord("d"):
            cliques_teste.clear()
            print("pontos de teste apagados")
        elif tecla == ord("l") and H is not None:
            destino = RAIZ / "calibracao" / "pontos_teste.json"
            registros = []
            print()
            print("pontos de teste:")
            for k, (tx, ty) in enumerate(cliques_teste):
                mx, my = pixel_para_metro(H, tx, ty)
                letra = LETRAS[k % 26]
                print(f"  {letra}  pixel ({tx:4d},{ty:4d})  ->  ({mx:8.4f}, {my:8.4f}) m")
                registros.append({"letra": letra, "pixel": [tx, ty],
                                  "metros": [round(mx, 4), round(my, 4)]})
            destino.write_text(json.dumps(registros, indent=2), encoding="utf-8")
            print(f"salvo em {destino}")
        elif tecla == ord("v"):
            ver_de_cima = not ver_de_cima
            if not ver_de_cima:
                cv2.destroyWindow("vista de cima")
        elif tecla == ord("s") and H is not None:
            # UM ARQUIVO POR PAPEL, E O DA CAMERA DO ALTO CONTINUA ONDE
            # ESTAVA. Trocar o nome do arquivo da alto quebraria o
            # `rodar.py`, o `mapear.py` e o `--mono` de uma vez — e nao ha
            # nada de errado com ele.
            destinos = [RAIZ / "calibracao" / f"homografia-{args.papel}.json"]
            if args.papel == "alto":
                destinos.append(SAIDA)

            corpo = json.dumps(
                {
                    "H": H.tolist(),
                    "papel": args.papel,
                    "fonte": fonte,
                    "pontos_imagem_px": pontos_ordenados,
                    "pontos_clicados_px": pontos_img,
                    "pontos_mundo_m": pontos_mundo.tolist(),
                    "origem_m": [float(ox), float(oy)],
                    "largura_m": args.largura_m,
                    "altura_m": args.altura_m,
                    "resolucao": resolucao or [640, 480],
                    "_nota": [
                        "O retangulo NAO precisa estar na origem.",
                        "",
                        "Este foi medido com trena a partir da MESMA marca de",
                        "fita que calibrou a camera do alto — por isso as",
                        "homografias caem no mesmo sistema de coordenadas por",
                        "construcao, e nao ha o que alinhar depois.",
                        "",
                        "    Dois instrumentos so concordam de graca quando",
                        "    foram referidos ao mesmo zero.",
                    ],
                },
                indent=2, ensure_ascii=False,
            )
            for destino in destinos:
                destino.write_text(corpo, encoding="utf-8")
                print(f"salvo em {destino}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
