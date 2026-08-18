"""Reconhece a estante com as cameras reais e grava na planta.

    python ferramentas/achar_ambiente.py                  olha e mostra
    python ferramentas/achar_ambiente.py --gravar         escreve no quarto.json
    python ferramentas/achar_ambiente.py --ver            + janela com o achado

RODA UMA VEZ, NAO A CADA QUADRO. E ISSO E A DECISAO, NAO UM DETALHE.

A estante nao se mexe. Procura-la sessenta vezes por segundo seria gastar o
orcamento de CPU — que ja esta apertado a 9,7 fps — para confirmar o obvio. O
movel e medido quando alguem manda medir, o resultado vai para
`loja/quarto.json`, e o laco principal so LE.

    O que nao muda deve ser medido uma vez e escrito. O que muda deve ser
    medido sempre. Confundir os dois custa fps de um lado ou verdade do outro.

Se a estante for movida, roda-se de novo. Sao dez segundos.

O QUE ESTA FERRAMENTA FAZ, EM ORDEM

    1. abre as tres cameras conforme `config/cameras.json`
    2. colhe alguns quadros e fica com a MEDIANA — um quadro solto pode ter
       uma sombra, um reflexo, alguem passando na frente
    3. acha os retangulos vistos de cima, ja em metros (a homografia)
    4. compara cada um com o gabarito de trena (`loja/estante.json`)
    5. escreve o vencedor em `loja/quarto.json`, se --gravar

UMA LIMITACAO DECLARADA, PORQUE ELA E REAL

A confirmacao pela frontal e pela lateral exige converter y de pixel em altura
de metro NAQUELAS cameras — e a escala vertical de hoje foi construida para a
camera do alto, medindo estatura. Enquanto essa conversao nao existir por
camera, esta ferramenta reconhece com a do alto sozinha e DIZ que foi uma so.

    Reconhecer com uma camera e arriscar. O programa que arrisca deve dizer
    que arriscou.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np                                          # noqa: E402

from percepcao.chao import carregar_homografia              # noqa: E402
from src.app.orquestrador import Orquestrador               # noqa: E402
from src.mundo.ambiente import Gabarito, reconhecer         # noqa: E402
from src.mundo.detectores import candidatos_do_alto         # noqa: E402
from src.nucleo import log as logmod                        # noqa: E402


def _quadro_estavel(app, papel, n=9, espera=0.12, limite_s=30.0):
    """A mediana de n quadros. Sombra, reflexo e quem passa somem na mediana.

    ESPERA POR TEMPO, NAO POR NUMERO DE VOLTAS. Consertado em 18/08.

    Antes este laco dava `n * 3` voltas de 0,12 s — 3,2 segundos no total — e
    desistia. So que a C920 do teto leva perto de NOVE segundos entre o
    programa registrar a camera e o evento CAMERA_CONNECTED aparecer: abrir o
    dispositivo, negociar resolucao e o auto-exposicao assentar.

        [09:44:55] cameras   registrada  papel=alto
        [09:45:04] EVENTO    CAMERA_CONNECTED  papel=alto

    A ferramenta desistia aos 3,2 s e imprimia "a camera do alto nao entregou
    quadro" — uma frase verdadeira sobre um fato que ainda nao tinha
    acontecido. Rodar duas vezes seguidas "funcionava" porque a segunda pegava
    a camera ja quente, o que fazia o defeito parecer capricho do hardware.

        Um limite de tentativas e um limite de tempo disfarcado, e o disfarce
        cai no dia em que o hardware demora. Se o que se espera e tempo,
        espere tempo.

    E enquanto espera, DIZ que esta esperando: silencio de trinta segundos e
    indistinguivel de travamento.
    """
    pilha = []
    comeco = time.monotonic()
    avisado = False
    while time.monotonic() - comeco < limite_s:
        instante = app.passo()
        q = instante.get(papel) if instante else None
        if q is not None and q.imagem is not None:
            pilha.append(q.imagem.astype(np.float32))
            if len(pilha) >= n:
                break
        else:
            decorrido = time.monotonic() - comeco
            if decorrido > 2.0 and not avisado:
                print(f"  esperando a camera '{papel}' acordar "
                      f"(ate {limite_s:.0f}s)...", flush=True)
                avisado = True
        time.sleep(espera)

    if not pilha:
        print(f"  a camera '{papel}' nao entregou quadro em "
              f"{time.monotonic() - comeco:.0f}s.")
        return None
    if len(pilha) < n:
        print(f"  atencao: mediana de {len(pilha)} quadros, e nao de {n}.")
    return np.median(np.stack(pilha), axis=0).astype(np.uint8)


def _extrapolado(x, y, calib, margem=0.05):
    """Este ponto esta FORA do retangulo em que a homografia foi ajustada?

    A homografia foi resolvida a partir de quatro cantos de um retangulo de
    1,65 x 1,32 m marcado no chao. Dentro dele, ela interpola entre medidas
    reais. Fora dele, ela EXTRAPOLA — e extrapolacao projetiva nao degrada
    devagar: o erro cresce rapido e de forma torta, esticando um eixo e
    encolhendo o outro conforme se aproxima da linha do horizonte.

        Uma calibracao mede o que ela viu. Perguntar a ela sobre o lado de
        fora nao devolve um numero pior: devolve um numero com outra regra.

    Por isso a resposta aqui nao e recusar — e AVISAR. Quem esta olhando a
    cena consegue julgar; um programa que apaga a medida sem explicar, nao.
    """
    lx = float(calib.get("largura_m") or 0.0)
    ly = float(calib.get("altura_m") or 0.0)
    if lx <= 0 or ly <= 0:
        return False
    return not (-margem <= x <= lx + margem and -margem <= y <= ly + margem)


def _portas(ambiente, chao, largura=0.45, fundura=0.55, folga=0.12):
    """A entrada e a saida, DEDUZIDAS da estante — nao digitadas.

        quero que ao lado da prateleira seja a entrada e a saida
                                                    — Eduardo, 13/08

    "Ao lado" so quer dizer alguma coisa depois que se sabe onde a estante
    esta e para onde ela olha. Por isso estas duas zonas nascem aqui, no
    mesmo instante em que o movel e medido, e nao num arquivo escrito a mao:
    se a estante mudar de lugar, elas mudam junto.

        Uma zona digitada envelhece calada no dia em que o movel se mexe.

    LIMITACAO DECLARADA: `Zona` e um retangulo alinhado com os eixos, e a
    estante pode estar girada. Entao o que sai daqui e a caixa alinhada em
    volta do ponto certo — a posicao e deduzida, o formato e aproximado.
    Para contar quem entrou e quem saiu isso basta; para medir area, nao.
    """
    ao_longo = np.array([np.cos(ambiente.rumo_da_face),
                         np.sin(ambiente.rumo_da_face)])
    centro = np.array([ambiente.x, ambiente.y])
    # meia largura da estante + meia largura da porta + uma folga entre elas
    afastamento = ambiente.largura / 2.0 + largura / 2.0 + folga
    # meio passo a frente da face, para a porta ficar no lado por onde se anda
    adiante = ambiente.normal * (fundura / 2.0)

    xmin, xmax, ymin, ymax = chao
    zonas = []
    for nome, ident, lado in (("Entrada", "entrada", -1.0),
                              ("Saida", "saida", +1.0)):
        c = centro + ao_longo * (afastamento * lado) + adiante
        z = {"id": ident, "nome": nome,
             "x0": round(float(np.clip(c[0] - largura / 2, xmin, xmax)), 3),
             "x1": round(float(np.clip(c[0] + largura / 2, xmin, xmax)), 3),
             "y0": round(float(np.clip(c[1] - fundura / 2, ymin, ymax)), 3),
             "y1": round(float(np.clip(c[1] + fundura / 2, ymin, ymax)), 3),
             "movel": "estante-aco"}
        # Zona que o clip esmagou nao e zona: e um risco no chao. Melhor nao
        # existir do que existir com area zero e nunca acusar ninguem.
        if z["x1"] - z["x0"] < 0.15 or z["y1"] - z["y0"] < 0.15:
            print(f"  {nome} cairia fora do chao calibrado — nao gravei.")
            continue
        zonas.append(z)
    return zonas


def _gravar(ambiente, caminho="loja/quarto.json"):
    """Escreve o movel achado na planta, substituindo o anterior."""
    p = Path(caminho)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["moveis"] = [m for m in d.get("moveis", []) if m.get("id") != "estante-aco"]
    d["moveis"].append({
        "id": "estante-aco",
        "nome": "Estante",
        "tipo": "estante",
        "x": round(ambiente.x, 3), "y": round(ambiente.y, 3),
        "largura": round(ambiente.largura, 3),
        "profundidade": round(ambiente.profundidade, 3),
        "altura": round(ambiente.altura, 3),
        "rumo_da_face": round(float(ambiente.rumo_da_face), 4),
        "prateleiras": [{"id": i, "altura": round(float(a), 3)}
                        for i, a in ambiente.prateleiras],
        "estante": "estante-aco-teste",
        "_medido_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_por": list(ambiente.cameras),
        "_nota": [
            "ACHADO PELAS CAMERAS, nao digitado. Ver ferramentas/achar_ambiente.py.",
            "As dimensoes vem do gabarito de trena (loja/estante.json); o que as",
            "cameras acrescentam e ONDE ele esta e para onde a face olha.",
            "",
            "Se a estante for movida, rode a ferramenta de novo. Editar este",
            "bloco a mao funciona e desfaz o motivo de ele existir."
        ]})

    c = d["chao"]
    d["zonas"] = [z for z in d.get("zonas", [])
                  if z.get("id") not in ("entrada", "saida")]
    d["zonas"] += _portas(ambiente,
                          (c["xmin"], c["xmax"], c["ymin"], c["ymax"]))

    d.pop("_a_medir", None)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="reconhece a estante com as cameras")
    p.add_argument("--planta", default="loja/quarto.json")
    p.add_argument("--captura", default="1280x720")
    p.add_argument("--gravar", action="store_true")
    p.add_argument("--ver", action="store_true")
    p.add_argument("--falsas", action="store_true")
    p.add_argument("--espera", type=float, default=30.0,
                   help="segundos a esperar a camera do alto acordar")
    p.add_argument("--log", default="WARNING")
    args = p.parse_args()

    logmod.configurar(args.log)
    w, h = (int(v) for v in args.captura.lower().split("x"))

    gab = Gabarito.de_arquivo("loja/estante.json")
    print(f"\n  GABARITO  {gab.largura:.2f} x {gab.profundidade:.2f} x "
          f"{gab.altura:.2f} m   ({len(gab.prateleiras)} prateleiras)")

    # `carregar_homografia` devolve (matriz, dicionario) e levanta SystemExit
    # quando o arquivo nao existe — entao a falta de calibracao ja se explica
    # sozinha, com o comando para consertar. Nao ha o que tratar aqui.
    H, calib = carregar_homografia()
    print(f"  AREA CALIBRADA  {calib.get('largura_m')} x "
          f"{calib.get('altura_m')} m   origem em (0,0)")

    app = Orquestrador(planta=args.planta, captura=(w, h), com_pose=False)
    if args.falsas:
        app.montar_cameras_falsas()
    else:
        app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    try:
        print("  olhando o ambiente...")
        alto = _quadro_estavel(app, "alto", limite_s=args.espera)
        if alto is None:
            print("  Confira se a C920 esta ligada e se nenhum outro programa")
            print("  esta com ela aberta. Para ver o que o sistema enxerga:")
            print("      python ferramentas/cameras.py\n")
            return

        candidatos = candidatos_do_alto(alto, H)
        print(f"\n  {len(candidatos)} retangulo(s) medido(s) no chao:")
        for c in sorted(candidatos, key=lambda v: -v.lado_maior * v.lado_menor):
            cabe = (gab.cabe(c.lado_maior, gab.largura)
                    and gab.cabe(c.lado_menor, gab.profundidade))
            marca = "  <-- cabe no gabarito" if cabe else ""
            if _extrapolado(c.centro[0], c.centro[1], calib):
                marca += "  [FORA DA AREA CALIBRADA]"
            print(f"    {c.lado_maior:.2f} x {c.lado_menor:.2f} m  "
                  f"em ({c.centro[0]:+.2f}, {c.centro[1]:+.2f}){marca}")

        achado = None
        for c in sorted(candidatos, key=lambda v: -v.lado_maior * v.lado_menor):
            achado = reconhecer(gab, do_alto=c)
            if achado is not None:
                break

        if achado is None:
            print("\n  NENHUM candidato tem o tamanho da estante.")
            print("  Isso e uma resposta, nao uma falha: ou ela esta fora do")
            print("  quadro da camera do alto, ou as bordas dela nao aparecem")
            print("  contra o chao. Confira a janela com --ver.\n")
        else:
            print(f"\n  ESTANTE em ({achado.x:+.2f}, {achado.y:+.2f}) m")
            print(f"  face olhando {np.degrees(achado.rumo_da_face):+.0f} graus")
            print(f"  {achado.largura:.2f} x {achado.profundidade:.2f} m  "
                  f"por: {'+'.join(achado.cameras)}")
            if not achado.confiavel:
                print("  UMA CAMERA SO — arriscado. Confira olhando a cena.")

            if _extrapolado(achado.x, achado.y, calib):
                lx, ly = calib.get("largura_m"), calib.get("altura_m")
                print()
                print("  !! ESTA ESTANTE ESTA FORA DA AREA CALIBRADA !!")
                print(f"     a homografia foi ajustada em 0..{lx} x 0..{ly} m,")
                print(f"     e o centro achado caiu em ({achado.x:+.2f}, "
                      f"{achado.y:+.2f}).")
                print()
                print("     Fora do retangulo medido a homografia extrapola, e")
                print("     extrapolacao projetiva estica um eixo e encolhe o")
                print("     outro. Compare com a trena:")
                print(f"       gabarito  {gab.largura:.2f} x "
                      f"{gab.profundidade:.2f} m")
                print(f"       medido    {achado.largura:.2f} x "
                      f"{achado.profundidade:.2f} m")
                print()
                print("     O conserto NAO e afrouxar a tolerancia: e recalibrar")
                print("     com um retangulo que inclua a pegada da estante.")
                print("       python calibracao/homografia.py")
            if args.gravar:
                _gravar(achado, args.planta)
                print(f"\n  gravado em {args.planta}")
            else:
                print("\n  (nao gravei — use --gravar)")

        if args.ver:
            import cv2
            from src.mundo.detectores import _bordas
            lado = np.hstack([alto, cv2.cvtColor(_bordas(alto),
                                                 cv2.COLOR_GRAY2BGR)])
            cv2.imshow("alto  |  bordas que o detector usa",
                       cv2.resize(lado, None, fx=0.6, fy=0.6))
            print("  qualquer tecla fecha a janela")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    finally:
        app.parar()
    print()


if __name__ == "__main__":
    main()
