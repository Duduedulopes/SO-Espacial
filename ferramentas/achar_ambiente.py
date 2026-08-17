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


def _quadro_estavel(app, papel, n=9, espera=0.12):
    """A mediana de n quadros. Sombra, reflexo e quem passa somem na mediana."""
    pilha = []
    for _ in range(n * 3):
        instante = app.passo()
        q = instante.get(papel) if instante else None
        if q is not None and q.imagem is not None:
            pilha.append(q.imagem.astype(np.float32))
            if len(pilha) >= n:
                break
        time.sleep(espera)
    if not pilha:
        return None
    return np.median(np.stack(pilha), axis=0).astype(np.uint8)


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
    d.pop("_a_medir", None)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="reconhece a estante com as cameras")
    p.add_argument("--planta", default="loja/quarto.json")
    p.add_argument("--captura", default="1280x720")
    p.add_argument("--gravar", action="store_true")
    p.add_argument("--ver", action="store_true")
    p.add_argument("--falsas", action="store_true")
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
        alto = _quadro_estavel(app, "alto")
        if alto is None:
            print("  a camera do alto nao entregou quadro.\n")
            return

        candidatos = candidatos_do_alto(alto, H)
        print(f"\n  {len(candidatos)} retangulo(s) medido(s) no chao:")
        for c in sorted(candidatos, key=lambda v: -v.lado_maior * v.lado_menor):
            cabe = (gab.cabe(c.lado_maior, gab.largura)
                    and gab.cabe(c.lado_menor, gab.profundidade))
            marca = "  <-- cabe no gabarito" if cabe else ""
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
