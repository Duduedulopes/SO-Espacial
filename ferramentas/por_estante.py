"""Poe a estante na planta a partir de duas medidas de trena.

    python ferramentas/por_estante.py --pe-esq 0.35 0.95 --pe-dir 1.25 1.02

DUAS MEDIDAS, TIRADAS DO CANTO DE ORIGEM DA CALIBRACAO

O canto (0,0) e o primeiro ponto que voce clicou ao calibrar a homografia, e
ele esta marcado no chao com fita. A partir dele, com a trena:

    --pe-esq   x, y ate o pe ESQUERDO da FRENTE da estante
    --pe-dir   x, y ate o pe DIREITO da FRENTE

"esquerdo" e "direito" de quem esta em pe diante dela, olhando para ela — a
mesma referencia de quem vai pegar um produto.

POR QUE TRENA E NAO CAMERA, DEPOIS DE UM DIA INTEIRO TENTANDO CAMERA

Em 18/08 foram gastas horas em reconstrucao multi-vista para descobrir estes
mesmos tres numeros. Nao funcionou, e o motivo nao foi o metodo: as tres
cameras deste arranjo quase nao veem as mesmas superficies, e reconstrucao
precisa de sobreposicao. O resultado foi um quarto de 4,5 m de largura por
0,66 de altura, com duas cameras empilhadas no mesmo ponto.

    A camera e o instrumento do que se move. A trena e o instrumento do que
    fica parado. Trocar os dois de lugar custa um dia.

E O REQUISITO CONTINUA VALENDO:

    a posicao dela nao muda, mais podera mudar, entao nao de a ela um ponto
    fixo                                            — Eduardo, 13/08

O numero nao entra no codigo: entra em `loja/quarto.json`, junto com a data
em que foi medido. Se a estante for empurrada, roda-se de novo — dois
minutos, e a planta inteira acompanha, portas incluidas.

O QUE ELE ESCREVE

    o movel, com as dimensoes de `loja/estante.json` — nunca as digitadas
    o rumo da face, deduzido da reta entre os dois pes
    a entrada e a saida, deduzidas da estante (ver achar_ambiente._portas)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np                                           # noqa: E402

from ferramentas.achar_ambiente import _portas               # noqa: E402
from src.mundo.ambiente import Ambiente, Gabarito            # noqa: E402


def estante_de(pe_esq, pe_dir, gabarito):
    """A estante posta no mundo a partir dos dois pes da frente.

    O centro NAO e o meio dos dois pes: os pes sao a frente, e o centro do
    movel esta meia profundidade atras deles. Confundir os dois poe a estante
    15 cm para dentro da area onde a pessoa anda — e a conta de "esta na
    frente da estante" passa a comecar dentro dela.
    """
    a, b = np.asarray(pe_esq, dtype=float), np.asarray(pe_dir, dtype=float)
    largura_medida = float(np.linalg.norm(b - a))

    ao_longo = (b - a) / (largura_medida or 1.0)
    rumo = math.atan2(ao_longo[1], ao_longo[0])
    normal = np.array([-math.sin(rumo), math.cos(rumo)])

    # A face olha para o lado de onde se anda — o lado da origem, porque
    # atras da estante ha parede. Mesma regra de `ambiente.reconhecer`.
    meio_da_frente = (a + b) / 2.0
    if float(normal @ (np.array([0.0, 0.0]) - meio_da_frente)) < 0:
        normal, rumo = -normal, rumo + math.pi

    centro = meio_da_frente - normal * (gabarito.profundidade / 2.0)
    return Ambiente(x=float(centro[0]), y=float(centro[1]),
                    rumo_da_face=float(math.atan2(math.sin(rumo),
                                                  math.cos(rumo))),
                    largura=gabarito.largura,
                    profundidade=gabarito.profundidade,
                    altura=gabarito.altura,
                    prateleiras=gabarito.prateleiras,
                    cameras=("trena",)), largura_medida


def gravar(estante, caminho, medida_esq, medida_dir):
    p = Path(caminho)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["moveis"] = [m for m in d.get("moveis", [])
                   if m.get("id") != "estante-aco"]
    d["moveis"].append({
        "id": "estante-aco", "nome": "Estante", "tipo": "estante",
        "x": round(estante.x, 3), "y": round(estante.y, 3),
        "largura": round(estante.largura, 3),
        "profundidade": round(estante.profundidade, 3),
        "altura": round(estante.altura, 3),
        "rumo_da_face": round(float(estante.rumo_da_face), 4),
        "prateleiras": [{"id": i, "altura": round(float(a), 3)}
                        for i, a in estante.prateleiras],
        "estante": "estante-aco-teste",
        "_medido_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_por": "trena, a partir do canto (0,0) da calibracao",
        "_pes_da_frente": {"esquerdo": list(medida_esq),
                           "direito": list(medida_dir)},
        "_nota": [
            "MEDIDO COM TRENA, e isso e uma escolha, nao uma desistencia.",
            "",
            "As tres cameras deste arranjo quase nao veem as mesmas",
            "superficies, e reconstrucao multi-vista precisa de sobreposicao.",
            "Um dia inteiro de tentativa devolveu um quarto de 4,5 m de",
            "largura por 0,66 de altura.",
            "",
            "    A camera e o instrumento do que se move. A trena e o",
            "    instrumento do que fica parado.",
            "",
            "Se a estante for empurrada, rode de novo com as medidas novas:",
            "    python ferramentas/por_estante.py --pe-esq X Y --pe-dir X Y",
            "As portas acompanham sozinhas."
        ]})

    c = d["chao"]
    d["zonas"] = [z for z in d.get("zonas", [])
                  if z.get("id") not in ("entrada", "saida")]
    d["zonas"] += _portas(estante,
                          (c["xmin"], c["xmax"], c["ymin"], c["ymax"]))
    d.pop("_a_medir", None)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="poe a estante na planta")
    p.add_argument("--pe-esq", nargs=2, type=float, required=True,
                   metavar=("X", "Y"), help="pe esquerdo da frente, em metros")
    p.add_argument("--pe-dir", nargs=2, type=float, required=True,
                   metavar=("X", "Y"), help="pe direito da frente, em metros")
    p.add_argument("--planta", default="loja/quarto.json")
    p.add_argument("--gravar", action="store_true")
    args = p.parse_args()

    gab = Gabarito.de_arquivo("loja/estante.json")
    estante, medida = estante_de(args.pe_esq, args.pe_dir, gab)

    print(f"\n  GABARITO   {gab.largura:.2f} x {gab.profundidade:.2f} x "
          f"{gab.altura:.2f} m")
    print(f"  MEDIDO     {medida:.2f} m entre os dois pes")

    # A CONFERENCIA QUE PAGA A ETAPA INTEIRA.
    #
    # A distancia entre os dois pes TEM que ser a largura da estante. Se nao
    # for, alguma medida saiu errada — e melhor descobrir aqui do que ver o
    # boneco atravessando o movel na tela.
    erro = abs(medida - gab.largura)
    if erro > 0.06:
        print(f"\n  ATENCAO: os dois pes distam {medida:.2f} m, mas a estante")
        print(f"  tem {gab.largura:.2f} m de largura — {erro * 100:.0f} cm de")
        print("  diferenca. Confira as medidas antes de gravar; uma delas")
        print("  provavelmente saiu do lugar errado.\n")
    else:
        print(f"  confere: {erro * 100:.1f} cm de diferenca\n")

    print(f"  ESTANTE em ({estante.x:+.2f}, {estante.y:+.2f}) m")
    print(f"  face olhando {math.degrees(estante.rumo_da_face):+.0f} graus")

    if args.gravar:
        gravar(estante, args.planta, args.pe_esq, args.pe_dir)
        print(f"\n  gravado em {args.planta} — com entrada e saida\n")
        print("  agora:  python rodar.py\n")
    else:
        print("\n  (nao gravei — use --gravar)\n")


if __name__ == "__main__":
    main()
