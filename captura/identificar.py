"""
Identificacao de cameras — por NOME, que nao muda.

O PROBLEMA

Indices de camera no Windows nao sao estaveis: mudam ao reconectar um cabo, ao
instalar um driver virtual, ao reiniciar. Em 08/08 isso fez duas janelas
mostrarem a MESMA camera e custou uma sessao inteira.

    Identificador que muda sozinho nao e identificador.

A SOLUCAO

O DirectShow enumera os dispositivos com nome, na mesma ordem que o OpenCV usa
como indice. Guardamos o NOME em `config/cameras.json`; o indice e resolvido a
cada execucao.

    {"alto": "HD Pro Webcam C920", "frontal": "VGA camera", ...}

POR QUE A ESCOLHA E NO TERMINAL

A primeira versao pedia teclas na janela do OpenCV. Falhou duas vezes no
Windows — antes com `[` e `]`, depois com A/F/L/N. Quando um mecanismo falha
duas vezes, troca-se o mecanismo: a janela virou uma FOTO e a resposta vem por
`input()`, que sempre funciona.

Uso:
    python captura/identificar.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from captura.dispositivos import listar, resolver  # noqa: E402
from captura.fonte import garantir_imagem_visivel  # noqa: E402

CONFIG = RAIZ / "config" / "cameras.json"
CONTATO = RAIZ / "config" / "cameras_vistas.png"

PAPEIS = ["alto", "frontal", "lateral"]


def abrir_todas(dispositivos, largura=1280, altura=720):
    """Abre cada camera UMA VEZ e mantem aberta.

    A versao anterior abria e fechava tres vezes por camera — na varredura, na
    escolha e no teste final. A C920 travava nesse vaivem e passava a devolver
    preto ate o cabo ser desconectado.

        Abrir dispositivo e caro e arriscado. Abra uma vez, use, feche no fim.

    Manter todas abertas tambem torna o teste "funcionam juntas?" automatico:
    se chegaram ate aqui, funcionam.
    """
    abertas = []
    for idx, nome in dispositivos:
        print(f"\n  [{idx}] {nome}")

        # SOMENTE DSHOW. NAO CAIA PARA MSMF.
        #
        # ERRO QUE ISTO CORRIGE (08/08, e vinha se repetindo)
        #
        # O pygrabber enumera pela ordem do DirectShow. O MSMF tem ordem
        # PROPRIA. Quando a abertura falhava no DSHOW e eu caia para MSMF, o
        # indice 1 passava a apontar para outro dispositivo fisico.
        #
        # Sintomas que isso produzia, e que eu andei atribuindo a outras causas:
        #   - a janela "alto" mostrando a cena da lateral
        #   - imagem esverdeada (outro formato de pixel)
        #   - aviso de "duas cameras mostram a mesma cena"
        #   - "VGA camera" reportando 1280x720 e a C920 reportando 640x480
        #
        # Um fallback que troca o significado do identificador e pior que
        # falhar. Falhar e visivel; trocar em silencio, nao.
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("      DSHOW nao abriu")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

        garantir_imagem_visivel(cap, verboso=False)
        ok, q = cap.read()

        if ok and q is not None and q.mean() > 8:
            print(f"      {q.shape[1]}x{q.shape[0]}  brilho {q.mean():.0f}")
            abertas.append([idx, nome, q, cap])
        else:
            print("      sem imagem utilizavel no DSHOW.")
            print("      -> desconecte o cabo desta camera por 10s e reconecte.")
            print("         (nao vou cair para MSMF: os indices sao outros la,")
            print("          e isso ja trocou cameras de lugar antes)")
            cap.release()
            time.sleep(0.4)
    return abertas


def parecidas(a, b, limiar=0.06):
    pa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (64, 48)).astype(float)
    pb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (64, 48)).astype(float)
    pa = (pa - pa.mean()) / (pa.std() + 1e-6)
    pb = (pb - pb.mean()) / (pb.std() + 1e-6)
    return float(np.abs(pa - pb).mean()) < limiar


def folha_de_contato(fotos, alt=320):
    tiras = []
    for idx, nome, q, *_ in fotos:
        esc = alt / q.shape[0]
        t = cv2.resize(q, None, fx=esc, fy=esc)
        faixa = np.full((52, t.shape[1], 3), 28, np.uint8)
        cv2.putText(faixa, f"[{idx}] {nome}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        cv2.putText(faixa, f"{q.shape[1]}x{q.shape[0]}  brilho {q.mean():.0f}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
        tiras.append(np.vstack([faixa, t]))

    larg = max(t.shape[1] for t in tiras)
    tiras = [np.pad(t, ((0, 0), (0, larg - t.shape[1]), (0, 0)),
                    constant_values=28) for t in tiras]
    return np.hstack(tiras)


def main() -> None:
    # SEM PERGUNTA INTERATIVA.
    #
    # Terceira tentativa de mecanismo, e a que nao pode falhar:
    #   1o) teclas na janela do OpenCV  -> o HighGUI nao entregava as teclas
    #   2o) input() no terminal         -> o visualizador de fotos roubava o foco
    #   3o) ARGUMENTOS na linha de comando
    #
    # Argumento nao depende de foco, de janela, nem de teclado funcionando em
    # lugar nenhum. Rode uma vez para ver a foto, e outra vez com a resposta.
    ap = argparse.ArgumentParser(
        description="Identifica as cameras e grava config/cameras.json")
    # Aceita INDICE ou NOME. Prefira nome: em 08/08 os indices reordenaram
    # entre duas execucoes seguidas, e a atribuicao saiu trocada.
    ap.add_argument("--alto", default=None,
                   help="indice OU nome da camera que olha o chao. Ex: C920")
    ap.add_argument("--frontal", default=None,
                   help="indice OU nome da camera que ve a pessoa de frente")
    ap.add_argument("--lateral", default=None,
                   help="indice OU nome da camera que ve a pessoa de lado")
    ap.add_argument("--sem-abrir", action="store_true",
                   help="nao abre a foto no visualizador")
    args = ap.parse_args()

    dispositivos = listar()
    if dispositivos is None:
        print("pygrabber nao instalado — sem nomes de dispositivo.\n"
              "  pip install pygrabber\n"
              "Seguindo com indices 0 a 3 (frageis).\n")
        dispositivos = [(i, f"indice {i}") for i in range(4)]
    else:
        print("cameras presentes no Windows:")
        for idx, nome in dispositivos:
            print(f"  [{idx}] {nome}")

    fotos = abrir_todas(dispositivos)
    if not fotos:
        raise SystemExit(
            "\nnenhuma camera entregou imagem.\n"
            "  - feche Iriun, Teams, app Camera\n"
            "  - finalize processos python.exe no Gerenciador de Tarefas\n"
            "  - rode: python captura/reparar.py")

    print()
    for i in range(len(fotos)):
        for j in range(i + 1, len(fotos)):
            if parecidas(fotos[i][2], fotos[j][2]):
                print(f"  AVISO: [{fotos[i][0]}] e [{fotos[j][0]}] mostram a "
                      f"mesma cena.")

    CONTATO.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CONTATO), folha_de_contato(fotos))

    por_indice = {reg[0]: reg[1] for reg in fotos}

    def resolver_escolha(valor):
        """Aceita indice ('1') ou nome ('C920', 'Iriun'), inteiro ou parcial."""
        if valor is None:
            return None
        v = str(valor).strip()
        if v.isdigit():
            return por_indice.get(int(v))
        alvo = v.lower()
        for nome in por_indice.values():
            if alvo == nome.lower():
                return nome
        for nome in por_indice.values():
            if alvo in nome.lower():
                return nome
        return None

    escolhas = {"alto": args.alto, "frontal": args.frontal,
                "lateral": args.lateral}
    atribuicao = {}
    for papel, valor in escolhas.items():
        if valor is None:
            continue
        nome = resolver_escolha(valor)
        if nome is None:
            print(f"  AVISO: '{valor}' ({papel}) nao corresponde a nenhuma "
                  f"camera com imagem, ignorado")
            continue
        atribuicao[papel] = nome

    # as cameras ficaram abertas o tempo todo — se chegaram aqui, funcionam juntas
    for reg in fotos:
        reg[3].release()

    if not atribuicao:
        print(f"\nfoto de todas as cameras salva em:\n  {CONTATO}")
        if not args.sem_abrir:
            try:
                os.startfile(str(CONTATO))
            except Exception:
                pass
        print("\nOLHE A FOTO e rode de novo dizendo qual e qual:")
        print("\n    python captura/identificar.py"
              " --alto N --frontal N --lateral N\n")
        print("  cameras que entregaram imagem:")
        for reg in fotos:
            print(f"      {reg[0]} = {reg[1]}")
        print("\n  exemplo:")
        idxs = [reg[0] for reg in fotos]
        ex = "python captura/identificar.py --alto {}".format(idxs[0])
        if len(idxs) > 1:
            ex += f" --frontal {idxs[1]}"
        if len(idxs) > 2:
            ex += f" --lateral {idxs[2]}"
        print(f"      {ex}")
        return

    if "alto" not in atribuicao:
        print("\nAVISO: sem camera 'alto'. Ela e a unica que da posicao no chao.")

    CONFIG.write_text(json.dumps(atribuicao, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\nsalvo em {CONFIG}:")
    for papel, nome in atribuicao.items():
        print(f"  {papel:8} {nome}")
    print("\nagora:  python percepcao/gemeo_multi.py")


def carregar():
    """Devolve {'alto': indice, ...}, resolvendo nomes para indices AGORA."""
    if not CONFIG.exists():
        raise SystemExit(
            f"nao achei {CONFIG}\nRode antes:  python captura/identificar.py")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    indices, faltando = resolver(config)

    for papel, nome in faltando:
        print(f"  AVISO: camera '{nome}' ({papel}) nao esta presente agora.")
    return indices


if __name__ == "__main__":
    main()
