"""
Calibra o azimute: uma caminhada deliberada, olhando para onde vai.

    python ferramentas/calibrar_azimute.py

Depois disso o sistema sabe distinguir andar para FRENTE de andar de LADO, que
e o ganho central da etapa B e a unica coisa que faltava para ele funcionar.

POR QUE APRENDER SOZINHO NAO FUNCIONA NESTA SALA

O `EstimadorDeAzimute` aprende comparando o rumo dos ombros com a direcao do
deslocamento de quem anda. Isso repousa inteiro sobre uma hipotese:

    quem anda, olha para onde vai

Numa loja, verdadeira. MEDIDO EM 11/08, numa area de 1,4 m diante de um
computador: falsa. A pessoa se desloca olhando para a tela, para a camera, para
onde a voz esta falando. O corpo quase nunca aponta para onde os pes vao.

O resultado foi pior que a abstencao. Enquanto as amostras ficavam espalhadas,
o estimador se calava. Quando passaram a ser aceitas, o grupo majoritario
passou a ser o ERRADO e ele respondeu com confianca:

    andar_frente    -> andando_tras      180 graus errado
    andar_esquerda  -> andando_tras       90 graus errado

    Aumentar a amostra de uma hipotese falsa nao a torna verdadeira; torna o
    erro confiante.

A SAIDA E A MESMA DA ESCALA VERTICAL

Uma acao deliberada, uma vez, com a verdade declarada. Nao ha aprendizado
possivel sem a regularidade que este ambiente nao tem — mas ha UMA caminhada
honesta, e ela basta.

O ESTIMADOR AUTOMATICO CONTINUA EXISTINDO

Ele nao foi removido, e nao deve ser: numa loja de verdade a hipotese volta a
valer, e la nao havera ninguem para calibrar. Aqui ele vira reserva, e o painel
passa a mostrar os dois lado a lado — se discordarem muito depois de gravado,
ou a camera foi movida ou o ambiente mudou.

    Calibrado manda no aprendido, e nao a media dos dois. Misturar uma medida
    honesta com um aprendizado viciado produz um terceiro numero pior que a
    medida, e ainda esconde qual dos dois estava errado.
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.acao.angulos import (                            # noqa: E402
    concentracao, diferenca_angular, media_circular,
)
from src.app.orquestrador import Orquestrador              # noqa: E402
from src.nucleo import log as logmod                       # noqa: E402
from src.nucleo.voz import (                               # noqa: E402
    Voz, apito_de_fim, apito_de_inicio,
)

DESTINO = RAIZ / "config" / "azimute.json"
LIMPAR = "\033[H\033[J"


def uma_travessia(app, voz, segundos, numero, total):
    """Uma caminhada em linha reta, olhando para onde vai.

    Devolve a lista de (rumo_dos_ombros_na_camera, rumo_do_deslocamento).
    """
    voz.dizer(f"Travessia {numero} de {total}. "
              f"Va ate uma borda e espere o apito.", esperar=True)

    print(f"{LIMPAR}TRAVESSIA {numero} de {total}\n")
    print("  1. VA ATE UMA BORDA da area")
    print("  2. VIRE O CORPO na direcao em que vai andar")
    print("  3. ao apito, ATRAVESSE OLHANDO PARA ONDE VAI\n")
    for falta in range(5, 0, -1):
        print(f"\r  comeca em {falta}...   ", end="", flush=True)
        time.sleep(1.0)
    print()

    voz.dizer("Atravesse!")
    apito_de_inicio()

    pares = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        if app.passo() is None:
            time.sleep(0.005)
            continue

        estados = list(app.gemeo.pessoas.values())
        leitura = None
        if len(estados) == 1:
            leitura = app.espacial.leituras.get(estados[0].id)

        # O rumo dos ombros vem da leitura de corpo (referencial da lente); o
        # rumo do deslocamento vem da mesma janela que o estimador usa.
        if leitura is not None and leitura.rumo_corpo_camera is not None:
            p = estados[0]
            andado, quanto = app.espacial.direcao.observar(
                p.id, p.x, p.y, time.monotonic())
            if andado is not None:
                pares.append((leitura.rumo_corpo_camera, andado))

        falta = segundos - (time.monotonic() - t0)
        print(f"{LIMPAR}TRAVESSIA {numero} de {total}      {falta:4.1f} s\n")
        print("  ATRAVESSE OLHANDO PARA ONDE VAI\n")
        print(f"  amostras boas: {len(pares)}")
        if not estados:
            print("\n  NINGUEM DETECTADO")
        elif leitura is None or leitura.rumo_corpo_camera is None:
            print("\n  OMBROS NAO VISTOS — fique de lado para a frontal")

    apito_de_fim()
    return pares


def main():
    p = argparse.ArgumentParser(description="Calibra o azimute da camera")
    p.add_argument("--travessias", type=int, default=3,
                   help="quantas caminhadas. Mais de uma porque uma sozinha "
                        "nao tem como ser conferida contra nada")
    p.add_argument("--segundos", type=float, default=6.0)
    p.add_argument("--planta", default="loja/bancada.json")
    p.add_argument("--sem-voz", action="store_true")
    p.add_argument("--log", default="AVISO")
    args = p.parse_args()

    logmod.configurar(args.log)
    voz = Voz(ligada=not args.sem_voz)

    app = Orquestrador(planta=args.planta, captura=(640, 480))
    app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    todos = []
    por_travessia = []
    try:
        print(f"{LIMPAR}CALIBRACAO DO AZIMUTE\n")
        print("  Voce vai atravessar a area algumas vezes, OLHANDO PARA ONDE VAI.")
        print("  E so isso — mas tem que ser de verdade: se voce andar olhando")
        print("  para a tela, a calibracao fica tao errada quanto o aprendizado")
        print("  automatico que ela existe para substituir.\n")
        input("  ENTER para comecar. ")

        for i in range(1, args.travessias + 1):
            pares = uma_travessia(app, voz, args.segundos, i, args.travessias)
            por_travessia.append(pares)
            todos += pares
            voz.dizer(f"{len(pares)} amostras.")
    except KeyboardInterrupt:
        print("\n  interrompido")
    finally:
        voz.calar()
        app.parar()

    if len(todos) < 15:
        print(f"\n  so {len(todos)} amostras — poucas para calibrar.")
        print("  A frontal precisa ver seus OMBROS enquanto voce atravessa.")
        print("  Se ela so pega voce de frente parado, ande de um lado para o")
        print("  outro DENTRO do campo dela.")
        return 1

    offsets = [diferenca_angular(andado, ombros) for ombros, andado in todos]
    offset = media_circular(offsets)
    conc = concentracao(offsets)

    print(f"{LIMPAR}CALIBRACAO DO AZIMUTE\n")
    print(f"  amostras          {len(todos)}")
    print(f"  AZIMUTE           {math.degrees(offset):+.1f} graus")
    print(f"  concordancia      {conc:.0%}\n")

    # CADA TRAVESSIA E UMA MEDIDA INDEPENDENTE, E ELAS TEM QUE CONCORDAR.
    #
    # Uma travessia sozinha produz um numero e nenhuma forma de duvidar dele.
    # Tres produzem tres numeros, e a discordancia entre eles E o erro da
    # medicao — a unica coisa que diz se a calibracao pode ser usada.
    print("  por travessia:")
    parciais = []
    for i, pares in enumerate(por_travessia, 1):
        if len(pares) < 5:
            print(f"    {i}. {len(pares)} amostras — ignorada, poucas")
            continue
        o = media_circular([diferenca_angular(a, b) for b, a in pares])
        parciais.append(o)
        desvio = math.degrees(abs(diferenca_angular(o, offset)))
        print(f"    {i}. {math.degrees(o):+7.1f} graus   "
              f"({len(pares)} amostras, {desvio:.0f} graus do consenso)")

    if len(parciais) >= 2:
        espalhamento = max(
            math.degrees(abs(diferenca_angular(a, b)))
            for a in parciais for b in parciais)
        print(f"\n  discordancia entre travessias: {espalhamento:.0f} graus")
        if espalhamento > 30:
            print("\n  ATENCAO: as travessias discordam demais.")
            print("  Isso significa que em pelo menos uma delas o corpo nao")
            print("  estava apontando para onde os pes iam. Refaca — e nao")
            print("  grave este numero, ele nao e melhor que o automatico.")
            return 1

    if conc < 0.7:
        print(f"\n  ATENCAO: concordancia de {conc:.0%} e baixa.")
        print("  As amostras individuais estao espalhadas. Refaca andando em")
        print("  linha reta, sem virar a cabeca nem o tronco durante o trajeto.")
        return 1

    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.write_text(json.dumps({
        "azimute_rad": round(float(offset), 5),
        "azimute_graus": round(math.degrees(offset), 2),
        "_o_que_e": ("giro da camera FRONTAL em relacao ao mundo. Converte o "
                     "rumo dos ombros, que sai no referencial da lente, para o "
                     "referencial do chao onde o Kalman trabalha."),
        "_por_que_calibrado": (
            "o aprendizado automatico assume que quem anda olha para onde vai. "
            "Numa area de 1,4 m diante de um computador isso e falso, e em "
            "11/08 ele convergiu para o grupo errado: andar_frente saiu como "
            "andando_tras."),
        "amostras": len(todos),
        "concordancia": round(conc, 3),
        "travessias": [round(math.degrees(o), 2) for o in parciais],
        "quando": datetime.now().isoformat(timespec="seconds"),
        "_atencao": [
            "Vale enquanto a camera FRONTAL nao for movida.",
            "O painel mostra o que o automatico TERIA aprendido ao lado deste "
            "numero. Se discordarem muito, alguem mexeu na camera.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n  gravado em {DESTINO}")
    print("  A partir de agora `andar_frente` e `andar_esquerda` sao")
    print("  distinguiveis, e o painel mostra `CALIBRADO` no lugar de ABSTIDO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
