"""
Decide, de uma vez, se o rumo do corpo sai certo ou invertido. UM BIT.

    python ferramentas/fixar_rumo.py

Tres travessias andando OLHANDO PARA ONDE VAI. Cada uma vota; a maioria grava
`config/rumo.json` e o assunto acaba.

POR QUE ISTO SUBSTITUIU O `calibrar_azimute.py`

Aquela ferramenta tentava medir um ANGULO, e falhou duas vezes com 105 e 148
graus de discordancia entre travessias. Mas os numeros que ela produziu
guardavam a resposta o tempo todo:

    -175,5   -138,3   -70,3
    -165,3    +77,8  -134,5

Como angulo, uma bagunca. Como BIT — "esta a mais ou a menos de 90 graus do
deslocamento?" — as duas execucoes deram 2 x 1. E o voto discordante de cada
uma e justamente o que cai perto de 90 graus, que e onde a pergunta nao tem
resposta: quem anda DE LADO tem o corpo perpendicular ao movimento.

Descartando essa faixa cega, as duas viram 2 x 0. Unanimes.

    O dado sempre soube responder a pergunta binaria. Era a pergunta continua
    que ele nao sustentava.

POR QUE O BIT EXISTE

`rumo_do_alto` calcula `frente = (dy, -dx)` a partir da linha dos ombros vista
de cima. Isso supoe um sistema de coordenadas DESTRO — e se a homografia foi
calibrada clicando os pontos na outra ordem, o resultado sai 180 graus virado.
Nao ha como deduzir; so o dado responde.

O APRENDIZADO CONTINUA RODANDO

Gravar nao desliga nada. O `SinalDoRumo` segue votando durante a operacao
normal, e o painel mostra os dois — se discordarem, alguem mexeu na camera do
alto ou refez a homografia.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.acao.corpo import SinalDoRumo                    # noqa: E402
from src.app.orquestrador import Orquestrador              # noqa: E402
from src.nucleo import log as logmod                       # noqa: E402
from src.nucleo.voz import (                               # noqa: E402
    Voz, apito_de_fim, apito_de_inicio,
)

DESTINO = RAIZ / "config" / "rumo.json"
LIMPAR = "\033[H\033[J"


def uma_travessia(app, voz, segundos, n, total):
    """Uma caminhada reta. Devolve (certo, invertido, ignorados)."""
    voz.dizer(f"Travessia {n} de {total}. Va ate uma borda e espere o apito.",
              esperar=True)

    print(f"{LIMPAR}TRAVESSIA {n} de {total}\n")
    print("  1. VA ATE UMA BORDA da area")
    print("  2. VIRE O CORPO na direcao em que vai andar")
    print("  3. ao apito, ATRAVESSE OLHANDO PARA ONDE VAI\n")
    for falta in range(5, 0, -1):
        print(f"\r  comeca em {falta}...   ", end="", flush=True)
        time.sleep(1.0)
    print()

    voz.dizer("Atravesse!")
    apito_de_inicio()

    # Um contador NOVO por travessia: cada uma tem que ser uma medida
    # independente, senao nao ha como saber se elas concordam.
    urna = SinalDoRumo(minimo_votos=1)
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        if app.passo() is None:
            time.sleep(0.005)
            continue
        e = app.espacial
        for pid, p in app.gemeo.pessoas.items():
            bruto = e._rumo_do_alto(p, e.rastros.rastros)
            andado, _ = e.direcao.observar(pid, p.x, p.y, time.monotonic())
            urna.votar(bruto, andado)

        falta = segundos - (time.monotonic() - t0)
        print(f"{LIMPAR}TRAVESSIA {n} de {total}      {falta:4.1f} s\n")
        print("  ATRAVESSE OLHANDO PARA ONDE VAI\n")
        print(f"  votos: {urna.certo} direto x {urna.invertido} invertido"
              f"   ({urna.ignorados} de lado, ignorados)")
        if not app.gemeo.pessoas:
            print("\n  NINGUEM DETECTADO")

    apito_de_fim()
    return urna


def main():
    p = argparse.ArgumentParser(description="Fixa o sinal do rumo do corpo")
    p.add_argument("--travessias", type=int, default=3)
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

    urnas = []
    try:
        print(f"{LIMPAR}FIXAR O SINAL DO RUMO\n")
        print("  Voce vai atravessar a area algumas vezes, OLHANDO PARA ONDE VAI.")
        print("  A pergunta e binaria — certo ou invertido — entao ela tolera")
        print("  muito mais imprecisao que a calibracao de angulo que falhou.")
        print("  Mas nao ande de re nem de lado: esses casos nao respondem.\n")
        input("  ENTER para comecar. ")

        for i in range(1, args.travessias + 1):
            u = uma_travessia(app, voz, args.segundos, i, args.travessias)
            urnas.append(u)
            veredicto = ("invertido" if u.invertido > u.certo else "direto")
            voz.dizer(f"{veredicto}.")
    except KeyboardInterrupt:
        print("\n  interrompido")
    finally:
        voz.calar()
        app.parar()

    validas = [u for u in urnas if u.total >= 3]
    if len(validas) < 2:
        print(f"\n  so {len(validas)} travessia(s) com votos suficientes.")
        print("  A camera do alto precisa ver seus OMBROS enquanto voce anda.")
        return 1

    print(f"{LIMPAR}FIXAR O SINAL DO RUMO\n")
    print("  por travessia:")
    vereditos = []
    for i, u in enumerate(validas, 1):
        v = -1 if u.invertido > u.certo else 1
        vereditos.append(v)
        print(f"    {i}. {'INVERTIDO' if v < 0 else 'direto':10} "
              f"({u.certo} x {u.invertido}, {u.ignorados} de lado)")

    invertidas = sum(1 for v in vereditos if v < 0)
    sinal = -1 if invertidas * 2 > len(vereditos) else 1

    # UNANIMIDADE NAO E EXIGIDA, MAS DISCORDANCIA E AVISO.
    #
    # Numa pergunta binaria, duas de tres ja e uma maioria solida. Mas se uma
    # travessia discordou, vale saber: ou o corpo nao estava apontando para
    # onde os pes iam, ou os ombros foram lidos errado naquele trecho.
    unanime = invertidas in (0, len(vereditos))
    print(f"\n  SINAL: {'INVERTIDO' if sinal < 0 else 'direto'}"
          f"   ({'unanime' if unanime else 'por maioria'})")
    if not unanime:
        print("  Uma travessia discordou. O resultado vale, mas se o boletim")
        print("  continuar trocando frente e tras, refaca com mais cuidado.")

    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.write_text(json.dumps({
        "sinal": sinal,
        "_o_que_e": ("+1 = o rumo do corpo vindo da camera do alto sai certo. "
                     "-1 = sai 180 graus virado, e o sistema corrige."),
        "_por_que": ("`frente = (dy, -dx)` supoe um sistema de coordenadas "
                     "DESTRO. Se a homografia foi calibrada clicando os pontos "
                     "na outra ordem, o rumo sai virado. Nao ha como deduzir."),
        "travessias": [
            {"certo": u.certo, "invertido": u.invertido,
             "ignorados": u.ignorados} for u in validas],
        "unanime": unanime,
        "quando": datetime.now().isoformat(timespec="seconds"),
        "_atencao": [
            "Vale enquanto a camera do ALTO nao for movida e a homografia nao "
            "for refeita.",
            "O painel mostra o que o aprendizado automatico diria ao lado "
            "deste valor. Se discordarem, alguma das duas coisas mudou.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n  gravado em {DESTINO}")
    print("  Rode o conferidor: `andar_frente` e `andar_tras` devem trocar de")
    print("  lugar se o sinal era invertido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
