"""
Conferidor — o sistema esta lendo CERTO o que a camera ve?

    python ferramentas/conferir.py                  as tres cameras, roteiro padrao
    python ferramentas/conferir.py --so-cameras     so a saude das cameras, 20 s
    python ferramentas/conferir.py --falsas         prova o aparato, sem hardware
    python ferramentas/conferir.py --comparar A B   dois boletins lado a lado

O QUE ELE FAZ, E POR QUE NAO E MAIS UM PAINEL

O painel do `rodar.py` mostra o que o sistema ESTA DIZENDO. Isso nao responde
se o que ele diz e verdade. Aqui a pessoa DECLARA antes o que vai fazer, o
programa cronometra a janela, anota o que foi lido e no fim compara.

    Sem registro do que aconteceu, nao ha como julgar o que o sistema disse
    que aconteceu.                                          — caderno, 10/08

DUAS FASES, E A ORDEM DECIDE O DIAGNOSTICO

    fase 1   as cameras estao entregando imagem util?
    fase 2   o sistema esta lendo certo o que elas entregam?

A fase 1 vem antes porque uma nota ruim com camera preta manda consertar o
classificador quando o problema esta no driver. Isso aconteceu de verdade: em
10/08 a camera lateral entregou 462 quadros com brilho 11 de 255, o MediaPipe
achou zero poses em todos, e eu sustentei por TRES execucoes a hipotese de que
era enquadramento. Era o DirectShow entregando preto.

    A imagem bonita no painel de Configuracoes do Windows nunca foi prova de
    nada.

Por isso a fase 1 reprova a camera antes de a fase 2 comecar, e a reprovacao
aparece no boletim: se a lateral estava cega, o boletim diz isso ao lado da
nota, e ninguem vai procurar defeito no lugar errado.

O BOLETIM E GRAVADO

`dados/confer/<carimbo>.json`. Duas execucoes viram comparacao, e comparacao e
a unica forma de saber se uma mudanca melhorou ou piorou alguma coisa. Em
10/08 tres rodadas de ajuste produziram 12 -> 16 -> 17 mudancas de locomocao,
e nenhuma delas podia ser julgada porque nao havia com o que comparar.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.acao.gabarito import Placar, roteiro_padrao       # noqa: E402
from src.app.orquestrador import Orquestrador              # noqa: E402
from src.nucleo import log as logmod                       # noqa: E402

LIMPAR = "\033[H\033[J"

# Faixa de brilho de uma cena real, medida em 10/08. Abaixo de 32 a imagem
# existe mas nao serve: `brilho_minimo=8` separa "sem imagem" de "com imagem",
# e nao separa "com imagem" de "com imagem UTIL". Entre os dois ha uma faixa
# onde o sistema funciona no papel e nao enxerga nada.
BRILHO_SUSPEITO = 32.0


# --------------------------------------------------------------- fase 1
def conferir_cameras(app, segundos=20.0):
    """Cada camera esta entregando imagem que serve? Devolve o laudo.

    NAO BASTA ESTAR ONLINE. As tres coisas que precisam ser verdade ao mesmo
    tempo, e cada uma ja falhou sozinha neste projeto:

        fps       a C920 caiu para 1,0 fps em luz fraca e impos essa taxa ao
                  sistema inteiro, porque o sincronizador espera a mais lenta
        brilho    o tablet abriu com 11 de 255 e ficou ONLINE
        poses     a lateral entregou 462 quadros e zero poses
    """
    print(f"{LIMPAR}FASE 1 — SAUDE DAS CAMERAS   ({segundos:.0f} s)\n")
    print("  Fique VISIVEL nas tres cameras, em pe, movimentando-se um pouco.")
    print("  Isto mede o que chega ao programa, nao o que aparece no Windows.\n")

    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        if app.passo() is None:
            time.sleep(0.005)
            continue
        falta = segundos - (time.monotonic() - t0)
        print(f"{LIMPAR}FASE 1 — SAUDE DAS CAMERAS      faltam {falta:4.1f} s\n")
        print("\n".join("  " + l for l in app.cameras.painel()))
        print()
        print("\n".join("  " + l for l in app.visao.painel()))

    return _laudo_das_cameras(app)


def _laudo_das_cameras(app):
    laudo = {}
    poses = {p: ex.t.metricas for p, ex in app.visao.executores.items()}

    for papel, fonte in app.cameras.fontes.items():
        m = fonte.metricas
        pm = poses.get(papel)
        queixas = []

        if fonte.estado.value != "online":
            queixas.append(f"nao esta online ({fonte.estado.value})")
        if m.recebidos == 0:
            queixas.append("nenhum quadro recebido")
        if 0 < m.fps < 8:
            queixas.append(f"{m.fps:.1f} fps — impoe essa taxa ao sistema todo")
        if m.recebidos and m.brilho < BRILHO_SUSPEITO:
            queixas.append(f"brilho {m.brilho:.0f} de 255 — imagem existe, "
                           f"mas provavelmente nao serve")
        # Zero pose com quadros chegando e a assinatura de imagem inutil. A
        # camera do alto nao roda pose, entao a checagem so vale onde ha.
        if pm is not None and pm.quadros > 30 and pm.saidas == 0:
            queixas.append(f"{pm.quadros} quadros e ZERO poses")

        laudo[papel] = {
            "estado": fonte.estado.value,
            "fps": round(m.fps, 1),
            "brilho": round(m.brilho, 1),
            "recebidos": m.recebidos,
            "falhas": m.falhas_leitura,
            "poses": pm.saidas if pm else None,
            "quadros_pose": pm.quadros if pm else None,
            "queixas": queixas,
        }
    return laudo


def mostrar_laudo(laudo):
    print(f"{LIMPAR}LAUDO DAS CAMERAS\n")
    for papel, d in laudo.items():
        marca = "OK  " if not d["queixas"] else "RUIM"
        pose = "-" if d["poses"] is None else f"{d['poses']}/{d['quadros_pose']}"
        print(f"  {marca} {papel:9} {d['fps']:5.1f} fps  "
              f"brilho {d['brilho']:5.1f}  quadros {d['recebidos']:5}  "
              f"poses {pose}")
        for q in d["queixas"]:
            print(f"         ! {q}")

    ruins = [p for p, d in laudo.items() if d["queixas"]]
    print()
    if ruins:
        print(f"  {len(ruins)} camera(s) com problema: {', '.join(ruins)}")
        print("  A nota da fase 2 vai carregar isso. Consertar a camera vem")
        print("  ANTES de mexer em qualquer limiar do classificador.")
    else:
        print("  As tres entregando imagem util. A fase 2 mede o sistema, "
              "nao o hardware.")
    return ruins


# --------------------------------------------------------------- fase 2
def rodar_roteiro(app, roteiro, placar):
    """Guia a pessoa passo a passo e anota o que o sistema leu."""
    for n, passo in enumerate(roteiro, 1):
        t0 = time.monotonic()
        while True:
            decorrido = time.monotonic() - t0
            if decorrido > passo.segundos:
                break
            if app.passo() is None:
                time.sleep(0.005)
                continue

            acoes = app.espacial.acoes
            prevendo = {pid: p.prevendo
                        for pid, p in app.gemeo.pessoas.items()}
            placar.anotar(passo, acoes, decorrido, prevendo)
            _tela(n, len(roteiro), passo, decorrido, acoes, prevendo)


def _tela(n, total, passo, decorrido, acoes, prevendo):
    falta = passo.segundos - decorrido
    aquecendo = decorrido < passo.acomodacao_s

    print(f"{LIMPAR}  ROTEIRO   passo {n} de {total}\n")
    print(f"      >>> {passo.instrucao} <<<")
    if passo.instrucao_extra:
        print(f"          {passo.instrucao_extra}")
    print()
    print("          " + ("acomodando, ainda nao conta..."
                          if aquecendo else
                          f"CONTANDO    faltam {falta:4.1f} s"))
    print()

    if not acoes:
        print("      lendo agora:  NINGUEM DETECTADO")
        return

    for pid, (a, _) in sorted(acoes.items()):
        marca = "  (posicao PREVISTA)" if prevendo.get(pid) else ""
        print(f"      lendo agora:  #{pid}  {a.locomocao} / {a.postura}"
              f"   conf {a.confianca:.0%}{marca}")
        for lado, estado, altura in (("E", a.braco_esquerdo, a.altura_mao_esq),
                                     ("D", a.braco_direito, a.altura_mao_dir)):
            metros = "    --" if altura is None else f"{altura:5.2f}m"
            print(f"                    braco {lado}  {estado:14} {metros}")


# --------------------------------------------------------------- registro
def gravar(laudo, placar, roteiro, app, pasta=None):
    pasta = Path(pasta or RAIZ / "dados" / "confer")
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"

    destino.write_text(json.dumps({
        "quando": datetime.now().isoformat(timespec="seconds"),
        "fps": round(app.fps, 2),
        "fps_regime": round(app.fps_regime, 2),
        "cameras": laudo,
        "roteiro": [p.acao for p in roteiro],
        "boletim": placar.para_dicionario(),
        "espacial": app.espacial.resumo(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def comparar(caminho_a, caminho_b):
    """Dois boletins lado a lado. Mudanca sem comparacao e fe, nao medida."""
    a = json.loads(Path(caminho_a).read_text(encoding="utf-8"))
    b = json.loads(Path(caminho_b).read_text(encoding="utf-8"))
    ca, cb = a["boletim"]["acoes"], b["boletim"]["acoes"]

    print(f"\n{'ACAO':22} {'ANTES':>8} {'DEPOIS':>8} {'DELTA':>8}")
    print("-" * 50)
    for acao in sorted(set(ca) | set(cb)):
        na = ca.get(acao, {}).get("nota")
        nb = cb.get(acao, {}).get("nota")
        if na is None or nb is None:
            print(f"{acao:22} {'-' if na is None else f'{na:.0%}':>8} "
                  f"{'-' if nb is None else f'{nb:.0%}':>8} {'novo':>8}")
            continue
        d = nb - na
        seta = "  " if abs(d) < 0.02 else ("UP" if d > 0 else "DOWN")
        print(f"{acao:22} {na:7.0%} {nb:7.0%} {d:+7.0%} {seta}")


# --------------------------------------------------------------- principal
def main():
    p = argparse.ArgumentParser(description="Conferidor do SO Espacial")
    p.add_argument("--planta", default="loja/bancada.json")
    p.add_argument("--captura", default="640x480")
    p.add_argument("--falsas", action="store_true",
                   help="sem hardware — prova o aparato, nao a percepcao")
    p.add_argument("--so-cameras", action="store_true")
    p.add_argument("--segundos-camera", type=float, default=20.0)
    p.add_argument("--comparar", nargs=2, metavar=("ANTES", "DEPOIS"))
    p.add_argument("--log", default="AVISO")
    args = p.parse_args()

    if args.comparar:
        comparar(*args.comparar)
        return

    logmod.configurar(args.log)
    w, h = (int(v) for v in args.captura.lower().split("x"))

    app = Orquestrador(planta=args.planta, captura=(w, h))
    if args.falsas:
        app.montar_cameras_falsas()
    else:
        app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    laudo, placar = {}, Placar()
    roteiro = roteiro_padrao()
    try:
        laudo = conferir_cameras(app, args.segundos_camera)
        ruins = mostrar_laudo(laudo)

        if args.so_cameras:
            return

        print("\n  A fase 2 leva cerca de "
              f"{sum(p.segundos for p in roteiro) / 60:.0f} minutos.")
        if ruins:
            print("  Com camera ruim, a nota mede o hardware e nao o sistema.")
        input("\n  ENTER para comecar, Ctrl+C para sair. ")

        rodar_roteiro(app, roteiro, placar)
    except KeyboardInterrupt:
        print("\n\n  interrompido — o boletim vale so ate aqui")
    finally:
        app.parar()

    print(f"{LIMPAR}BOLETIM\n")
    print("\n".join(placar.linhas()))

    if laudo:
        ruins = [p for p, d in laudo.items() if d["queixas"]]
        if ruins:
            print(f"\nATENCAO: cameras com problema nesta sessao: "
                  f"{', '.join(ruins)}")
            print("A nota acima carrega esse defeito. Conserte a camera antes")
            print("de concluir qualquer coisa sobre o classificador.")

    destino = gravar(laudo, placar, roteiro, app)
    print(f"\ngravado em {destino}")
    print("compare com outra execucao:")
    print(f"  python ferramentas/conferir.py --comparar OUTRO.json {destino.name}")


if __name__ == "__main__":
    main()
