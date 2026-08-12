"""Aprende como CADA prateleira se parece nas tres cameras.

    python ferramentas/aprender_prateleiras.py

O MESMO ROTEIRO DO GABARITO, COM OUTRA PERGUNTA

`conferir_altura.py` te leva pelas cinco prateleiras e pergunta:

    quantos centimetros o sistema errou?

Esta ferramenta percorre o mesmo caminho e pergunta:

    o que ACONTECE nas tres cameras quando a mao esta ali?

A diferenca decidiu o projeto. A regua precisava do chao, da escala e da
postura ao mesmo tempo — e as tres quebraram, uma por dia, sempre produzindo
um numero plausivel e errado. A assinatura nao precisa de nenhuma das tres.

    A regua mede o quanto erramos. A assinatura mede o que acontece. A
    segunda responde a pergunta que a loja faz: QUAL prateleira.

O QUE ELE GRAVA, E NADA DISSO E METRICO

    alcance        pulso no corpo, em fracao de tronco (0 = quadril, 1 = ombro)
    coxa           verticalidade — 1 em pe, ~0,2 agachado
    braco          ao_lado / estendido / levantado
    encolhimento   quanto a caixa encolheu, visto do alto
    quem viu       qual camera enxergou o pulso — e qual PERDEU

Perder o pulso e evidencia: uma webcam de mesa perde o que sobe demais ou
desce demais, e qual delas perdeu diz de que lado da faixa a mao estava.

A HONESTIDADE DO RELATORIO: TREINO E TESTE SEPARADOS

Conferir o metodo com os mesmos quadros que o ensinaram sempre da nota alta —
ele decorou. Entao a primeira metade de cada prateleira vira assinatura, e a
SEGUNDA metade e classificada como se fosse nova.

    Metrica medida no proprio treino nao mede o metodo: mede a memoria dele.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from ferramentas.conferir_altura import (                      # noqa: E402
    _mostrar_chamada, chamada_das_cameras,
)
from src.acao.prateleira import (                              # noqa: E402
    Assinatura, ClassificadorDePrateleira, evidencia_de, pesos_medidos,
)
from src.app.orquestrador import Orquestrador                  # noqa: E402
from src.nucleo import log as logmod                           # noqa: E402
from src.nucleo.voz import (                                   # noqa: E402
    Voz, apito_de_fim, apito_de_inicio,
)

LIMPAR = "\033[H\033[J"
DESTINO = RAIZ / "config" / "prateleiras.json"

# Tolerancia minima. Um agrupamento apertado demais recusaria a proxima
# repeticao do mesmo gesto — ninguem pousa a mao duas vezes no mesmo ponto.
TOLERANCIA_MINIMA = {"alcance": 0.12, "alcance_2d": 0.12,
                     "coxa": 0.06, "encolhimento": 0.05}


def encolhimento_de(app, pessoa_id):
    """Quanto a caixa encolheu, vindo do motor.

    A primeira versao montava a conta aqui fora, com
    `caixas_por_id.get(pessoa_id)`. Mas aquele dicionario e indexado pelo id
    do RASTREADOR, e nao pelo da pessoa — dois espacos de identificadores com
    a mesma cara. A busca nunca casava, e o sinal saiu `--` na colheita
    inteira de 12/08 sem levantar erro nenhum.

        Id que parece id mas e de outro dominio nao falha: devolve None e some.
    """
    return app.espacial.encolhimentos.get(pessoa_id)


def colher(app, voz, prateleira, segundos, n, total, lado):
    """Uma prateleira: coleta evidencias enquanto a mao esta la."""
    nome, altura = prateleira["nome"], prateleira["altura"]

    voz.dizer(f"{nome}. {altura*100:.0f} centimetros.", esperar=True)
    print(f"{LIMPAR}  {n} de {total}:  {nome}   ({altura:.2f} m)\n")
    print(f"      PEGUE algo dessa prateleira com a mao {lado.upper()},")
    print("      do jeito que voce pegaria de verdade. Segure.\n")
    for falta in range(4, 0, -1):
        print(f"\r      comeca em {falta}...   ", end="", flush=True)
        time.sleep(1.0)
    print()

    voz.dizer("Ja!")
    apito_de_inicio()

    colhidas, sem_pessoa = [], 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        if app.passo() is None:
            time.sleep(0.005)
            continue

        leituras = app.espacial.leituras
        if not leituras:
            sem_pessoa += 1
            continue

        pid = sorted(leituras)[0]
        ev = evidencia_de(leituras[pid], lado=lado,
                          encolhimento=encolhimento_de(app, pid))
        if ev is not None and not ev.vazia():
            colhidas.append(ev)

        falta = segundos - (time.monotonic() - t0)
        print(f"{LIMPAR}  {n} de {total}:  {nome}   {falta:4.1f} s\n")
        print(f"      PEGUE com a mao {lado.upper()} e SEGURE\n")
        if colhidas:
            u = colhidas[-1]
            print(f"      alcance {_ou(u.alcance)}   coxa {_ou(u.coxa)}   "
                  f"encolhimento {_ou(u.encolhimento)}")
            print(f"      braco {u.braco or '--'}      "
                  f"viu: frontal {u.viu_frontal}  lateral {u.viu_lateral}")
        print(f"\n      amostras {len(colhidas)}")
        if sem_pessoa:
            print(f"      quadros sem ninguem: {sem_pessoa}")

    apito_de_fim()
    voz.dizer("Peguei." if colhidas else "Nao consegui ver.")

    # AS BORDAS DA JANELA NAO SAO O GESTO: SAO A ENTRADA E A SAIDA DELE.
    #
    # Medido em 12/08: a p5 saiu com tolerancia 0,59 no alcance — enorme, e
    # grande o bastante para a faixa dela cobrir a p4 inteira. O bracO nao
    # aparece pronto no apito: ele sobe, para, e desce. Os quadros do sobe e
    # do desce dizem "estou a caminho", nao "estou aqui".
    #
    #     Uma janela de tempo nao e uma janela de gesto. Medir a transicao
    #     junto com o alvo alarga a faixa ate ela invadir a vizinha.
    return _miolo(colhidas), sem_pessoa


def _miolo(evidencias, fora=0.2):
    """Descarta a primeira e a ultima fatia da janela. Ver `colher`."""
    n = len(evidencias)
    if n < 10:
        return evidencias
    corte = int(n * fora)
    return evidencias[corte:n - corte]


def _ou(v, casas=2):
    return "--" if v is None else f"{v:.{casas}f}"


def resumir(prateleira, evidencias):
    """Transforma as evidencias colhidas na `Assinatura` daquela prateleira.

    Centro = mediana, que ignora o quadro esquisito. Tolerancia = metade do
    intervalo interquartil, com piso: a mediana diz onde o gesto mora e o IQR
    diz o quanto ele varia quando a MESMA pessoa repete o MESMO gesto.
    """
    def faixa(campo):
        vs = [getattr(e, campo) for e in evidencias
              if getattr(e, campo) is not None]
        if len(vs) < 4:
            return None
        q = statistics.quantiles(vs, n=4)
        tol = max((q[2] - q[0]) / 2.0, TOLERANCIA_MINIMA[campo])
        return (round(statistics.median(vs), 4), round(tol, 4))

    bracos = Counter(e.braco for e in evidencias if e.braco)
    n = sum(bracos.values())

    def fracao(campo):
        vs = [getattr(e, campo) for e in evidencias
              if getattr(e, campo) is not None]
        return round(sum(vs) / len(vs), 3) if vs else None

    return Assinatura(
        id=prateleira["id"], nome=prateleira["nome"],
        altura=prateleira["altura"],
        alcance=faixa("alcance"), alcance_2d=faixa("alcance_2d"),
        coxa=faixa("coxa"),
        encolhimento=faixa("encolhimento"),
        bracos={k: round(v / n, 3) for k, v in bracos.items()} if n else {},
        visto_frontal=fracao("viu_frontal"),
        visto_lateral=fracao("viu_lateral"),
        amostras=len(evidencias),
    )


def conferir(volta_treino, volta_teste, prateleiras):
    """As cinco se separam? Treina numa VOLTA e testa em OUTRA.

    O DEFEITO DA PRIMEIRA VERSAO, ENCONTRADO EM 12/08

    Ela cortava os 6 segundos de UM gesto ao meio: os primeiros 3 s ensinavam,
    os ultimos 3 s conferiam. Mas os ultimos 3 s sao o mesmo instante, a mesma
    posicao no chao, a mesma luz. Estudar uma questao e depois responder
    aquela mesma questao.

    O dado mostrou o preco. A visibilidade do pulso, entre duas colheitas
    separadas por 50 minutos:

        12:09   0,00  0,48  0,67  0,72  0,09
        12:59   0,36  0,00  0,25  0,98  0,31

    Nada a ver uma com a outra — e era o sinal com o MAIOR peso medido. Ele
    separava dentro da sessao e nao descrevia a prateleira: descrevia onde a
    pessoa estava de pe naquele dia. O teste antigo nao tinha como ver isso,
    porque as duas metades vinham do mesmo minuto.

        Treino e teste tirados do mesmo gesto continuo medem consistencia,
        nao reprodutibilidade. Sao perguntas diferentes, e a facil nao serve
        para decidir se o metodo vai funcionar amanha.

    Agora sao duas voltas completas pelas cinco prateleiras, com a pessoa
    saindo e voltando entre elas. A segunda volta e uma situacao nova de
    verdade — e o acerto contra ela e o unico numero que autoriza ligar isto
    no laco real.
    """
    treino = ClassificadorDePrateleira(janela=1)
    for p in prateleiras:
        evs = volta_treino.get(p["id"], [])
        if len(evs) >= 8:
            treino.declarar(resumir(p, evs))

    if len(treino.assinaturas) < 2:
        return None, None

    matriz, acertos, total = {}, 0, 0
    for p in prateleiras:
        evs = volta_teste.get(p["id"], [])
        if len(evs) < 4:
            continue
        c = ClassificadorDePrateleira(treino.assinaturas, janela=1)
        votos = Counter()
        for i, ev in enumerate(evs):
            palpite = c.observar(f"t{i}", ev)
            if palpite:
                votos[palpite.prateleira] += 1
        matriz[p["id"]] = votos
        acertos += votos.get(p["id"], 0)
        total += sum(votos.values())

    return matriz, (acertos / total if total else 0.0)


def boletim(colheita, prateleiras, assinaturas, voltas=None):
    linhas = ["ASSINATURA DE CADA PRATELEIRA", "",
              f"{'PRATELEIRA':22} {'ALCANCE':>14} {'COXA':>12} "
              f"{'ENCOLHIM.':>12}  BRACO"]

    for a in assinaturas:
        def f(par):
            return "--" if par is None else f"{par[0]:+.2f}+-{par[1]:.2f}"

        braco = max(a.bracos, key=a.bracos.get) if a.bracos else "--"
        linhas.append(f"{a.nome[:22]:22} {f(a.alcance):>14} "
                      f"{f(a.coxa):>12} {f(a.encolhimento):>12}  "
                      f"{braco} ({a.amostras} amostras)")

    linhas += ["", "  QUEM ENXERGOU O PULSO EM CADA ALTURA:"]
    for a in assinaturas:
        linhas.append(f"    {a.nome[:22]:22} frontal {_ou(a.visto_frontal, 0 if a.visto_frontal is None else 2):>6}"
                      f"   lateral {_ou(a.visto_lateral, 0 if a.visto_lateral is None else 2):>6}")
    linhas += ["    Perder o pulso e EVIDENCIA: qual camera perdeu diz de que",
               "    lado da faixa a mao estava."]

    # QUAL SINAL REALMENTE SEPARA. E o diagnostico mais util do relatorio:
    # ele diz onde investir camera e luz, em vez de deixar adivinhar.
    pesos = pesos_medidos(assinaturas)
    linhas += ["", "  PODER DE CADA SINAL  (medido, nao escolhido):"]
    for k, v in sorted(pesos.items(), key=lambda kv: -kv[1]):
        barra = "#" * min(30, int(v * 3))
        marca = "" if v > 0 else "   <-- mudo nesta colheita"
        linhas.append(f"    {k:16} {v:6.2f}  {barra}{marca}")

    if voltas and len(voltas) >= 2:
        matriz, taxa = conferir(voltas[0], voltas[1], prateleiras)
        titulo = "(treino na VOLTA 1, teste na VOLTA 2 — gesto novo)"
    else:
        # Sem segunda volta, so da para medir consistencia interna. O numero
        # sai, e sai marcado: ele nao autoriza ligar nada.
        meio = {k: v[:len(v) // 2] for k, v in colheita.items()}
        resto = {k: v[len(v) // 2:] for k, v in colheita.items()}
        matriz, taxa = conferir(meio, resto, prateleiras)
        titulo = "(UMA VOLTA SO: mede consistencia, NAO reprodutibilidade)"

    if matriz is None:
        linhas += ["", "  Amostras insuficientes para conferir a separacao."]
        return linhas, 0.0

    linhas += ["", f"  AS CINCO SE SEPARAM?  {titulo}",
               "", "    verdade ->  o que o metodo respondeu"]
    for pid, votos in matriz.items():
        total = sum(votos.values()) or 1
        resposta = "  ".join(f"{k} {v/total:.0%}"
                             for k, v in votos.most_common(3))
        certo = votos.get(pid, 0) / total
        marca = "  OK" if certo >= 0.8 else "  <<< CONFUNDE"
        linhas.append(f"    {pid:10} {resposta}{marca}")

    linhas += ["", f"  ACERTO GERAL: {taxa:.0%}"]
    if taxa >= 0.8 and voltas and len(voltas) >= 2:
        linhas += ["", "  As prateleiras se separam num gesto que o metodo NUNCA",
                   "  viu. Isto autoriza ligar o classificador no laco real."]
    elif taxa >= 0.8:
        linhas += ["", "  Bom — mas medido dentro da MESMA volta. Rode com duas",
                   "  voltas antes de confiar: consistencia nao e reprodutibilidade."]
    else:
        linhas += ["", "  Ainda se confundem. Olhe a matriz acima: QUAIS duas se",
                   "  misturam diz o que falta. Se forem vizinhas, e resolucao;",
                   "  se forem distantes, ha sinal errado ou camera cega."]
    return linhas, taxa


def main():
    p = argparse.ArgumentParser(description="Aprende a assinatura das prateleiras")
    p.add_argument("--estante", default="loja/estante.json")
    p.add_argument("--lado", choices=("direita", "esquerda"), default="direita")
    p.add_argument("--segundos", type=float, default=6.0)
    p.add_argument("--planta", default="loja/bancada.json")
    p.add_argument("--pular", action="append", default=[])
    p.add_argument("--voltas", type=int, default=2,
                   help="quantas passadas pelas cinco. 2 = treina numa, "
                        "testa na outra (o unico jeito de medir reproducao)")
    p.add_argument("--sem-voz", action="store_true")
    p.add_argument("--log", default="AVISO")
    args = p.parse_args()

    estante = json.loads((RAIZ / args.estante).read_text(encoding="utf-8"))
    prateleiras = [x for x in estante["prateleiras"] if x["id"] not in args.pular]

    logmod.configurar(args.log)
    voz = Voz(ligada=not args.sem_voz)

    app = Orquestrador(planta=args.planta, captura=(640, 480))
    app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    colheita, voltas = {}, []
    try:
        estados = chamada_das_cameras(app)
        print(f"{LIMPAR}APRENDER AS PRATELEIRAS\n")
        fora = _mostrar_chamada(estados)
        if fora and input("  Digite CONTINUAR para aprender assim mesmo, "
                          "ou ENTER para sair. ").strip().upper() != "CONTINUAR":
            print("\n  saindo. Ponha a camera de pe e rode de novo.")
            return 1

        print(f"  {estante['nome']}, {len(prateleiras)} prateleiras.\n")
        print("  Pegue de cada prateleira como pegaria um produto de verdade.")
        print("  Agache se for natural agachar. Estique se for natural esticar.")
        print("  NAO force postura: o que o sistema aprender aqui e o que ele")
        print("  vai esperar la na frente.\n")
        print("  Antes, ANDE alguns segundos pela area — a camera do alto")
        print("  precisa medir sua estatura em pe.\n")
        input("  ENTER para comecar. ")

        voz.dizer("Ande um pouco pela area primeiro.", esperar=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 12:
            if app.passo() is None:
                time.sleep(0.005)
                continue
            print(f"{LIMPAR}  ANDE pela area   {12 - (time.monotonic()-t0):4.1f} s\n")
            print(f"  {app.espacial.resumo()['escala']}")

        for volta in range(1, args.voltas + 1):
            if volta > 1:
                voz.dizer(f"Volta {volta}. Saia da area e volte, "
                          "para o teste valer.", esperar=True)
                print(f"{LIMPAR}  VOLTA {volta} de {args.voltas}\n")
                print("  SAIA da area e VOLTE antes de continuar.")
                print("  E isso que torna esta volta um gesto NOVO — se voce")
                print("  ficar parado no mesmo lugar, o teste vira decoreba.\n")
                input("  ENTER quando estiver posicionado de novo. ")

            desta = {}
            for i, prat in enumerate(prateleiras, 1):
                evs, _ = colher(app, voz, prat, args.segundos, i,
                                len(prateleiras), args.lado)
                desta[prat["id"]] = evs
                colheita.setdefault(prat["id"], []).extend(evs)
            voltas.append(desta)
    except KeyboardInterrupt:
        print("\n  interrompido")
    finally:
        voz.calar()
        app.parar()

    assinaturas = [resumir(x, colheita[x["id"]]) for x in prateleiras
                   if len(colheita.get(x["id"], [])) >= 4]
    if not assinaturas:
        print("\n  Nenhuma prateleira teve amostras suficientes.")
        return 1

    linhas, taxa = boletim(colheita, prateleiras, assinaturas, voltas)
    print(f"{LIMPAR}" + "\n".join(linhas))

    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.write_text(json.dumps({
        "_como": "assinatura de cada prateleira nas tres cameras, nao metros",
        "_nao_e": ("faixa de altura. Nenhum campo aqui e metrico — ver "
                   "src/acao/prateleira.py"),
        "estante": estante["id"],
        "lado": args.lado,
        "quando": datetime.now().isoformat(timespec="seconds"),
        "voltas": len(voltas),
        "acerto_fora_do_treino": round(taxa, 3),
        # O BRUTO VIAJA JUNTO COM O RESUMO.
        #
        # A primeira versao gravou so a assinatura. Quando o acerto deu 16% e
        # eu precisei entender POR QUE, o dado ja tinha sido descartado — e a
        # unica saida era pedir ao Eduardo que refizesse a coleta inteira.
        #
        #     Resumo responde a pergunta que voce ja sabia fazer. O bruto
        #     responde a proxima, que so aparece depois do resultado ruim.
        "cruas": {pid: [{k: v for k, v in vars(e).items() if v is not None}
                        for e in evs]
                  for pid, evs in colheita.items()},
        "prateleiras": [{
            "id": a.id, "nome": a.nome, "altura": a.altura,
            "alcance": a.alcance, "alcance_2d": a.alcance_2d,
            "coxa": a.coxa,
            "encolhimento": a.encolhimento, "bracos": a.bracos,
            "visto_frontal": a.visto_frontal, "visto_lateral": a.visto_lateral,
            "amostras": a.amostras,
        } for a in assinaturas],
        "_atencao": [
            "Vale para ESTA pessoa e ESTE arranjo de cameras.",
            "Mexeu em camera? Refaca. Outra pessoa muito mais alta ou baixa?",
            "O alcance e proporcao de corpo e viaja bem; a visibilidade nao.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n  gravado em {DESTINO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
