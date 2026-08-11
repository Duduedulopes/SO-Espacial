"""
Confere a ALTURA DA MAO contra uma estante de alturas conhecidas.

    python ferramentas/conferir_altura.py

Este e o gabarito fisico que faltava desde o comeco. A altura da mao em metros
e o numero mais importante do projeto — e o que vai decidir qual produto foi
pego — e ate agora era o unico que nunca tinha sido comparado com a realidade.

    Todo boletim ate hoje terminou com a mesma linha: "CONFIRA COM FITA
    METRICA. O sistema nao tem como saber se este numero esta certo."

Agora tem.

O QUE ELE MEDE, E POR QUE ISSO E DIFERENTE DE ACERTAR O CENTIMETRO

Para cada prateleira, ele reporta tres coisas:

    VIES        o sistema le sistematicamente alto ou baixo? Quanto?
    DISPERSAO   quanto a leitura treme com a mao parada no mesmo lugar
    ACERTO      a prateleira CERTA seria identificada?

A terceira e a que decide o projeto. Com vao de 40 cm entre prateleiras, um
vies de 5 cm nao atrapalha nada — basta ser conhecido e constante. Uma
dispersao de 20 cm atrapalha tudo, mesmo com vies zero.

    Vies constante se subtrai. Dispersao, nao.

O PULSO NAO E A MAO, E ISSO E UM VIES ESPERADO

O sistema mede o PULSO — e a camada de acao inteira e construida sobre juntas,
nao sobre pontas de dedo. Quando a mao pousa numa prateleira, o pulso fica
alguns centimetros acima da superficie.

Isso e vies, nao erro: e a mesma diferenca em toda prateleira, e some na
subtracao. A ferramenta mede quanto ele vale em vez de supor.

COMO EXECUTAR

Fique de frente para a estante, de lado para a camera frontal, e pouse a mao
na prateleira que a voz pedir — como quem pega um produto, sem forcar postura.
Segure ate ouvir o apito duplo.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.app.orquestrador import Orquestrador              # noqa: E402
from src.nucleo import log as logmod                       # noqa: E402
from src.nucleo.voz import (                               # noqa: E402
    Voz, apito_de_fim, apito_de_inicio,
)

LIMPAR = "\033[H\033[J"
DESTINO = RAIZ / "dados" / "confer"


def medir_uma(app, voz, prateleira, segundos, n, total, lado):
    """Pousa a mao numa prateleira e coleta o que o sistema le."""
    nome, verdade = prateleira["nome"], prateleira["altura"]
    campo = "altura_mao_dir" if lado == "direita" else "altura_mao_esq"

    voz.dizer(f"{nome}. {verdade*100:.0f} centimetros.", esperar=True)
    print(f"{LIMPAR}  {n} de {total}:  {nome}   ({verdade:.2f} m)\n")
    print(f"      POUSE A MAO {lado.upper()} nessa prateleira,")
    print("      como quem pega um produto. Segure.\n")
    for falta in range(4, 0, -1):
        print(f"\r      comeca em {falta}...   ", end="", flush=True)
        time.sleep(1.0)
    print()

    voz.dizer("Ja!")
    apito_de_inicio()

    # OS DOIS SILENCIOS NAO SAO O MESMO SILENCIO.
    #
    # Em 11/08 a prateleira 4 saiu como SEM LEITURA com `sem_leitura = 0`, e o
    # boletim nao tinha como explicar a contradicao: se ninguem ficou sem ser
    # detectado, por que nao houve nenhuma amostra?
    #
    # Porque ha duas maneiras de nao sair numero, e elas pedem consertos
    # OPOSTOS:
    #
    #     sem_pessoa   a camera do alto perdeu a pessoa      -> enquadramento
    #     sem_braco    a pessoa foi lida, o pulso foi        -> alcance, luz,
    #                  recusado por nao ter sido visto          oclusao
    #
    # A primeira e uma falha do rastreio. A segunda e a recusa funcionando
    # como projetada — e recusa funcionando nao devia aparecer com a mesma
    # cara de falha.
    #
    #     Somar dois motivos diferentes num contador so nao simplifica o
    #     relatorio: apaga a pergunta seguinte.
    lidas, medidas, sem_pessoa, sem_braco = [], 0, 0, 0
    # DE ONDE VEIO CADA LEITURA. Sem isso o boletim entrega um numero sem
    # procedencia, e nao ha como saber se as tres cameras trabalharam ou se
    # so uma respondeu enquanto as outras estavam cegas.
    fontes = {}
    escalas = {}
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        if app.passo() is None:
            time.sleep(0.005)
            continue

        acoes = app.espacial.acoes
        if not acoes:
            sem_pessoa += 1
            continue
        acao = acoes[sorted(acoes)[0]][0]
        v = getattr(acao, campo)
        if v is None:
            sem_braco += 1
        else:
            lidas.append(v)
            medidas += int(acao.altura_medida)
            leitura = app.espacial.leituras.get(sorted(acoes)[0])
            if leitura is not None:
                f = getattr(leitura, "fonte_braco_dir" if lado == "direita"
                            else "fonte_braco_esq", "") or "?"
                fontes[f] = fontes.get(f, 0) + 1
                e = getattr(leitura, "fonte_escala", "") or "?"
                escalas[e] = escalas.get(e, 0) + 1

        falta = segundos - (time.monotonic() - t0)
        atual = f"{lidas[-1]:.2f} m" if lidas else "--"
        print(f"{LIMPAR}  {n} de {total}:  {nome}   verdade {verdade:.2f} m"
              f"      {falta:4.1f} s\n")
        print(f"      POUSE A MAO {lado.upper()} nessa prateleira\n")
        print(f"      lendo agora: {atual}      amostras {len(lidas)}")
        if not acoes:
            print("\n      NINGUEM DETECTADO")

    apito_de_fim()
    if lidas:
        voz.dizer(f"{statistics.median(lidas)*100:.0f} centimetros.")
    else:
        voz.dizer("Nao consegui medir.")

    return {
        "prateleira": prateleira["id"],
        "nome": nome,
        "verdade": verdade,
        "lidas": lidas,
        "sem_pessoa": sem_pessoa,
        "sem_braco": sem_braco,
        "fracao_medida": (medidas / len(lidas)) if lidas else 0.0,
        "fontes": fontes,
        "escalas": escalas,
    }


def boletim(resultados, prateleiras):
    linhas = ["ALTURA DA MAO CONTRA A ESTANTE", "",
              f"{'PRATELEIRA':22} {'VERDADE':>8} {'LIDO':>8} {'VIES':>8} "
              f"{'DISPERSAO':>10}  QUADROS CERTOS"]
    alturas = [p["altura"] for p in prateleiras]
    vieses, acertos, validos = [], 0, 0

    for r in resultados:
        if len(r["lidas"]) < 5:
            # SEM LEITURA precisa dizer QUAL dos dois silencios foi.
            pessoa, braco = r.get("sem_pessoa", 0), r.get("sem_braco", 0)
            if pessoa > braco:
                motivo = "perdeu a pessoa"
            elif braco:
                motivo = "nao viu o pulso"
            else:
                motivo = "sem quadro"
            linhas.append(f"{r['nome']:22} {r['verdade']:7.2f}m "
                          f"{'--':>8} {'--':>8} {'--':>10}  "
                          f"SEM LEITURA ({motivo}: {pessoa}p/{braco}b)")
            continue

        validos += 1
        lido = statistics.median(r["lidas"])
        vies = lido - r["verdade"]
        vieses.append(vies)
        disp = (statistics.quantiles(r["lidas"], n=4)[2]
                - statistics.quantiles(r["lidas"], n=4)[0]
                if len(r["lidas"]) >= 4 else 0.0)

        # A PERGUNTA QUE DECIDE O PROJETO: a prateleira certa seria escolhida?
        # Nao "o centimetro bateu", mas "o produto certo seria cobrado".
        #
        # E A CONTA E POR QUADRO, NAO PELA MEDIANA.
        #
        # A primeira versao julgava so a mediana — e uma simulacao com 20 cm de
        # dispersao passou com nota maxima, porque a mediana caia no lugar
        # certo enquanto quase metade dos quadros individuais caia na
        # prateleira vizinha.
        #
        # Em producao nao ha mediana: cada quadro decide sozinho se a mao
        # entrou naquela faixa. Julgar pela mediana mediria um sistema que nao
        # existe.
        #
        #     Metrica agregada esconde exatamente o erro que o uso real
        #     sofre um a um.
        def prateleira_de(v):
            return min(alturas, key=lambda a: abs(a - v))

        por_quadro = sum(abs(prateleira_de(v) - r["verdade"]) < 1e-9
                         for v in r["lidas"]) / len(r["lidas"])
        certo = por_quadro >= 0.9
        acertos += int(certo)
        mais_perto = prateleira_de(lido)

        veredicto = (f"{por_quadro:4.0%}" if certo
                     else f"{por_quadro:4.0%} -> {mais_perto:.2f}m")
        linhas.append(
            f"{r['nome']:22} {r['verdade']:7.2f}m {lido:7.2f}m "
            f"{vies:+7.2f}m {disp:9.2f}m  {veredicto}")

    # QUEM RESPONDEU O QUE. A COMPLEMENTARIDADE TEM QUE APARECER.
    #
    # Observado pelo Eduardo, 11/08: "se for lida apenas pela camera frontal,
    # invalida o teste — elas precisam trabalhar juntas".
    #
    # O codigo ja consulta as duas vistas de pose e usa o alto para a escala.
    # O que faltava era o relatorio DIZER isso. Um numero sem procedencia nao
    # pode ser auditado, e uma complementaridade que nao aparece no boletim
    # nao pode ser verificada.
    total_f, total_e = {}, {}
    for r in resultados:
        for k, v in r.get("fontes", {}).items():
            total_f[k] = total_f.get(k, 0) + v
        for k, v in r.get("escalas", {}).items():
            total_e[k] = total_e.get(k, 0) + v

    if total_f:
        n = sum(total_f.values())
        linhas += ["", "  QUEM RESPONDEU O BRACO:"]
        for k, v in sorted(total_f.items(), key=lambda kv: -kv[1]):
            linhas.append(f"    {k:20} {v:5}  {100*v/n:4.0f}%")
        if len(total_f) == 1:
            linhas.append("    UMA VISTA SO respondeu tudo. A outra estava cega")
            linhas.append("    ou nao viu o pulso — confira o laudo antes de")
            linhas.append("    concluir que a complementaridade funcionou.")
    if total_e:
        n = sum(total_e.values())
        linhas += ["", "  DE ONDE VEIO A ESCALA (altura do quadril):"]
        for k, v in sorted(total_e.items(), key=lambda kv: -kv[1]):
            linhas.append(f"    {k:20} {v:5}  {100*v/n:4.0f}%")

    # A FAIXA UTIL E UMA ESPECIFICACAO, NAO UMA RECLAMACAO.
    #
    # Observado pelo Eduardo antes de rodar: a camera frontal esta apoiada na
    # mesa e nao vai alcancar a prateleira de baixo. Ele esta certo — mas o
    # teste nao precisa que a camera veja a PRATELEIRA, e sim o PULSO naquela
    # altura. O efeito e o mesmo, e o numero que sai dali e valioso:
    #
    #     "o sistema mede a altura da mao entre X e Y metros"
    #
    # Isso e uma especificacao do arranjo. Uma loja projeta as prateleiras
    # dentro da faixa util das cameras, ou poe mais cameras. Saber onde a faixa
    # comeca e acaba vale mais que fingir que ela e infinita.
    #
    #     Limite medido vira requisito de projeto. Limite ignorado vira
    #     defeito em producao.
    medidas = [r["verdade"] for r in resultados if len(r["lidas"]) >= 5]
    cegas = [r for r in resultados if len(r["lidas"]) < 5]
    if medidas:
        linhas.append("")
        linhas.append(f"  FAIXA UTIL MEDIDA: {min(medidas):.2f} m a "
                      f"{max(medidas):.2f} m")
        if cegas:
            fora = ", ".join(f"{r['verdade']:.2f} m" for r in cegas)
            linhas.append(f"  fora da faixa: {fora} — o pulso sai do quadro da")
            linhas.append("  frontal nessas alturas. Nao e defeito de codigo:")
            linhas.append("  e o alcance da camera onde ela esta.")

        # QUANTOS QUADROS RESPONDERAM, POR PRATELEIRA.
        #
        # "Teve leitura" e "foi medida" nao sao a mesma coisa. Em 11/08 a
        # prateleira 4 entregou 5 amostras e recusou 35 — 12% de resposta — e
        # entrou na faixa util com a mesma cara de uma prateleira que
        # respondeu 82%.
        #
        # E a amostra que sobra nao e aleatoria: sobrevivem os quadros em que
        # o pulso estava DENTRO do quadro, ou seja, os menos extremos do
        # gesto. Recusar o que saiu da imagem esta certo; tratar o resto como
        # amostra imparcial, nao.
        #
        #     Taxa de recusa alta nao encolhe a amostra: enviesa a amostra,
        #     e o vies aponta sempre para o centro do quadro.
        linhas.append("")
        linhas.append("  QUANTO CADA ALTURA RESPONDEU:")
        for r in resultados:
            tentativas = (len(r["lidas"]) + r.get("sem_braco", 0)
                          + r.get("sem_pessoa", 0))
            if not tentativas:
                continue
            taxa = len(r["lidas"]) / tentativas
            alerta = "   <<< amostra enviesada" if taxa < 0.4 else ""
            linhas.append(f"    {r['nome']:22} {taxa:4.0%} "
                          f"({len(r['lidas'])} de {tentativas}){alerta}")

    if not validos:
        linhas += ["", "  Nenhuma prateleira medida. A camera frontal precisa",
                   "  ver seu PULSO enquanto voce pousa a mao — nao a",
                   "  prateleira. Se nem a altura do peito funcionou, o",
                   "  problema e enquadramento e nao alcance."]
        return linhas, None

    vies_medio = statistics.median(vieses)
    linhas += ["", f"  PRATELEIRAS COM 90%+ DOS QUADROS CERTOS:  "
                   f"{acertos} de {validos}",
               f"  VIES SISTEMATICO:      {vies_medio:+.2f} m"]

    # VIES CONSTANTE SE SUBTRAI; DISPERSAO NAO.
    #
    # Um vies de 5 cm igual em todas as prateleiras nao atrapalha nada — basta
    # ser conhecido. O que atrapalha e a leitura tremer, ou o vies MUDAR de
    # prateleira para prateleira, porque ai nao ha constante a subtrair.
    if len(vieses) >= 2:
        espalhamento = max(vieses) - min(vieses)
        linhas.append(f"  o vies varia {espalhamento:.2f} m entre prateleiras")
        if espalhamento < 0.08:
            linhas += ["",
                       "  O vies e CONSTANTE. Isso e bom: constante se subtrai.",
                       f"  Somar {-vies_medio:+.2f} m a leitura corrigiria todas."]
        else:
            linhas += ["",
                       "  O vies MUDA conforme a altura. Nao ha constante a",
                       "  subtrair — a relacao entre pulso e prateleira nao e",
                       "  a mesma em cima e embaixo, provavelmente porque a",
                       "  postura do braco muda."]

    linhas += _reta_medida(resultados)

    if acertos == validos:
        linhas += ["", "  A ETAPA D E VIAVEL com estes numeros: cada prateleira",
                   "  foi identificada corretamente. Declarar as faixas na",
                   "  planta e comparar passa a ser aritmetica."]
    else:
        linhas += ["", f"  {validos - acertos} prateleira(s) confundida(s).",
                   "  Antes de concluir que falhou, olhe a RETA acima: leitura",
                   "  ERRADA e leitura INUTIL sao coisas diferentes."]
    return linhas, vies_medio


def _reta_medida(resultados, minimo=3):
    """A leitura nao precisa estar CERTA. Precisa ser SEPARAVEL.

    O QUE A ESTANTE MOSTROU EM 11/08, DEPOIS DE AS CAMERAS FUNCIONAREM

        verdade   0,15   0,55   0,95   1,35   1,90
        lido      0,40   0,95   0,97   1,13   1,41

    O bloco do VIES olhou isso e disse a coisa certa pela razao certa: o vies
    varia 0,89 m entre prateleiras, logo nao ha constante a subtrair. Verdade.

    Mas ele parou cedo. Ajustando uma reta:

        lido = 0,509 x verdade + 0,473        ponto fixo em 0,96 m

    O ponto fixo cai em cima do QUADRIL. Ou seja: a ancora esta certa, e o que
    esta comprimido — pela metade — e o deslocamento do pulso EM RELACAO ao
    quadril. Erro que cresce proporcional a distancia do quadril, para cima e
    para baixo, com a mesma inclinacao dos dois lados.

    Isso e assinatura de estimador de pose regredindo para a media: a pose
    neutra (mao na altura do quadril) e a mais comum no treino e sai exata; as
    poses extremas sao raras e saem puxadas para o centro. Explica tambem a
    dispersao BAIXA nas pontas — estimativa colapsada e muito repetivel.

    E AQUI O QUE IMPORTA PARA A ETAPA D

    Nao precisamos do centimetro. Precisamos saber QUAL PRATELEIRA. Uma funcao
    monotona e repetivel entre a verdade e a leitura e invertivel, e uma
    compressao uniforme nao apaga a ordem: encolhe os vaos, e a pergunta passa
    a ser se o vao encolhido ainda cabe fora do ruido.

        Um instrumento errado e util quando o erro e estavel. Um instrumento
        instavel nao serve nem quando acerta a media.

    E o mesmo argumento que ja tinha sido aceito para o fator 5,25 da escala:
    constante empirica nao precisa ter nome fisico, precisa ser estavel e ser
    medida do mesmo jeito que sera usada.

    Por isso o veredicto final desta ferramenta NAO pode ser "acertou o
    centimetro". Tem que ser a margem: vao entre prateleiras vizinhas, ja
    comprimido, dividido pela dispersao tipica.
    """
    pontos = [(r["verdade"], statistics.median(r["lidas"]),
               _dispersao(r["lidas"]))
              for r in resultados if len(r["lidas"]) >= 5]
    if len(pontos) < minimo:
        return []

    xs = [p[0] for p in pontos]
    ys = [p[1] for p in pontos]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return []

    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = my - a * mx

    # R2 diz se a RETA descreve o que aconteceu. Sem isso, inverter a reta
    # seria aplicar um modelo a dados que nao sao uma reta — e o resultado
    # sairia com a mesma aparencia de rigor.
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    linhas = ["", "  A RETA MEDIDA (leitura em funcao da verdade):",
              f"    lido = {a:.3f} x verdade {b:+.3f}     R2 = {r2:.3f}"]

    if abs(1 - a) > 1e-6:
        linhas.append(f"    ponto fixo (onde acerta sozinho): {b/(1-a):.2f} m")
    if a < 0.85:
        linhas += [f"    o deslocamento do pulso sai COMPRIMIDO a {a:.0%} do real,",
                   "    e o ponto fixo tende a cair no quadril: a ancora esta",
                   "    certa e a extensao do braco e que encolhe."]

    # A MARGEM: O UNICO NUMERO QUE DECIDE A ETAPA D.
    ordenados = sorted(zip(xs, ys))
    vaos = [ordenados[i + 1][1] - ordenados[i][1]
            for i in range(len(ordenados) - 1)]
    dispersoes = [p[2] for p in pontos if p[2] > 0]
    if not vaos or not dispersoes:
        return linhas

    menor_vao = min(vaos)
    tipica = statistics.median(dispersoes)
    margem = menor_vao / tipica if tipica > 1e-9 else float("inf")

    linhas += ["",
               f"    menor vao entre prateleiras vizinhas, NA LEITURA:"
               f" {menor_vao:.2f} m",
               f"    dispersao tipica da leitura:            {tipica:.2f} m",
               f"    MARGEM = {margem:.1f}x"]

    if r2 < 0.9:
        linhas += ["", "    R2 baixo: isto NAO e uma reta. Nao inverta — a",
                   "    relacao entre verdade e leitura ainda nao esta",
                   "    descrita, e um ajuste ruim so daria aparencia de rigor."]
    elif margem >= 3:
        linhas += ["", "    Margem folgada. A leitura esta errada e SERVE:",
                   "    inverter a reta poe cada prateleira no lugar. O que",
                   "    falta e repetir a medicao para saber se a reta e",
                   "    ESTAVEL entre sessoes — uma reta medida uma vez e uma",
                   "    coincidencia com dois parametros."]
    else:
        linhas += ["", "    Margem apertada: o vao comprimido nao cabe fora do",
                   "    ruido. Inverter a reta nao salva — prateleiras vizinhas",
                   "    continuariam se confundindo. Faltam vaos maiores, menos",
                   "    dispersao, ou uma vista que nao comprima."]
    return linhas


def _dispersao(lidas):
    if len(lidas) < 4:
        return 0.0
    q = statistics.quantiles(lidas, n=4)
    return q[2] - q[0]


def chamada_das_cameras(app, espera_s=8.0):
    """Quem esta de pe, ANTES do teste comecar — nao depois dele.

    POR QUE ISTO PRECISOU EXISTIR, 11/08

    O tablet caiu da rede no meio da sessao. O sistema fez tudo certo:
    registrou a falha, agendou nova tentativa, dobrou o intervalo. Mas o
    conferidor seguiu ate o `ENTER para comecar` como se nada tivesse
    acontecido, e o rodape do log estava a doze linhas de distancia da
    pergunta.

    Se o ENTER fosse apertado ali, o teste inteiro rodaria com DUAS cameras e
    o boletim so contaria isso no fim, depois de cinco prateleiras e alguns
    minutos de agachamento. O numero sairia — e sairia sem a vista que existe
    justamente para cobrir o que a frontal perde.

        A hora de descobrir que falta uma camera e antes de a pessoa
        agachar cinco vezes, nao depois.

    E O TESTE DEGRADADO CONTINUA VALENDO — DESDE QUE SEJA ESCOLHIDO

    Rodar com duas cameras nao e proibido: mede o que duas cameras medem, e as
    vezes e exatamente o que se quer saber. O que nao pode e acontecer sozinho.
    Por isso a falta nao aborta nada — ela troca o `ENTER` distraido por uma
    palavra digitada de proposito.

        Degradar em silencio produz um numero que ninguem sabe interpretar.
        Degradar com consentimento produz uma medicao com escopo declarado.
    """
    def estados():
        return {p: f.estado.value for p, f in app.cameras.fontes.items()}

    t0 = time.monotonic()
    while time.monotonic() - t0 < espera_s:
        app.passo()
        if all(e == "online" for e in estados().values()):
            break
        time.sleep(0.05)
    return estados()


def _mostrar_chamada(estados):
    """Imprime a chamada e devolve os papeis que nao responderam."""
    fora = [p for p, e in estados.items() if e != "online"]

    print("  CAMERAS:")
    for papel, estado in estados.items():
        marca = "" if estado == "online" else "   <<< NAO VAI RESPONDER NADA"
        print(f"    {papel:10} {estado}{marca}")
    print()

    if not fora:
        return fora

    # O QUE CADA VISTA CUSTA, DITO PELO NOME. Um aviso generico ("camera
    # offline") nao ajuda a decidir; saber QUAL pergunta fica sem resposta,
    # sim.
    perde = {
        "alto": ("a posicao no chao, a estatura em metros e o rumo do corpo. "
                 "Sem ela a altura da mao volta a sair estimada pelo tronco"),
        "frontal": "a altura do pulso vista de frente — a fonte principal",
        "lateral": ("a reserva que responde quando a frontal perde o pulso. "
                    "Sem ela, complementaridade nenhuma e demonstrada"),
    }
    for papel in fora:
        print(f"  SEM '{papel}' voce perde {perde.get(papel, 'essa vista')}.")
    print()
    return fora


def main():
    p = argparse.ArgumentParser(description="Confere a altura da mao")
    p.add_argument("--estante", default="loja/estante.json")
    p.add_argument("--lado", choices=("direita", "esquerda"), default="direita")
    p.add_argument("--segundos", type=float, default=5.0)
    p.add_argument("--planta", default="loja/bancada.json")
    p.add_argument("--pular", action="append", default=[],
                   help="id de prateleira que voce nao alcanca. Ex: --pular p5")
    p.add_argument("--sem-voz", action="store_true")
    p.add_argument("--log", default="AVISO")
    args = p.parse_args()

    estante = json.loads((RAIZ / args.estante).read_text(encoding="utf-8"))
    prateleiras = [p for p in estante["prateleiras"]
                   if p["id"] not in args.pular]

    logmod.configurar(args.log)
    voz = Voz(ligada=not args.sem_voz)

    app = Orquestrador(planta=args.planta, captura=(640, 480))
    app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    resultados = []
    try:
        estados = chamada_das_cameras(app)

        print(f"{LIMPAR}CONFERIR A ALTURA DA MAO\n")
        fora = _mostrar_chamada(estados)
        if fora and input("  Digite CONTINUAR para medir assim mesmo, "
                          "ou ENTER para sair. ").strip().upper() != "CONTINUAR":
            print("\n  saindo. Ponha a camera de pe e rode de novo.")
            return 1

        print(f"  {estante['nome']}, {len(prateleiras)} prateleiras.\n")
        print("  Fique DE LADO para a camera frontal, de frente para a estante.")
        print("  A cada pedido, pouse a mao na prateleira dita e SEGURE.")
        print("  Nao force postura: pegue como pegaria um produto.\n")
        print("  Antes, ANDE alguns segundos pela area — a escala vertical")
        print("  precisa te ver em pe e andando para medir sua estatura.\n")
        input("  ENTER para comecar. ")

        voz.dizer("Ande um pouco pela area primeiro.", esperar=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 12:
            if app.passo() is None:
                time.sleep(0.005)
                continue
            print(f"{LIMPAR}  ANDE pela area   {12 - (time.monotonic()-t0):4.1f} s\n")
            print(f"  {app.espacial.resumo()['escala']}")

        for i, prat in enumerate(prateleiras, 1):
            resultados.append(medir_uma(app, voz, prat, args.segundos,
                                        i, len(prateleiras), args.lado))
    except KeyboardInterrupt:
        print("\n  interrompido")
    finally:
        voz.calar()
        app.parar()

    linhas, vies = boletim(resultados, estante["prateleiras"])
    print(f"{LIMPAR}" + "\n".join(linhas))

    escala = app.espacial.resumo()["escala"]
    print(f"\n  {escala}")
    if "NAO CALIBRADA" in escala:
        print("  A altura saiu ESTIMADA pelo tronco (+-8 cm). Com a escala")
        print("  calibrada o erro cai para +-3 cm — rode calibrar_escala.py.")

    DESTINO.mkdir(parents=True, exist_ok=True)
    destino = DESTINO / f"altura_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    destino.write_text(json.dumps({
        "quando": datetime.now().isoformat(timespec="seconds"),
        "estante": estante["id"],
        "lado": args.lado,
        "escala": escala,
        "vies_mediano": vies,
        "prateleiras": [
            {k: v for k, v in r.items() if k != "lidas"}
            | {"lidas": [round(x, 3) for x in r["lidas"]]}
            for r in resultados],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  gravado em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
