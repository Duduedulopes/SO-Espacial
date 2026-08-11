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

from src.acao.gabarito import (                            # noqa: E402
    Placar, roteiro_padrao, voltar_ao_meio,
)
from src.app.orquestrador import Orquestrador              # noqa: E402
from src.nucleo import log as logmod                       # noqa: E402
from src.nucleo.voz import Voz, apito_de_fim, apito_de_inicio  # noqa: E402

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
def rodar_roteiro(app, roteiro, placar, voz=None, esperar_enter=False,
                  preparo_s=4.0):
    """Guia a pessoa passo a passo e anota o que o sistema leu.

    A INSTRUCAO CHEGA PELO OUVIDO, NAO PELA TELA

    A primeira versao mandava andar pela sala e ler a tela ao mesmo tempo. A
    pessoa ficou perto do monitor, mal se deslocou, e dez acoes reprovaram por
    uma causa que era o proprio aparato. Ver `src/nucleo/voz.py`.

    O CICLO DE CADA PASSO

        fala a instrucao   ->  preparo (ou ENTER)  ->  APITO
        conta a janela     ->  APITO DUPLO         ->  fala o resultado

    Os apitos existem porque a fala tem duracao variavel e nao serve de marco
    temporal: quem escuve "ande para frente" nao sabe se o cronometro comecou
    na primeira ou na ultima silaba.

    E o resultado e falado no fim porque, com a pessoa longe do computador,
    um boletim so no final obrigaria a refazer tudo para descobrir que um
    passo falhou.
    """
    voz = voz or Voz(ligada=False)

    for n, passo in enumerate(roteiro, 1):
        voz.dizer(f"Passo {n}. {passo.instrucao}")
        _preparar(n, len(roteiro), passo, esperar_enter, preparo_s, voz)

        apito_de_inicio()
        t0 = time.monotonic()
        while True:
            decorrido = time.monotonic() - t0
            if decorrido > passo.segundos:
                break
            if app.passo() is None:
                time.sleep(0.005)
                continue

            acoes = app.espacial.acoes
            pessoas = dict(app.gemeo.pessoas)
            placar.anotar(passo, acoes, decorrido, pessoas)
            _tela(n, len(roteiro), passo, decorrido, acoes, pessoas)

        apito_de_fim()
        voz.dizer(_veredicto_falado(placar, passo))


def rodar_travado(app, roteiro, placar, voz=None, limite_s=25.0,
                  confirmar_s=0.8, preparo_s=3.0):
    """Cada passo ESPERA ate o sistema reconhecer a acao. Sem cronometro.

    IDEIA DO EDUARDO, 11/08, E ELA MUDA A PERGUNTA DO TESTE

    Com cronometro fixo a pergunta e "que fracao dos quadros estava certa?".
    Essa resposta mistura um sistema LENTO com um sistema ERRADO — os dois dao
    40%, e os consertos sao opostos.

    Travando, a pergunta vira "quanto tempo o sistema levou para reconhecer?".
    E `nunca reconheceu` deixa de ser uma porcentagem baixa para virar uma
    falha alta e clara.

        Nota baixa nao diz se o sistema e lento ou se esta errado. O tempo diz.

    E ha um ganho pratico: quem executa nao precisa mais adivinhar quanto tempo
    segurar a posicao. Segura ate ouvir que confirmou.

    POR QUE HA LIMITE MESMO ASSIM

    Sem limite, uma acao que o sistema nao sabe ler trava a sessao para sempre
    e nada e medido. O limite generoso deixa a acao falhar e o roteiro seguir —
    e o tempo gasto ate desistir vira o proprio dado.
    """
    voz = voz or Voz(ligada=False)

    for n, passo in enumerate(roteiro, 1):
        voz.dizer(f"Passo {n}. {passo.instrucao}")

        if passo.reposiciona or passo.eixo is None:
            _contar(app, passo, passo.segundos, placar, n, len(roteiro))
            continue

        _preparo_curto(n, len(roteiro), passo, preparo_s)
        apito_de_inicio()

        t0 = time.monotonic()
        t_primeira = t_confirmada = None
        acumulado = 0.0
        ultimo = t0

        while True:
            decorrido = time.monotonic() - t0
            if decorrido > limite_s:
                break
            if app.passo() is None:
                time.sleep(0.005)
                continue

            agora = time.monotonic()
            dt, ultimo = agora - ultimo, agora

            acoes = app.espacial.acoes
            pessoas = dict(app.gemeo.pessoas)
            # `acomodacao_s=0` porque a espera JA e a acomodacao: nao ha janela
            # a descontar quando o passo so termina depois de dar certo.
            placar.anotar(passo, acoes, decorrido + 999, pessoas)

            certo = _esta_certo(passo, acoes, pessoas)
            if certo:
                if t_primeira is None:
                    t_primeira = decorrido
                acumulado += dt
                if acumulado >= confirmar_s:
                    t_confirmada = decorrido
                    break
            else:
                # SUSTENTAR, E NAO SO ACERTAR UM QUADRO.
                #
                # Um unico quadro certo no meio do ruido nao e reconhecimento.
                # Zerar o acumulo a cada discordancia e a mesma regra do
                # `Estavel` do classificador, aplicada aqui.
                acumulado = 0.0

            _tela_travada(n, len(roteiro), passo, decorrido, acoes, pessoas,
                          certo, acumulado, confirmar_s, limite_s)

        espera = time.monotonic() - t0
        placar.marcar_tempo(passo, t_primeira, t_confirmada, espera)
        apito_de_fim()
        voz.dizer(_veredicto_travado(passo, placar, t_confirmada))


def _esta_certo(passo, acoes, pessoas):
    if not acoes:
        return False
    pid = sorted(acoes)[0]
    if getattr(pessoas.get(pid), "prevendo", 0):
        return False          # posicao prevista nao confirma nada
    return getattr(acoes[pid][0], passo.eixo, None) in passo.certo


def _contar(app, passo, segundos, placar, n, total):
    """Passo sem nota: so espera o tempo declarado."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        if app.passo() is None:
            time.sleep(0.005)
            continue
        _tela(n, total, passo, time.monotonic() - t0,
              app.espacial.acoes, dict(app.gemeo.pessoas))


def _preparo_curto(n, total, passo, preparo_s):
    print(f"{LIMPAR}  PASSO {n} de {total}\n")
    print(f"      >>> {passo.instrucao} <<<")
    print("          SEGURE ate ouvir que confirmou\n")
    for falta in range(int(preparo_s), 0, -1):
        print(f"\r      comeca em {falta}...   ", end="", flush=True)
        time.sleep(1.0)
    print()


def _tela_travada(n, total, passo, decorrido, acoes, pessoas, certo,
                  acumulado, confirmar_s, limite_s):
    print(f"{LIMPAR}  PASSO {n} de {total}      {decorrido:5.1f}s / "
          f"{limite_s:.0f}s\n")
    print(f"      >>> {passo.instrucao} <<<")
    print("          SEGURE ate ouvir que confirmou\n")

    if certo:
        barras = int(20 * min(1.0, acumulado / confirmar_s))
        print(f"      CERTO  [{'#' * barras}{'.' * (20 - barras)}]  "
              f"segurando {acumulado:.1f}s de {confirmar_s:.1f}s")
    else:
        print("      ainda nao...")
    print()

    if not acoes:
        print("      lendo agora:  NINGUEM DETECTADO")
        return
    for pid, (a, _) in sorted(acoes.items()):
        atual = getattr(a, passo.eixo, "?")
        print(f"      lendo agora:  #{pid}  {passo.eixo} = {atual}")
        print(f"                    esperado: "
              f"{' ou '.join(passo.certo)}")


def _veredicto_travado(passo, placar, t_confirmada):
    if t_confirmada is not None:
        return f"Confirmado em {t_confirmada:.0f} segundos."

    c = placar.contagens.get(passo.acao)
    confusao = c.pior_confusao[0] if c else None
    if confusao:
        return f"Nao reconheceu. Leu {confusao.replace('_', ' ')}."
    return "Nao reconheceu."


def _preparar(n, total, passo, esperar_enter, preparo_s, voz):
    """Tempo entre ouvir a instrucao e a contagem comecar."""
    print(f"{LIMPAR}  PASSO {n} de {total}\n")
    print(f"      >>> {passo.instrucao} <<<")
    if passo.instrucao_extra:
        print(f"          {passo.instrucao_extra}")
    print()

    # PASSO DE REPOSICIONAMENTO NAO TEM PREPARO. Ele JA e o preparo.
    #
    # Contar quatro segundos antes de "va ate a borda de tras" e fazer a pessoa
    # esperar duas vezes pela mesma coisa. Com dezoito passos, esse preparo
    # somava mais de um minuto de espera pura.
    if passo.reposiciona:
        return

    # O ENTER PRENDE A PESSOA AO TECLADO — E O TECLADO ESTA NUMA BORDA.
    #
    # Observado pelo Eduardo em 11/08: `--passo-a-passo` deixa quem executa na
    # borda da frente, junto do notebook, e a instrucao seguinte manda ir ate a
    # borda de TRAS. As duas brigam, e a segunda perde.
    #
    # Com voz, o ENTER deixou de ser necessario: a contagem regressiva e
    # ouvida de qualquer lugar da sala. O modo continua existindo para
    # depuracao, mas nao e mais o caminho recomendado.
    if esperar_enter:
        input("      ENTER quando estiver em posicao (o apito comeca a contar) ")
        return

    for falta in range(int(preparo_s), 0, -1):
        print(f"\r      comeca em {falta}...   ", end="", flush=True)
        time.sleep(1.0)
    print()


def _veredicto_falado(placar, passo):
    """Uma frase curta, para quem esta do outro lado da sala.

    Diz a nota E o que o sistema leu no lugar quando errou. Sem a segunda
    parte, "vinte por cento" nao diz se o problema foi nao ver ninguem, ver e
    classificar errado, ou o sistema ter se abstido.
    """
    if passo.eixo is None:
        return "Aquecimento terminado."

    c = placar.contagens.get(passo.acao)
    if not c or not c.total:
        return "Nao consegui medir esse passo."

    nota = round(c.nota * 100)
    if nota >= 85:
        return f"Certo. {nota} por cento."

    confusao, frac = c.pior_confusao
    if confusao:
        return f"Falhou. {nota} por cento. Leu {confusao.replace('_', ' ')}."
    if c.aproveitamento > 0.8:
        return f"{nota} por cento, mas sem erro: o sistema se absteve."
    return f"Falhou. {nota} por cento."


def _tela(n, total, passo, decorrido, acoes, pessoas):
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
        p = pessoas.get(pid)
        marca = "  (posicao PREVISTA)" if (p and p.prevendo) else ""
        print(f"      lendo agora:  #{pid}  {a.locomocao} / {a.postura}"
              f"   {a.velocidade_ms:.2f} m/s{marca}")
        for lado, estado, altura in (("E", a.braco_esquerdo, a.altura_mao_esq),
                                     ("D", a.braco_direito, a.altura_mao_dir)):
            metros = "    --" if altura is None else f"{altura:5.2f}m"
            print(f"                    braco {lado}  {estado:14} {metros}")


def diagnostico_da_cascata(app):
    """O que estava APRENDIDO quando a sessao rodou. Vem antes das notas.

    POR QUE ESTA SECAO EXISTE

    MEDIDO EM 11/08: o boletim mostrou dez acoes reprovadas e a leitura natural
    foi "dez coisas quebradas". Eram dez sintomas de UMA causa — a pessoa mal
    se deslocou, entao o azimute e a inclinacao nunca convergiram, e tudo que
    depende deles degradou junto.

    Sem esta secao, o boletim manda consertar dez lugares. Com ela, manda
    andar mais.

        Nota que nao mostra a cadeia de dependencia manda consertar o lugar
        errado — mesmo defeito do `rejeitadas plausibilidade 358` de 10/08,
        que era verdadeiro e nao apontava conserto nenhum.
    """
    e = app.espacial.resumo()
    corpo = app.espacial.corpo
    incl = app.espacial.inclinacao

    linhas = ["O QUE O SISTEMA APRENDEU NESTA SESSAO",
              f"  azimute da camera   {e['corpo']}",
              f"  inclinacao          {e['inclinacao']}",
              f"  altura de pessoa    {e['altura']}"]

    faltando = []
    if not corpo.azimute.confiavel:
        faltando.append(
            "AZIMUTE nao convergiu. Sem ele o sistema nao sabe para que lado a\n"
            "    camera aponta, e NAO consegue distinguir andar para frente de\n"
            "    andar de lado — responde `andando` e conta como POBRE.")
    if not incl.confiavel:
        faltando.append(
            "INCLINACAO nao convergiu. Sem ela o esqueleto chega girado pelo\n"
            "    angulo da lente, e bracos e agachamento saem errados.")

    if faltando:
        linhas += ["", "  ISTO EXPLICA AS REPROVACOES ABAIXO:"]
        linhas += [f"  -> {t}" for t in faltando]
        linhas += [
            "",
            "  Os dois so aprendem com quem ANDA acima de 0,25 m/s. Ande de",
            "  verdade no passo de aquecimento — alguns metros de ida e volta,",
            "  nao passos no lugar. Nada abaixo depende de codigo enquanto",
            "  estas duas linhas nao estiverem `ativo`."]
    else:
        linhas += ["", "  Os dois convergiram: as notas abaixo medem o "
                       "classificador, nao a falta de caminhada."]
    return linhas


def escolher(nomes):
    """Filtra o roteiro, mas NUNCA tira o aquecimento.

    O `EstimadorDeAzimute` so aprende com quem anda, e o `EstimadorDeInclinacao`
    tambem. Rodar `--acao andar_frente` sozinho mediria um sistema que ainda
    nao teve materia-prima — e reprovaria por uma coisa que o recorte causou.

    MEDIDO EM 11/08: sem caminhada suficiente, dez acoes reprovaram de uma vez.
    Era uma causa so, e ela estava no comeco da cadeia.

        Um componente que exige movimento nao pode ser julgado por uma sessao
        sem movimento.                                        — caderno, 10/08
    """
    roteiro = roteiro_padrao()
    if not nomes:
        return roteiro

    pedidos = set(nomes)
    desconhecidos = pedidos - {p.acao for p in roteiro}
    if desconhecidos:
        raise SystemExit(
            f"acao desconhecida: {', '.join(sorted(desconhecidos))}\n"
            f"  veja os nomes com:  python ferramentas/conferir.py --listar")

    # O AQUECIMENTO ENTRA SEMPRE; O RETORNO ENTRA ONDE FOR PRECISO.
    #
    # No roteiro inteiro, `andar_frente` e seguido de `andar_tras` e o par se
    # desfaz sozinho. Recortado com `--acao andar_frente`, esse retorno some —
    # e repetir o passo empurraria a pessoa para fora do quadro.
    #
    # Entao o retorno e acrescentado depois de todo passo que desloca, exceto
    # quando o proprio passo seguinte ja e o par que desfaz o deslocamento.
    # Os retornos do roteiro completo sao DESCARTADOS aqui e recolocados
    # abaixo. Mante-los seria empilhar dois seguidos: um do roteiro e um da
    # regra, com a pessoa esperando doze segundos entre dois passos.
    selecionados = [p for p in roteiro
                    if not p.reposiciona and (p.eixo is None
                                              or p.acao in pedidos)]

    escolhido = []
    for i, passo in enumerate(selecionados):
        escolhido.append(passo)
        if not passo.desloca:
            continue
        proximo = selecionados[i + 1] if i + 1 < len(selecionados) else None
        if proximo is not None and proximo.desloca:
            continue          # o par seguinte ja traz a pessoa de volta
        escolhido.append(voltar_ao_meio())

    # Um retorno no fim nao serve para nada: a sessao acabou.
    if escolhido and escolhido[-1].reposiciona:
        escolhido.pop()
    return escolhido


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

    p.add_argument("--acao", action="append", metavar="NOME",
                   help="testa so estas acoes. O aquecimento entra sempre, "
                        "porque o azimute depende dele. Pode repetir.")
    p.add_argument("--listar", action="store_true",
                   help="mostra os nomes das acoes e sai")
    p.add_argument("--passo-a-passo", action="store_true",
                   help="espera ENTER antes de cada passo, em vez de contagem")
    p.add_argument("--preparo", type=float, default=4.0,
                   help="segundos entre ouvir a instrucao e comecar a contar")
    p.add_argument("--sem-voz", action="store_true")

    p.add_argument("--cronometrado", action="store_true",
                   help="modo antigo: cada passo dura um tempo fixo e a nota e "
                        "a fracao de quadros certos. O padrao e TRAVADO — "
                        "espera reconhecer e mede quanto demorou")
    p.add_argument("--limite", type=float, default=25.0,
                   help="segundos ate desistir de um passo travado")
    p.add_argument("--confirmar", type=float, default=0.8,
                   help="segundos de leitura certa SUSTENTADA para confirmar")
    args = p.parse_args()

    if args.listar:
        for passo in roteiro_padrao():
            eixo = passo.eixo or "(aquecimento, sem nota)"
            print(f"  {passo.acao:22} {eixo:16} {passo.instrucao}")
        return

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
    roteiro = escolher(args.acao)
    voz = Voz(ligada=not args.sem_voz)
    try:
        laudo = conferir_cameras(app, args.segundos_camera)
        ruins = mostrar_laudo(laudo)

        if args.so_cameras:
            return

        print(f"\n  {len(roteiro)} passo(s).")
        print("  As instrucoes sao FALADAS — voce nao precisa olhar a tela.")
        if args.cronometrado:
            total = sum(p.segundos + args.preparo for p in roteiro)
            print(f"  Modo CRONOMETRADO: cada passo dura um tempo fixo. "
                  f"~{total / 60:.0f} min.")
            print("  Um apito comeca a contagem; dois apitos terminam.")
        else:
            print("  Modo TRAVADO: cada passo espera ate o sistema RECONHECER.")
            print("  Segure a posicao ate ouvir que confirmou. Nao ha pressa —")
            print(f"  ele desiste sozinho depois de {args.limite:.0f}s.")
        if ruins:
            print("  Com camera ruim, o resultado mede o hardware, nao o sistema.")
        input("\n  ENTER para comecar, Ctrl+C para sair. ")

        voz.dizer("Comecando. Pode se afastar do computador.", esperar=True)
        if args.cronometrado:
            rodar_roteiro(app, roteiro, placar, voz,
                          esperar_enter=args.passo_a_passo,
                          preparo_s=args.preparo)
        else:
            rodar_travado(app, roteiro, placar, voz,
                          limite_s=args.limite, confirmar_s=args.confirmar)
        voz.dizer("Terminado.")
    except KeyboardInterrupt:
        print("\n\n  interrompido — o boletim vale so ate aqui")
    finally:
        voz.calar()
        app.parar()

    print(f"{LIMPAR}BOLETIM\n")
    print("\n".join(diagnostico_da_cascata(app)))
    print()
    tempos = placar.linhas_de_tempo()
    if tempos:
        print("\n".join(tempos))
        print()
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
