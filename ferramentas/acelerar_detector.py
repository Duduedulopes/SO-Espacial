"""Mede o detector nos tres formatos e escolhe o mais rapido que ACERTA.

    python ferramentas/acelerar_detector.py            # so mede
    python ferramentas/acelerar_detector.py --gravar    # mede e adota

O QUE ESTA FERRAMENTA RESOLVE

O detector da camera do alto e o gargalo do sistema, com folga. Medido na
corrida de 19/08:

    ciclo em regime                152 ms   ->  6,6 quadros/s
    detector do alto               130 ms   ->  86% do ciclo
    camera do alto entrega          15,0 quadros/s
    o detector consome               7,7 quadros/s
    descartados na fila            721 de 2743 quadros

Metade do que a camera ve e jogada fora sem ninguem olhar. E o efeito nao
para no numero de quadros: cada milissegundo de detector e milimetro de
atraso do boneco, e a 1 m/s 130 ms sao 13 cm.

O CAMINHO NAO E TROCAR DE MODELO

Trocar `yolo11n-pose` por algo menor troca velocidade por acerto, e acerto
aqui e a diferenca entre ver um tornozelo e nao ver. O caminho e rodar A
MESMA ARITMETICA com um executor melhor:

    PyTorch     generico, otimizado para treinar
    ONNX        grafo congelado, executor especializado em inferencia
    OpenVINO    o mesmo, com os kernels da Intel

Sao os mesmos pesos e a mesma conta. O que muda e quem executa.

POR QUE MEDIR EM VEZ DE ESCOLHER

A literatura diz "ate 3x" para os dois, e qual dos dois ganha depende do
processador — OpenVINO costuma liderar em Intel, ONNX e mais parelho fora.
"Costuma" nao e uma medida da maquina do Eduardo.

    Uma constante de desempenho escrita no codigo e uma medicao feita na
    maquina de outra pessoa.

E MAIS RAPIDO NAO BASTA

Um executor que devolve caixas diferentes nao e uma otimizacao: e outro
detector. Entao alem do tempo esta ferramenta confere, nos mesmos quadros:

    quantas pessoas cada formato achou
    onde ficou o centro de cada caixa
    onde ficaram os 17 pontos

Quem discordar alem da tolerancia e reprovado por mais rapido que seja.

    Ganho de velocidade sem conferencia de saida e so uma forma educada de
    trocar o problema.
"""
import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2                                                   # noqa: E402
import numpy as np                                           # noqa: E402

# Quanto duas saidas podem diferir e ainda serem a mesma resposta.
#
# Nao e zero: exportar troca a ordem das somas em ponto flutuante, e o
# resultado muda no ultimo bit e depois no arredondamento para pixel. Dois
# pixels e menos que a espessura de um tornozelo na imagem de 320.
#
#     Exigir bit a bit de um grafo reexecutado reprova a otimizacao por ela
#     ser uma otimizacao.
TOLERANCIA_PX = 2.0
ESPACO_MINIMO_GB = 1.5


def _achar_quadros(raiz, papel="alto"):
    """Os melhores quadros disponiveis, e diz de onde vieram.

    TRES FONTES, EM ORDEM DE HONESTIDADE:

        1. dados/quadros/alto-*-com-pessoa.jpg   gente de verdade, camera certa
        2. dados/quadros/alto-*.jpg              camera certa, talvez sem gente
        3. dados/levantamento/alto.png           o levantamento

    Por que a ordem importa, e nao e capricho:

    So a camera do ALTO passa por este detector. As outras duas vao para o
    MediaPipe, que e outro modelo e outro custo — medir nelas mediria uma
    coisa que nunca acontece.

    E a conferencia de saida precisa de GENTE. Sem ninguem no quadro as tres
    versoes concordam em "nao vi nada", o que e verdade e nao prova nada; e o
    teste dos ids nem roda, porque nao ha id para sobreviver.

        Duas medidas que concordam sobre o vazio nao concordam sobre coisa
        alguma.

    Nao busca `*.png` solto na pasta de proposito: em 19/08 eu deixei tres
    diagramas meus em `dados/levantamento` e a ferramenta teria cronometrado
    o detector em cima das minhas proprias figuras.
    """
    quadros = raiz / "dados" / "quadros"
    com_gente = sorted(quadros.glob(f"{papel}-*-com-pessoa.jpg"))
    if com_gente:
        return com_gente, "gravados com pessoa em cena", True
    qualquer = sorted(quadros.glob(f"{papel}-*.jpg"))
    if qualquer:
        return qualquer, "gravados, sem pessoa marcada", False
    levantamento = raiz / "dados" / "levantamento" / f"{papel}.png"
    if levantamento.exists():
        return [levantamento], "do levantamento", False
    return [], "", False


def _quadros(caminhos, quantos):
    """Carrega e repete ate ter `quantos`.

    Repetir nao falseia o que se mede: o custo por quadro e por quadro, e uma
    imagem so nao tem amostra para mediana — e mediana e o ponto.
    """
    if not caminhos:
        raise SystemExit(
            "\n  nao achei quadro nenhum da camera do alto.\n\n"
            "  o melhor jeito de conseguir uns, com voce em cena:\n"
            "      python rodar.py --salvar-quadros 0.5 --segundos 25\n\n"
            "  ou, so o levantamento:\n"
            "      python ferramentas/achar_ambiente.py --so-salvar\n")
    imagens = [cv2.imread(str(c)) for c in caminhos]
    imagens = [i for i in imagens if i is not None]
    if not imagens:
        raise SystemExit("\n  achei os arquivos e nenhum abriu como imagem.\n")
    return [imagens[k % len(imagens)] for k in range(quantos)]


def _medir(caminho_modelo, quadros, imgsz, conf, aquecer=5):
    """Cronometra e observa numa passada so. Devolve um dicionario.

    MEDE `track`, E NAO `predict`. Consertado antes da primeira corrida.

    O sistema nao chama `predict` em lugar nenhum: chama `track(persist=True)`,
    porque precisa do id. O rastreador custa — associacao, Kalman por caixa,
    manutencao de tracks — e medir `predict` daria um numero menor que o do
    painel, que compara com 130 ms.

        Cronometrar uma chamada que o programa nao faz mede um programa que
        nao existe.

    De quebra, medir com `track` faz o teste dos ids sair da mesma passada, e
    entao o modelo e carregado UMA vez por formato em vez de duas.

    MEDIANA, NAO MEDIA. Em 19/08 o painel registrou um quadro de 1047 ms num
    detector de 130 — uma amostra envenena a media e nao a mediana.

    E AQUECE ANTES, com `predict`. A primeira inferencia do Ultralytics custou
    15,2 s naquela mesma maquina: ele faz na estreia coisas que nunca mais
    repete. Contar estreia como regime ja produziu um diagnostico errado neste
    projeto. O aquecimento usa `predict` de proposito — aquecer com `track`
    criaria estado de rastreio a partir de uma imagem preta, e o primeiro
    quadro real ja nasceria com historico inventado (ver `detector._aquecer`).
    """
    from ultralytics import YOLO

    modelo = YOLO(str(caminho_modelo))
    vazio = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(aquecer):
        modelo.predict(vazio, imgsz=imgsz, verbose=False)

    tempos, vistas, ids = [], None, []
    for k, img in enumerate(quadros):
        t = time.perf_counter()
        r = modelo.track(img, persist=True, conf=conf, classes=[0],
                         imgsz=imgsz, verbose=False)[0]
        tempos.append((time.perf_counter() - t) * 1000.0)
        if k == 0:
            vistas = _resumir(r)
        if r.boxes is not None and r.boxes.id is not None:
            ids.append({int(v) for v in r.boxes.id.tolist()})

    return {"mediana": statistics.median(tempos),
            "menor": min(tempos), "maior": max(tempos),
            "vistas": vistas, "ids_sobrevivem": _ids_sobrevivem(ids)}


def _ids_sobrevivem(ids):
    """Algum id atravessou todos os quadros em que houve deteccao?

    O id que sai do `track` E a identidade da pessoa no resto do programa. Um
    formato que roda rapido e troca o id a cada quadro quebra o rastreio, o
    limbo e a contagem de quem entrou na zona — sem parecer quebrado.

        Um numero que muda quando nao deveria e pior que um erro: o erro
        aparece.

    `None` quando nao houve deteccao nenhuma: ai nao ha id para sobreviver, e
    dizer que o teste passou seria mentir sobre um teste que nao rodou.
    """
    if not ids:
        return None
    if len(ids) == 1:
        return True
    return bool(set.intersection(*ids))


def _numeros(t):
    """Tensor do torch ou array do numpy -> lista. Nao supoe o backend.

    O Ultralytics embrulha a saida nas classes dele em qualquer formato, mas
    supor `.cpu()` e supor que sempre havera torch por baixo — que e
    exatamente o que esta ferramenta existe para deixar de ser verdade.
    """
    if hasattr(t, "cpu"):
        t = t.cpu()
    return np.asarray(t).tolist()


def _resumir(r):
    """O que o detector viu, num formato comparavel entre formatos."""
    caixas = r.boxes
    if caixas is None or len(caixas) == 0:
        return {"quantas": 0, "centros": [], "juntas": []}
    centros = [_numeros(b.xyxy[0]) for b in caixas]
    juntas = []
    if r.keypoints is not None:
        juntas = [_numeros(p.xy[0]) for p in r.keypoints]
    return {"quantas": len(caixas), "centros": centros, "juntas": juntas}


def _concordam(a, b, tolerancia=TOLERANCIA_PX):
    """As duas saidas descrevem a mesma cena? Devolve (bool, motivo)."""
    if a["quantas"] != b["quantas"]:
        return False, (f"achou {b['quantas']} pessoas onde o PyTorch achou "
                       f"{a['quantas']}")
    # COMPARAR AS CONTAGENS ANTES DE COMPARAR OS PARES.
    #
    # `zip` para na lista mais curta e nao reclama. Um modelo sem cabeca de
    # pose devolveria `juntas = []`, o `zip` nao renderia par nenhum, e a
    # conferencia diria "confere" sobre zero comparacoes.
    #
    #     Um laco que nao iterou nao concordou: ele nao aconteceu.
    if len(a["juntas"]) != len(b["juntas"]):
        return False, (f"tem {len(b['juntas'])} esqueletos onde o PyTorch tem "
                       f"{len(a['juntas'])}")

    for k, (ca, cb) in enumerate(zip(a["centros"], b["centros"])):
        d = max(abs(x - y) for x, y in zip(ca, cb))
        if d > tolerancia:
            return False, f"a caixa {k} saiu {d:.1f} px fora"
    for k, (ja, jb) in enumerate(zip(a["juntas"], b["juntas"])):
        pa, pb = np.array(ja, dtype=float), np.array(jb, dtype=float)
        if pa.shape != pb.shape:
            return False, f"o esqueleto {k} tem outro formato"
        d = float(np.abs(pa - pb).max()) if pa.size else 0.0
        if d > tolerancia:
            return False, f"a junta do esqueleto {k} saiu {d:.1f} px fora"
    return True, "confere"


# O que cada formato precisa: para ESCREVER o arquivo, e para RODAR ele.
PRECISA = {
    "onnx": (("onnx", "onnxslim"), ("onnxruntime",)),
    "openvino": (("openvino",), ("openvino",)),
}


def _falta(formato):
    """Os pacotes ausentes para este formato. Vazio quer dizer pronto.

    POR QUE CONFERIR EM VEZ DE DEIXAR O ULTRALYTICS INSTALAR SOZINHO

    Ele faz `check_requirements`, que dispara um `pip install` no meio da
    exportacao. Em 18/08 este projeto passou a tarde com o disco cheio, e um
    pip que comeca sozinho e para no meio deixa o ambiente pela metade — sem
    que ninguem tenha pedido nada.

        Instalacao que comeca sem ser pedida e a que ninguem esta olhando
        quando falha.

    Aqui a ferramenta pula o formato e imprime o comando. Quem decide gastar
    400 MB e o dono do disco.
    """
    import importlib.util

    faltando = []
    for grupo in PRECISA.get(formato, ((), ())):
        for pacote in grupo:
            if (importlib.util.find_spec(pacote) is None
                    and pacote not in faltando):
                faltando.append(pacote)
    return faltando


def _exportar(origem, formato, imgsz):
    """Exporta e devolve o caminho, ou (None, motivo)."""
    from ultralytics import YOLO
    try:
        saida = YOLO(str(origem)).export(format=formato, imgsz=imgsz,
                                         verbose=False)
        return Path(saida), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _espaco_livre_gb():
    return shutil.disk_usage(RAIZ).free / 1e9


def main():
    p = argparse.ArgumentParser(
        description="mede o detector em PyTorch, ONNX e OpenVINO")
    p.add_argument("--modelo", default="yolo11n-pose.pt")
    p.add_argument("--pasta", default=None,
                   help="por padrao procura sozinho os quadros da camera do alto")
    p.add_argument("--quadros", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=320,
                   help="TEM que ser o mesmo do rodar.py")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--gravar", action="store_true")
    args = p.parse_args()

    livre = _espaco_livre_gb()
    print(f"\n  espaco livre {livre:.1f} GB")
    if livre < ESPACO_MINIMO_GB:
        raise SystemExit(
            f"\n  menos de {ESPACO_MINIMO_GB} GB livres. As exportacoes sao\n"
            f"  pequenas (uns 12 MB cada), mas o disco cheio no meio de uma\n"
            f"  escrita deixa arquivo pela metade. Libere espaco antes.\n")

    if args.pasta:
        caminhos = (sorted((RAIZ / args.pasta).glob("*.jpg"))
                    + sorted((RAIZ / args.pasta).glob("*.png")))
        origem, tem_gente = args.pasta, False
    else:
        caminhos, origem, tem_gente = _achar_quadros(RAIZ)

    quadros = _quadros(caminhos, args.quadros)
    print(f"  {len(caminhos)} quadro(s) {origem}, imgsz {args.imgsz}")
    if not tem_gente:
        print("\n  ATENCAO: nenhum quadro com pessoa marcada.")
        print("  O tempo sai certo, mas a conferencia de saida fica fraca:")
        print("  sem ninguem em cena os tres formatos concordam em 'nao vi")
        print("  nada', e o teste dos ids nem roda. Para medir de verdade:")
        print("      python rodar.py --salvar-quadros 0.5 --segundos 25")
        print("  (ande na frente da camera do alto) e rode isto de novo.")
    print()

    candidatos = [("PyTorch", Path(args.modelo))]
    for formato, rotulo in (("onnx", "ONNX"), ("openvino", "OpenVINO")):
        faltando = _falta(formato)
        if faltando:
            print(f"  {rotulo}: pulando, falta  {' '.join(faltando)}")
            print(f"      pip install {' '.join(faltando)}")
            continue
        print(f"  exportando para {rotulo}...", end=" ", flush=True)
        caminho, erro = _exportar(args.modelo, formato, args.imgsz)
        if caminho is None:
            print(f"NAO DEU — {erro}")
            continue
        print(f"ok  {caminho.name}")
        candidatos.append((rotulo, caminho))
    print()

    referencia, referencia_rastreia = None, None
    linhas = []
    for rotulo, caminho in candidatos:
        try:
            m = _medir(caminho, quadros, args.imgsz, args.conf)
        except Exception as e:
            linhas.append({"rotulo": rotulo, "caminho": caminho, "ms": None,
                           "nota": f"nao rodou: {type(e).__name__}: {e}",
                           "ok": False})
            continue

        if referencia is None:
            referencia = m["vistas"]
            referencia_rastreia = m["ids_sobrevivem"]
            ok, motivo = True, "referencia"
        else:
            ok, motivo = _concordam(referencia, m["vistas"])

        # O TESTE DOS IDS SO REPROVA QUEM FAZ PIOR QUE A REFERENCIA.
        #
        # Se o proprio PyTorch nao segurou os ids nestes quadros, o teste nao
        # esta discriminando nada — esta descrevendo os quadros. Reprovar
        # todo mundo por isso seria culpar os candidatos pelo instrumento.
        if m["ids_sobrevivem"] is False and referencia_rastreia is True:
            ok, motivo = False, "perdeu os ids entre quadros"
        elif m["ids_sobrevivem"] is None and ok:
            motivo += "  (sem gente: ids nao testados)"

        linhas.append({"rotulo": rotulo, "caminho": caminho,
                       "ms": m["mediana"], "menor": m["menor"],
                       "maior": m["maior"], "ok": ok,
                       "nota": motivo if ok else f"REPROVADO: {motivo}"})

    # A BASE DA COLUNA DE GANHO E O PYTORCH, E NAO O PRIMEIRO QUE RODOU.
    # Se o PyTorch falhar, nao ha contra o que comparar — melhor coluna vazia
    # que uma razao contra um denominador que ninguem escolheu.
    base = next((l["ms"] for l in linhas
                 if l["rotulo"] == "PyTorch" and l["ms"] is not None), None)

    print(f"  {'formato':<10} {'mediana':>9} {'melhor':>8} {'pior':>8} "
          f"{'ganho':>7}   conferencia")
    print(f"  {'-' * 10} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 7}   {'-' * 40}")
    for l in linhas:
        if l["ms"] is None:
            print(f"  {l['rotulo']:<10} {'—':>9} {'—':>8} {'—':>8} {'—':>7}"
                  f"   {l['nota']}")
            continue
        ganho = f"{base / l['ms']:.2f}x" if base else "—"
        print(f"  {l['rotulo']:<10} {l['ms']:7.1f}ms {l['menor']:6.1f}ms "
              f"{l['maior']:6.1f}ms {ganho:>7}   {l['nota']}")

    aprovados = sorted((l for l in linhas if l["ms"] is not None and l["ok"]),
                       key=lambda l: l["ms"])
    if not aprovados:
        raise SystemExit("\n  nenhum formato passou na conferencia.\n")
    melhor = aprovados[0]

    print(f"\n  MAIS RAPIDO QUE ACERTA: {melhor['rotulo']}  "
          f"{melhor['ms']:.1f} ms por quadro")
    if base and melhor["ms"] < base:
        economia = base - melhor["ms"]
        # O CICLO NAO SE EXTRAPOLA DAQUI, E DIZER QUE SE EXTRAPOLA E PIOR QUE
        # NAO DIZER NADA.
        #
        # A tentacao e somar: "o ciclo era 152 com 130 de detector, entao com
        # 50 fica 72". Mas aquele 130 foi medido noutra corrida, com as duas
        # cameras disputando CPU e o resto do sistema rodando junto. Misturar
        # a medida de agora, isolada, com o painel de ontem produz um numero
        # com duas casas e nenhuma procedencia.
        #
        #     Numero preciso feito de duas medidas incomparaveis mente com
        #     mais confianca do que um chute.
        #
        # O que se pode afirmar: o detector economiza tanto por quadro. O que
        # isso vale no ciclo inteiro, quem responde e o painel do `rodar.py`.
        print(f"  economiza {economia:.0f} ms por quadro "
              f"({base / melhor['ms']:.2f}x)")
        print(f"  a {1000 / melhor['ms']:.1f} quadros/s ele "
              f"{'ainda nao' if melhor['ms'] > 66 else 'ja'} acompanha os "
              f"15,0 que a camera entrega")
        print("  quanto disso vira fps do sistema, o painel do rodar.py diz")

    alvo = RAIZ / "config" / "detector.json"
    if not args.gravar:
        print("\n  (nao gravei — use --gravar)\n")
        return

    if melhor["rotulo"] == "PyTorch":
        # APAGAR A ESCOLHA ANTIGA, E NAO SO DEIXAR DE ESCREVER A NOVA.
        # Sem isto, uma medicao que elege o PyTorch deixaria o sistema rodando
        # o ONNX de uma medicao anterior — e o painel nao contaria a diferenca.
        if alvo.exists():
            alvo.unlink()
            print("\n  o PyTorch ganhou — apaguei a escolha anterior.\n")
        else:
            print("\n  o PyTorch ganhou — nao ha nada a gravar.\n")
        return

    alvo.parent.mkdir(exist_ok=True)
    try:
        relativo = str(Path(melhor["caminho"]).resolve().relative_to(RAIZ))
    except ValueError:
        relativo = str(Path(melhor["caminho"]).resolve())
    comparacao = (f"PyTorch {base:.0f} ms  ->  {melhor['rotulo']} "
                  f"{melhor['ms']:.0f} ms" if base else
                  f"{melhor['rotulo']} {melhor['ms']:.0f} ms "
                  f"(o PyTorch nao rodou nesta medicao)")
    alvo.write_text(json.dumps({
        "modelo": relativo,
        "_formato": melhor["rotulo"],
        "_ms_por_quadro": round(melhor["ms"], 1),
        "_imgsz": args.imgsz,
        "_medido_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_nota": [
            "MEDIDO NESTA MAQUINA, nao escolhido no codigo.",
            "",
            comparacao,
            "",
            "Sao os mesmos pesos e a mesma conta; o que muda e quem executa.",
            "A conferencia exigiu as mesmas caixas e as mesmas 17 juntas",
            "dentro de 2 px, e os ids sobrevivendo entre quadros — um formato",
            "rapido que perde o id quebraria o rastreio sem parecer quebrado.",
            "",
            "    Uma constante de desempenho escrita no codigo e uma medicao",
            "    feita na maquina de outra pessoa.",
            "",
            f"IMPORTANTE: exportado para imgsz={args.imgsz}. Rodar com outro",
            "imgsz nao vai acelerar, e pode nem funcionar. Se mudar o --imgsz",
            "do rodar.py, rode esta ferramenta de novo.",
            "",
            "PARA VOLTAR AO PYTORCH: apague este arquivo.",
        ]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n  gravado em config/detector.json")
    print("  (para voltar atras, apague esse arquivo)")
    print("\n  agora:  python rodar.py\n")


if __name__ == "__main__":
    main()
