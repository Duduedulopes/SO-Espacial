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


def _quadros(pasta, quantos):
    """Os quadros de teste. Reais, e da camera do alto.

    Medir com imagem preta mediria o caminho rapido: sem deteccao nao ha
    pos-processamento, e o pos-processamento e parte do custo. A imagem
    precisa ter gente — ou pelo menos moveis — como a de verdade tem.
    """
    caminhos = sorted(Path(pasta).glob("*.png")) + sorted(Path(pasta).glob("*.jpg"))
    if not caminhos:
        raise SystemExit(
            f"\n  nao achei imagem nenhuma em {pasta}.\n"
            f"  rode antes:  python ferramentas/achar_ambiente.py --so-salvar\n")
    imagens = [cv2.imread(str(c)) for c in caminhos]
    imagens = [i for i in imagens if i is not None]
    # repete o conjunto ate ter `quantos`: o que se mede e o custo por quadro,
    # e um so nao tem amostra para mediana
    return [imagens[k % len(imagens)] for k in range(quantos)]


def _medir(caminho_modelo, quadros, imgsz, conf, aquecer=5):
    """Mediana de ms por quadro, e o que ele viu no primeiro quadro.

    MEDIANA, NAO MEDIA. Em 19/08 o painel registrou um quadro de 1047 ms num
    detector de 130 — uma amostra envenena a media e nao a mediana.

    E AQUECE ANTES. A primeira inferencia do Ultralytics custou 15,2 s
    naquela mesma maquina: ele faz na estreia coisas que nunca mais repete.
    Contar estreia como regime ja produziu um diagnostico errado neste
    projeto (ver `detector._aquecer`).
    """
    from ultralytics import YOLO

    modelo = YOLO(str(caminho_modelo))
    vazio = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(aquecer):
        modelo.predict(vazio, imgsz=imgsz, verbose=False)

    tempos, vistas = [], None
    for k, img in enumerate(quadros):
        t = time.perf_counter()
        r = modelo.predict(img, imgsz=imgsz, conf=conf, classes=[0],
                           verbose=False)[0]
        tempos.append((time.perf_counter() - t) * 1000.0)
        if k == 0:
            vistas = _resumir(r)
    return statistics.median(tempos), min(tempos), max(tempos), vistas


def _resumir(r):
    """O que o detector viu, num formato comparavel entre formatos."""
    caixas = r.boxes
    if caixas is None or len(caixas) == 0:
        return {"quantas": 0, "centros": [], "juntas": []}
    centros = [tuple(float(v) for v in b.xyxy[0].tolist()) for b in caixas]
    juntas = []
    if r.keypoints is not None:
        juntas = [p.xy[0].cpu().numpy().tolist() for p in r.keypoints]
    return {"quantas": len(caixas), "centros": centros, "juntas": juntas}


def _concordam(a, b, tolerancia=TOLERANCIA_PX):
    """As duas saidas descrevem a mesma cena? Devolve (bool, motivo)."""
    if a["quantas"] != b["quantas"]:
        return False, f"achou {b['quantas']} pessoas onde o PyTorch achou {a['quantas']}"
    for k, (ca, cb) in enumerate(zip(a["centros"], b["centros"])):
        d = max(abs(x - y) for x, y in zip(ca, cb))
        if d > tolerancia:
            return False, f"a caixa {k} saiu {d:.1f} px fora"
    for k, (ja, jb) in enumerate(zip(a["juntas"], b["juntas"])):
        pa, pb = np.array(ja), np.array(jb)
        if pa.shape != pb.shape:
            return False, f"o esqueleto {k} tem outro formato"
        d = float(np.abs(pa - pb).max()) if pa.size else 0.0
        if d > tolerancia:
            return False, f"a junta do esqueleto {k} saiu {d:.1f} px fora"
    return True, "confere"


def _rastreia(caminho_modelo, quadros, imgsz, conf):
    """Os ids sobrevivem entre quadros neste formato?

    O sistema nao usa `predict`: usa `track(persist=True)`, e o id que sai
    dali E a identidade da pessoa no resto do programa. Um formato que roda
    rapido e perde o id a cada quadro quebra o rastreio, o limbo e a
    contagem de quem entrou na zona — sem parecer quebrado.

        Um numero que muda quando nao deveria e pior que um erro: o erro
        aparece.
    """
    from ultralytics import YOLO

    modelo = YOLO(str(caminho_modelo))
    ids = []
    for img in quadros[:12]:
        r = modelo.track(img, persist=True, conf=conf, classes=[0],
                         imgsz=imgsz, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            ids.append({int(v) for v in r.boxes.id.tolist()})
    if not ids:
        return None                     # nada detectado: nao da para afirmar
    return bool(set.intersection(*ids)) if len(ids) > 1 else True


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
    p.add_argument("--pasta", default="dados/levantamento")
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
            f"  pequenas (uns 12 MB cada), mas o Ultralytics instala\n"
            f"  onnxruntime e openvino sozinho se faltarem, e ai sao uns\n"
            f"  400 MB. Libere espaco antes.\n")

    quadros = _quadros(RAIZ / args.pasta, args.quadros)
    print(f"  {len(quadros)} quadros de {args.pasta}, imgsz {args.imgsz}\n")

    candidatos = [("PyTorch", Path(args.modelo))]
    for formato, rotulo in (("onnx", "ONNX"), ("openvino", "OpenVINO")):
        print(f"  exportando para {rotulo}...", end=" ", flush=True)
        caminho, erro = _exportar(args.modelo, formato, args.imgsz)
        if caminho is None:
            print(f"NAO DEU — {erro}")
            continue
        print(f"ok  {caminho.name}")
        candidatos.append((rotulo, caminho))
    print()

    referencia = None
    linhas = []
    for rotulo, caminho in candidatos:
        try:
            mediana, menor, maior, vistas = _medir(caminho, quadros,
                                                   args.imgsz, args.conf)
        except Exception as e:
            linhas.append((rotulo, caminho, None, None, None,
                           f"nao rodou: {type(e).__name__}: {e}"))
            continue

        if referencia is None:
            referencia, ok, motivo = vistas, True, "referencia"
        else:
            ok, motivo = _concordam(referencia, vistas)

        rastreio = _rastreia(caminho, quadros, args.imgsz, args.conf)
        if rastreio is False:
            ok, motivo = False, "perdeu os ids entre quadros"
        elif rastreio is None and motivo == "confere":
            motivo = "confere (sem gente nos quadros: rastreio nao testado)"

        linhas.append((rotulo, caminho, mediana, menor, maior,
                       motivo if ok else f"REPROVADO: {motivo}"))

    print(f"  {'formato':<10} {'mediana':>9} {'melhor':>8} {'pior':>8}   conferencia")
    print(f"  {'-' * 10} {'-' * 9} {'-' * 8} {'-' * 8}   {'-' * 40}")
    base = None
    for rotulo, _c, mediana, menor, maior, nota in linhas:
        if mediana is None:
            print(f"  {rotulo:<10} {'—':>9} {'—':>8} {'—':>8}   {nota}")
            continue
        if base is None:
            base = mediana
        ganho = f"{base / mediana:.2f}x" if mediana else "—"
        print(f"  {rotulo:<10} {mediana:7.1f}ms {menor:6.1f}ms {maior:6.1f}ms"
              f"   {nota}   {ganho}")

    aprovados = [(m, r, c) for r, c, m, _n, _x, nota in linhas
                 if m is not None and not nota.startswith("REPROVADO")]
    if not aprovados:
        raise SystemExit("\n  nenhum formato passou na conferencia.\n")

    aprovados.sort()
    melhor_ms, melhor_rotulo, melhor_caminho = aprovados[0]
    pytorch_ms = next((m for r, _c, m, _n, _x, _t in linhas
                       if r == "PyTorch" and m is not None), None)

    print(f"\n  MAIS RAPIDO QUE ACERTA: {melhor_rotulo}  {melhor_ms:.1f} ms")
    if pytorch_ms:
        # O que isso vale no sistema inteiro, e nao so neste arquivo. O ciclo
        # em regime era 152 ms com 130 de detector; o resto nao muda.
        resto = 152.0 - 130.0
        novo_ciclo = resto + melhor_ms
        print(f"  ciclo estimado: {152.0:.0f} ms -> {novo_ciclo:.0f} ms"
              f"   ({1000 / 152:.1f} -> {1000 / novo_ciclo:.1f} quadros/s)")
        print(f"  atraso do cano a 1 m/s: {13:.0f} cm -> "
              f"{melhor_ms / 10:.0f} cm")

    if args.gravar and melhor_rotulo != "PyTorch":
        alvo = RAIZ / "config" / "detector.json"
        alvo.parent.mkdir(exist_ok=True)
        try:
            relativo = str(Path(melhor_caminho).resolve().relative_to(RAIZ))
        except ValueError:
            relativo = str(Path(melhor_caminho).resolve())
        alvo.write_text(json.dumps({
            "modelo": relativo,
            "_formato": melhor_rotulo,
            "_ms_por_quadro": round(melhor_ms, 1),
            "_imgsz": args.imgsz,
            "_medido_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "_nota": [
                "MEDIDO NESTA MAQUINA, nao escolhido no codigo.",
                "",
                f"PyTorch {pytorch_ms:.0f} ms  ->  {melhor_rotulo} "
                f"{melhor_ms:.0f} ms",
                "",
                "Sao os mesmos pesos e a mesma conta; o que muda e quem",
                "executa. A conferencia exigiu as mesmas caixas e as mesmas",
                "17 juntas dentro de 2 px, e os ids sobrevivendo entre",
                "quadros — um formato rapido que perde o id quebraria o",
                "rastreio sem parecer quebrado.",
                "",
                "    Uma constante de desempenho escrita no codigo e uma",
                "    medicao feita na maquina de outra pessoa.",
                "",
                f"IMPORTANTE: exportado para imgsz={args.imgsz}. Rodar com",
                "outro imgsz nao vai acelerar, e pode nem funcionar. Se mudar",
                "o --imgsz do rodar.py, rode esta ferramenta de novo.",
            ]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  gravado em config/detector.json")
        print(f"\n  agora:  python rodar.py\n")
    elif args.gravar:
        print("\n  o PyTorch ganhou — nao ha nada a gravar.\n")
    else:
        print("\n  (nao gravei — use --gravar)\n")


if __name__ == "__main__":
    main()
