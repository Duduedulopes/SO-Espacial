"""
Acha o tablet na rede, confirma que ele entrega VIDEO, e corrige o config.

    python ferramentas/achar_ip.py               varre e mostra o que achou
    python ferramentas/achar_ip.py --gravar      escreve em config/cameras.json
    python ferramentas/achar_ip.py --papel lateral --gravar

O PROBLEMA, E POR QUE ELE VAI VOLTAR SE NAO FOR RESOLVIDO AQUI

O IP do tablet vem do DHCP do roteador. Ele muda toda vez que o aparelho
reconecta, e nada avisa. Em 11/08 o `cameras.json` dizia `192.168.1.2` e o
tablet mostrava `192.168.1.9` na tela — a camera simplesmente nao conectava, e
o sintoma foi o mesmo de qualquer outra falha de rede.

Este projeto ja pagou exatamente esse defeito uma vez, com outra roupa:

    Nomes das cameras no Windows. Sem isto so ha indices numericos, que mudam
    sozinhos e ja custaram uma sessao inteira de depuracao.
                                                        — requirements.txt

Indice de DirectShow e IP de DHCP sao a mesma coisa: **identidade atribuida por
terceiros, que muda sem avisar.** La a saida foi usar o nome do dispositivo.
Aqui e procurar.

    Identidade que muda sozinha nao pode ser escrita a mao num arquivo.

PORTA ABERTA NAO E PROVA DE VIDEO

Varrer a rede e achar a porta 8080 aberta responde "tem alguem ai", nao "tem
video ai". Um roteador com painel web na 8080 apareceria como candidato.

Pior: o servidor pode responder e entregar imagem PRETA — foi o que a camera
virtual do Windows fez em 10/08, entregando video bonito ao painel de
Configuracoes e preto ao DirectShow, e a hipotese errada sobreviveu a tres
execucoes.

Entao cada candidato e aberto de verdade, um quadro e lido, e o BRILHO e
medido. So entra no resultado quem passou nas tres coisas.
"""

import argparse
import json
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

CONFIG = RAIZ / "config" / "cameras.json"

# Caminhos que os aplicativos de camera IP mais comuns servem. `IP Webcam`
# (Android) usa /video; DroidCam usa /mjpegfeed; alguns servem a raiz.
CAMINHOS = ("/video", "/videofeed", "/mjpegfeed?640x480", "/live", "")
PORTAS = (8080, 4747, 8081, 554)

# Mesma faixa do conferidor: `brilho_minimo=8` separa "sem imagem" de "com
# imagem", e nao separa "com imagem" de "com imagem UTIL".
BRILHO_SUSPEITO = 32.0


class SemRede(Exception):
    """Nao deu para descobrir a rede local. Traz instrucao, nao so o erro."""


def meu_ip():
    """IPv4 desta maquina na rede local.

    O truque do UDP para 8.8.8.8 nao envia pacote nenhum: so faz o sistema
    escolher a interface de saida, que e a informacao que interessa. Funciona
    sem internet e sem depender do nome do host, que no Windows costuma
    resolver para 127.0.0.1.

    QUANDO ELE FALHA, FALHA COM INSTRUCAO

    Sem rede nenhuma, `connect` levanta OSError e a versao anterior desta
    funcao deixava o traceback subir. Traceback nao diz o que fazer. A saida
    existe e e simples — passar `--prefixo` a mao — entao quem falha tem a
    obrigacao de dizer isso.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        ip = None

    if not ip or ip.startswith("127."):
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            ip = None

    if not ip or ip.startswith("127."):
        raise SemRede(
            "nao consegui descobrir a rede desta maquina.\n"
            "  Veja o IP que o tablet mostra na tela e passe o prefixo:\n"
            "      python ferramentas/achar_ip.py --prefixo 192.168.1")
    return ip


def porta_aberta(ip, porta, timeout=0.35):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((ip, porta)) == 0


def varrer(prefixo, portas=PORTAS, trabalhadores=64):
    """Devolve [(ip, porta)] com a porta aberta. Rapido: 254 x len(portas)."""
    alvos = [(f"{prefixo}.{n}", p) for n in range(1, 255) for p in portas]
    achados = []

    with ThreadPoolExecutor(max_workers=trabalhadores) as pool:
        for (ip, porta), aberto in zip(
                alvos, pool.map(lambda a: porta_aberta(*a), alvos)):
            if aberto:
                achados.append((ip, porta))
    return achados


def confirmar_video(url, timeout_s=6.0):
    """Abre a URL e le UM quadro. Devolve o laudo, ou None se nao for video.

    Le e descarta alguns quadros antes de medir. A primeira imagem de uma
    conexao MJPEG costuma ser parcial ou vir do buffer anterior — e julgar o
    brilho antes de o fluxo estabilizar foi exatamente o erro do `_aquecer()`
    da camera em 08/08, quando o diagnostico virou a causa do defeito.
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return None

        t0, quadro = time.monotonic(), None
        lidos = 0
        while time.monotonic() - t0 < timeout_s and lidos < 5:
            ok, q = cap.read()
            if ok and q is not None:
                quadro, lidos = q, lidos + 1
        if quadro is None:
            return None

        brilho = float(np.mean(cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)))
        h, w = quadro.shape[:2]
        return {"url": url, "largura": w, "altura": h,
                "brilho": round(brilho, 1),
                "util": brilho >= BRILHO_SUSPEITO}
    finally:
        cap.release()


def procurar(prefixo=None):
    prefixo = prefixo or ".".join(meu_ip().split(".")[:3])
    print(f"varrendo {prefixo}.1 a {prefixo}.254 nas portas "
          f"{', '.join(map(str, PORTAS))} ...")

    abertas = varrer(prefixo)
    if not abertas:
        return []
    print(f"  {len(abertas)} porta(s) aberta(s): "
          + ", ".join(f"{i}:{p}" for i, p in abertas))
    print("  confirmando quais entregam video de verdade ...\n")

    laudos = []
    for ip, porta in abertas:
        for caminho in CAMINHOS:
            laudo = confirmar_video(f"http://{ip}:{porta}{caminho}")
            if laudo:
                laudos.append(laudo)
                break
    return laudos


def gravar(url, papel):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    atual = cfg.get(papel)
    if isinstance(atual, dict):
        antes = atual.get("fonte")
        atual["fonte"] = url
    else:
        antes = atual
        cfg[papel] = {"fonte": url, "captura": "640x480"}

    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"\nconfig/cameras.json atualizado")
    print(f"  {papel}:  {antes}")
    print(f"        ->  {url}")


def main():
    p = argparse.ArgumentParser(description="Acha a camera IP na rede local")
    p.add_argument("--papel", default="lateral")
    p.add_argument("--prefixo", default=None,
                   help="ex: 192.168.1 — padrao e a rede desta maquina")
    p.add_argument("--gravar", action="store_true")
    args = p.parse_args()

    try:
        laudos = procurar(args.prefixo)
    except SemRede as e:
        print(e)
        return 1
    except OSError as e:
        print(f"nao consegui varrer a rede: {e}")
        return 1

    if not laudos:
        print("Nenhuma fonte de video encontrada.\n")
        print("  Verifique, nesta ordem:")
        print("   1. o tablet e o PC estao na MESMA rede Wi-Fi?")
        print("      (rede de convidados nao enxerga a rede principal)")
        print("   2. o aplicativo de camera esta com o servidor LIGADO?")
        print("      a tela do tablet mostra 'Video connections: 0' ate")
        print("      alguem conectar — isso e normal, nao e erro")
        print("   3. o firewall do Windows esta bloqueando a saida?")
        print("   4. tente abrir a URL do tablet no navegador do PC primeiro")
        return 1

    print(f"{len(laudos)} fonte(s) de video encontrada(s):\n")
    for d in laudos:
        marca = "OK  " if d["util"] else "PRETA"
        print(f"  {marca} {d['url']:44} {d['largura']}x{d['altura']}  "
              f"brilho {d['brilho']}")
        if not d["util"]:
            print("        ! responde e entrega imagem escura. Aponta a camera")
            print("          para uma cena iluminada e rode de novo — em 10/08")
            print("          uma fonte assim ficou ONLINE e deu ZERO poses.")

    uteis = [d for d in laudos if d["util"]]
    if args.gravar:
        if not uteis:
            print("\nnada gravado: nenhuma fonte entrega imagem util.")
            return 1
        gravar(uteis[0]["url"], args.papel)
    elif uteis:
        print(f"\npara gravar no config:")
        print(f"  python ferramentas/achar_ip.py --papel {args.papel} --gravar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
