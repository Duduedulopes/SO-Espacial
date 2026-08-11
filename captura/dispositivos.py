"""
Cameras por NOME, nao por indice.

O PROBLEMA, que custou uma sessao inteira em 08/08

Indices de camera nao sao estaveis. Mudam ao reconectar um cabo, ao instalar
um driver virtual, ao reiniciar. Duas janelas chegaram a mostrar a mesma
camera porque `--alto 0 --lateral 2` apontava para o lugar errado.

    Identificador que muda sozinho nao e identificador.

A SOLUCAO

O DirectShow enumera os dispositivos com NOME, e na MESMA ordem que o OpenCV
usa como indice. Entao da para montar a tabela nome -> indice, e guardar nome
na configuracao.

    "HD Pro Webcam C920"  -> indice 0
    "VGA camera"          -> indice 1
    "Iriun Webcam"        -> indice 2

Se amanha os indices trocarem, a tabela e refeita e nada quebra.

Precisa de:  pip install pygrabber
Sem ele, cai para indices numericos e avisa.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


def listar():
    """[(indice, nome)] na ordem em que o OpenCV/DSHOW enxerga."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return None
    try:
        return list(enumerate(FilterGraph().get_input_devices()))
    except Exception:
        return None


def indice_de(nome, dispositivos=None):
    """Indice atual de uma camera, pelo nome. None se nao estiver presente.

    Compara sem diferenciar maiusculas e aceita nome parcial — o Windows as
    vezes acrescenta sufixos como "(2)" quando ha dispositivos repetidos.
    """
    devs = dispositivos if dispositivos is not None else listar()
    if not devs:
        return None
    alvo = nome.strip().lower()
    for idx, n in devs:
        if n.strip().lower() == alvo:
            return idx
    for idx, n in devs:
        if alvo in n.strip().lower() or n.strip().lower() in alvo:
            return idx
    return None


def resolver(config):
    """Converte {'alto': 'HD Pro Webcam C920', ...} em {'alto': 0, ...}.

    Devolve (indices, faltando). Cameras ausentes nao viram erro: o sistema
    deve rodar com o que houver, e dizer o que falta.
    """
    devs = listar()
    indices, faltando = {}, []

    for papel, valor in config.items():
        if isinstance(valor, int):          # configuracao antiga, por indice
            indices[papel] = valor
            continue
        idx = indice_de(valor, devs)
        if idx is None:
            faltando.append((papel, valor))
        else:
            indices[papel] = idx

    return indices, faltando


def imprimir():
    devs = listar()
    if devs is None:
        print("pygrabber nao instalado — sem nomes de dispositivo.")
        print("  pip install pygrabber")
        return
    if not devs:
        print("nenhuma camera encontrada pelo DirectShow.")
        return
    print("cameras presentes:")
    for idx, nome in devs:
        print(f"  indice {idx}  \"{nome}\"")


if __name__ == "__main__":
    imprimir()
