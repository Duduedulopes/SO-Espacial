"""
Cameras por NOME. Indice e detalhe de acesso, nao identidade.

O CUSTO DE NAO TER ISTO

Em 08/08 os indices reordenaram entre duas execucoes SEGUIDAS, sem ninguem
mexer em nada:

    antes:  [0] C920    [1] VGA     [2] Iriun
    depois: [0] Iriun   [1] C920    [2] VGA

Uma configuracao gravada como `{"alto": 0}` passou a apontar para a camera
errada, e o sistema rodou mostrando a vista lateral no lugar da de cima.

    Identificador que muda sozinho nao e identificador.

REGRA DE BACKEND

O DirectShow enumera com nome, na mesma ordem que o OpenCV usa como indice
quando aberto com CAP_DSHOW. O MSMF tem ordem PROPRIA.

    Um indice so significa alguma coisa junto com o backend que o gerou.

Foi assim que um fallback "se DSHOW falhar, tenta MSMF" trocou cameras de
lugar e produziu cinco sintomas diferentes que pareciam cinco bugs.
"""

from src.nucleo.erros import CameraNaoEncontrada
from src.nucleo.log import Log

log = Log("dispositivos")


def listar():
    """[(indice, nome)] na ordem do DirectShow. None se pygrabber faltar."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        log.aviso("pygrabber ausente — sem nomes de dispositivo",
                  instale="pip install pygrabber")
        return None
    try:
        return list(enumerate(FilterGraph().get_input_devices()))
    except Exception as e:
        log.erro("falha ao enumerar dispositivos", erro=str(e))
        return None


def indice_de(nome, dispositivos=None):
    """Indice atual pelo nome. Aceita nome parcial e ignora caixa.

    O Windows as vezes acrescenta sufixos como "(2)" quando ha dispositivos
    repetidos — por isso a busca parcial.
    """
    devs = dispositivos if dispositivos is not None else listar()
    if not devs:
        return None
    alvo = str(nome).strip().lower()
    for idx, n in devs:
        if n.strip().lower() == alvo:
            return idx
    for idx, n in devs:
        if alvo in n.strip().lower() or n.strip().lower() in alvo:
            return idx
    return None


def exigir_indice(nome):
    """Igual a `indice_de`, mas levanta em vez de devolver None.

    Usado na abertura, onde seguir sem a camera nao faz sentido — e onde uma
    excecao com nome e sugestao vale mais que um None silencioso.
    """
    idx = indice_de(nome)
    if idx is None:
        presentes = [n for _, n in (listar() or [])]
        raise CameraNaoEncontrada(
            f"camera '{nome}' nao esta presente",
            presentes=presentes)
    return idx


def imprimir():
    devs = listar()
    if devs is None:
        print("pygrabber nao instalado:  pip install pygrabber")
        return
    if not devs:
        print("nenhuma camera encontrada pelo DirectShow.")
        return
    print("cameras presentes:")
    for idx, nome in devs:
        print(f"  [{idx}] {nome}")


if __name__ == "__main__":
    imprimir()
