"""A TELA DA BANCA. Um comando, uma janela.

    python apresentar.py                 janela normal, as tres cameras
    python apresentar.py --falsas        sem hardware — para ensaiar
    python apresentar.py --tela-cheia    sem bordas, para a banca

POR QUE ISTO NAO E UMA BANDEIRA DO `rodar.py`

    o tempo esta correndo ... ideias para facilitar esses processos, fazer de
    forma integrada e automatizada                      — Eduardo

`rodar.py` e a bancada de trabalho: catorze bandeiras, quatro janelas soltas,
painel de texto rolando no terminal. E o que se quer quando o objetivo e
depurar, e e exatamente o que nao se quer com uma plateia olhando.

O risco real de uma demonstracao nao e o sistema falhar: e quem apresenta ter
que escolher, ao vivo, quais janelas arrastar para onde. Entao este arquivo
tira a escolha do caminho: uma janela so, ja composta. Tela cheia fica em
`--tela-cheia`, porque um modo que dificulta sair nao pode ser o padrao.

    Numa demonstracao, toda decisao que sobra para a hora e uma chance de
    errar na frente de todo mundo.

O QUE ELE MOSTRA QUE O `rodar.py` NAO MOSTRAVA

QUANTAS unidades — a terceira pergunta, que ate hoje nao tinha resposta em
lugar nenhum (`src/acao/pegadas.py`), e a PROCEDENCIA de cada leitura ao lado
dela. Se o boneco levantar o braco sozinho, da para ver na hora qual camera
disse isso.
"""

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.acao.pegadas import ContadorDePegadas                 # noqa: E402
from src.acao.vocabulario import Braco, Locomocao              # noqa: E402
from src.app.orquestrador import Orquestrador                  # noqa: E402
from src.gemeo import boneco                                   # noqa: E402
from src.gemeo.suave import Suavizador                         # noqa: E402
from src.nucleo import log as logmod                           # noqa: E402

JANELA = "SO Espacial"
RESPIROS_POR_SEGUNDO = 0.30

CAMERAS = (("alto", "posicao, rumo, estatura"),
           ("frontal", "bracos, altura da mao"),
           ("lateral", "profundidade do alcance"))

BONITO = {Braco.AO_LADO: "ao lado", Braco.LEVANTADO: "LEVANTADO",
          Braco.ESTENDIDO: "ESTENDIDO", Braco.DESCONHECIDO: "?"}


def _bracos(acao):
    """Uma linha dizendo o que os dois bracos estao fazendo."""
    esq, dir_ = acao.braco_esquerdo, acao.braco_direito
    partes = [f"{n} {BONITO.get(v, v)}"
              for n, v in (("esq", esq), ("dir", dir_))
              if v != Braco.AO_LADO]
    return "  ".join(partes) or "ao lado"


def _fonte_dos_bracos(leitura, acao):
    """Qual camera respondeu — do lado que ESTA alcancando.

    Declarar a fonte do braco parado seria mostrar a procedencia da resposta
    que ninguem esta questionando. Quem levanta o braco decide a prateleira; e
    a fonte DELE que precisa estar na tela.
    """
    if leitura is None:
        return ""
    if acao.braco_direito != Braco.AO_LADO:
        return getattr(leitura, "fonte_braco_dir", "")
    if acao.braco_esquerdo != Braco.AO_LADO:
        return getattr(leitura, "fonte_braco_esq", "")
    return (getattr(leitura, "fonte_braco_dir", "")
            or getattr(leitura, "fonte_braco_esq", ""))


def _altura_da_mao(acao):
    alturas = [a for a in (acao.altura_mao_esq, acao.altura_mao_dir)
               if a is not None]
    return f"{max(alturas):.2f} m" if alturas else "-"


def _grau(rad):
    """Graus na volta, sempre entre -180 e +180.

    MEDIDO EM 12/08, na primeira corrida da tela:  `rumo  -189 graus`

    Nao existe -189 graus. O numero e o mesmo que +171, e a diferenca importa:
    quem le -189 procura um defeito de leitura onde ha so um defeito de
    apresentacao — e desconfia do resto da coluna junto.

    O suavizador ja devolve o angulo na volta (atan2), EXCETO na primeira
    amostra de cada pessoa, que ele repassa crua. Era esse quadro que estava
    na tela. Normalizar aqui cobre a origem toda de uma vez, porque quem
    escreve o angulo e quem sabe que ele precisa caber numa volta.
    """
    import math
    if rad is None:
        return "-"
    g = (math.degrees(rad) + 180.0) % 360.0 - 180.0
    return f"{g:+.0f} graus"


def main():
    p = argparse.ArgumentParser(description="SO Espacial — tela de apresentacao")
    p.add_argument("--planta", default="loja/bancada.json")
    p.add_argument("--captura", default="1280x720")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--lado", choices=("direita", "esquerda"), default="direita")
    p.add_argument("--falsas", action="store_true",
                   help="fontes sinteticas — para ensaiar sem hardware")
    # TELA CHEIA E OPT-IN, NAO PADRAO.
    #
    #     quando eu iniciei, nao consegui forcar a parar o programa
    #                                                 — Eduardo, 12/08
    #
    # Ela abriu sem borda, cobrindo o terminal, e a unica saida era uma tecla
    # que so chega quando a JANELA tem foco — coisa que ninguem adivinha
    # quando a janela acabou de tapar tudo. Um modo que dificulta sair nao pode
    # ser o que acontece quando voce nao pede nada.
    #
    #     O padrao e para o dia comum. O modo da banca dura vinte minutos por
    #     mes e voce digita uma bandeira para entrar nele.
    p.add_argument("--tela-cheia", action="store_true",
                   help="para a apresentacao. ESC ou q saem (clique na janela "
                        "primeiro)")
    p.add_argument("--tela", default="1600x900")
    p.add_argument("--log", default="WARNING",
                   help="WARNING por padrao: numa apresentacao o terminal "
                        "cheio de INFO rolando atras da janela so atrapalha")
    args = p.parse_args()

    import cv2
    from visual.apresentacao import Apresentacao
    from visual.cena3d import Cena3D, Esqueleto

    logmod.configurar(args.log)
    w, h = (int(v) for v in args.captura.lower().split("x"))
    tw, th = (int(v) for v in args.tela.lower().split("x"))

    app = Orquestrador(planta=args.planta, captura=(w, h), imgsz=args.imgsz,
                       conf=args.conf, lado_lateral=args.lado, com_pose=True)
    if args.falsas:
        app.montar_cameras_falsas()
    else:
        app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    # CHAMADA ANTES DE COMECAR.
    #
    # `iniciar()` espera UMA camera e segue — sensato para depurar, perigoso
    # aqui. A camera do alto e quem da posicao no chao, rumo e estatura: sem
    # ela nao ha gemeo, so bonecos parados na origem. Se ela for a mais lenta a
    # subir, o aviso "sem camera 'alto'" sai por CORRIDA, nao por falha, e a
    # apresentacao comeca quebrada sem ninguem saber por que.
    #
    # Entao aqui se espera pelas tres, e o que faltou aparece escrito antes de
    # a janela abrir — enquanto ainda da tempo de mexer num cabo.
    faltando = [p for p, _ in CAMERAS if not app.cameras.tem(p)]
    if faltando:
        app.cameras.esperar_online(timeout=8.0, minimo=len(CAMERAS))
        faltando = [p for p, _ in CAMERAS if not app.cameras.tem(p)]
    print("\n  CAMERAS")
    for papel, entrega in CAMERAS:
        marca = "  ok  " if papel not in faltando else "  --  "
        print(f"  {marca} {papel:8} {entrega}")
    if "alto" in faltando:
        print("\n  A CAMERA DO ALTO NAO SUBIU. Sem ela nao ha posicao no chao,")
        print("  nem rumo, nem estatura — o gemeo abre, mas nao mede nada.")
    print()

    tela = Apresentacao(tw, th)
    cw, ch = tela.tamanho_da_cena
    cena = Cena3D(cw, ch, chao=app.planta.chao, calor_hz=4.0)
    app.planta.aplicar_na_cena(cena)
    suavizador = Suavizador()
    contador = ContadorDePegadas()

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    if args.tela_cheia:
        cv2.setWindowProperty(JANELA, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    print("  para sair: clique na janela e aperte ESC ou q "
          "— ou Ctrl+C aqui no terminal\n")

    t0 = time.monotonic()
    vazios = 0
    try:
        while True:
            instante = app.passo()

            # QUADRO QUE NAO CHEGOU NAO E FIM DE PROGRAMA.
            #
            # `passo()` devolve None sempre que nenhuma camera tinha quadro
            # pronto naquele giro — normal, transitorio, e frequente logo apos
            # a abertura, quando as tres ainda estao subindo. `rodar.py` sempre
            # soube disso (dorme 5 ms e continua). Este arquivo nasceu ontem
            # com `break` no lugar do `continue`, e por isso a tela fechava no
            # primeiro giro sem dizer o porque.
            #
            #     Copiar a estrutura de um laco sem copiar o que ele faz nos
            #     casos que nao sao o caso feliz e como copiar so a parte do
            #     codigo que se entende.
            #
            # Silencio prolongado, esse sim, e defeito — e agora ele fala.
            if instante is None:
                vazios += 1
                if vazios % 400 == 0:
                    print(f"  nenhuma camera entrega quadro ha "
                          f"{vazios * 0.005:.0f}s — confira as conexoes")
                time.sleep(0.005)
                continue
            vazios = 0

            agora = time.monotonic()
            fase = (agora * RESPIROS_POR_SEGUNDO) % 1.0

            esqueletos, pessoas = [], []
            for pes in app.gemeo.pessoas.values():
                item = app.espacial.acoes.get(pes.id)
                if item is None:
                    continue
                acao = item[0]
                leitura = app.espacial.leituras.get(pes.id)
                palpite = app.espacial.palpites.get(pes.id)
                prateleira = palpite.prateleira if palpite else None

                # O contador so e alimentado com o palpite do MOMENTO. Ele
                # guarda sozinho o ultimo antes da mao descer — ver pegadas.py.
                contador.observar(pes.id, acao.braco_esquerdo,
                                  acao.braco_direito, prateleira)

                x, y, rumo = suavizador.suavizar(
                    pes.id, pes.x, pes.y, getattr(leitura, "rumo_corpo", None))
                esqueletos.append(Esqueleto(
                    id=pes.id,
                    juntas=boneco.montar(
                        estatura=app.espacial.escala.estatura(pes.id),
                        x=x, y=y, rumo=(rumo or 0.0),
                        postura=acao.postura, locomocao=acao.locomocao,
                        braco_esq=acao.braco_esquerdo,
                        braco_dir=acao.braco_direito,
                        altura_mao_esq=acao.altura_mao_esq,
                        altura_mao_dir=acao.altura_mao_dir, fase=fase),
                    prevendo=bool(pes.prevendo), rumo=rumo,
                    andando=acao.locomocao not in (Locomocao.PARADO,
                                                   Locomocao.DESCONHECIDA),
                    historico=list(app.gemeo.trilhas.get(pes.id, ()))))

                pessoas.append(dict(
                    id=pes.id,
                    postura=acao.postura, locomocao=acao.locomocao,
                    prateleira=prateleira,
                    firme=bool(palpite and palpite.firme),
                    quantas=contador.quantas(pes.id),
                    bracos=_bracos(acao),
                    fonte_bracos=_fonte_dos_bracos(leitura, acao),
                    altura_mao=_altura_da_mao(acao),
                    fonte_escala=getattr(leitura, "fonte_escala", ""),
                    rumo=_grau(rumo),
                    fonte_rumo=getattr(leitura, "fonte_rumo", "")))

            vivos = set(app.gemeo.pessoas)
            suavizador.esquecer(vivos)
            contador.esquecer(vivos)

            quadro = cena.desenhar(esqueletos, "", calor=app.gemeo.calor,
                                   zonas=app.planta.zonas)
            vistas = [(papel, entrega,
                       getattr(instante.get(papel), "imagem", None))
                      for papel, entrega in CAMERAS]

            decorrido = max(1e-6, agora - t0)
            rodape = (f"{app.quadros / decorrido:.1f} fps   "
                      f"defasagem {instante.defasagem_ms:.0f} ms   "
                      f"{contador.total} un no total")
            cv2.imshow(JANELA, tela.desenhar(quadro, pessoas, vistas, rodape))

            k = cv2.waitKeyEx(1) & 0xFFFFFF
            if k in (27, ord("q")):
                break
            # O X DA JANELA TAMBEM TEM QUE FUNCIONAR.
            #
            # Sem isto, fechar no X destroi a janela e o laco continua girando
            # para sempre, desenhando numa janela que nao existe mais — o
            # processo fica preso sem nada na tela para clicar. A saida mais
            # obvia que existe era a unica que nao servia.
            if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break
            cena.tecla(k)
    except KeyboardInterrupt:
        pass
    finally:
        app.parar()
        cv2.destroyAllWindows()

    print(f"\n{contador.total} unidades contadas.")
    for p in sorted({pid for pid, _ in contador._contagem}):
        print(f"  #{p}  {contador.por_prateleira(p)}")


if __name__ == "__main__":
    main()
