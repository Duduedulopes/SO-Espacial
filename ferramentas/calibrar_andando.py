"""Calibra a lateral e a frontal com voce PARANDO em varios lugares.

    python ferramentas/calibrar_andando.py --segundos 120
    python ferramentas/calibrar_andando.py --segundos 120 --gravar

O QUE FAZER: pare, conte ate tres, ande para outro lugar, pare de novo. Dez a
quinze paradas, o mais espalhadas que as cameras alcancarem, de pe e olhando
para frente. Nao precisa clicar em nada nem imprimir nada.

POR QUE PARANDO, E NAO ANDANDO — MEDIDO EM 20/08

A conta supoe que voce e um BASTAO VERTICAL: que o ombro esta exatamente
acima do meio dos tornozelos. Andando, isso e falso — o tronco inclina e os
pes se afastam. Medido na caminhada real: 15 a 25 cm de desvio, em TODO
quadro. Nao e ruido que se rejeite; e vies.

    A hipotese nao estava errada por pouco: estava sendo aplicada exatamente
    no momento em que ela nao vale.

Parado, o corpo e vertical de verdade e os pes ficam juntos. Andar continua
sendo necessario — para cobrir area — mas o que ele mede sao as PARADAS.

POR QUE ISSO FUNCIONA

Um tabuleiro de xadrez e um objeto de dimensoes conhecidas posto na cena. Ja
ha um objeto de dimensoes conhecidas na cena: voce. Tem 1,80 m (medidos e
gravados em `config/escala.json`), fica em pe, e as tres cameras olham para
voce ao mesmo tempo.

    Quando o proprio objeto de interesse tem dimensao conhecida, ele e o
    padrao de calibracao. Trazer outro e trazer um problema a mais.

A cada instante:

    a camera do alto     diz ONDE voce esta, em metros          (homografia)
    a antropometria      diz a que altura ficam ombro e nariz   (boneco.py)
    as outras cameras    dizem em que PIXEL cada ponto cai

Cada quadro rende ate tres correspondencias 3D->2D por camera. Cem quadros
dao trezentas, e `cv2.calibrateCamera` resolve K, distorcao, R e t com isso.

TRES ALTURAS, E NAO DUAS

Tornozelo em z=0, ombro em 0,82 da estatura, nariz em 0,925. Pontos
coplanares nao determinam a focal — se so o pe entrasse, todos estariam no
piso e a solucao seria indeterminada. Cada altura a mais condiciona melhor.

    Um plano so nao tem profundidade para revelar. E preciso que o objeto
    saia do plano para que a lente se denuncie.

E O NUMERO QUE DIZ SE PRESTA: ERRO DE REPROJECAO

Nao e o do tabuleiro. O canto de um xadrez e achado com precisao de
sub-pixel; aqui o "canto" e um tornozelo de detector, que oscila uns 3 px. O
erro nao pode ficar abaixo do ruido da entrada — ver `andando.ERRO_MAXIMO_PX`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np                                            # noqa: E402

from src.gemeo.boneco import (ALTURA_JOELHO, ALTURA_NARIZ,     # noqa: E402
                              ALTURA_OMBRO, ALTURA_QUADRIL,
                              ALTURA_TORNOZELO)
from src.mundo.andando import (Coleta, diagnostico,           # noqa: E402
                               homografia_da_pose, resolver)

# CINCO ALTURAS AO LONGO DO EIXO DO CORPO, E NAO DUAS.
#
# ERRO MEU, MEDIDO EM 20/08: eu usava so ombro e nariz, que ficam a 17 cm um
# do outro. Dois planos a 17 cm, com a camera a 1,5 m, sao praticamente um
# plano so — e a focal fica INDETERMINADA. Ela saiu 166 px numa corrida e
# 37 359 px noutra: o otimizador pode por a camera em qualquer lugar ao
# longo de um raio.
#
# Escrevi "um plano so nao tem profundidade para revelar" no cabecalho deste
# arquivo e nao percebi que 17 cm e um plano so.
#
#     Duas medidas quase no mesmo lugar nao sao duas medidas. Sao uma, com
#     um numero a mais para dar confianca.
#
# Cotovelo e pulso ficam de fora de proposito: eles BALANCAM, e nao estao
# sobre o eixo vertical do corpo. As cinco abaixo estao.
PONTOS = {
    "nariz": ((0,), ALTURA_NARIZ),
    "ombro": ((5, 6), ALTURA_OMBRO),
    "quadril": ((11, 12), ALTURA_QUADRIL),
    "joelho": ((13, 14), ALTURA_JOELHO),
    "tornozelo": ((15, 16), ALTURA_TORNOZELO),
}

# Acima desta velocidade, o corpo nao e um bastao vertical.
#
# MEDIDO: andando, o desvio entre o meio dos tornozelos e o meio dos ombros
# chega a 15-25 cm — o tronco inclina e os pes se afastam. Isso nao e ruido
# que se rejeita: e vies presente em TODO quadro de caminhada.
#
#     A hipotese nao estava errada por pouco: estava sendo aplicada
#     exatamente no momento em que ela nao vale.
#
# Parado, o corpo e vertical de verdade e os pes estao juntos. Entao a
# instrucao muda de "ande" para "PARE em varios lugares".
PARADO_ABAIXO_DE = 0.10

# Confianca minima do ponto para ele virar correspondencia.
#
# Alta de proposito. Uma junta inventada pelo modelo nao avisa que foi
# inventada, e uma correspondencia errada nao piora a calibracao um pouco:
# ela puxa a solucao inteira, porque o ajuste e global.
#
#     Numa media, um ponto ruim dilui. Num ajuste, ele arrasta.
CONFIANCA_MINIMA = 0.6


def _ponto(juntas, conf, indices, minimo=CONFIANCA_MINIMA):
    """A media dos indices que passaram na confianca. Ou None."""
    if juntas is None or conf is None:
        return None
    bons = [juntas[i] for i in indices if float(conf[i]) >= minimo]
    if not bons:
        return None
    p = np.mean(bons, axis=0)
    return float(p[0]), float(p[1])


def _pixeis_da_vista(pose):
    """{nome_da_junta: (u, v)} da pose crua de uma camera."""
    if pose is None or pose.juntas_2d is None:
        return None
    j, c = pose.juntas_2d, pose.conf_2d
    achados = {nome: _ponto(j, c, indices)
               for nome, (indices, _fracao) in PONTOS.items()}
    return achados if any(v is not None for v in achados.values()) else None


FRACAO_DA_ESTATURA = {nome: fracao for nome, (_i, fracao) in PONTOS.items()}


def _juntar(cruas, papel, pe_no_chao, pixeis):
    """Guarda a FRACAO da estatura, e nao a altura em metros.

    ERRO MEU, VISTO NA PRIMEIRA CORRIDA: `alturas 13`.

    `escala.estatura(id)` e uma estimativa que CONVERGE — ela muda a cada
    quadro enquanto junta amostras. Multiplicando por ela na hora, o mesmo
    ombro fisico virava treze alturas diferentes no mundo, e a nuvem que
    deveria ser tres planos limpos virava tres borroes.

        Uma grandeza que ainda esta convergindo nao pode ser usada como
        regua enquanto converge. Ou se espera, ou se guarda a razao e se
        multiplica no fim.

    Guardar a fracao e multiplicar uma vez, no fim, com a estatura final,
    deixa os planos exatos e ainda permite recalcular sem andar de novo.
    """
    c = cruas.setdefault(papel, {"pontos": [], "quadros": 0})
    c["quadros"] += 1
    x, y = pe_no_chao
    for nome, uv in pixeis.items():
        if uv is not None:
            c["pontos"].append((float(x), float(y),
                                FRACAO_DA_ESTATURA[nome], uv, nome))


def _fechar(cruas, estatura):
    """As fracoes viram metros, de uma vez, com a estatura final."""
    fora = {}
    for papel, c in cruas.items():
        coleta = Coleta(quadros=c["quadros"])
        for x, y, fracao, uv, _nome in c["pontos"]:
            coleta.juntar((x, y, fracao * estatura), uv)
        fora[papel] = coleta
    return fora


def _gravar(papel, K, dist, R, t, tamanho, erro, quadros, pares):
    """Escreve nos MESMOS arquivos que o resto do programa ja le."""
    calib = RAIZ / "calibracao"
    calib.mkdir(exist_ok=True)
    H = homografia_da_pose(K, R, t)
    if H is None:
        return None

    nota = [
        "CALIBRADA COM A PESSOA ANDANDO. Sem tabuleiro, sem fita, sem clique.",
        "",
        f"{pares} correspondencias 3D->2D de {quadros} quadros de caminhada.",
        f"erro de reprojecao: {erro:.2f} px",
        "",
        "A pessoa e o objeto de calibracao: 1,80 m de estatura medida, em pe,",
        "vista pelas tres cameras ao mesmo tempo. A do teto diz onde ela esta",
        "em metros; a antropometria diz a que altura ficam ombro e nariz.",
        "",
        "    Quando o proprio objeto de interesse tem dimensao conhecida, ele",
        "    e o padrao de calibracao.",
        "",
        "Metodo: PAMI 2006, 'Camera Calibration from Video of a Walking Human'.",
    ]
    (calib / f"homografia-{papel}.json").write_text(json.dumps({
        "H": H.tolist(), "papel": papel,
        "resolucao": [int(tamanho[0]), int(tamanho[1])],
        "_por": "caminhada", "_erro_px": round(erro, 2), "_nota": nota,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (calib / f"intrinseca-{papel}.json").write_text(json.dumps({
        "K": K.tolist(), "dist": np.asarray(dist).ravel().tolist(),
        "resolucao": [int(tamanho[0]), int(tamanho[1])],
        "rms_px": round(erro, 3), "_por": "caminhada", "_nota": nota,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return calib / f"homografia-{papel}.json"


def _despejar(cruas, estatura, destino):
    """Grava os pares crus. Para eu poder analisar em vez de adivinhar.

        Quando o resultado nao faz sentido, o proximo passo nao e outra
        hipotese: e olhar o dado que a produziu.
    """
    if not cruas:
        return
    caminho = RAIZ / destino
    caminho.parent.mkdir(parents=True, exist_ok=True)
    corpo = {"estatura_m": estatura,
             "fracoes": FRACAO_DA_ESTATURA,
             "por_papel": {
                 papel: {"quadros": c["quadros"],
                         "pontos": [{"x": x, "y": y, "fracao": f,
                                     "u": uv[0], "v": uv[1], "junta": nome}
                                    for x, y, f, uv, nome in c["pontos"]]}
                 for papel, c in cruas.items()}}
    caminho.write_text(json.dumps(corpo, indent=1), encoding="utf-8")
    print(f"  pares crus em {destino}")


def main():
    p = argparse.ArgumentParser(
        description="calibra as outras cameras com voce andando")
    p.add_argument("--segundos", type=float, default=90.0)
    p.add_argument("--papeis", nargs="*", default=["frontal", "lateral"])
    p.add_argument("--captura", default="1280x720")
    p.add_argument("--gravar", action="store_true")
    p.add_argument("--salvar-pares", default="dados/pares_calibracao.json",
                   help="onde despejar os pares crus, para analise")
    args = p.parse_args()

    from src.app.orquestrador import Orquestrador

    print(__doc__)
    larg, alt = (int(v) for v in args.captura.lower().split("x"))
    app = Orquestrador(captura=(larg, alt), com_pose=True)
    app.montar_cameras_reais().montar_visao().iniciar()

    cruas = {}
    estaturas = []
    andando = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.segundos:
            instante = app.passo()
            if instante is None:
                time.sleep(0.005)
                continue

            # UMA PESSOA SO. Com duas em cena, o sistema nao sabe qual pose da
            # lateral pertence a qual pessoa do alto — e uma correspondencia
            # trocada nao estraga um ponto: estraga a calibracao inteira.
            pessoas = list(app.gemeo.pessoas.values())
            if len(pessoas) != 1:
                continue
            pessoa = pessoas[0]
            if pessoa.prevendo:
                continue                 # posicao prevista nao e posicao vista
            if pessoa.velocidade >= PARADO_ABAIXO_DE:
                andando += 1
                continue                 # em movimento o corpo nao e vertical

            estatura = app.espacial.escala.estatura(pessoa.id)
            if not estatura:
                continue
            estaturas.append(float(estatura))

            poses = app.espacial.poses_por_papel
            for papel in args.papeis:
                pixeis = _pixeis_da_vista(poses.get(papel))
                if pixeis:
                    _juntar(cruas, papel, (pessoa.x, pessoa.y), pixeis)

            se = time.monotonic() - t0
            if int(se) != int(se - 0.2):
                quanto = "  ".join(f"{k}:{len(v['pontos'])}"
                                   for k, v in cruas.items())
                print(f"\r  {se:5.1f}s   PARE em varios lugares   "
                      f"pares {quanto}   (descartados andando: {andando})   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n  interrompido")
    finally:
        app.parar()

    # A ESTATURA FINAL, UMA SO, E PELA MEDIANA.
    #
    # Ela converge ao longo da caminhada; a mediana das amostras e o valor
    # que a serie inteira sustenta, e nao o ultimo palpite.
    estatura_final = float(np.median(estaturas)) if estaturas else 0.0
    print(f"\n\n  estatura: mediana {estatura_final:.3f} m  de "
          f"{len(estaturas)} amostras")
    if estaturas:
        print(f"            varia de {min(estaturas):.2f} a "
              f"{max(estaturas):.2f} m")
    print()
    coletas = _fechar(cruas, estatura_final) if estatura_final else {}
    _despejar(cruas, estatura_final, args.salvar_pares)
    if not coletas:
        raise SystemExit(
            "  nao juntei par nenhum.\n"
            "  Ou nenhuma camera te viu, ou o rastro nao sobreviveu.\n"
            "  Rode `python rodar.py` antes e confira que ha 1 pessoa.\n")

    algum = False
    for papel, coleta in sorted(coletas.items()):
        fonte = app.cameras.fontes.get(papel)
        # A resolucao REAL que a fonte entregou, e nao a pedida — a mesma
        # licao do `homografia.py`: pedir 1280x720 e receber 640x480
        # acontece, e gravar a pedida documenta uma intencao como medida.
        tamanho = ((int(fonte.largura), int(fonte.altura)) if fonte
                   else (640, 480))
        achado = resolver(coleta, tamanho)
        linhas, bom = diagnostico(coleta, achado, tamanho)
        print(f"  {papel.upper()}")
        print("\n".join(linhas))
        if bom and args.gravar:
            K, dist, R, t, erro = achado
            destino = _gravar(papel, K, dist, R, t, tamanho, erro,
                              coleta.quadros, len(coleta))
            print(f"  gravado em {destino}")
            algum = True
        elif bom:
            print("  (nao gravei — use --gravar)")
        print()

    if algum:
        print("  agora:  python rodar.py\n")


if __name__ == "__main__":
    main()
