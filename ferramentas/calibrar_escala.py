"""
Calibra a escala vertical: uma pessoa de altura conhecida, uma vez.

    python ferramentas/calibrar_escala.py --estatura 1.78

Depois disso a camera do alto mede a estatura de QUALQUER pessoa em metros, e
a altura da mao deixa de depender de uma proporcao chutada sobre o tronco.

O QUE ELE FAZ

Pede que voce fique EM PE e ande um pouco pela area por vinte segundos. Nesse
tempo o `FiltroDePlausibilidade` aprende a razao geometrica da sua caixa, e a
conta fecha:

    Hc = sua_estatura / razao_observada

Uma pessoa medida uma vez calibra a camera para todas as outras. Ver
`src/acao/escala.py` para a geometria.

POR QUE NAO MEDIR A CAMERA COM TRENA

Daria certo e seria pior. Medir a altura da lente exige subir numa cadeira, e
o numero que importa nao e a distancia ao teto: e a altura acima do PLANO que
a homografia calibrou — que pode nao ser exatamente o piso, dependendo de onde
os pontos de calibracao foram marcados.

    Calibrar pelo que o sistema mede evita a discordancia entre a regua e o
    modelo. Se o modelo estiver torto, a calibracao entorta junto e o
    resultado continua certo.

ANDE. NAO FIQUE PARADO.

O filtro so aprende com quem PERCORREU distancia — decisao de 08/08, depois de
uma cadeira com roupas ensinar o filtro que "pessoa aqui tem esse tamanho".
Parado, nenhuma amostra entra e a calibracao nao acontece.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.acao.escala import EscalaVertical                 # noqa: E402
from src.app.orquestrador import Orquestrador              # noqa: E402
from src.nucleo import log as logmod                       # noqa: E402
from src.nucleo.voz import Voz                             # noqa: E402

DESTINO = RAIZ / "config" / "escala.json"
LIMPAR = "\033[H\033[J"


def coletar(app, segundos, voz):
    """Roda o laco e devolve a razao mediana da caixa de quem esta em cena."""
    from collections import deque

    razoes = deque(maxlen=300)
    t0 = time.monotonic()
    voz.dizer("Fique em pe e ande devagar pela area, sem agachar.")

    while time.monotonic() - t0 < segundos:
        if app.passo() is None:
            time.sleep(0.005)
            continue

        filtro = app.espacial.plausibilidade
        for caixa in app.espacial.caixas_por_id.values():
            r = filtro.razao(caixa)
            if r:
                razoes.append(r)

        falta = segundos - (time.monotonic() - t0)
        print(f"{LIMPAR}CALIBRACAO DA ESCALA VERTICAL      faltam {falta:4.1f} s\n")
        print("  FIQUE EM PE e ande devagar pela area. Nao agache.\n")
        print(f"  amostras da caixa: {len(razoes)}")
        print(f"  {app.espacial.resumo()['altura']}")
        if not app.espacial.caixas_por_id:
            print("\n  NINGUEM DETECTADO — entre no campo da camera do alto")

    return razoes


def main():
    p = argparse.ArgumentParser(description="Calibra a escala vertical")
    p.add_argument("--estatura", type=float, required=True,
                   help="sua altura em metros, medida com trena. Ex: 1.78")
    p.add_argument("--segundos", type=float, default=25.0)
    p.add_argument("--planta", default="loja/bancada.json")
    p.add_argument("--sem-voz", action="store_true")
    p.add_argument("--log", default="AVISO")
    args = p.parse_args()

    logmod.configurar(args.log)
    voz = Voz(ligada=not args.sem_voz)

    app = Orquestrador(planta=args.planta, captura=(640, 480))
    app.montar_cameras_reais()
    app.montar_visao()
    app.iniciar()

    try:
        razoes = coletar(app, args.segundos, voz)
    finally:
        voz.calar()
        app.parar()

    if len(razoes) < 30:
        print(f"\n  so {len(razoes)} amostras — poucas para calibrar.")
        print("  A camera do alto precisa te ver EM PE e voce precisa ANDAR.")
        return 1

    import numpy as np

    mediana = float(np.median(razoes))
    dispersao = float(np.percentile(razoes, 75)
                      - np.percentile(razoes, 25)) / mediana

    try:
        altura = EscalaVertical.calibrar(args.estatura, mediana)
    except ValueError as e:
        print(f"\n  {e}")
        return 1

    print(f"{LIMPAR}CALIBRACAO DA ESCALA VERTICAL\n")
    print(f"  sua estatura declarada    {args.estatura:.2f} m")
    print(f"  razao mediana da caixa    {mediana:.4f}  "
          f"({len(razoes)} amostras, dispersao {dispersao:.0%})")
    print(f"  ALTURA DA CAMERA DO ALTO  {altura:.2f} m\n")

    # A dispersao e o unico sinal de que a calibracao NAO deve ser confiada.
    # Caixa tremendo muito significa deteccao instavel, e a mediana de dados
    # instaveis e um numero preciso sobre nada.
    if dispersao > 0.25:
        print(f"  ATENCAO: dispersao de {dispersao:.0%} e alta. A caixa esta")
        print("  instavel — provavelmente voce saiu do quadro em parte dos")
        print("  quadros. Refaca ficando inteiro no campo da camera do alto.")
    if not 1.2 <= altura <= 4.0:
        print(f"  ATENCAO: {altura:.2f} m e uma altura de camera improvavel.")
        print("  Confira se a estatura declarada esta certa.")

    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.write_text(json.dumps({
        "altura_camera_m": round(altura, 4),
        "_como": "estatura_conhecida / razao_observada da camera do alto",
        "estatura_de_referencia_m": args.estatura,
        "razao_mediana": round(mediana, 5),
        "amostras": len(razoes),
        "dispersao": round(dispersao, 3),
        "quando": datetime.now().isoformat(timespec="seconds"),
        "_atencao": [
            "Este numero vale enquanto a camera do alto NAO for movida.",
            "Mexeu na camera ou recalibrou a homografia? Refaca isto.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n  gravado em {DESTINO}")
    print("  A partir de agora a altura da mao sai MEDIDA, sem o til.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
