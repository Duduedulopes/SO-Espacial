"""As tres cameras mapeiam o quarto. Roda uma vez, sem clique nenhum.

    python ferramentas/mapear.py            olha e mostra
    python ferramentas/mapear.py --gravar   escreve loja/mapa.npz

ANTES DA PRIMEIRA VEZ

    pip install torch torchvision
    pip install --no-deps git+https://github.com/facebookresearch/vggt.git
    pip install huggingface_hub safetensors einops

O VGGT NAO ESTA NO PyPI — e repositorio do GitHub.

E O `--no-deps` NAO E GAMBIARRA, e a parte que mais importa aqui.

O VGGT pina `numpy<2`. Em Python novo nao existe wheel pronta do numpy 1.26,
entao o pip tenta COMPILAR do zero, e para com

    ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ...]

que parece falta de compilador e e, na verdade, um pino conservador de outra
biblioteca. O codigo do VGGT roda com numpy 2.

    Instalar um compilador para satisfazer um pino que ninguem precisa e
    consertar o sintoma no lugar mais caro possivel.

LICENCA: o peso padrao (`facebook/VGGT-1B`) e nao-comercial e baixa sem
cadastro nenhum — cobre trabalho academico, que e o uso de hoje. Para uso
comercial existe `--comercial`, que usa o `VGGT-1B-Commercial` e exige um
formulario no HuggingFace. Mesma qualidade; muda so a licenca.

Os pesos baixam sozinhos na primeira execucao (~2 GB) e ficam guardados.

O QUE ACONTECE, EM ORDEM

    1. colhe um quadro estavel de cada camera
    2. VGGT devolve a pose das tres e uma nuvem densa — sem calibracao
    3. o chao da nuvem e casado com a homografia, que ja esta em metros
    4. sai o mapa: cameras situadas e ambiente reconstruido, tudo em metros

O PASSO 3 E O QUE PRENDE O MAPA A REALIDADE.

O VGGT resolve a geometria a menos de uma similaridade: forma certa, tamanho e
orientacao arbitrarios. A homografia da camera do alto ja define metro neste
quarto, medido com trena. Casar um no outro custa dez pontos do chao.

    Um numero que ja existe em algum lugar nao deve ser reescrito noutro.
    Deve ser LIDO de onde ele mora.
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from percepcao.chao import carregar_homografia, para_metros    # noqa: E402
from src.mundo.mapeamento import amarrar                       # noqa: E402

PAPEIS = ("alto", "frontal", "lateral")


def _vggt(imagens, peso):
    """Poses e nuvem, sem calibracao. Devolve (nuvem, poses, pontos_do_alto).

    `pontos_do_alto` sao os pontos da nuvem que a camera do alto enxerga, com
    o pixel de cada um — e sao eles que viram ancora, porque so a camera do
    alto tem homografia.
    """
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as e:
        raise SystemExit(
            f"\n  falta {e.name}. O VGGT nao esta no PyPI — e do GitHub:\n\n"
            f"      pip install torch torchvision\n"
            f"      pip install --no-deps "
            f"git+https://github.com/facebookresearch/vggt.git\n"
            f"      pip install huggingface_hub safetensors einops\n\n"
            f"  O --no-deps evita que o pip tente compilar numpy 1.26 do zero\n"
            f"  por causa de um pino conservador. Sem ele o erro e\n"
            f"  'Unknown compiler(s)', que parece outra coisa.\n")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  VGGT em {dispositivo}"
          f"{' (sem GPU: vai levar alguns minutos)' if dispositivo == 'cpu' else ''}")

    # DOIS PESOS, E A ESCOLHA E DE LICENCA, NAO DE QUALIDADE.
    #
    #   facebook/VGGT-1B              nao-comercial, baixa sem cadastro
    #   facebook/VGGT-1B-Commercial   permite uso comercial, exige formulario
    #                                 no HuggingFace (aprovacao automatica)
    #
    # O desempenho e equivalente. O padrao aqui e o gratuito porque hoje isto
    # e trabalho academico — e trabalho academico esta coberto por ele.
    #
    #     Exigir a licenca comercial antes de existir uso comercial e cobrar
    #     hoje o preco de um problema de amanha.
    #
    # No dia em que a Smart Store for vendida ou licenciada, `--comercial`
    # troca o peso. E uma linha, e o resto do codigo nao muda.
    modelo = VGGT.from_pretrained(peso).to(dispositivo).eval()
    lote = load_and_preprocess_images(imagens).to(dispositivo)

    with torch.no_grad():
        saida = modelo(lote)
    extri, _ = pose_encoding_to_extri_intri(saida["pose_enc"],
                                            lote.shape[-2:])
    mapa_de_pontos = saida["world_points"][0].cpu().numpy()
    confianca = saida["world_points_conf"][0].cpu().numpy()
    extri = extri[0].cpu().numpy()

    # So os pontos em que a rede confia. O resto e preenchimento de textura
    # lisa, e parede lisa e o que mais existe num quarto.
    firme = confianca > np.quantile(confianca, 0.5)
    nuvem = mapa_de_pontos[firme].reshape(-1, 3)

    poses = {}
    for i, papel in enumerate(PAPEIS[:len(extri)]):
        r, t = extri[i][:3, :3], extri[i][:3, 3]
        poses[papel] = (-r.T @ t, r.T @ np.array([0.0, 0.0, 1.0]))

    do_alto = mapa_de_pontos[0][firme[0]].reshape(-1, 3)
    pixels = np.argwhere(firme[0])[:, ::-1].astype(float)
    return nuvem, poses, do_alto, pixels


def _ancoras(pontos, pixels, h, quantas=60):
    """Pontos que a camera do alto ve NO CHAO, na nuvem e em metros.

    A homografia so vale para o chao — foi ignorar isso que produziu, em
    18/08, uma estante a 1,79 m na diagonal. Aqui a regra e obedecida: entram
    apenas os pontos que a nuvem coloca no piso.
    """
    baixos = np.argsort(pontos[:, 2])[:quantas * 3]
    na_nuvem, no_mundo = [], []
    for i in baixos:
        try:
            m = para_metros(h, float(pixels[i][0]), float(pixels[i][1]))
        except Exception:
            continue
        if m is None:
            continue
        na_nuvem.append(pontos[i])
        no_mundo.append(np.asarray(m).ravel()[:2])
        if len(na_nuvem) >= quantas:
            break
    return np.array(na_nuvem), np.array(no_mundo)


def main():
    p = argparse.ArgumentParser(description="as 3 cameras mapeiam o quarto")
    p.add_argument("--pasta", default="dados/levantamento")
    p.add_argument("--saida", default="loja/mapa.npz")
    p.add_argument("--gravar", action="store_true")
    p.add_argument("--comercial", action="store_true",
                   help="peso de licenca comercial; exige "
                        "formulario no HuggingFace")
    args = p.parse_args()

    pasta = RAIZ / args.pasta
    imagens = [str(pasta / f"{papel}.png") for papel in PAPEIS
               if (pasta / f"{papel}.png").exists()]
    if len(imagens) < 2:
        raise SystemExit(f"\n  preciso de ao menos 2 imagens em {args.pasta}.\n"
                         f"  rode antes:  python ferramentas/achar_ambiente.py"
                         f" --so-salvar\n")
    print(f"\n  {len(imagens)} vistas: "
          f"{', '.join(Path(i).stem for i in imagens)}")

    h, calib = carregar_homografia()
    print(f"  homografia: {calib.get('largura_m')} x {calib.get('altura_m')} m")

    peso = ('facebook/VGGT-1B-Commercial' if args.comercial
            else 'facebook/VGGT-1B')
    nuvem, poses, do_alto, pixels = _vggt(imagens, peso)
    print(f"  nuvem bruta: {len(nuvem)} pontos, {len(poses)} poses")

    na_nuvem, no_mundo = _ancoras(do_alto, pixels, h)
    print(f"  ancoras no chao: {len(na_nuvem)}")

    mapa = amarrar(nuvem, poses, na_nuvem, no_mundo)
    if mapa is None:
        raise SystemExit("\n  a amarracao nao fechou. A nuvem nao tem um chao\n"
                         "  reconhecivel, ou as ancoras cairam fora dele.\n")

    print(f"\n  MAPA  escala {mapa.escala:.3f}   "
          f"residuo {mapa.residuo_m * 100:.1f} cm   "
          f"{len(mapa.nuvem)} pontos")
    x0, x1, y0, y1 = mapa.chao
    print(f"  chao  x {x0:+.2f} a {x1:+.2f}   y {y0:+.2f} a {y1:+.2f} m")
    print(f"  altura reconstruida ate {mapa.nuvem[:, 2].max():.2f} m")
    for papel, (c, _) in mapa.poses.items():
        print(f"    {papel:<8} em ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}) m")
    if not mapa.pronto:
        print("\n  MENOS DE DUAS CAMERAS — o mapa nao se sustenta.")

    if args.gravar:
        np.savez(RAIZ / args.saida, nuvem=mapa.nuvem, escala=mapa.escala,
                 residuo_m=mapa.residuo_m,
                 **{f"pose_{k}": v[0] for k, v in mapa.poses.items()})
        print(f"\n  gravado em {args.saida}\n")
    else:
        print("\n  (nao gravei — use --gravar)\n")


if __name__ == "__main__":
    main()
