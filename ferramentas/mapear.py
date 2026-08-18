"""As tres cameras mapeiam o quarto. Roda uma vez, sem clique nenhum.

    python ferramentas/mapear.py            olha e mostra
    python ferramentas/mapear.py --gravar   escreve loja/mapa.npz

ANTES DA PRIMEIRA VEZ

    git clone --recursive https://github.com/naver/dust3r.git
    pip install roma einops

E SO ISSO — nao ha `pip install` do proprio DUSt3R.

O repositorio nao tem `setup.py` nem `pyproject.toml`: ele nao e um pacote,
e uma pasta de codigo para ser usada de onde esta. `pip install -e dust3r`
responde

    ERROR: does not appear to be a Python project

que soa como repositorio quebrado e e so uma expectativa errada de quem
instala. Este arquivo poe a pasta no caminho de import sozinho.

POR QUE DUSt3R E NAO VGGT

O VGGT e melhor e foi a primeira escolha. O peso dele tem 5 GB, e depois de
quatro tentativas nao coube nos 8 GB livres da maquina — o downloader precisa
do dobro, e cada falha deixava lixo para a seguinte. O DUSt3R e da mesma
familia, resolve o mesmo problema, e o peso tem 2,3 GB.

    O metodo certo que nao roda na maquina que existe nao e o metodo certo.
    E uma preferencia.

LICENCA: CC BY-NC-SA, nao-comercial — cobre o uso academico de hoje.

O QUE ACONTECE, EM ORDEM

    1. le os quadros salvos em dados/levantamento
    2. DUSt3R devolve a pose das tres e uma nuvem densa, sem calibracao
    3. o chao da nuvem e casado com a homografia, que ja esta em metros
    4. sai o mapa: cameras situadas e ambiente reconstruido, tudo em metros

O PASSO 3 E O QUE PRENDE O MAPA A REALIDADE.

Estas redes resolvem a geometria a menos de uma similaridade: forma certa,
tamanho e orientacao arbitrarios. A homografia da camera do alto ja define
metro neste quarto, medida com trena. Casar uma na outra custa sessenta
pontos do chao.

    Um numero que ja existe em algum lugar nao deve ser reescrito noutro.
    Deve ser LIDO de onde ele mora.
"""
import argparse
import os
import sys
from pathlib import Path

# O DOWNLOADER CLASSICO, E NAO O XET. Decidido em 18/08, medindo.
#
# O xet baixa comprimido e depois RECONSTROI o arquivo:
#
#     downloading bytes:   2,63 GB
#     reconstructing file: 5,03 GB
#
# Precisa do espaco dos dois ao mesmo tempo — mais de 7 GB para um peso de 5.
# E quando falta, ele nao para: vai ate o meio e morre com "Background writer
# channel closed", deixando dois gigabytes de lixo que a proxima tentativa
# encontra pela frente.
#
#     Um download que falha sem limpar o que escreveu nao custa uma tentativa:
#     custa todas as seguintes, cada vez com menos espaco.
#
# O downloader classico escreve direto, sem a etapa de reconstrucao.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from percepcao.chao import carregar_homografia, para_metros    # noqa: E402
from src.mundo.mapeamento import amarrar                       # noqa: E402

PAPEIS = ("alto", "frontal", "lateral")


def _dust3r(imagens):
    """Poses e nuvem, sem calibracao. Devolve (nuvem, poses, do_alto, pixels).

    DUSt3R, da mesma familia do VGGT e pelo mesmo motivo: recebe imagens de
    cameras sem calibracao e devolve pose e geometria juntas. Trocado em
    18/08 por um motivo que nao e tecnico — o peso do VGGT tem 5 GB e nao
    coube no disco, depois de quatro tentativas; o do DUSt3R tem 2,3 GB.

        O metodo certo que nao roda na maquina que existe nao e o metodo
        certo. E uma preferencia.

    O que muda daqui para baixo: nada. `amarrar` acha o chao, casa com a
    homografia e devolve metros — e nao pergunta de qual rede veio a nuvem.

    LICENCA: CC BY-NC-SA, nao-comercial. Cobre o uso academico de hoje. No
    dia em que a Smart Store for vendida, este e um dos itens a revisitar.
    """
    # A PASTA CLONADA ENTRA NO CAMINHO DE IMPORT.
    #
    # O DUSt3R nao se instala; ele mora onde foi clonado. `croco` e submodulo
    # dele e precisa entrar tambem — sem isso o import falha em `models.croco`
    # com uma mensagem que parece falta de dependencia.
    for pasta in (RAIZ / "dust3r", RAIZ / "dust3r" / "croco"):
        if pasta.is_dir() and str(pasta) not in sys.path:
            sys.path.insert(0, str(pasta))

    try:
        import torch
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
    except ImportError as e:
        raise SystemExit(
            f"\n  falta {e.name}. O DUSt3R e do GitHub, nao do PyPI:\n\n"
            f"      git clone --recursive "
            f"https://github.com/naver/dust3r.git\n"
            f"      pip install roma einops\n\n"
            f"  Clone DENTRO de {RAIZ.name}. Nao ha pip install do DUSt3R:\n"
            f"  ele nao tem setup.py, e uma pasta de codigo, e este arquivo\n"
            f"  a poe no caminho de import sozinho.\n")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  DUSt3R em {dispositivo}"
          f"{' (sem GPU: alguns minutos)' if dispositivo == 'cpu' else ''}")

    modelo = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(dispositivo)
    vistas = load_images(imagens, size=512)

    # `complete` compara TODAS as vistas duas a duas. Com tres imagens sao
    # tres pares — barato, e evita que a lateral fique so amarrada a frontal
    # quando o que ela compartilha de verdade e com a do alto.
    pares = make_pairs(vistas, scene_graph="complete", symmetrize=True)
    saida = inference(pares, modelo, dispositivo, batch_size=1)

    cena = global_aligner(saida, device=dispositivo,
                          mode=GlobalAlignerMode.PointCloudOptimizer)
    cena.compute_global_alignment(init="mst", niter=300, schedule="cosine",
                                  lr=0.01)

    matrizes = cena.get_im_poses().detach().cpu().numpy()   # camera -> mundo
    pontos = [p.detach().cpu().numpy() for p in cena.get_pts3d()]
    mascaras = [m.detach().cpu().numpy() for m in cena.get_masks()]

    poses = {}
    for i, papel in enumerate(PAPEIS[:len(matrizes)]):
        m = matrizes[i]
        poses[papel] = (m[:3, 3], m[:3, :3] @ np.array([0.0, 0.0, 1.0]))

    nuvem = np.vstack([p[m] for p, m in zip(pontos, mascaras)])

    # a vista 0 e a do alto — a unica com homografia, e por isso a unica que
    # pode virar ancora
    do_alto = pontos[0][mascaras[0]]
    pixels = np.argwhere(mascaras[0])[:, ::-1].astype(float)

    # os pixels sao do quadro redimensionado para 512; a homografia foi feita
    # no tamanho original. Sem esta conversao a ancora aponta para o lugar
    # errado, e o mapa inteiro sai deslocado sem erro nenhum na tela.
    alt, larg = mascaras[0].shape
    original = cv2.imread(imagens[0])
    pixels[:, 0] *= original.shape[1] / larg
    pixels[:, 1] *= original.shape[0] / alt
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

    nuvem, poses, do_alto, pixels = _dust3r(imagens)
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
