"""As tres cameras mapeiam o quarto. Roda uma vez, sem clique nenhum.

    python ferramentas/mapear.py            olha e mostra
    python ferramentas/mapear.py --gravar   escreve loja/mapa.npz

ANTES DA PRIMEIRA VEZ

    git clone --recursive https://github.com/naver/dust3r.git
    pip install roma einops scipy trimesh matplotlib tqdm

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
from src.mundo.mapeamento import (amarrar,                     # noqa: E402
                                  plano_dominante)

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
        # DUAS FALHAS DIFERENTES, DUAS MENSAGENS DIFERENTES.
        #
        # A primeira versao respondia "clone o repositorio" para qualquer
        # import que faltasse — inclusive `scipy`, que ja estava clonado e so
        # precisava de um pip. Quem leu a mensagem foi mandado repetir um
        # passo que ja tinha dado.
        #
        #     Uma mensagem de erro que nao distingue as causas nao esta
        #     ajudando: esta apostando na mais provavel e errando o resto.
        if e.name and e.name.split(".")[0] in ("dust3r", "croco", "models"):
            raise SystemExit(
                f"\n  nao achei o DUSt3R. Ele nao se instala por pip — e uma\n"
                f"  pasta de codigo, e tem que ser clonada dentro de\n"
                f"  {RAIZ.name}:\n\n"
                f"      git clone --recursive "
                f"https://github.com/naver/dust3r.git\n")
        raise SystemExit(
            f"\n  o DUSt3R esta aqui, mas falta uma dependencia dele: "
            f"{e.name}\n\n      pip install {e.name}\n\n"
            f"  As que ele costuma pedir, de uma vez:\n"
            f"      pip install roma einops scipy trimesh matplotlib tqdm\n")

    # CONFERE O DISCO ANTES DE BAIXAR, e nao depois de treze minutos.
    #
    # Seis tentativas morreram por falta de espaco em 18/08, cada uma depois
    # de baixar gigabytes. E o numero enganava: cada falha deixava restolho
    # para a seguinte, entao o espaco livre CAIA a cada tentativa — e quando
    # o processo morria, o Python apagava o temporario dele e o espaco
    # voltava. Quem olhava o Windows depois via 7 GB; quem comecava a
    # proxima encontrava 2.
    #
    #     Conferir um recurso depois de gasta-lo nao e conferir: e narrar.
    #
    # Meia hora de download perdida por uma verificacao de tres linhas.
    import shutil
    livre = shutil.disk_usage(Path.home()).free / 1e9
    if livre < 3.0:
        raise SystemExit(
            f"\n  so ha {livre:.1f} GB livres, e o peso precisa de 2,3 mais\n"
            f"  folga. Provavelmente sao downloads pela metade de tentativas\n"
            f"  anteriores — cada falha deixa o pedaco para tras:\n\n"
            f'      Remove-Item "$env:USERPROFILE\\.cache\\huggingface" '
            f"-Recurse -Force\n")
    print(f"  disco: {livre:.1f} GB livres")

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

    O CHAO E ACHADO, NAO SUPOSTO. Consertado em 18/08, depois de rodar.

    A primeira versao pegava os 60 pontos de MENOR z e os chamava de chao:

        baixos = np.argsort(pontos[:, 2])[:quantas * 3]

    Mas o z do DUSt3R nao e altura — e profundidade a partir da camera, num
    eixo que a rede escolhe sozinha. Os "mais baixos" eram os mais PROXIMOS
    da lente, espalhados por parede, movel e piso. A ancora inteira era ruido,
    e o mapa saiu com o teto a 0,65 m e a camera do teto abaixo do chao.

        Um eixo so e altura depois que alguem decide qual e o chao. Antes
        disso, ordenar por ele e ordenar por nada.

    Agora o plano dominante e ajustado por RANSAC nos pontos desta camera —
    num quarto ele e o piso — e as ancoras saem dos pontos que caem NELE.
    """
    achado = plano_dominante(pontos, tolerancia=0.02)
    if achado is None:
        return np.zeros((0, 3)), np.zeros((0, 2))
    _, _, no_chao = achado

    indices = np.where(no_chao)[0]
    if len(indices) > quantas:
        # espalhados, e nao os primeiros: ancora concentrada num canto fixa a
        # rotacao mal, mesmo com residuo pequeno
        indices = indices[np.linspace(0, len(indices) - 1, quantas, dtype=int)]

    na_nuvem, no_mundo = [], []
    for i in indices:
        try:
            m = para_metros(h, float(pixels[i][0]), float(pixels[i][1]))
        except Exception:
            continue
        if m is None:
            continue
        na_nuvem.append(pontos[i])
        no_mundo.append(np.asarray(m).ravel()[:2])
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
