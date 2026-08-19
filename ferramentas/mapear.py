"""As tres cameras montam o ambiente. A estante e a regua.

    python ferramentas/mapear.py            olha e mostra
    python ferramentas/mapear.py --gravar   escreve em loja/quarto.json
    python ferramentas/mapear.py --vggt     usa o VGGT (peso de 5 GB)

ANTES DA PRIMEIRA VEZ

    git clone --recursive https://github.com/naver/dust3r.git
    pip install roma einops scipy trimesh matplotlib tqdm

Nao ha `pip install` do proprio DUSt3R: ele nao tem setup.py, e uma pasta de
codigo, e este arquivo a poe no caminho de import sozinho.

O QUE ACONTECE

    1. le os tres quadros salvos em dados/levantamento
    2. a rede devolve uma nuvem densa, sem calibracao nenhuma
    3. o plano dominante vira o CHAO, e a cena deita sobre ele
    4. tiradas as paredes, o que sobe e a ESTANTE
    5. altura dela / 1,90 m de trena  ->  a ESCALA, e tudo vira metro
    6. o contorno do piso  ->  a AREA de movimento

A ESTANTE E A REGUA, E ISSO E O QUE MUDOU EM 18/08.

Seis corridas tentaram tirar o metro da homografia, encaixando a nuvem no
retangulo de 1,65 x 1,32 m. Todas terminaram em zero ancoras — e o motivo
final foi humilde: a mascara de confianca da rede descarta piso liso, e
aquele retangulo e quase todo piso liso.

Mas o metro nunca precisou vir de la. A estante mede 1,90 m com trena e esta
no meio da cena.

    Quando um objeto de dimensao conhecida esta na cena, ele e a regua. Ir
    buscar a escala noutro instrumento e atravessar a rua para pegar o que
    esta na mao.

O que sai daqui e o ambiente que as cameras viram — chao, estante e area de
movimento, em metros, no formato que o `rodar.py` ja le.
"""
import argparse
import json
import math
import os
import sys
import time
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

from percepcao.chao import caixa_do_contorno, pegada_no_chao   # noqa: E402
from src.mundo.ambiente import Gabarito                        # noqa: E402
from src.mundo.mapeamento import montar                        # noqa: E402

PAPEIS = ("alto", "frontal", "lateral")


def _vggt(imagens):
    """O mesmo contrato do _dust3r, pelo VGGT. Devolve (nuvem, poses, do_alto, pixels).

    OS DOIS CONVIVEM PORQUE A FRONTEIRA ESTA NO LUGAR CERTO.

    `amarrar` acha o chao, casa com a homografia e devolve metros — e nao
    pergunta de qual rede veio a nuvem. Entao trocar de modelo custa uma
    funcao com esta assinatura, e nada mais no programa muda.

        Quando dois caminhos entregam a mesma coisa, escolher entre eles
        deixa de ser arquitetura e vira uma opcao de linha de comando.

    O VGGT e melhor nas avaliacoes e mais caro em disco: 5 GB contra 2,3. Com
    espaco, vale comparar os dois na MESMA cena — se discordarem muito, o
    problema esta na captura, nao no modelo.
    """
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as e:
        raise SystemExit(
            f"\n  falta {e.name}. O VGGT tambem e do GitHub:\n\n"
            f"      pip install --no-deps "
            f"git+https://github.com/facebookresearch/vggt.git\n")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  VGGT em {dispositivo} — peso de 5 GB na primeira vez")

    modelo = VGGT.from_pretrained("facebook/VGGT-1B").to(dispositivo).eval()
    lote = load_and_preprocess_images(imagens).to(dispositivo)
    with torch.no_grad():
        saida = modelo(lote)

    mapa_de_pontos = saida["world_points"][0].cpu().numpy()
    confianca = saida["world_points_conf"][0].cpu().numpy()
    # TUDO que a camera enxergou, sem cortar pela confianca — ver a nota
    # em `_dust3r`.
    firme = confianca > -np.inf
    nuvem = mapa_de_pontos.reshape(-1, 3)

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    m, _ = pose_encoding_to_extri_intri(saida["pose_enc"], lote.shape[-2:])
    m = m[0].cpu().numpy()
    poses = {}
    for i, papel in enumerate(PAPEIS[:len(m)]):
        r, t = m[i][:3, :3], m[i][:3, 3]
        poses[papel] = (-r.T @ t, r.T @ np.array([0.0, 0.0, 1.0]))

    # grade inteira, sem a mascara de confianca — ver a nota em `_dust3r`
    alt, larg = firme[0].shape
    do_alto = mapa_de_pontos[0]
    grade = np.stack(np.meshgrid(np.arange(larg), np.arange(alt)), axis=-1)
    pixels = grade.reshape(-1, 2).astype(float)
    original = cv2.imread(imagens[0])
    pixels[:, 0] *= original.shape[1] / larg
    pixels[:, 1] *= original.shape[0] / alt
    return nuvem, poses, do_alto, pixels


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

    # TUDO QUE AS CAMERAS ENXERGARAM. Sem mascara de confianca.
    #
    #     a camera precisa reproduzir TODO O CHAO QUE ELA ENXERGAR e as
    #     outras cameras precisam reproduzir TUDO QUE ELAS ENXERGAREM
    #                                             — Eduardo, 18/08
    #
    # A mascara guardava so os pontos de alta confianca, e a confianca destas
    # redes despenca em superficie lisa: parede branca, ladrilho, bandeja de
    # aco. Ou seja, ela descartava exatamente o comodo e guardava as quinas.
    #
    #     A confianca da rede mede o quanto a PROFUNDIDADE dali e incerta.
    #     Usa-la para decidir o que EXISTE e trocar a pergunta.
    #
    # Ponto de profundidade incerta ainda esta na direcao certa: ele erra em
    # quanto esta longe, nao em existir. Para desenhar o comodo isso basta, e
    # sem ele o comodo fica com buraco onde ha parede lisa.
    nuvem = np.vstack([p.reshape(-1, 3) for p in pontos])

    # a vista 0 e a do alto — a unica com homografia, e por isso a unica que
    # pode virar ancora
    # A ANCORA USA A GRADE INTEIRA, SEM A MASCARA DE CONFIANCA.
    #
    # Este e o defeito das seis corridas de 18/08 que terminaram em zero.
    #
    # A mascara do DUSt3R guarda so os pontos de alta confianca — e a
    # confianca destas redes despenca em superficie lisa e sem textura. O chao
    # do quarto e ladrilho liso: a rede confia na estante, nos moveis e nas
    # quinas, e DESCARTA justamente o piso.
    #
    # O retangulo calibrado, por sua vez, e quase todo piso liso. Medido: dos
    # 307.200 pixels da imagem, 92.240 caem dentro dele — e nenhum deles
    # sobrevivia a mascara.
    #
    #     A confianca da rede mede o quanto a PROFUNDIDADE dali e incerta.
    #     Nao mede se o pixel serve de referencia. Filtrar ancora por ela e
    #     usar uma regua para responder outra pergunta.
    #
    # E a profundidade baixa nao atrapalha aqui: a ancora nao precisa da
    # altura do ponto, so da correspondencia entre o pixel e o metro. Quem
    # da o metro e a homografia, que e exata no chao.
    #
    # A nuvem que vai para o desenho continua filtrada — ali a confianca
    # importa, porque ali o que se ve e a profundidade.
    alt, larg = mascaras[0].shape
    do_alto = pontos[0]
    grade = np.stack(np.meshgrid(np.arange(larg), np.arange(alt)), axis=-1)
    pixels = grade.reshape(-1, 2).astype(float)

    # os pixels sao do quadro redimensionado para 512; a homografia foi feita
    # no tamanho original. Sem esta conversao a ancora aponta para o lugar
    # errado, e o mapa inteiro sai deslocado sem erro nenhum na tela.
    original = cv2.imread(imagens[0])
    pixels[:, 0] *= original.shape[1] / larg
    pixels[:, 1] *= original.shape[0] / alt
    return nuvem, poses, do_alto, pixels


MONO_PADRAO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"


def _profundidade_monocular(caminho, modelo=MONO_PADRAO):
    """Uma imagem -> profundidade em METROS. Devolve (mapa, None) ou (None, erro).

    POR QUE ISTO EXISTE AO LADO DO DUSt3R, E NAO DEPOIS DELE

    Reconstrucao multi-vista precisa que as cameras vejam AS MESMAS
    SUPERFICIES. As tres deste arranjo quase nao veem — a do alto olha o
    piso, a frontal olha a pessoa, a lateral olha o perfil da estante. O
    DUSt3R nao estava mal configurado: estava sendo usado fora da hipotese
    dele, e por isso devolveu um comodo de 2,1 m2 num sistema proprio.

    Profundidade monocular metrica nao tem esse requisito. Uma imagem entra,
    metros saem.

    O CUIDADO QUE ESTA FUNCAO EXISTE PARA TOMAR

    O `pipeline` do transformers devolve duas coisas com nomes parecidos:

        depth              uma IMAGEM, normalizada de 0 a 255, para olhar
        predicted_depth    o tensor, em METROS

    Usar a primeira daria um mapa lindo e sem unidade, e o erro so
    apareceria la adiante como uma escala absurda — depois de passar por
    tres etapas que nao teriam culpa nenhuma.

        Duas saidas com nomes parecidos e unidades diferentes sao um erro
        esperando o momento mais caro para acontecer.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as e:
        return None, (f"falta um pacote ({e}).\n"
                      f"      pip install transformers pillow")

    try:
        imagem = Image.open(caminho).convert("RGB")
        processador = AutoImageProcessor.from_pretrained(modelo)
        rede = AutoModelForDepthEstimation.from_pretrained(modelo)
        rede.eval()
        entrada = processador(images=imagem, return_tensors="pt")
        with torch.no_grad():
            saida = rede(**entrada)

        alvo = [(imagem.height, imagem.width)]
        if hasattr(processador, "post_process_depth_estimation"):
            pos = processador.post_process_depth_estimation(
                saida, target_sizes=alvo)
            mapa = pos[0]["predicted_depth"].cpu().numpy()
        else:
            bruto = saida.predicted_depth.unsqueeze(1)
            mapa = torch.nn.functional.interpolate(
                bruto, size=alvo[0], mode="bicubic",
                align_corners=False)[0, 0].cpu().numpy()
        return np.asarray(mapa, dtype=float), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser(description="as 3 cameras mapeiam o quarto")
    p.add_argument("--pasta", default="dados/levantamento")
    p.add_argument("--planta", default="loja/quarto.json")
    p.add_argument("--gravar", action="store_true")
    p.add_argument("--vggt", action="store_true",
                   help="usa o VGGT em vez do DUSt3R (peso de 5 GB)")
    p.add_argument("--mono", action="store_true",
                   help="profundidade monocular na camera do alto; nao precisa "
                        "de sobreposicao entre as vistas")
    p.add_argument("--modelo-mono", default=MONO_PADRAO)
    args = p.parse_args()

    if args.mono:
        return _mapear_mono(args)

    pasta = RAIZ / args.pasta
    imagens = [str(pasta / f"{papel}.png") for papel in PAPEIS
               if (pasta / f"{papel}.png").exists()]
    if len(imagens) < 2:
        raise SystemExit(f"\n  preciso de ao menos 2 imagens em {args.pasta}.\n"
                         f"  rode antes:  python ferramentas/achar_ambiente.py"
                         f" --so-salvar\n")
    print(f"\n  {len(imagens)} vistas: "
          f"{', '.join(Path(i).stem for i in imagens)}")

    gab = Gabarito.de_arquivo("loja/estante.json")
    print(f"  a regua: a estante tem {gab.altura:.2f} m de altura, de trena")

    rede = _vggt if args.vggt else _dust3r
    nuvem, poses, mapa_do_alto, _ = rede(imagens)
    print(f"  nuvem: {len(nuvem)} pontos, {len(poses)} poses")

    # A PONTE COM O MUNDO DO GEMEO.
    #
    # Sem ela o ambiente sai certo em forma e tamanho e FLUTUANDO num sistema
    # proprio — e foi assim que o boneco passou a atravessar a estante. O
    # gemeo e rastreado pela homografia; a estante vinha da reconstrucao; os
    # dois nunca estiveram no mesmo mundo.
    h, calib = None, None
    try:
        from percepcao.chao import carregar_homografia
        h, calib = carregar_homografia()
        print(f"  ponte: homografia de {calib.get('largura_m')} x "
              f"{calib.get('altura_m')} m")
    except SystemExit:
        print("  SEM HOMOGRAFIA: o ambiente vai sair num mundo proprio, e o")
        print("  gemeo nao vai coincidir com ele.")

    # O TAMANHO DO COMODO NAO SAI DA NUVEM. SAI DE QUEM MEDE POSICAO.
    #
    # Ate 19/08 o piso era a extensao dos pontos que o DUSt3R conseguiu casar
    # entre as vistas: 2,1 m2. A camera do alto, pela homografia que ja
    # existia, mede 8,4 m2 do mesmo chao — quatro vezes mais.
    #
    #     A reconstrucao diz o que TEM na sala. Quem sabe o tamanho dela e o
    #     campo de visao de quem mede posicao.
    #
    # LIMITE HONESTO: so a camera do alto entra nesta conta, porque so ela tem
    # homografia. A frontal e a lateral tambem enxergam piso e nao sabem dizer
    # em metros onde ele esta. No dia em que uma delas for calibrada, o comodo
    # cresce de novo — e o codigo que faz isso e o mesmo.
    contorno = None
    if h is not None:
        larg_px, alt_px = (calib or {}).get("resolucao", (640, 480))
        contorno = pegada_no_chao(h, int(larg_px), int(alt_px))
        if contorno is not None:
            area = float(abs(np.dot(contorno[:, 0], np.roll(contorno[:, 1], -1))
                             - np.dot(contorno[:, 1],
                                      np.roll(contorno[:, 0], -1))) / 2.0)
            print(f"  piso: a camera do alto mede {area:.1f} m2 de chao")

    amb = montar(nuvem, gab, mapa_do_alto=mapa_do_alto, homografia=h,
                 calib=calib)
    if amb is None:
        raise SystemExit(
            "\n  nao consegui montar o ambiente. Ou a nuvem nao tem um chao\n"
            "  reconhecivel, ou nao ha nada alto o bastante para ser a\n"
            "  estante. Confira as tres imagens em dados/levantamento.\n")

    x0, x1, y0, y1 = _caixa(amb, contorno)
    ex, ey, rumo = amb.estante
    print(f"\n  AMBIENTE   escala {amb.escala:.2f}")
    if amb.no_mundo_do_gemeo:
        print(f"  ponte      {amb.ancoras} pares, residuo "
              f"{amb.residuo_m * 100:.1f} cm")
        concordam = amb.as_duas_reguas_concordam
        print(f"  as duas reguas: homografia {amb.escala:.2f}  x  "
              f"altura da estante {amb.escala_da_estante:.2f}   "
              f"{'concordam' if concordam else 'DISCORDAM'}")
        if concordam is False:
            print("    Duas reguas independentes discordando e a informacao")
            print("    mais util desta cena. Confira antes de gravar.")
    else:
        print("  SEM PONTE — o ambiente esta num mundo proprio, e o gemeo")
        print("  nao vai coincidir com ele na tela.")
    print(f"  chao       {x1 - x0:.2f} x {y1 - y0:.2f} m de area para andar")
    print(f"  altura     {amb.altura_da_cena:.2f} m ate o ponto mais alto")
    print(f"  estante    em ({ex:+.2f}, {ey:+.2f}) m, "
          f"face a {math.degrees(rumo):+.0f} graus")
    print(f"  nuvem      {len(amb.nuvem)} pontos em metros")

    if args.gravar:
        _gravar_planta(amb, gab, args.planta, contorno)
        np.savez(RAIZ / "loja" / "nuvem.npz", pontos=amb.nuvem)
        print(f"\n  gravado em {args.planta} e loja/nuvem.npz")
        print("\n  agora:  python rodar.py\n")
    else:
        print("\n  (nao gravei — use --gravar)\n")


def _mapear_mono(args):
    """O caminho monocular: uma imagem da camera do alto, e a trena.

    CADA PASSO IMPRIME UM NUMERO QUE DA PARA CONFERIR

        a altura da camera      confira com a trena, no teto
        o chao plano            quanto ele ficou torto, em cm
        a altura da estante     tem que dar os 1,90 que a trena mediu

    Nenhum dos tres foi usado para chegar nos outros. Sao tres reguas
    independentes sobre a mesma cena, e e a concordancia entre elas que
    autoriza gravar.

        Um resultado que nao produz nenhum numero conferivel nao pode ser
        aceito nem recusado — so acreditado.
    """
    from percepcao.chao import carregar_homografia
    from src.mundo.profundidade import (ambiente_do_mono, camera_da_homografia,
                                        nuvem_do_alto)

    caminho = RAIZ / args.pasta / "alto.png"
    if not caminho.exists():
        raise SystemExit(
            f"\n  nao achei {caminho}.\n"
            f"  rode antes:  python ferramentas/achar_ambiente.py --so-salvar\n")

    gab = Gabarito.de_arquivo("loja/estante.json")
    print("\n  MONOCULAR — uma imagem, sem precisar de sobreposicao")
    print(f"  a regua: a estante tem {gab.altura:.2f} m de altura, de trena")

    h, calib = carregar_homografia()
    larg_px, alt_px = calib.get("resolucao", (640, 480))

    cam = camera_da_homografia(h, int(larg_px), int(alt_px))
    if cam is None:
        raise SystemExit(
            "\n  nao consegui deduzir a camera da homografia. Isso acontece\n"
            "  quando o retangulo calibrado esta quase de frente para a\n"
            "  lente: sem perspectiva nao ha informacao de escala nenhuma.\n"
            "  Recalibre clicando um retangulo mais esticado no chao.\n")

    print("\n  CAMERA deduzida do chao que a trena mediu")
    print(f"    altura      {cam.altura_m:.2f} m   <- CONFIRA COM A TRENA")
    print(f"    posicao     ({cam.posicao[0]:+.2f}, {cam.posicao[1]:+.2f}) m")
    print(f"    focal       {cam.focal:.0f} px")
    if cam.discordancia > 0.25:
        print(f"    ATENCAO: as duas estimativas da focal discordam "
              f"{cam.discordancia * 100:.0f}%.")
        print("    A lente pode nao ter pixel quadrado ou o centro optico no")
        print("    meio da imagem. O resto vai sair torto.")
    else:
        print(f"    conferencia {cam.discordancia * 100:.1f}% entre as duas "
              f"estimativas da focal")

    print(f"\n  rodando {args.modelo_mono}")
    print("  (a primeira vez baixa o modelo; depois fica em cache)")
    mapa, erro = _profundidade_monocular(caminho, args.modelo_mono)
    if mapa is None:
        raise SystemExit(f"\n  nao deu: {erro}\n")
    print(f"  profundidade {mapa.shape[1]}x{mapa.shape[0]}, "
          f"{np.nanmin(mapa):.2f} a {np.nanmax(mapa):.2f} (unidade da rede)")

    nuvem = nuvem_do_alto(mapa, h, tamanho_original=(int(larg_px), int(alt_px)),
                          camera=cam)
    if nuvem is None:
        raise SystemExit(
            "\n  a nuvem nao fechou. A escala nao pode ser recuperada do\n"
            "  chao — o que acontece se a maior parte da imagem NAO for\n"
            "  piso, ou se o modelo devolveu profundidade relativa em vez\n"
            "  de metrica. Confira que o modelo tem 'Metric' no nome.\n")

    print(f"\n  NUVEM  {len(nuvem.pontos)} pontos, ja em metros no mundo do gemeo")
    print(f"    escala da rede corrigida por {nuvem.escala:.3f}x")
    print(f"    {nuvem.fracao_de_chao * 100:.0f}% da imagem era chao")
    print(f"    chao plano em {nuvem.residuo_chao_m * 100:.1f} cm   <- a nota")

    contorno = pegada_no_chao(h, int(larg_px), int(alt_px))
    amb = ambiente_do_mono(nuvem, gab)
    if amb is None:
        raise SystemExit(
            "\n  achei o chao e nao achei a estante. Ou ela nao aparece nesta\n"
            "  vista, ou o que sobe na cena nao tem a forma dela.\n")

    ex, ey, rumo = amb.estante
    altura_vista = amb.escala_da_estante * gab.altura
    print(f"\n  ESTANTE em ({ex:+.2f}, {ey:+.2f}) m, "
          f"face a {math.degrees(rumo):+.0f} graus")
    print(f"    altura na nuvem {altura_vista:.2f} m  contra "
          f"{gab.altura:.2f} da trena")
    concordam = amb.as_duas_reguas_concordam
    print(f"    as duas reguas {'concordam' if concordam else 'DISCORDAM'}")
    if concordam is False:
        print("    Duas reguas independentes discordando e a informacao mais")
        print("    util desta cena. Confira antes de gravar.")

    if args.gravar:
        _gravar_planta(amb, gab, args.planta, contorno)
        np.savez(RAIZ / "loja" / "nuvem.npz", pontos=amb.nuvem)
        print(f"\n  gravado em {args.planta} e loja/nuvem.npz")
        print("\n  agora:  python rodar.py\n")
    else:
        print("\n  (nao gravei — use --gravar)\n")


def _caixa(amb, contorno):
    """A caixa que contem TUDO: a pegada da camera e a nuvem.

    Nao e uma das duas, e a uniao. A caixa dimensiona a grade do calor e o
    enquadramento da camera virtual — se algo medido cair fora dela, esse algo
    some do mapa de calor sem aviso.

        Uma caixa que nao contem tudo que foi medido recorta em silencio, que
        e o defeito que este projeto passou um dia inteiro caçando.
    """
    caixas = []
    if amb is not None and amb.chao is not None:
        caixas.append(amb.chao)
    c = caixa_do_contorno(contorno)
    if c is not None:
        caixas.append(c)
    if not caixas:
        return (0.0, 1.0, 0.0, 1.0)
    return (min(b[0] for b in caixas), max(b[1] for b in caixas),
            min(b[2] for b in caixas), max(b[3] for b in caixas))


def _gravar_planta(amb, gab, caminho, contorno=None):
    """Escreve o ambiente na planta que o `rodar.py` ja le.

    O chao, a estante com as medidas de trena, e a entrada e a saida
    deduzidas dela. E o mesmo formato de sempre — o que mudou foi de onde os
    numeros vieram.
    """
    from ferramentas.achar_ambiente import _portas
    from src.mundo.ambiente import Ambiente

    x0, x1, y0, y1 = _caixa(amb, contorno)
    ex, ey, rumo = amb.estante

    caminho = RAIZ / caminho
    d = json.loads(caminho.read_text(encoding="utf-8"))
    d["chao"] = {"xmin": round(x0, 3), "xmax": round(x1, 3),
                 "ymin": round(y0, 3), "ymax": round(y1, 3),
                 "_nota": [
                     "A CAIXA que contem tudo: a pegada da camera do alto e a",
                     "nuvem das tres vistas. Serve para dimensionar a grade do",
                     "calor e o enquadramento — nao para desenhar o piso.",
                     "",
                     "Para desenhar o piso existe `contorno`, que e o",
                     "quadrilatero de verdade. A caixa dele tem quase o dobro",
                     "da area: projecao de retangulo nao volta a ser",
                     "retangulo.",
                 ]}
    if contorno is not None and len(contorno) >= 3:
        d["contorno"] = [[round(float(x), 3), round(float(y), 3)]
                         for x, y in contorno]
        d["_contorno_nota"] = [
            "O PISO QUE A CAMERA DO ALTO REALMENTE MEDE, em metros.",
            "",
            "Ate 19/08 o comodo saia da extensao da nuvem do DUSt3R: 2,1 m2.",
            "A mesma homografia que ja existia mede 8,4 m2 do mesmo chao.",
            "",
            "    A reconstrucao diz o que TEM na sala. Quem sabe o tamanho",
            "    dela e o campo de visao de quem mede posicao.",
            "",
            "O retangulo de 1,65 x 1,32 da calibracao nunca foi o limite da",
            "medida: era o alcance da trena. Fora dele a homografia continua",
            "valendo — medido, o erro no pior canto da imagem e 4,7 cm.",
            "",
            "So a camera do alto entra aqui, porque so ela tem homografia. A",
            "frontal e a lateral tambem veem piso e nao sabem dizer em metros",
            "onde ele esta.",
        ]
    d["moveis"] = [{
        "id": "estante-aco", "nome": "Estante", "tipo": "estante",
        "x": round(ex, 3), "y": round(ey, 3),
        "largura": round(gab.largura, 3),
        "profundidade": round(gab.profundidade, 3),
        "altura": round(gab.altura, 3),
        "rumo_da_face": round(float(rumo), 4),
        "prateleiras": [{"id": i, "altura": round(float(a), 3)}
                        for i, a in gab.prateleiras],
        "estante": "estante-aco-teste",
        "_medido_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_por": "as 3 cameras; a altura de trena deu a escala",
        "_nota": [
            "POSICAO achada pelas cameras. DIMENSOES da trena.",
            "",
            "A escala do mundo saiu da altura da estante: 1,90 m medidos",
            "divididos pela altura dela na nuvem. Nenhuma homografia entrou",
            "nesta conta.",
            "",
            "    Quando um objeto de dimensao conhecida esta na cena, ele e",
            "    a regua.",
        ]}]

    est = Ambiente(x=ex, y=ey, rumo_da_face=float(rumo),
                   largura=gab.largura, profundidade=gab.profundidade,
                   altura=gab.altura, prateleiras=gab.prateleiras,
                   cameras=("alto", "frontal", "lateral"))
    d["zonas"] = _portas(est, (x0, x1, y0, y1))
    d.pop("_a_medir", None)
    caminho.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                       encoding="utf-8")


if __name__ == "__main__":
    main()
