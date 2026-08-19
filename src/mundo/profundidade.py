"""O ambiente a partir de UMA imagem, ancorado no chao que a trena mediu.

    da uma olhada nos estudos mais recentes e tente nos atualizar
                                                    — Eduardo, 19/08

POR QUE ISTO EXISTE, DEPOIS DE UM DIA INTEIRO DE DUSt3R

Reconstrucao multi-vista precisa que as cameras vejam AS MESMAS SUPERFICIES.
As tres deste arranjo quase nao veem: a do alto olha o piso, a frontal olha a
pessoa, a lateral olha o perfil da estante. O DUSt3R nao falhou por ma
configuracao — ele foi usado fora da hipotese dele, e devolveu um comodo de
2,1 m2 flutuando num sistema de coordenadas proprio.

    Um metodo aplicado fora da hipotese dele nao erra: ele responde outra
    pergunta, com a mesma cara de resposta.

A profundidade monocular metrica nao tem esse requisito. Uma imagem entra,
metros saem. E o estado da arte de 2025 (Depth Pro, Depth Anything V2 Metric,
UniDepth) faz isso com escala absoluta.

O QUE ESTE ARQUIVO ACRESCENTA, E QUE NENHUM DELES FAZ

A rede devolve profundidade em metros NO SISTEMA DELA — origem na camera,
eixos da camera, e uma escala que erra alguns por cento porque foi aprendida
de dados sinteticos. Para o gemeo isso nao serve: o boneco anda no sistema da
HOMOGRAFIA, medido com trena.

Aqui os dois se encontram, e o encontro nao precisa de nenhuma correspondencia
nova — porque a homografia JA DESCREVE A CAMERA:

    1. o plano metrico + a homografia   ->  a distancia focal
    2. a focal + a homografia           ->  R, t, e ONDE A CAMERA ESTA
    3. o chao conhecido                 ->  a correcao de escala da rede
    4. retroprojecao                    ->  a nuvem, ja em metros no mundo
                                            onde o gemeo anda

O passo 2 devolve de brinde uma verificacao que o projeto nunca teve: a
ALTURA DA CAMERA. Se a conta disser 2,5 m e a trena disser 2,5 m, tudo o que
veio antes esta certo. Se disser 4 m, algo esta errado antes de qualquer
nuvem ser desenhada.

    Um metodo que produz, de graca, um numero que da para conferir com a
    trena vale mais que um metodo mais preciso que nao produz nenhum.

E A PONTE DEIXA DE EXISTIR

Ela so foi necessaria porque o DUSt3R inventava um mundo. Aqui a nuvem nasce
no mundo da homografia — nao por transformacao, por construcao.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Quanto as duas estimativas independentes da focal podem discordar.
#
# Elas saem de duas restricoes diferentes sobre a mesma matriz (as colunas de
# rotacao serem ortogonais, e terem o mesmo comprimento). Se discordarem
# muito, as hipoteses — pixel quadrado, centro otico no meio da imagem — nao
# valem para esta lente, e a focal recuperada nao vale.
#
#     Uma medida sozinha nunca esta errada. Ela so passa a estar quando ha
#     uma segunda.
DISCORDANCIA_MAXIMA = 0.25

# A que distancia do chao um ponto ainda conta como chao, ao ajustar a escala.
FOLGA_DO_CHAO = 0.06


@dataclass
class CameraDoAlto:
    """A camera, deduzida do plano que a trena mediu."""
    K: np.ndarray                     # 3x3, intrinseca
    R: np.ndarray                     # 3x3, mundo -> camera
    t: np.ndarray                     # 3, mundo -> camera
    posicao: np.ndarray               # 3, onde a camera esta, em metros
    focal: float
    discordancia: float               # entre as duas estimativas da focal
    tamanho: tuple                    # (largura_px, altura_px)

    @property
    def altura_m(self):
        """A altura da camera sobre o chao. CONFIRA COM A TRENA."""
        return float(self.posicao[2])

    @property
    def confiavel(self):
        return (self.discordancia <= DISCORDANCIA_MAXIMA
                and 0.5 < self.altura_m < 8.0)

    def para_camera(self, pontos):
        p = np.asarray(pontos, dtype=float).reshape(-1, 3)
        return p @ self.R.T + self.t

    def para_o_mundo(self, pontos_camera):
        p = np.asarray(pontos_camera, dtype=float).reshape(-1, 3)
        return (p - self.t) @ self.R


def _b_de(g_a, g_b, cx, cy):
    """Os dois termos de `gaT B gb`, separados pelo que multiplica 1/f2.

    B = K^-T K^-1 com K = [[f,0,cx],[0,f,cy],[0,0,1]] se parte em dois:
    uma matriz inteira multiplicada por 1/f2, mais um 1 no canto. Separar
    permite resolver para 1/f2 em linha reta, sem otimizacao.
    """
    a1, a2, a3 = g_a
    b1, b2, b3 = g_b
    com_w = (a1 * b1 + a2 * b2
             - cx * (a1 * b3 + a3 * b1)
             - cy * (a2 * b3 + a3 * b2)
             + (cx * cx + cy * cy) * a3 * b3)
    sem_w = a3 * b3
    return com_w, sem_w


def intrinseca_da_homografia(homografia, largura_px, altura_px):
    """A distancia focal, deduzida do plano metrico. Devolve (K, discordancia).

    UMA HOMOGRAFIA DE UM PLANO METRICO JA CONTEM A FOCAL

    Se `G` leva metros do chao a pixels, entao `G = lambda * K [r1 r2 t]`,
    onde r1 e r2 sao as duas primeiras colunas da rotacao. E colunas de
    rotacao obedecem duas regras que nao dependem de nada:

        sao PERPENDICULARES         r1 . r2 = 0
        tem o MESMO COMPRIMENTO     |r1| = |r2|

    Cada regra vira uma equacao em 1/f2. Com pixel quadrado, sem
    cisalhamento e centro otico no meio da imagem, sobra UMA incognita e
    sobram DUAS equacoes — entao ha resposta fechada e ainda sobra uma
    conferencia. E o metodo do Zhang com uma imagem so.

        Duas equacoes para uma incognita nao e desperdicio: a segunda e a
        unica coisa que diz se a primeira valeu.

    As hipoteses sao razoaveis para uma webcam e NAO SAO PARA QUALQUER LENTE.
    Por isso a discordancia entre as duas estimativas volta junto: e ela que
    denuncia quando a suposicao nao vale.

    Devolve (None, discordancia) quando nao ha focal real — o que acontece,
    por exemplo, se o plano estiver quase de frente para a camera, porque ai
    a perspectiva nao carrega informacao de escala nenhuma.
    """
    h = np.asarray(homografia, dtype=float)
    if h.shape != (3, 3) or abs(np.linalg.det(h)) < 1e-12:
        return None, float("inf")

    # `homografia` leva PIXEL -> METRO; a decomposicao quer METRO -> PIXEL.
    g = np.linalg.inv(h)
    g1, g2 = g[:, 0], g[:, 1]
    cx, cy = largura_px / 2.0, altura_px / 2.0

    m12, n12 = _b_de(g1, g2, cx, cy)
    m11, n11 = _b_de(g1, g1, cx, cy)
    m22, n22 = _b_de(g2, g2, cx, cy)

    candidatos = []
    if abs(m12) > 1e-12:
        candidatos.append(-n12 / m12)                    # perpendiculares
    if abs(m11 - m22) > 1e-12:
        candidatos.append(-(n11 - n22) / (m11 - m22))    # mesmo comprimento

    validos = [w for w in candidatos if w > 1e-12]
    if not validos:
        return None, float("inf")

    if len(validos) == 2:
        f1, f2 = 1.0 / math.sqrt(validos[0]), 1.0 / math.sqrt(validos[1])
        discordancia = abs(f1 - f2) / max(f1, f2)
        focal = math.sqrt(f1 * f2)          # media geometrica: e uma escala
    else:
        focal = 1.0 / math.sqrt(validos[0])
        # Uma estimativa so nao e uma conferencia, e nao pode se passar por
        # uma. Sai como discordancia infinita, que reprova em `confiavel`.
        discordancia = float("inf")

    if not np.isfinite(focal) or focal <= 0:
        return None, float("inf")

    K = np.array([[focal, 0.0, cx],
                  [0.0, focal, cy],
                  [0.0, 0.0, 1.0]])
    return K, float(discordancia)


def focal_pela_altura(homografia, largura_px, altura_px, altura_medida_m,
                      minimo=150.0, maximo=4000.0, passos=60):
    """A focal que faz a camera cair na altura que a TRENA mediu.

    O MELHOR RESULTADO DE 19/08, E ELE VEIO DE UMA FITA METRICA

    Deduzir a focal de uma homografia so exige supor pixel quadrado e centro
    optico no meio da imagem. Na C920, com o barril que ela tem, isso deu:

        focal deduzida     758 px  ->  camera a 2,73 m
        trena                          camera a 2,23 m

    Cinquenta centimetros, 22% de erro, e ele contaminaria toda posicao
    calculada adiante.

    Mas a relacao vale nos dois sentidos. Se a focal errada produz a altura
    errada, entao a altura CERTA determina a focal certa — e altura de camera
    e uma das poucas coisas deste projeto que se mede em um minuto, com uma
    fita, sem software nenhum.

        Uma grandeza dificil de medir se resolve por outra facil de medir,
        quando as duas estao presas pela mesma geometria.

    E A CURVA NAO E MONOTONA, o que eu supus e estava errado. Medido:

        f=200 -> 0,93 m    f=800  -> 2,81 m    f=1500 -> 3,29 m  (pico)
        f=500 -> 2,08 m    f=1200 -> 3,23 m    f=3000 -> 2,66 m

    Ela sobe, satura e DESCE. Entao cada altura tem DUAS focais possiveis, e
    uma busca binaria ingenua sobre a faixa inteira pode cair em qualquer uma
    — ou em nenhuma, se os extremos ficarem do mesmo lado.

    A fisica desempata: a raiz grande, para a altura de 2,23 m, fica acima de
    1500 px, ou seja menos de 25 graus de campo. Webcam nenhuma e teleobjetiva.
    Entao a busca acontece no ramo que SOBE, do minimo ate o pico.

        Duas solucoes matematicas nao sao ambiguidade quando uma delas
        descreve um aparelho que nao existe.

    E O RESULTADO SE CONFERE SOZINHO. Com a altura de 2,23 m a focal deu
    551 px, ou seja 72 graus de diagonal — e a C920 tem cerca de 70 em 4:3.
    A trena, a geometria e a folha de dados do fabricante concordando sobre
    um numero que nenhuma das tres usou para chegar la.

    NAO SUBSTITUI O TABULEIRO. Isto devolve a focal; `calibracao/intrinseca.py`
    devolve tambem a distorcao radial, que e o que torce as bordas. Devolve o
    que da para ter agora, e nao o melhor que existe.
    """
    if not (altura_medida_m and altura_medida_m > 0):
        return None

    def altura_com(f):
        K = np.array([[f, 0.0, largura_px / 2.0],
                      [0.0, f, altura_px / 2.0],
                      [0.0, 0.0, 1.0]])
        cam = camera_da_homografia(homografia, largura_px, altura_px, K=K)
        return cam.altura_m if cam is not None else None

    # Onde fica o pico. Varredura grossa em escala logaritmica: a focal e uma
    # escala, e amostrar linearmente gastaria metade dos pontos na parte plana.
    escala_log = np.geomspace(minimo, maximo, 60)
    alturas = [altura_com(float(f)) for f in escala_log]
    validos = [(f, a) for f, a in zip(escala_log, alturas) if a is not None]
    if not validos:
        return None
    f_pico, a_pico = max(validos, key=lambda par: par[1])

    a_min = altura_com(minimo)
    if a_min is None or not (a_min <= altura_medida_m <= a_pico):
        # A altura pedida nao e alcancavel no ramo que sobe. Nao ha resposta, e
        # inventar a mais proxima esconderia que a homografia ou a medida estao
        # erradas.
        return None

    lo, hi = minimo, float(f_pico)
    for _ in range(passos):
        meio = (lo + hi) / 2.0
        a = altura_com(meio)
        if a is None:
            return None
        if a < altura_medida_m:
            lo = meio
        else:
            hi = meio
    focal = (lo + hi) / 2.0

    return np.array([[focal, 0.0, largura_px / 2.0],
                     [0.0, focal, altura_px / 2.0],
                     [0.0, 0.0, 1.0]])


def camera_da_homografia(homografia, largura_px, altura_px, K=None):
    """A camera inteira: intrinseca, pose e ONDE ELA ESTA. Ou None.

    Depois da focal, o resto e algebra direta:

        r1 = lambda K^-1 g1      r2 = lambda K^-1 g2      t = lambda K^-1 g3
        r3 = r1 x r2

    com lambda = 1/|K^-1 g1|, e o sinal escolhido para a cena ficar NA FRENTE
    da camera. Sem essa escolha a decomposicao aceita, com o mesmo erro
    algebrico, uma camera enterrada no chao olhando para cima.

        Uma equacao com duas solucoes nao esta resolvida ate alguem dizer
        qual das duas e o mundo.

    A rotacao sai quase ortonormal e e forcada a ser exatamente ortonormal
    pela SVD — o pouco que falta e o erro de medida da propria homografia, e
    deixa-lo entrar torceria todas as distancias adiante.
    """
    discordancia = 0.0
    if K is None:
        # MEDIDA VENCE DEDUZIDA, SEMPRE.
        #
        # `calibracao/intrinseca.py` mede K com tabuleiro e ainda devolve a
        # distorcao radial. Deduzir a focal de uma homografia so exige supor
        # pixel quadrado e centro optico no meio — hipoteses que a C920, com
        # o barril dela, nao honra: na homografia real de 19/08 as duas
        # estimativas discordaram 15,8% e o campo de visao saiu 46 graus
        # contra os ~60 que a lente tem.
        K = intrinseca_medida(largura_px=largura_px, altura_px=altura_px)
        if K is None:
            K, discordancia = intrinseca_da_homografia(homografia, largura_px,
                                                       altura_px)
        if K is None:
            return None

    h = np.asarray(homografia, dtype=float)

    # A MAO DO SISTEMA DE COORDENADAS, E ELA NAO E ESCOLHA NOSSA.
    #
    # O (0,0) e os eixos do chao vieram da ordem em que o Eduardo clicou os
    # quatro cantos ao calibrar. Nada obriga essa ordem a produzir um sistema
    # DESTRO com z para cima — e o dele nao produz.
    #
    # MEDIDO EM 19/08, na homografia real: a decomposicao devolveu a camera a
    # 2,73 m ABAIXO do piso. O modulo estava certo (o teto dele tem 2,7 m);
    # so o sinal e que vinha do sistema ser canhoto.
    #
    #     Um resultado com o modulo certo e o sinal errado nao e um erro de
    #     conta: e uma convencao que ninguem declarou.
    #
    # A saida e trocar o sentido de y, resolver no sistema destro, e desfazer
    # a troca na resposta. Duas linhas, e o resto do arquivo nao precisa saber.
    for espelho in (1.0, -1.0):
        virar = np.diag([1.0, espelho, 1.0])
        cam = _decompor(virar @ h, K, largura_px, altura_px, discordancia)
        if cam is None:
            continue
        if espelho < 0:
            cam = _desvirar_y(cam)
        # A UNICA PROVA FISICA QUE EXISTE AQUI: a camera esta acima do chao.
        if cam.altura_m > 0.05:
            return cam
    return None


def _decompor(homografia, K, largura_px, altura_px, discordancia):
    """`G = lambda K [r1 r2 t]` resolvido, com a cena na frente da camera."""
    g = np.linalg.inv(np.asarray(homografia, dtype=float))
    Ki = np.linalg.inv(K)
    c1, c2, c3 = Ki @ g[:, 0], Ki @ g[:, 1], Ki @ g[:, 2]

    norma = np.linalg.norm(c1)
    if norma < 1e-12:
        return None
    lam = 1.0 / norma

    r1, r2, t = lam * c1, lam * c2, lam * c3
    # A cena tem que estar na FRENTE: o chao visto pela camera tem z > 0.
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)

    R = np.column_stack([r1, r2, r3])
    u, _s, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:              # reflexao nao e rotacao
        u[:, 2] *= -1
        R = u @ vt

    posicao = -R.T @ t
    return CameraDoAlto(K=K, R=R, t=t, posicao=posicao,
                        focal=float(K[0, 0]), discordancia=discordancia,
                        tamanho=(largura_px, altura_px))


def _desvirar_y(cam):
    """Desfaz a troca de sentido de y, na pose e na posicao."""
    D = np.diag([1.0, -1.0, 1.0])
    R = cam.R @ D
    return CameraDoAlto(K=cam.K, R=R, t=cam.t, posicao=D @ cam.posicao,
                        focal=cam.focal, discordancia=cam.discordancia,
                        tamanho=cam.tamanho)


def intrinseca_medida(caminho=None, largura_px=None, altura_px=None):
    """A K medida com tabuleiro, se alguem ja mediu. Senao None.

    `calibracao/intrinseca.py` existe neste projeto desde o comeco e nunca foi
    rodado — o `intrinseca.json` nao existe. O docstring dele ja descrevia
    esta mesma decomposicao, e ele mede o que aqui e SUPOSTO:

        pixel quadrado, sem cisalhamento, centro optico no meio da imagem

    Para a C920, com o barril que ela tem, essas hipoteses custam caro: na
    homografia real de 19/08 as duas estimativas da focal discordaram 15,8% e
    o campo de visao saiu 46 graus, contra os ~60 que a lente tem em 4:3.

        Deduzir sob hipotese o que se pode medir em dez minutos e escolher
        carregar a hipotese para sempre.
    """
    import json
    from pathlib import Path

    if caminho:
        candidatos = [Path(caminho)]
    else:
        # O NOME DO ARQUIVO SEGUE O PAPEL, E NAO O INDICE DO WINDOWS.
        #
        # `intrinseca.py` salvava em `cam0.json` — um nome que depende de qual
        # USB o Windows enumerou primeiro. Com duas webcams ligadas, o mesmo
        # arquivo descreveria lentes diferentes em dias diferentes, sem nada
        # mudar de aparencia.
        calib = Path(__file__).resolve().parent.parent.parent / "calibracao"
        candidatos = [calib / "intrinseca-alto.json", calib / "intrinseca.json"]
    p = next((c for c in candidatos if c.exists()), None)
    if p is None:
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        K = np.array(d["K"], dtype=float)
        if K.shape != (3, 3):
            return None
        # A K FOI MEDIDA NUMA RESOLUCAO E PODE SER USADA NOUTRA.
        #
        # Focal e centro optico sao em PIXELS: mudar a resolucao muda os dois,
        # proporcionalmente. Usar a K de 1280x720 numa imagem de 640x480 sem
        # reescalar poria o centro optico fora do quadro — e o erro pareceria
        # uma lente estranha, nao uma conta esquecida.
        medida = d.get("resolucao")
        if medida and largura_px and altura_px:
            sx = float(largura_px) / float(medida[0])
            sy = float(altura_px) / float(medida[1])
            K = K * np.array([[sx, sx, sx], [sy, sy, sy], [1.0, 1.0, 1.0]])
        return K
    except Exception:
        return None


# ------------------------------------------------------------ a nuvem
@dataclass
class NuvemMonocular:
    """A nuvem em metros, no mundo da homografia, e o que se sabe sobre ela."""
    pontos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    camera: CameraDoAlto | None = None
    escala: float = 1.0               # correcao aplicada a saida da rede
    residuo_chao_m: float = 0.0       # quao plano o chao ficou
    fracao_de_chao: float = 0.0       # quanto da imagem virou chao

    @property
    def pronta(self):
        return (len(self.pontos) > 100 and self.camera is not None
                and self.camera.confiavel and self.residuo_chao_m < 0.10)


def profundidade_do_chao(camera, largura_px, altura_px, passo=1):
    """Que profundidade CADA pixel teria se ali fosse chao.

    Esta e a previsao contra a qual a rede vai ser medida. Ela nao vem de
    modelo nenhum: vem da geometria da camera, que veio da trena.

    A profundidade e a coordenada z na camera — a convencao dos modelos
    metricos, que devolvem distancia ao longo do eixo optico e nao distancia
    ate a lente.
    """
    us = np.arange(0, largura_px, passo, dtype=float)
    vs = np.arange(0, altura_px, passo, dtype=float)
    uu, vv = np.meshgrid(us, vs)
    pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1)

    # raio na camera, e onde ele encosta no plano z=0 do mundo
    raios = pixels @ np.linalg.inv(camera.K).T
    # no mundo: C + s * (R^T raio). Queremos a componente z zero.
    dirs = raios @ camera.R
    C = camera.posicao
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -C[2] / dirs[:, 2]
    z_camera = np.where(s > 0, s, np.nan)
    return z_camera.reshape(len(vs), len(us)), (us, vs)


def _escala_pelo_chao(medida, prevista, passos=3):
    """A escala que faz a rede concordar com o chao conhecido. Robusta.

    A rede erra a escala em alguns por cento porque aprendeu de dados
    sinteticos. O chao da camera do alto e conhecido em metros, e ocupa a
    maior parte da imagem — entao a razao `medida / prevista` tem uma moda
    clara no valor certo, contaminada pelos pixels que NAO sao chao.

    Mediana, e depois duas rodadas descartando quem se afasta dela. Media
    aqui seria decidida pela estante, que e justamente o que nao e chao.

        Um estimador que a minoria consegue mover nao esta medindo a
        maioria.

    Devolve (escala, fracao_de_pixels_que_eram_chao).
    """
    m = np.asarray(medida, dtype=float).ravel()
    p = np.asarray(prevista, dtype=float).ravel()
    bom = np.isfinite(m) & np.isfinite(p) & (p > 1e-6) & (m > 1e-6)
    if bom.sum() < 50:
        return None, 0.0

    razao = m[bom] / p[bom]
    escala = float(np.median(razao))
    dentro = np.ones(razao.shape, dtype=bool)
    for _ in range(passos):
        dentro = np.abs(razao - escala) <= 0.05 * escala
        if dentro.sum() < 30:
            break
        escala = float(np.median(razao[dentro]))
    return escala, float(dentro.sum()) / len(razao)


def nuvem_do_alto(profundidade, homografia, tamanho_original=None,
                  camera=None, passo=2, altura_maxima=3.0):
    """Mapa de profundidade -> nuvem em metros, no mundo do gemeo.

    `profundidade`   (H, W) em metros, saida de um modelo metrico monocular
    `homografia`     pixel -> metro no chao, a mesma do `rodar.py`
    `tamanho_original`  resolucao para a qual a homografia foi calibrada

    A ORDEM, E CADA PASSO E CONFERIVEL

        1. a camera sai da homografia            -> confira a altura na trena
        2. o chao previsto sai da camera         -> nao vem de modelo nenhum
        3. a escala sai da razao com o previsto   -> robusta, so o chao decide
        4. a retroprojecao usa a escala corrigida -> metros de verdade

    O residuo do chao (quao plano ele ficou depois de tudo) e a nota do
    conjunto. Se ele for grande, alguma coisa antes esta errada, e o desenho
    nao deve ser confiado ainda que fique bonito.
    """
    d = np.asarray(profundidade, dtype=float)
    if d.ndim != 2 or d.size == 0:
        return None
    alt_d, larg_d = d.shape
    larg_o, alt_o = tamanho_original if tamanho_original else (larg_d, alt_d)

    if camera is None:
        camera = camera_da_homografia(homografia, larg_o, alt_o)
    if camera is None:
        return None

    # A homografia foi calibrada numa resolucao; o mapa de profundidade pode
    # vir noutra. Trabalhar em pixels da CALIBRACAO e reamostrar a
    # profundidade evita reescalar a matriz, que ja foi fonte de erro aqui.
    prev, (us, vs) = profundidade_do_chao(camera, larg_o, alt_o, passo)
    cols = np.clip((us * larg_d / larg_o).astype(int), 0, larg_d - 1)
    linhas = np.clip((vs * alt_d / alt_o).astype(int), 0, alt_d - 1)
    med = d[np.ix_(linhas, cols)]

    escala, fracao = _escala_pelo_chao(med, prev)
    if escala is None or not (0.05 < escala < 20.0):
        return None

    z = med / escala                       # profundidade corrigida, em metros
    uu, vv = np.meshgrid(us, vs)
    pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1)
    raios = pixels @ np.linalg.inv(camera.K).T
    pontos_cam = raios * z.ravel()[:, None]
    mundo = camera.para_o_mundo(pontos_cam)

    bom = np.isfinite(mundo).all(axis=1) & (z.ravel() > 0)
    # NAO SE RECORTA A VISTA DA CAMERA — a licao de 18/08, que custou tres
    # correcoes. So sai o que e impossivel: ponto abaixo do chao alem do
    # ruido, ou mais alto que o pe direito.
    bom &= (mundo[:, 2] > -0.25) & (mundo[:, 2] < altura_maxima)
    pontos = mundo[bom]
    if len(pontos) < 100:
        return None

    perto_do_chao = pontos[np.abs(pontos[:, 2]) < FOLGA_DO_CHAO]
    residuo = (float(np.sqrt(np.mean(perto_do_chao[:, 2] ** 2)))
               if len(perto_do_chao) > 30 else float("inf"))

    return NuvemMonocular(pontos=pontos, camera=camera, escala=escala,
                          residuo_chao_m=residuo, fracao_de_chao=fracao)


def ambiente_do_mono(nuvem, gabarito):
    """A `NuvemMonocular` no formato que o resto do programa ja consome.

    TRES PASSOS DO CAMINHO ANTIGO DESAPARECEM AQUI, E ISSO E O PONTO

    `mapeamento.montar` precisa, na ordem: achar o chao por RANSAC, girar a
    cena para deitar esse chao, e dividir pela altura da estante para achar o
    metro. Nenhum dos tres tem o que fazer com esta nuvem:

        o chao ja e z = 0        porque veio da homografia
        a cena ja esta de pe     pelo mesmo motivo
        a unidade ja e metro     a escala foi corrigida contra o chao

    E A PONTE TAMBEM NAO E MAIS NECESSARIA. Ela existia porque o DUSt3R
    entregava um mundo proprio que precisava ser casado com o do gemeo. Aqui
    nao ha dois mundos para casar: a nuvem NASCE no da homografia.

        Uma etapa que so existe para reconciliar dois sistemas de
        coordenadas deixa de existir quando so ha um.

    Sobra achar a estante — e a altura dela vira a CONFERENCIA em vez da
    regua. Se a nuvem disser 1,90 m, a trena e a rede concordam sobre uma
    coisa que nenhuma das duas usou para chegar la.
    """
    from src.mundo.mapeamento import Ambiente3D, achar_estante

    if nuvem is None or not len(nuvem.pontos):
        return None

    achada = achar_estante(nuvem.pontos, gabarito.largura,
                           gabarito.profundidade)
    if achada is None:
        return None
    cx, cy, rumo, altura = achada

    return Ambiente3D(
        nuvem=nuvem.pontos,
        escala=1.0,                            # ja veio em metro
        estante=(float(cx), float(cy), float(rumo)),
        altura_da_cena=float(nuvem.pontos[:, 2].max()),
        # A OUTRA REGUA. Aqui ela nao mede: confere. Um for a altura da
        # estante na nuvem dividida pela trena, e as duas concordam.
        escala_da_estante=float(altura / gabarito.altura)
        if gabarito.altura > 0 else 0.0,
        residuo_m=float(nuvem.residuo_chao_m),
        # Cada pixel de chao que decidiu a escala e uma correspondencia entre
        # o que a rede viu e o que a trena mediu. Nao ha ponte separada
        # porque estas SAO a ponte, e elas sao milhares.
        ancoras=int(nuvem.fracao_de_chao * len(nuvem.pontos)),
    )
