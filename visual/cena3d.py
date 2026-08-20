"""
Renderizador 3D — a cena do gemeo digital, estilo AiFi.

CONTINUACAO DIRETA DO BLOCO 1.

    homografia  = projecao entre DOIS PLANOS   (3x3)
    isto aqui   = projecao do ESPACO no plano  (4x4 + divisao)

E o mesmo princípio, um grau acima: coordenadas homogeneas, matriz, e a
divisao final por w que produz a perspectiva.

Nao depende de camera, YOLO ou pose. Recebe esqueletos em metros e desenha.
Roda sozinho com dados sinteticos:

    python visual/cena3d.py

SISTEMA DE COORDENADAS DO MUNDO
    x  para a direita, em metros
    y  para a frente,  em metros   (o plano do chao e z=0)
    z  para CIMA,      em metros

Atencao: o chao aqui e o plano z=0, e a homografia devolve (x, y). Entao o
ponto do pe entra como (x, y, 0) — a ligacao entre os dois mundos e direta.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------- esqueleto
# Indices no padrao COCO-17, o mesmo do yolo11n-pose.
NOMES = ["nariz", "olho_e", "olho_d", "orelha_e", "orelha_d",
         "ombro_e", "ombro_d", "cotovelo_e", "cotovelo_d",
         "pulso_e", "pulso_d", "quadril_e", "quadril_d",
         "joelho_e", "joelho_d", "tornozelo_e", "tornozelo_d"]

OSSOS = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # bracos
    (5, 6), (5, 11), (6, 12), (11, 12),     # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),  # pernas
    (0, 5), (0, 6),                          # pescoco
]

# TEMA CLARO, DA REFERENCIA DA AiFi.
#
# O fundo escuro fazia o boneco parecer instrumentacao — tela de radar, coisa
# de sala de controle. O claro com grade branca faz o mesmo dado parecer
# ESPACO: um piso onde alguem esta, nao um grafico sobre fundo preto.
#
#     Fundo escuro pede que o observador leia numeros. Fundo claro pede que
#     ele reconheca uma cena. Para uma banca, a segunda leitura e a que vale.
#
# Cores em BGR, que e a ordem do OpenCV.
# Na referencia TODOS sao roxos — a cor identifica o SISTEMA, nao a pessoa.
# Quem distingue quem e a etiqueta. Uma paleta de seis cores gritantes faria
# duas pessoas parecerem duas coisas diferentes, e sao a mesma coisa.
CORES = [
    (122, 42, 108),          # roxo da referencia
    (138, 58, 122), (108, 36, 96), (150, 72, 134),
    (96, 30, 86), (128, 50, 116),
]

FUNDO = (247, 246, 246)
CHAO = (236, 235, 236)
GRADE = (255, 255, 255)
TEXTO = (90, 88, 90)


@dataclass
class Esqueleto:
    """Uma pessoa em 3D.

    juntas: array (17, 3) em METROS, no mundo (z=0 e o chao).
    O jeito de montar: pegue a pose relativa ao quadril (o que os modelos
    monoculares entregam bem) e some a posicao do chao que a homografia deu.
    """
    id: int
    juntas: np.ndarray
    visivel: np.ndarray | None = None      # (17,) bool
    prevendo: bool = False
    historico: list = field(default_factory=list)   # rastro no chao, [(x,y)]
    # PARA ONDE ELE ESTA VIRADO, E SE ESTA INDO.
    #
    # Nao da para ler isso das juntas. As 17 juntas de alguem de costas e as de
    # alguem de frente sao quase o mesmo desenho, e as pernas pararam de andar
    # em 12/08 — entao a silhueta nao diz mais nem se a pessoa se move. O que o
    # sistema SABE (`rumo_corpo`, `locomocao`) tem que chegar ate aqui por
    # escrito, senao nao chega.
    rumo: float | None = None              # radianos, None = nao sei
    andando: bool = False


# ---------------------------------------------------------------- camera
class CameraVirtual:
    """Camera pinhole que se move em orbita ao redor de um alvo.

    A matriz de projecao faz o mesmo que a homografia fazia, com uma dimensao
    a mais: leva coordenadas homogeneas do mundo para a imagem, e a divisao
    final por w e o que cria a perspectiva.
    """

    def __init__(self, largura, altura, fov_graus=50.0):
        self.w, self.h = largura, altura
        f = 0.5 * largura / np.tan(np.deg2rad(fov_graus) / 2)
        self.K = np.array([[f, 0, largura / 2],
                           [0, f, altura / 2],
                           [0, 0, 1.0]])
        self.alvo = np.array([0.5, 0.5, 0.9])
        self.dist = 5.0
        self.azimute = np.deg2rad(-60)
        self.elevacao = np.deg2rad(28)

    @property
    def posicao(self):
        d, a, e = self.dist, self.azimute, self.elevacao
        return self.alvo + np.array([d * np.cos(e) * np.cos(a),
                                     d * np.cos(e) * np.sin(a),
                                     d * np.sin(e)])

    def _extrinseca(self):
        olho = self.posicao
        frente = self.alvo - olho
        frente /= np.linalg.norm(frente)
        cima_mundo = np.array([0, 0, 1.0])
        direita = np.cross(frente, cima_mundo)
        direita /= np.linalg.norm(direita)
        cima = np.cross(direita, frente)
        # linhas: x da imagem = direita, y da imagem = -cima, z = frente
        R = np.stack([direita, -cima, frente])
        t = -R @ olho
        return R, t

    def projetar(self, pontos):
        """(N,3) em metros -> (N,2) em pixels, e (N,) profundidade."""
        R, t = self._extrinseca()
        cam = (R @ np.asarray(pontos, dtype=float).T).T + t
        z = cam[:, 2]
        seguro = np.maximum(z, 1e-4)
        px = (self.K @ (cam / seguro[:, None]).T).T
        return px[:, :2], z


# ---------------------------------------------------------------- cena
def _clarear(cor, quanto):
    """Mistura a cor com branco."""
    return tuple(int(c + (255 - c) * quanto) for c in cor)


def _escurecer(cor, quanto):
    return tuple(int(c * (1.0 - quanto)) for c in cor)


def _esfera(img, centro, raio, cor, luz=(-0.45, -0.45)):
    """Circulo com SOMBREAMENTO, nao circulo chapado.

    O que separa "bolinha" de "esfera" e o gradiente: claro do lado da luz,
    escuro do lado oposto. Sem ele o boneco parece recortado em papel.

    Feito com circulos concentricos deslocados na direcao da luz — barato, e
    visualmente indistinguivel de um gradiente radial de verdade neste
    tamanho. Um shader daria o mesmo resultado por um custo que este projeto
    nao precisa pagar.
    """
    x, y = int(centro[0]), int(centro[1])
    if raio < 2:
        cv2.circle(img, (x, y), max(1, raio), cor, -1, cv2.LINE_AA)
        return

    # Base escura: e ela que aparece na borda oposta a luz.
    cv2.circle(img, (x, y), raio, _escurecer(cor, 0.35), -1, cv2.LINE_AA)

    camadas = max(3, min(9, raio // 2))
    for k in range(camadas):
        f = (k + 1) / camadas                     # 0 -> 1 do centro a borda
        r = int(raio * (1.0 - f * 0.72))
        if r < 1:
            break
        cx = int(x + luz[0] * (raio - r) * 0.85)
        cy = int(y + luz[1] * (raio - r) * 0.85)
        cv2.circle(img, (cx, cy), r, _clarear(cor, 0.10 + f * 0.42),
                   -1, cv2.LINE_AA)


def _osso(img, a, b, ra, rb, cor):
    """Osso como TRONCO DE CONE, nao como linha de espessura constante.

    `ra` e `rb` sao as MEIAS-LARGURAS nas duas pontas, ja em pixels. Passar a
    largura pronta em vez do raio da junta foi o que permitiu separar as duas
    decisoes: o quanto uma junta pesa, e o quanto um osso e grosso.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    comprimento = float(np.hypot(*d))
    if comprimento < 1e-3:
        return

    # Normal ao osso: e nela que a largura e aplicada.
    n = np.array([-d[1], d[0]]) / comprimento
    # OSSO FINO, ESFERA GRANDE — e a proporcao que faz a silhueta.
    #
    # A primeira versao usou 0,80 do raio e o boneco ficou inflado: cabeca e
    # ombros viraram uma massa so, sem pescoco, e o braco levantado colou no
    # tronco. Volume nao e gordura.
    #
    #     Na referencia da AiFi as juntas sao contas e os ossos sao o cordao.
    #     E o CONTRASTE entre os dois que da forma; igualar os dois apaga a
    #     silhueta que se queria ganhar.
    wa, wb = ra, rb

    quad = np.array([a + n * wa, b + n * wb, b - n * wb, a - n * wa])
    cv2.fillConvexPoly(img, quad.astype(np.int32), cor, cv2.LINE_AA)


class Cena3D:
    def __init__(self, largura=960, altura=620, chao=(-1.5, 3.5, -1.5, 3.5),
                 calor_hz=4.0, contorno=None):
        self.cam = CameraVirtual(largura, altura)
        self.chao = chao

        # O PISO E UM QUADRILATERO TORTO, NAO UM RETANGULO.
        #
        # A pegada de uma camera no chao e a imagem de um retangulo por uma
        # perspectiva — e isso nunca volta a ser retangulo. A caixa que a
        # contem mede 16,4 m2 onde a camera mede 8,4: desenhar a caixa poria
        # metade do comodo onde a camera nunca olhou.
        #
        #     Chao desenhado onde nada foi medido e a mesma mentira do
        #     recorte, ao contrario: um afirma de menos, o outro de mais.
        #
        # `None` mantem o retangulo de antes, para quem ainda nao tem contorno.
        self.contorno = (np.asarray(contorno, dtype=float).reshape(-1, 2)
                         if contorno is not None and len(contorno) >= 3
                         else None)

        # A CAMERA MIRA O CENTRO DO PISO DECLARADO, NAO UM PONTO FIXO.
        #
        # `CameraVirtual` nasceu com alvo em (0.5, 0.5), que era o centro da
        # bancada de teste — e funcionou enquanto o chao foi aquele. Com
        # `loja/quarto.json` (0..2 em x e y) o alvo passou a cair no canto e
        # metade da tela virou fundo vazio.
        #
        #     Constante que so vale para um caso deixa de ser constante no
        #     dia em que aparece o segundo caso.
        #
        # A distancia ja nao e uma regra de bolso: e resolvida em
        # `_distancia_que_enquadra`, que precisa do alvo ja posto.
        x0, x1, y0, y1 = chao
        self.cam.alvo = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.95])
        self.cam.dist = self._distancia_que_enquadra()
        # (x, y, largura, profundidade, altura, rotulo, rumo, alturas)
        self.moveis = []

        # ---- caches ----
        # MEDIDO em 08/08: o desenho custava 87 ms por quadro, MAIS que o YOLO.
        #
        # Culpado principal: cada face de cada movel fazia `img.copy()` da
        # imagem inteira para poder misturar com transparencia. Tres moveis
        # de cinco faces = 15 copias de 960x620 por quadro.
        #
        # Mas chao, grade e moveis so mudam quando a CAMERA VIRTUAL se move.
        # Entao desenhamos uma vez e guardamos. Por quadro sobra copiar a base
        # e desenhar quem se mexe.
        self._base = None          # geometria + calor, composto
        self._geometria = None     # so chao, grade e moveis
        self._chave_base = None

        # O mapa de calor tambem era caro: desfoque gaussiano + mapa de cores +
        # duas reprojecoes de 960x620, tudo a cada quadro. Ele muda devagar —
        # atualizar 4x por segundo e visualmente identico.
        self._calor_img = None
        self._calor_alfa = None
        self._calor_t = 0.0
        self.calor_hz = calor_hz

    def _distancia_que_enquadra(self, altura_util=1.85, margem=0.05,
                                minimo=3.0, maximo=40.0):
        """A distancia em que TUDO cabe na tela, e nem um palmo mais.

        A REGRA DE BOLSO QUEBROU NA SEGUNDA SALA, COMO ELA MESMA AVISOU

        Ate 19/08 aqui estava `dist = clip(largura * 2, 5.5, 18)`, com um
        comentario dizendo:

            Constante que so vale para um caso deixa de ser constante no dia
            em que aparece o segundo caso.

        O segundo caso chegou: o piso passou de 1,5 m para 4,1 m de lado. A
        regra devolveu 8,1 m — longe demais, e o boneco de 1,75 m virou um
        risco. Nao havia como acertar os dois com o mesmo numero, porque nao
        e um numero: e uma pergunta de geometria com resposta exata.

            Um valor afinado a mao acerta o caso em que foi afinado. So a
            conta acerta o proximo.

        COMO SE RESOLVE

        O que precisa caber e o piso — nos dois niveis. No chao, porque e o
        comodo; a 1,85 m, porque e onde fica a cabeca de quem estiver no
        canto mais distante, e cortar a cabeca de alguem no canto e o defeito
        que a regra antiga ja tinha do outro lado.

        A extensao projetada encolhe quando a camera se afasta, sempre. Entao
        basta a menor distancia que faz tudo caber — busca binaria, vinte e
        cinco passos, resposta ao milimetro. Roda uma vez, na montagem.
        """
        pes = self.pes_do_chao()
        pontos = np.array([[x, y, z] for x, y in pes for z in (0.0, altura_util)],
                          dtype=float)

        def cabe(d):
            antes, self.cam.dist = self.cam.dist, float(d)
            try:
                p, z = self.cam.projetar(pontos)
            finally:
                self.cam.dist = antes
            if (z <= 0.05).any():
                return False
            mx, my = self.cam.w * margem, self.cam.h * margem
            return bool((p[:, 0] >= mx).all() and (p[:, 0] <= self.cam.w - mx).all()
                        and (p[:, 1] >= my).all()
                        and (p[:, 1] <= self.cam.h - my).all())

        if cabe(minimo):
            return minimo
        if not cabe(maximo):
            return maximo
        baixo, alto = minimo, maximo
        for _ in range(25):
            meio = (baixo + alto) / 2.0
            if cabe(meio):
                alto = meio
            else:
                baixo = meio
        return float(alto)

    def _chave_camera(self):
        c = self.cam
        # Os moveis inteiros entram na chave, e nao so QUANTOS sao: mover a
        # estante sem mudar a contagem deixaria o desenho velho em cache, e o
        # movel apareceria no lugar antigo ate a camera virtual girar.
        return (round(c.azimute, 4), round(c.elevacao, 4), round(c.dist, 3),
                tuple(np.round(c.alvo, 3)), tuple(self.moveis))

    def invalidar(self):
        self._base = None

    def add_movel(self, x, y, larg, prof, alt, rotulo="", rumo=0.0,
                  prateleiras=()):
        """x, y sao o CENTRO do movel. `rumo` em radianos, para onde a face olha.

        A convencao de centro e a de `estado.planta.Movel` e a de
        `src.mundo.ambiente.Ambiente` — as tres precisam concordar, e ate
        14/08 esta aqui discordava calada. Ver a nota em `Movel`.
        """
        self.moveis.append((x, y, larg, prof, alt, rotulo, float(rumo),
                            tuple(float(a) for _, a in prateleiras)))

    # Meia-volta seria olhar a face de frente e achatada. Trinta graus dao a
    # vista de 3/4 que mostra largura E profundidade no mesmo quadro.
    TRES_QUARTOS = np.deg2rad(30.0)

    def olhar_pela_face(self, offset=None):
        """Poe a camera virtual do lado do CLIENTE, nao do lado da parede.

        MEDIDO em 20/08: com azimute -60 fixo, a camera nascia em
        (+2,71, -3,32) e a face da estante apontava para (-0,52, +0,85). O
        produto escalar dava -0,99 — dead behind. A pessoa, corretamente
        posta na frente da estante pelo motor, saia atras dela na tela.

            A estante aparecia na frente do gemeo digital e na verdade e ao
            contrario, eu estou na frente e ela esta atras.   — Eduardo, 20/08

        Nao era erro de geometria nem de ordem de desenho: o modelo estava
        certo e o ponto de vista, errado. Ninguem monta uma loja para ser
        vista pelo fundo da gondola.

            Um angulo de camera escolhido antes de existir um movel e um
            palpite; depois que o movel tem face, e uma conta.

        O `rumo_da_face` ja diz para onde o movel olha. A camera passa a
        nascer desse lado. As setas continuam mandando — isto e o inicio,
        nao uma trava.

        Devolve True se mexeu.
        """
        if not self.moveis:
            return False
        # O movel PRINCIPAL e o de maior face (largura x altura): e dele que
        # se pega produto, e e ele que a camera precisa mostrar de frente.
        _x, _y, _larg, _prof, _alt, _rot, rumo, _alturas = max(
            self.moveis, key=lambda m: m[2] * m[4])
        # a face olha para (-sen rumo, cos rumo); a camera fica desse lado
        alvo = self.TRES_QUARTOS if offset is None else float(offset)
        self.cam.azimute = float(np.arctan2(np.cos(rumo), -np.sin(rumo))
                                 + alvo)
        self.cam.dist = self._distancia_que_enquadra()
        self.invalidar()
        return True

    @staticmethod
    def pes_do_movel(m):
        """Os quatro cantos do movel no chao, em metros.

        Separado do desenho de proposito: e a unica parte de `_movel` que pode
        estar certa ou errada, e desenhar numa imagem para conferir um numero
        seria medir com a regua errada.
        """
        x, y, w, d = m[0], m[1], m[2], m[3]
        rumo = m[6]
        # cantos no referencial do proprio movel: (ao_longo, adiante)
        meia = np.array([[-w / 2, -d / 2], [+w / 2, -d / 2],
                         [+w / 2, +d / 2], [-w / 2, +d / 2]], dtype=float)
        co, si = np.cos(rumo), np.sin(rumo)
        return meia @ np.array([[co, si], [-si, co]]) + np.array([x, y])

    # ---------- pecas ----------
    def pintar_chao(self, img, textura, alfa, extent):
        """Projeta uma imagem de cima no plano do chao.

        Mesmo truque do bloco 1, ao contrario: em vez de levar a imagem para o
        chao, levamos uma imagem do chao para a tela. Os quatro cantos do
        retangulo no mundo viram quatro pontos na tela, e isso define uma
        homografia — a mesma matriz 3x3 de sempre.
        """
        if textura is None:
            return
        x0, x1, y0, y1 = extent
        destino, z = self.cam.projetar([[x0, y0, 0], [x1, y0, 0],
                                        [x1, y1, 0], [x0, y1, 0]])
        if (z <= 0).any():
            return

        h, w = textura.shape[:2]
        origem = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                          dtype=np.float32)
        H = cv2.getPerspectiveTransform(origem, destino.astype(np.float32))

        cor = cv2.warpPerspective(textura, H, (self.cam.w, self.cam.h))
        a = cv2.warpPerspective(alfa, H, (self.cam.w, self.cam.h))[..., None]
        np.copyto(img, (img * (1 - a) + cor * a).astype(np.uint8))

    def desenhar_zonas(self, img, zonas):
        for z in zonas:
            cantos = [[z.x0, z.y0, 0], [z.x1, z.y0, 0],
                      [z.x1, z.y1, 0], [z.x0, z.y1, 0]]
            p, prof = self.cam.projetar(cantos)
            if (prof <= 0).any():
                continue
            p = p.astype(np.int32)
            cor = (60, 150, 235) if z.ocupacao else (185, 182, 185)
            cv2.polylines(img, [p], True, cor, 2 if z.ocupacao else 1, cv2.LINE_AA)
            txt = f"{z.nome}  {z.ocupacao}  {z.tempo_total:.0f}s"
            cv2.putText(img, txt, tuple(p[0] + [4, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, tuple(p[0] + [4, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, cor, 1, cv2.LINE_AA)

    def pes_do_chao(self):
        """Os cantos do piso em metros: o contorno se houver, a caixa se nao."""
        if self.contorno is not None:
            return self.contorno
        x0, x1, y0, y1 = self.chao
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

    def _grade(self, img, mascara=None):
        """A grade de 1 m, recortada ao piso.

        A grade e a regua da cena: e por ela que se ve que o comodo tem quatro
        metros e nao dois. Mas ela desenhada alem do piso vira risco no vazio,
        e sugere chao onde a camera nao olhou — o mesmo problema que o
        contorno existe para resolver. Entao ela passa pela mesma mascara.
        """
        x0, x1, y0, y1 = self.chao
        linhas = []
        for x in np.arange(np.ceil(x0), x1 + .001, 1.0):
            linhas.append(([x, y0, 0], [x, y1, 0]))
        for y in np.arange(np.ceil(y0), y1 + .001, 1.0):
            linhas.append(([x0, y, 0], [x1, y, 0]))

        alvo = img if mascara is None else img.copy()
        for a, b in linhas:
            (pa, pb), z = self.cam.projetar([a, b])
            if z.min() <= 0:
                continue
            cv2.line(alvo, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     GRADE, 1, cv2.LINE_AA)
        if mascara is not None:
            np.copyto(img, alvo, where=mascara[..., None].astype(bool))

    def _movel(self, img, m):
        """Uma caixa GIRADA em torno do proprio centro.

        A estante do quarto nao esta alinhada com os eixos da homografia, e
        nao ha motivo para estar: ela foi encostada onde coube. Desenha-la
        quadrada com a sala mostraria uma estante que nao existe, e — pior —
        poria a face olhando para o lado errado, que e justamente o dado que
        decide se a pessoa esta na frente dela.

            A largura corre ao longo de (cos, sin); a profundidade, ao longo
            da normal (-sin, cos). E a mesma convencao de `ambiente.relacao`,
            e ela precisa ser a mesma, senao o desenho contradiz a conta.
        """
        h, rotulo = m[4], m[5]
        alturas = m[7] if len(m) > 7 else ()
        pes = self.pes_do_movel(m)

        if alturas:
            self._estante(img, pes, h, alturas, rotulo)
            return

        c = np.array([[px, py, 0] for px, py in pes]
                     + [[px, py, h] for px, py in pes], dtype=float)
        p, z = self.cam.projetar(c)
        if z.min() <= 0:
            return
        p = p.astype(int)

        faces = [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for f in faces:
            poly = p[list(f)]
            sobre = img.copy()
            # Movel CLARO e quase translucido, como as prateleiras brancas da
            # referencia. No tema claro um bloco escuro rouba a atencao do
            # corpo — e o corpo e o assunto.
            cv2.fillConvexPoly(sobre, poly, (226, 224, 226))
            cv2.addWeighted(sobre, 0.72, img, 0.28, 0, img)
            cv2.polylines(img, [poly], True, (198, 194, 198), 1, cv2.LINE_AA)

        if rotulo:
            topo = p[[4, 5, 6, 7]].mean(axis=0).astype(int)
            cv2.putText(img, rotulo, tuple(topo), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (150, 146, 150), 1, cv2.LINE_AA)

    def _estante(self, img, pes, altura, alturas, rotulo=""):
        """A estante de aco como ela e: cinco bandejas e quatro montantes.

            vc ja poderia fazer extamente ela como um objeto do nosso ambiente
            virtual MAIS PRECISA FICAR IGUAL!          — Eduardo, 18/08

        Um bloco macico e uma estante e um armario e uma geladeira. Desenhado
        assim, o gemeo mostrava um paralelepipedo onde existe uma prateleira
        vazada — e a pergunta que o sistema inteiro responde e DE QUAL DAS
        CINCO a mao veio. Se as cinco nao aparecem, a resposta nao tem onde
        pousar.

            Um bloco fechado nao representa uma estante: representa o espaco
            que ela ocupa. Sao coisas diferentes, e a diferenca e exatamente
            a informacao que interessa.

        As alturas nao sao inventadas nem espacadas por conta: vem de
        `loja/estante.json`, medidas com trena — 0,15 / 0,55 / 0,95 / 1,35 /
        1,90. Desenhar niveis igualmente espacados seria bonito e mentiroso.
        """
        # Cada bandeja e o mesmo retangulo do chao, subido ate a altura dela.
        for k, z in enumerate(sorted(alturas)):
            c = np.array([[px, py, z] for px, py in pes], dtype=float)
            p, prof = self.cam.projetar(c)
            if prof.min() <= 0:
                continue
            p = p.astype(int)
            sobre = img.copy()
            cv2.fillConvexPoly(sobre, p, (222, 220, 222))
            cv2.addWeighted(sobre, 0.80, img, 0.20, 0, img)
            cv2.polylines(img, [p], True, (176, 172, 176), 1, cv2.LINE_AA)

        # Montantes: quatro cantoneiras do chao ao topo. Sao eles que dao a
        # silhueta de estante — sem os quatro, as bandejas flutuam.
        for canto in range(4):
            px, py = pes[canto]
            c = np.array([[px, py, 0.0], [px, py, altura]], dtype=float)
            p, prof = self.cam.projetar(c)
            if prof.min() <= 0:
                continue
            p = p.astype(int)
            cv2.line(img, tuple(p[0]), tuple(p[1]), (158, 154, 158), 2,
                     cv2.LINE_AA)

        if rotulo:
            c = np.array([[pes[:, 0].mean(), pes[:, 1].mean(), altura]])
            p, prof = self.cam.projetar(c)
            if prof.min() > 0:
                cv2.putText(img, rotulo, tuple(p[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 146, 150), 1,
                            cv2.LINE_AA)

    # Seta no chao: a que distancia da pessoa, quanto mede, quanto abre.
    # O recuo tira a seta de baixo das pernas. Menos que isso e ela fica
    # escondida atras do proprio corpo em metade dos angulos de camera.
    SETA_RECUO, SETA_COMPRIMENTO, SETA_LARGURA = 0.46, 0.34, 0.30

    def _seta(self, img, e, cor):
        """Uma seta no chao dizendo para onde ele esta virado.

            quero que tenha uma mini seta na frente dele apontando para qual
            direcao ele esta indo/se movimentando/parado olhando
                                                        — Eduardo, 12/08

        O RUMO NAO CABE NO CORPO. ENTAO ELE FICA NO CHAO.

        De costas ou de frente, o boneco desenha quase o mesmo borrao — tres
        esferas e umas linhas. E desde que as pernas pararam de andar, nem o
        movimento aparece na silhueta. Duas informacoes que o sistema mede e
        que o desenho estava jogando fora.

        Botar isso no corpo (um "nariz", um ombro destacado) so funcionaria de
        certos angulos, e brigaria com a silhueta limpa que ele desenhou. No
        chao a seta e legivel de qualquer angulo da camera, nao disputa espaco
        com o corpo, e ainda ganha a dimensao que faltava: ela ACENDE quando a
        pessoa anda e apaga quando ela para.

            Nao saber para onde alguem olha e diferente de saber que ele nao
            se mexe. A seta mostra as duas, sem confundir uma com a outra.

        Rumo desconhecido nao desenha seta nenhuma — o vazio e a resposta.
        """
        if e.rumo is None:
            return

        pes = e.juntas[[15, 16]].mean(axis=0)          # tornozelos, COCO-17
        c, s = float(np.cos(e.rumo)), float(np.sin(e.rumo))
        frente = np.array([-s, c])             # o mesmo giro que boneco._girar
        lado = np.array([c, s])

        base = pes[:2] + frente * self.SETA_RECUO
        ponta = base + frente * self.SETA_COMPRIMENTO
        meia = lado * (self.SETA_LARGURA / 2.0)

        chao = np.array([[*ponta, 0.012],
                         [*(base + meia), 0.012],
                         [*(base - meia), 0.012]])
        p, z = self.cam.projetar(chao)
        if (z <= 0).any():
            return

        # Parado ainda desenha, so que apagado: "olhando para la, sem sair do
        # lugar" e uma resposta, nao ausencia de resposta.
        forca = 1.0 if e.andando else 0.55
        tom = tuple(int(FUNDO[i] + (cor[i] - FUNDO[i]) * forca) for i in range(3))
        cv2.fillConvexPoly(img, p.astype(np.int32), tom, cv2.LINE_AA)
        # Contorno cheio nas duas: sem ele a seta apagada some no fundo claro,
        # e "parado olhando para la" viraria "nao sei para onde ele olha" —
        # exatamente a distincao que a seta existe para fazer.
        cv2.polylines(img, [p.astype(np.int32)], True, cor, 1, cv2.LINE_AA)

    def _rastro(self, img, e, cor, faixas=6):
        """Rastro que esmaece para o passado, em POUCAS chamadas de desenho.

        A versao anterior desenhava uma linha antisserrilhada por segmento —
        160 chamadas por pessoa, por quadro. Com duas pessoas, 320 desenhos so
        de rastro.

        Aqui o historico e cortado em 6 faixas de brilho e cada faixa vira UMA
        polilinha. Visualmente quase igual, ~27x menos chamadas.
        """
        if len(e.historico) < 2:
            return
        pts = np.array([[x, y, 0.01] for x, y in e.historico[-160:]])
        p, z = self.cam.projetar(pts)
        p = p[z > 0].astype(np.int32)
        n = len(p)
        if n < 2:
            return

        passo = max(2, n // faixas)
        for f in range(faixas):
            ini, fim = f * passo, min(n, (f + 1) * passo + 1)
            if fim - ini < 2:
                continue
            t = (f + 1) / faixas
            c = tuple(int(FUNDO[j] + (cor[j] - FUNDO[j]) * (0.15 + 0.7 * t))
                      for j in range(3))
            cv2.polylines(img, [p[ini:fim]], False, c, 1, cv2.LINE_AA)

    # Raio de cada junta, em fracao do raio base. Vem da anatomia: a cabeca
    # e a maior massa visivel, ombros e quadris sao articulacoes largas, e as
    # extremidades afinam. Sem essa variacao o boneco vira um colar de contas
    # iguais — que e exatamente a cara de diagrama que queremos perder.
    # SO TRES COISAS TEM MASSA: cabeca, ombros e quadril.
    #
    #     eu desenhei para vc entender          — Eduardo, 12/08
    #
    # No desenho dele, cotovelo, pulso, joelho e tornozelo sao apenas LINHA.
    # E ele esta certo: uma esfera em cada junta compete com as outras por
    # atencao, e o olho perde a leitura do corpo.
    #
    #     Marcar tudo e o mesmo que nao marcar nada. O que da forma nao e a
    #     quantidade de pontos: e QUAIS deles ganham peso.
    #
    # Sobra um pictograma — le-se num instante, de longe, num projetor.
    PESO_DA_JUNTA = {
        0: 2.60,                # cabeca, a maior massa
        5: 1.05, 6: 1.05,       # ombros
        11: 1.00, 12: 1.00,     # quadris
    }

    def _raio(self, i, z):
        """Raio da esfera em pixels. Zero para junta que nao ganha massa."""
        peso = self.PESO_DA_JUNTA.get(i)
        if peso is None:
            return 0
        base = np.clip(62.0 / max(z, 0.3), 4.0, 24.0)
        return max(2, int(base * peso))

    def _espessura(self, z):
        """Largura do osso. Constante no corpo, so encolhe com a distancia.

        Os ossos do desenho do Eduardo tem todos a mesma grossura — sao TRACO,
        nao membro. Variar a espessura por junta traria de volta o efeito de
        corpo inflado.
        """
        return float(np.clip(16.0 / max(z, 0.3), 1.5, 6.0))

    def _esqueleto(self, img, e):
        cor = CORES[e.id % len(CORES)]
        p, z = self.cam.projetar(e.juntas)
        vis = e.visivel if e.visivel is not None else np.ones(len(e.juntas), bool)
        vis = vis & (z > 0)
        p = p.astype(int)

        # sombra no chao, sob o quadril
        base = e.juntas[[11, 12]].mean(axis=0)
        (s,), zs = self.cam.projetar([[base[0], base[1], 0.0]])
        if zs[0] > 0:
            eixo = max(4, int(320 / max(zs[0], .3) * 0.05))
            sobre = img.copy()
            cv2.ellipse(sobre, tuple(s.astype(int)), (eixo, int(eixo * .45)),
                        0, 0, 360, (0, 0, 0), -1)
            cv2.addWeighted(sobre, 0.35, img, 0.65, 0, img)

        # ESFERAS E OSSOS COM VOLUME, EM VEZ DE LINHA FINA.
        #
        #     n acho ele algo bonito ou legal para se apresentar, eu gosto
        #     muito da ideia da AiFi e o esqueleto deles   — Eduardo, 12/08
        #
        # Ele tem razao, e a diferenca nao e enfeite: linha de 2 px le-se como
        # DIAGRAMA, e esfera com volume le-se como CORPO. Numa apresentacao,
        # quem assiste decide em dois segundos se aquilo e uma pessoa ou um
        # grafico — e decide pelo peso visual, nao pela precisao.
        #
        #     O mesmo dado desenhado com volume conta outra historia. Forma e
        #     argumento, nao acabamento.
        #
        # Nada aqui muda a leitura: sao os mesmos 17 pontos que o `boneco.py`
        # monta a partir da descricao. Muda so a espessura do traco.
        cor_viva = cor if not e.prevendo else tuple(int(c * .45) for c in cor)

        # DE TRAS PARA A FRENTE. Sem ordenar, um pulso atras do tronco aparece
        # desenhado por cima dele, e o corpo perde a profundidade que a
        # perspectiva acabou de calcular.
        for a, b in sorted(OSSOS, key=lambda ab: -(z[ab[0]] + z[ab[1]])):
            if not (vis[a] and vis[b]):
                continue
            esp = self._espessura((z[a] + z[b]) / 2)
            _osso(img, p[a], p[b], esp, esp, cor_viva)

        for i in sorted(range(len(p)), key=lambda k: -z[k]):
            # Olhos e orelhas ficam de fora: a AiFi desenha a cabeca como UMA
            # esfera, e quatro pontinhos no rosto so poluem a silhueta.
            r = self._raio(i, z[i])
            if not vis[i] or r <= 0:
                continue
            _esfera(img, p[i], r, cor_viva)

        # A etiqueta escapa a esfera da cabeca em vez de usar um recuo fixo.
        # Com a cabeca pesando 2.60 o recuo antigo (+14) caia DENTRO dela, e o
        # texto sumia. Quem cresce e o raio; entao e o raio que da o recuo.
        cabeca, raio = p[0], self._raio(0, z[0])
        etq = f"#{e.id}" + ("  prevendo" if e.prevendo else "")
        canto = (int(cabeca[0] + raio + 6), int(cabeca[1] - raio + 4))
        cv2.putText(img, etq, canto, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, FUNDO, 3, cv2.LINE_AA)
        cv2.putText(img, etq, canto, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, cor, 1, cv2.LINE_AA)

    # ---------- principal ----------
    def _construir_base(self):
        """Chao, grade e moveis. So muda quando a camera virtual se move."""
        img = np.full((self.cam.h, self.cam.w, 3), FUNDO, dtype=np.uint8)

        pes = self.pes_do_chao()
        quad, z = self.cam.projetar([[x, y, 0.0] for x, y in pes])
        mascara = None
        if (z > 0).all():
            poly = quad.astype(np.int32)
            cv2.fillPoly(img, [poly], CHAO)
            mascara = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mascara, [poly], 255)

        self._grade(img, mascara)
        for m in self.moveis:
            self._movel(img, m)
        return img

    def _fundo(self, calor, agora):
        """Chao + grade + moveis + calor, ja compostos numa imagem so.

        Tudo isto muda devagar. Compor uma vez e copiar por quadro custa ~1 ms;
        recompor custa ~90. A recomposicao acontece quando a camera virtual se
        move ou quando chega a hora de atualizar o calor.
        """
        chave = self._chave_camera()
        vencido = (self.calor_hz > 0 and calor is not None
                   and (agora - self._calor_t) >= 1.0 / self.calor_hz)

        if self._base is None or chave != self._chave_base:
            self._geometria = self._construir_base()
            self._chave_base = chave
            vencido = True

        if self._base is None or vencido:
            self._calor_t = agora
            img = self._geometria.copy()
            if calor is not None:
                cor, alfa = calor.imagem()
                self.pintar_chao(img, cor, alfa, calor.ext)
            self._base = img

        return self._base

    def desenhar_marcadores(self, img, marcadores):
        """Pessoas SEM pose: posicao conhecida, formato desconhecido.

        POR QUE ISTO PRECISOU EXISTIR

        Em 10/08 o Eduardo entrou em cena com o filho. O detector viu os dois
        (1293 saidas em 740 quadros) e o rastreador seguiu os dois. Mas a
        janela ficou VAZIA: com duas pessoas, a associacao entre a vista do
        alto e as vistas de pose deixa de ser confiavel, entao nenhum
        esqueleto e montado — e a cena so sabia desenhar esqueletos.

        O sistema tinha a informacao e mostrava nada. Pior que mostrar pouco.

            Falta de dado deve virar desenho mais pobre, nao tela vazia.

        Um pino no chao com o id diz a verdade inteira: "ha alguem aqui, e eu
        nao sei a pose dele". Deliberadamente diferente de um esqueleto, para
        ninguem confundir medida com estimativa.
        """
        for pid, x, y, prevendo in marcadores:
            cor = CORES[pid % len(CORES)]
            base = np.array([[x, y, 0.0]])
            topo = np.array([[x, y, 1.70]])
            (pb, _), (pt, _) = self.cam.projetar(base), self.cam.projetar(topo)
            p0 = tuple(np.int32(pb[0]))
            p1 = tuple(np.int32(pt[0]))

            # tracejado quando a posicao e previsao, nao medida
            if prevendo:
                v = np.array(p1) - np.array(p0)
                for k in range(0, 10, 2):
                    a = tuple(np.int32(np.array(p0) + v * (k / 10)))
                    b = tuple(np.int32(np.array(p0) + v * ((k + 1) / 10)))
                    cv2.line(img, a, b, cor, 2, cv2.LINE_AA)
            else:
                cv2.line(img, p0, p1, cor, 2, cv2.LINE_AA)

            cv2.circle(img, p0, 6, cor, -1, cv2.LINE_AA)
            cv2.circle(img, p0, 6, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(img, f"#{pid} sem pose", (p1[0] + 8, p1[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, cor, 1, cv2.LINE_AA)

    def desenhar(self, esqueletos, titulo="", calor=None, zonas=(),
                 marcadores=()):
        import time as _t
        agora = _t.monotonic()

        img = self._fundo(calor, agora).copy()

        if zonas:
            self.desenhar_zonas(img, zonas)

        if marcadores:
            self.desenhar_marcadores(img, marcadores)

        # desenha de tras para a frente
        ordem = sorted(esqueletos,
                       key=lambda e: -self.cam.projetar([e.juntas[0]])[1][0])
        # Rastro e seta vao ANTES de todos os corpos, nao antes de cada corpo:
        # sao marcas de chao, e chao fica embaixo de todo mundo. Intercalar
        # faria a seta de quem esta na frente passar por cima de quem esta
        # atras.
        for e in ordem:
            self._rastro(img, e, CORES[e.id % len(CORES)])
            self._seta(img, e, CORES[e.id % len(CORES)])
        for e in ordem:
            self._esqueleto(img, e)

        if titulo:
            cv2.putText(img, titulo, (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, TEXTO, 1, cv2.LINE_AA)
        cv2.putText(img, "setas giram a camera   +/- zoom   ESC sai",
                    (14, self.cam.h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120, 120, 120), 1, cv2.LINE_AA)
        return img

    def tecla(self, k):
        """Controle de orbita. Devolve False no ESC."""
        if k == 27:
            return False
        if k in (81, 2424832):   self.cam.azimute -= .06
        elif k in (83, 2555904): self.cam.azimute += .06
        elif k in (82, 2490368): self.cam.elevacao = min(1.45, self.cam.elevacao + .04)
        elif k in (84, 2621440): self.cam.elevacao = max(.05, self.cam.elevacao - .04)
        elif k in (ord("+"), ord("=")): self.cam.dist = max(1.5, self.cam.dist - .25)
        elif k == ord("-"): self.cam.dist = min(18, self.cam.dist + .25)
        return True


# ---------------------------------------------------------------- demo
def _pose_andando(t, altura=1.75):
    """Esqueleto sintetico, so para provar o renderizador sem camera."""
    q = altura * 0.53                      # altura do quadril
    passo = np.sin(t * 4)
    braco = np.sin(t * 4 + np.pi)
    lo = altura * 0.115                     # meia largura de ombros
    lq = altura * 0.075

    j = np.zeros((17, 3))
    j[0] = [0, 0, altura]                                   # nariz
    j[1] = [-.03, 0, altura - .02]; j[2] = [.03, 0, altura - .02]
    j[3] = [-.07, 0, altura - .05]; j[4] = [.07, 0, altura - .05]
    j[5] = [-lo, 0, altura * .83]; j[6] = [lo, 0, altura * .83]
    j[7] = [-lo - .04, .10 * braco, altura * .66]
    j[8] = [lo + .04, -.10 * braco, altura * .66]
    j[9] = [-lo - .06, .20 * braco, altura * .50]
    j[10] = [lo + .06, -.20 * braco, altura * .50]
    j[11] = [-lq, 0, q]; j[12] = [lq, 0, q]
    j[13] = [-lq, .17 * passo, q * .55]
    j[14] = [lq, -.17 * passo, q * .55]
    j[15] = [-lq, .30 * passo, max(0.02, .06 + .12 * max(0, passo))]
    j[16] = [lq, -.30 * passo, max(0.02, .06 + .12 * max(0, -passo))]
    return j


if __name__ == "__main__":
    cena = Cena3D()
    # x, y sao o CENTRO — estes numeros ja vem convertidos do canto antigo.
    cena.add_movel(-0.9, 1.4, 0.6, 2.4, 1.6, "gondola A")
    cena.add_movel(2.5, 1.4, 0.6, 2.4, 1.6, "gondola B")
    cena.add_movel(1.0, 3.05, 1.6, 0.5, 0.9, "checkout")
    cena.add_movel(0.8, 0.4, 0.9, 0.3, 1.9, "estante girada", rumo=0.6)

    hist = {1: [], 2: []}
    t = 0.0
    while True:
        t += 0.033
        ags = []
        for k, (raio, fase, vel) in enumerate([(1.0, 0, .8), (1.6, 2.2, -.55)], 1):
            cx = 0.9 + raio * np.cos(vel * t + fase)
            cy = 1.0 + raio * np.sin(vel * t + fase)
            rumo = np.arctan2(np.cos(vel * t + fase) * raio * vel,
                              -np.sin(vel * t + fase) * raio * vel)
            j = _pose_andando(t + k)
            R = np.array([[np.cos(rumo), -np.sin(rumo), 0],
                          [np.sin(rumo), np.cos(rumo), 0],
                          [0, 0, 1.0]])
            j = (R @ j.T).T + np.array([cx, cy, 0])
            hist[k].append((cx, cy))
            hist[k] = hist[k][-160:]
            ags.append(Esqueleto(id=k, juntas=j, historico=hist[k],
                                 prevendo=(k == 2 and int(t) % 7 == 0)))

        cv2.imshow("cena 3D", cena.desenhar(ags, "demonstracao sem camera"))
        if not cena.tecla(cv2.waitKeyEx(16) & 0xFFFFFF):
            break
    cv2.destroyAllWindows()
