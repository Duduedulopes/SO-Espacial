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

CORES = [
    (232, 200, 53), (90, 180, 255), (120, 230, 130),
    (220, 130, 240), (110, 220, 240), (240, 170, 110),
]

FUNDO = (28, 24, 22)
CHAO = (46, 40, 37)
GRADE = (64, 56, 51)
TEXTO = (200, 200, 200)


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
class Cena3D:
    def __init__(self, largura=960, altura=620, chao=(-1.5, 3.5, -1.5, 3.5),
                 calor_hz=4.0):
        self.cam = CameraVirtual(largura, altura)
        self.chao = chao
        self.moveis = []          # (x, y, largura, profundidade, altura, rotulo)

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

    def _chave_camera(self):
        c = self.cam
        return (round(c.azimute, 4), round(c.elevacao, 4), round(c.dist, 3),
                tuple(np.round(c.alvo, 3)), len(self.moveis))

    def invalidar(self):
        self._base = None

    def add_movel(self, x, y, larg, prof, alt, rotulo=""):
        self.moveis.append((x, y, larg, prof, alt, rotulo))

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
            cor = (120, 220, 255) if z.ocupacao else (95, 95, 95)
            cv2.polylines(img, [p], True, cor, 2 if z.ocupacao else 1, cv2.LINE_AA)
            txt = f"{z.nome}  {z.ocupacao}  {z.tempo_total:.0f}s"
            cv2.putText(img, txt, tuple(p[0] + [4, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, tuple(p[0] + [4, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, cor, 1, cv2.LINE_AA)

    def _grade(self, img):
        x0, x1, y0, y1 = self.chao
        linhas = []
        for x in np.arange(np.ceil(x0), x1 + .001, 1.0):
            linhas.append(([x, y0, 0], [x, y1, 0]))
        for y in np.arange(np.ceil(y0), y1 + .001, 1.0):
            linhas.append(([x0, y, 0], [x1, y, 0]))

        for a, b in linhas:
            (pa, pb), z = self.cam.projetar([a, b])
            if z.min() <= 0:
                continue
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     GRADE, 1, cv2.LINE_AA)

    def _movel(self, img, m):
        x, y, w, d, h, rotulo = m
        c = np.array([
            [x, y, 0], [x + w, y, 0], [x + w, y + d, 0], [x, y + d, 0],
            [x, y, h], [x + w, y, h], [x + w, y + d, h], [x, y + d, h],
        ], dtype=float)
        p, z = self.cam.projetar(c)
        if z.min() <= 0:
            return
        p = p.astype(int)

        faces = [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for f in faces:
            poly = p[list(f)]
            sobre = img.copy()
            cv2.fillConvexPoly(sobre, poly, (72, 66, 62))
            cv2.addWeighted(sobre, 0.85, img, 0.15, 0, img)
            cv2.polylines(img, [poly], True, (105, 98, 92), 1, cv2.LINE_AA)

        if rotulo:
            topo = p[[4, 5, 6, 7]].mean(axis=0).astype(int)
            cv2.putText(img, rotulo, tuple(topo), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (170, 165, 160), 1, cv2.LINE_AA)

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

        # ossos como tubos: espessura cai com a distancia
        for a, b in OSSOS:
            if not (vis[a] and vis[b]):
                continue
            esp = int(np.clip(60 / max((z[a] + z[b]) / 2, .3), 2, 9))
            cv2.line(img, tuple(p[a]), tuple(p[b]),
                     cor if not e.prevendo else tuple(int(c * .5) for c in cor),
                     esp, cv2.LINE_AA)

        # juntas como esferas
        for i in range(len(p)):
            if not vis[i]:
                continue
            r = int(np.clip(46 / max(z[i], .3), 3, 12))
            if i in (0,):
                r = int(r * 1.7)          # cabeca
            cv2.circle(img, tuple(p[i]), r, cor, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p[i]), max(1, r // 3),
                       (255, 255, 255), -1, cv2.LINE_AA)

        cabeca = p[0]
        etq = f"#{e.id}" + ("  prevendo" if e.prevendo else "")
        cv2.putText(img, etq, (cabeca[0] + 14, cabeca[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, etq, (cabeca[0] + 14, cabeca[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)

    # ---------- principal ----------
    def _construir_base(self):
        """Chao, grade e moveis. So muda quando a camera virtual se move."""
        img = np.full((self.cam.h, self.cam.w, 3), FUNDO, dtype=np.uint8)

        x0, x1, y0, y1 = self.chao
        quad, z = self.cam.projetar([[x0, y0, 0], [x1, y0, 0],
                                     [x1, y1, 0], [x0, y1, 0]])
        if (z > 0).all():
            cv2.fillConvexPoly(img, quad.astype(int), CHAO)

        self._grade(img)
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
        for e in ordem:
            self._rastro(img, e, CORES[e.id % len(CORES)])
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
    cena.add_movel(-1.2, 0.2, 0.6, 2.4, 1.6, "gondola A")
    cena.add_movel(2.2, 0.2, 0.6, 2.4, 1.6, "gondola B")
    cena.add_movel(0.2, 2.8, 1.6, 0.5, 0.9, "checkout")

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
