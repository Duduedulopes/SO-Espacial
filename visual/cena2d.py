"""
Renderizador 2D do gemeo digital — vista de cima, estilo AiFi.

SEPARACAO DELIBERADA: este modulo nao sabe nada de camera, YOLO, homografia ou
Kalman. Ele recebe uma lista de Agente — posicao em metros, velocidade, para
onde olha — e desenha.

Por que isso importa: o ESTADO vira dado puro. Hoje quem desenha e o OpenCV.
Amanha pode ser Unity, Godot, uma pagina web ou um jogo. Nada do resto do
sistema precisa mudar.

E o mesmo principio do "Fonte como interface", agora do outro lado do cano:

    camera -> percepcao -> ESTADO (metros, m/s) -> renderizador

Uso:
    cena = Cena2D(xmin=-1, xmax=2, ymin=-1, ymax=2, px_por_m=140)
    img = cena.desenhar([Agente(id=1, x=0.5, y=0.8, vx=0.4, vy=0.0), ...])
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---- paleta (BGR) ----
FUNDO = (26, 22, 20)
GRADE_FINA = (44, 39, 36)
GRADE_GROSSA = (62, 55, 50)
CALIBRADA = (150, 140, 60)
TEXTO = (200, 200, 200)
TEXTO_FRACO = (120, 120, 120)

CORES = [
    (232, 200, 53),   # ciano
    (90, 180, 255),   # laranja
    (120, 230, 130),  # verde
    (220, 130, 240),  # rosa
    (110, 220, 240),  # amarelo
    (240, 170, 110),  # azul claro
]


@dataclass
class Agente:
    """Uma pessoa no mundo. Tudo em metros e metros por segundo."""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    incerteza: float = 0.0
    prevendo: int = 0                       # quadros sem medicao
    historico: list = field(default_factory=list)
    rumo: float | None = None               # radianos; None = usa a velocidade

    @property
    def velocidade(self):
        return float(np.hypot(self.vx, self.vy))


class Cena2D:
    def __init__(self, xmin, xmax, ymin, ymax, px_por_m=140,
                 area_calibrada=None):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.ppm = px_por_m
        self.larg = int((xmax - xmin) * px_por_m)
        self.alt = int((ymax - ymin) * px_por_m)
        self.area_calibrada = area_calibrada   # (largura_m, altura_m)
        self._rumos: dict[int, float] = {}     # ultimo rumo conhecido por id

    def px(self, x, y):
        return int((x - self.xmin) * self.ppm), int((y - self.ymin) * self.ppm)

    # ------------------------------------------------------------------
    def desenhar(self, agentes, titulo=""):
        img = np.full((self.alt, self.larg, 3), FUNDO, dtype=np.uint8)
        self._grade(img)
        self._area_calibrada(img)

        for a in agentes:
            self._rastro(img, a)
        for a in agentes:
            self._pessoa(img, a)

        self._escala(img)
        if titulo:
            cv2.putText(img, titulo, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, TEXTO, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------
    def _grade(self, img):
        # 25 cm fina, 1 m grossa
        passo_f = self.ppm * 0.25
        x = np.ceil(self.xmin / 0.25) * 0.25
        while x <= self.xmax:
            gx = self.px(x, 0)[0]
            cv2.line(img, (gx, 0), (gx, self.alt), GRADE_FINA, 1)
            x += 0.25
        y = np.ceil(self.ymin / 0.25) * 0.25
        while y <= self.ymax:
            gy = self.px(0, y)[1]
            cv2.line(img, (0, gy), (self.larg, gy), GRADE_FINA, 1)
            y += 0.25

        x = np.ceil(self.xmin)
        while x <= self.xmax:
            gx = self.px(x, 0)[0]
            cv2.line(img, (gx, 0), (gx, self.alt), GRADE_GROSSA, 1)
            cv2.putText(img, f"{x:.0f}", (gx + 3, self.alt - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXTO_FRACO, 1, cv2.LINE_AA)
            x += 1
        y = np.ceil(self.ymin)
        while y <= self.ymax:
            gy = self.px(0, y)[1]
            cv2.line(img, (0, gy), (self.larg, gy), GRADE_GROSSA, 1)
            cv2.putText(img, f"{y:.0f}", (5, gy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXTO_FRACO, 1, cv2.LINE_AA)
            y += 1

    def _area_calibrada(self, img):
        if not self.area_calibrada:
            return
        lm, am = self.area_calibrada
        pts = np.array([self.px(0, 0), self.px(lm, 0),
                        self.px(lm, am), self.px(0, am)], dtype=np.int32)
        cv2.polylines(img, [pts], True, CALIBRADA, 1, cv2.LINE_AA)
        cv2.putText(img, "area calibrada", (pts[0][0] + 5, pts[0][1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CALIBRADA, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _rastro(self, img, a):
        if len(a.historico) < 2:
            return
        cor = CORES[a.id % len(CORES)]
        pts = a.historico[-160:]
        n = len(pts)
        # esmaece o passado
        for k in range(1, n):
            f = k / n
            c = tuple(int(FUNDO[j] + (cor[j] - FUNDO[j]) * (0.15 + 0.75 * f))
                      for j in range(3))
            cv2.line(img, self.px(*pts[k - 1]), self.px(*pts[k]), c,
                     1 if f < 0.6 else 2, cv2.LINE_AA)

    def _pessoa(self, img, a):
        cor = CORES[a.id % len(CORES)]
        cx, cy = self.px(a.x, a.y)

        # rumo: usa a velocidade quando anda, senao guarda o ultimo
        if a.rumo is not None:
            rumo = a.rumo
        elif a.velocidade > 0.12:
            rumo = float(np.arctan2(a.vy, a.vx))
            self._rumos[a.id] = rumo
        else:
            rumo = self._rumos.get(a.id, -np.pi / 2)

        prevendo = a.prevendo > 0

        # incerteza
        r_inc = int(max(a.incerteza, 0.02) * self.ppm)
        if r_inc > 4:
            sobre = img.copy()
            cv2.circle(sobre, (cx, cy), r_inc, cor, -1, cv2.LINE_AA)
            cv2.addWeighted(sobre, 0.10, img, 0.90, 0, img)
            cv2.circle(img, (cx, cy), r_inc, cor, 1, cv2.LINE_AA)

        # cone de visao — comeca na borda do corpo, para nao ficar por baixo
        meio = np.deg2rad(32)
        ini = 0.20 * self.ppm
        alc = 1.05 * self.ppm
        cone = np.array([
            [cx + ini * np.cos(rumo), cy + ini * np.sin(rumo)],
            [cx + alc * np.cos(rumo - meio), cy + alc * np.sin(rumo - meio)],
            [cx + alc * np.cos(rumo), cy + alc * np.sin(rumo)],
            [cx + alc * np.cos(rumo + meio), cy + alc * np.sin(rumo + meio)],
        ], dtype=np.int32)
        sobre = img.copy()
        cv2.fillPoly(sobre, [cone], cor)
        cv2.addWeighted(sobre, 0.14, img, 0.86, 0, img)

        # ---- corpo visto de cima ----
        # Proporcoes humanas reais: ombros ~45 cm de largura por ~25 de
        # profundidade; cabeca ~18 cm de diametro. Vista de cima, a cabeca
        # cobre boa parte do tronco — por isso ela e desenhada em tom mais
        # escuro, e nao branca: senao vira um borrao sem direcao.
        eixo = (max(5, int(0.225 * self.ppm)), max(3, int(0.125 * self.ppm)))
        ang = np.rad2deg(rumo)
        escuro = tuple(int(c * 0.45) for c in cor)

        if prevendo:
            cv2.ellipse(img, (cx, cy), eixo, ang, 0, 360, cor, 1, cv2.LINE_AA)
        else:
            cv2.ellipse(img, (cx, cy), eixo, ang, 0, 360, cor, -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), eixo, ang, 0, 360, (245, 245, 245), 1, cv2.LINE_AA)

        # cabeca, levemente a frente do centro dos ombros
        r_cab = max(3, int(0.09 * self.ppm))
        hx = int(cx + 0.025 * self.ppm * np.cos(rumo))
        hy = int(cy + 0.025 * self.ppm * np.sin(rumo))
        cv2.circle(img, (hx, hy), r_cab, escuro if not prevendo else cor,
                   -1 if not prevendo else 1, cv2.LINE_AA)

        # marcador de frente: cunha curta saindo da cabeca
        nariz = np.array([
            [hx + 0.20 * self.ppm * np.cos(rumo), hy + 0.20 * self.ppm * np.sin(rumo)],
            [hx + 0.08 * self.ppm * np.cos(rumo + 2.4), hy + 0.08 * self.ppm * np.sin(rumo + 2.4)],
            [hx + 0.08 * self.ppm * np.cos(rumo - 2.4), hy + 0.08 * self.ppm * np.sin(rumo - 2.4)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [nariz], (245, 245, 245) if not prevendo else cor, cv2.LINE_AA)

        # etiqueta
        etq = f"#{a.id}"
        if prevendo:
            etq += "  prevendo"
        elif a.velocidade > 0.12:
            etq += f"  {a.velocidade:.1f} m/s"
        cv2.putText(img, etq, (cx + eixo[0] + 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, etq, (cx + eixo[0] + 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, cor, 1, cv2.LINE_AA)

    def _escala(self, img):
        n = int(1.0 * self.ppm)
        x0, y0 = 14, self.alt - 20
        cv2.line(img, (x0, y0), (x0 + n, y0), TEXTO, 2, cv2.LINE_AA)
        cv2.line(img, (x0, y0 - 4), (x0, y0 + 4), TEXTO, 2)
        cv2.line(img, (x0 + n, y0 - 4), (x0 + n, y0 + 4), TEXTO, 2)
        cv2.putText(img, "1 m", (x0 + n + 8, y0 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXTO, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # demonstracao sem camera: duas pessoas andando em circulo.
    # Prova que o renderizador nao depende de nada do resto do sistema —
    # e por isso que ele poderia alimentar um jogo.
    import time

    cena = Cena2D(-1, 3, -1, 3, px_por_m=150, area_calibrada=(1.0, 1.0))
    hist = {1: [], 2: []}
    t = 0.0
    while True:
        t += 0.033
        ags = []
        for k, (raio, fase, vel) in enumerate([(0.9, 0.0, 1.0), (1.4, 2.0, -0.7)], start=1):
            x = 1.0 + raio * np.cos(vel * t + fase)
            y = 1.0 + raio * np.sin(vel * t + fase)
            vx = -raio * vel * np.sin(vel * t + fase)
            vy = raio * vel * np.cos(vel * t + fase)
            hist[k].append((x, y))
            hist[k] = hist[k][-160:]
            ags.append(Agente(id=k, x=x, y=y, vx=vx, vy=vy,
                              incerteza=0.05, historico=hist[k]))

        cv2.imshow("cena 2D - ESC sai", cena.desenhar(ags, "demonstracao sem camera"))
        if cv2.waitKey(16) == 27:
            break
    cv2.destroyAllWindows()
