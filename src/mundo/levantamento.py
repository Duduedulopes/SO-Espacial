"""O LEVANTAMENTO DO AMBIENTE — a leitura que acontece antes do sistema rodar.

    precisamos resolver isso com uma leitura de ambiente primeiro... antes do
    sistema rodar as 3 cameras precisamos fazer uma leitura para entender como
    e o ambiente, como estao as coisas nesse ambiente e o que ela deve
    reproduzir no ambiente virtual        — Eduardo, 18/08

A INVERSAO QUE FAZ ISSO FUNCIONAR: A ESTANTE MEDE AS CAMERAS.

Ate 18/08 o sistema pedia as cameras para medirem a estante, e o resultado foi
uma estante de 1,01 x 0,23 m atravessada na diagonal — porque a camera do teto
enxerga a BANDEJA DE CIMA, e a homografia so vale para o chao.

Mas a estante e um corpo rigido de dimensoes conhecidas com trena: 0,92 x 0,30
x 1,90 m, cinco bandejas em alturas medidas. Um objeto assim, visto por uma
camera, determina a POSE dessa camera — e essa e a informacao que faltava para
as tres trabalharem juntas.

    Nao faltava uma camera melhor. Faltava saber onde cada uma estava. Duas
    vistas sem pose relativa nao se fundem: nao ha conta que as junte.

O QUE O LEVANTAMENTO ENTREGA, e a ordem importa porque cada item destrava o
seguinte:

    1. POSE DE CADA CAMERA      onde esta e para onde olha, em metros
    2. O CHAO EM CADA VISTA     consequencia da pose
    3. ESCALA VERTICAL POR CAMERA   pixel -> metro de altura, em cada uma
    4. OS MOVEIS FIXOS          posicao e rumo; as medidas ja sao da trena
    5. QUEM ENXERGA QUAL PRATELEIRA   decide qual camera responde o que
    6. A NUVEM DE PONTOS        o ambiente reconstruido

A NUVEM E O RESULTADO, NAO O ATALHO.

Triangular um ponto exige ve-lo em duas vistas COM POSE CONHECIDA. Entao a
nuvem nao resolve a pose: ela nasce de a pose estar resolvida. Quem tenta pela
ordem inversa esta fazendo reconstrucao a partir do nada, que e um problema
muito mais dificil e mal condicionado com tres webcams.

    A nuvem de pontos e o retrato do que o sistema entendeu. Ela nao e o
    instrumento que entende.

E ela e ESPARSA, de proposito e com limite declarado: quinas, arestas, os
montantes da estante, os cantos do chao. Nuvem densa exige textura e linhas de
base largas, e uma parede lisa de quarto nao da nem uma coisa nem outra.

RODA UMA VEZ E VIRA ARQUIVO.

O ambiente nao se mexe. Levanta-lo a cada quadro seria gastar o orcamento de
CPU para confirmar o obvio.

    O que nao muda deve ser medido uma vez e escrito. O que muda deve ser
    medido sempre.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Quantos pontos 2D<->3D o solucionador de pose exige.
#
# Quatro e o minimo matematico do PnP. Seis e o minimo PRATICO: com quatro,
# qualquer erro de um ponto vira erro de pose inteiro, e nao sobra residuo
# para denunciar. Com seis ha sobra, e a sobra e o que permite dizer "esta
# pose esta ruim".
PONTOS_MINIMOS = 6

# Erro de reprojecao aceitavel, em pixels. Acima disso a pose e recusada.
#
# Nao e rigor estetico: uma pose com 12 px de residuo numa imagem de 640
# significa que os pontos nao descrevem o objeto que dizem descrever, e todo
# metro derivado dela sera ficcao com aparencia de medida.
RESIDUO_MAXIMO_PX = 8.0

# Angulo minimo entre dois raios para a triangulacao valer, em graus.
#
# Duas cameras quase alinhadas veem quase a mesma coisa, e a profundidade sai
# do ruido em vez de sair da geometria. Cinco graus e pouco; abaixo disso o
# ponto triangulado e opiniao.
ANGULO_MINIMO_GRAUS = 5.0


# ------------------------------------------------------------------ intrinseca
def intrinseca_estimada(largura_px, altura_px, fov_graus=60.0):
    """A matriz K quando ninguem calibrou a lente. E um chute DECLARADO.

    A distancia focal de uma webcam comum cai perto de um campo de visao
    horizontal de 55 a 65 graus. Com f = (largura/2) / tan(fov/2) e o centro
    optico no meio da imagem, chega-se a uma K plausivel.

        Isto NAO e calibracao intrinseca. Uma K chutada da uma pose com erro
        sistematico — os angulos saem bons, a escala em profundidade sai
        enviesada. Para dizer QUEM ESTA NA FRENTE DE QUE serve; para medir
        centimetro em profundidade, nao.

    O jeito certo, quando o centimetro importar, e um tabuleiro de xadrez e
    `cv2.calibrateCamera`. Ate la, o erro tem nome e esta escrito aqui.
    """
    f = (largura_px / 2.0) / math.tan(math.radians(fov_graus) / 2.0)
    return np.array([[f, 0.0, largura_px / 2.0],
                     [0.0, f, altura_px / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


# ------------------------------------------------------------------ o gabarito
def pontos_do_gabarito(gabarito):
    """Os cantos da estante em 3D, no referencial DELA. Em metros.

    Origem no centro da pegada, no chao. A largura corre em x, a profundidade
    em y, a altura em z — a mesma convencao de `ambiente.Ambiente`, e ela
    precisa ser a mesma, senao a pose sai girada em relacao ao resto.

    Devolve {nome: (x, y, z)}. O nome importa: e por ele que quem marca os
    pontos na imagem diz QUAL canto marcou.
    """
    mx, my = gabarito.largura / 2.0, gabarito.profundidade / 2.0
    pontos = {}
    # os quatro pes, no chao
    for nome, (sx, sy) in (("pe_esq_frente", (-1, -1)), ("pe_dir_frente", (+1, -1)),
                           ("pe_dir_fundo", (+1, +1)), ("pe_esq_fundo", (-1, +1))):
        pontos[nome] = (sx * mx, sy * my, 0.0)
    # os cantos de cada bandeja, nas alturas de trena
    for pid, altura in gabarito.prateleiras:
        for lado, (sx, sy) in (("esq_frente", (-1, -1)), ("dir_frente", (+1, -1)),
                               ("dir_fundo", (+1, +1)), ("esq_fundo", (-1, +1))):
            pontos[f"{pid}_{lado}"] = (sx * mx, sy * my, float(altura))
    return pontos


# ------------------------------------------------------------------ a pose
@dataclass
class PoseDaCamera:
    """Onde uma camera esta e para onde olha, no referencial do mundo.

    Guarda rvec e tvec no formato do OpenCV — a transformacao do MUNDO para a
    CAMERA. A posicao da camera no mundo e outra coisa, e sai de
    `posicao`: quem confunde as duas ganha uma camera no lugar espelhado.
    """
    papel: str
    rvec: np.ndarray            # 3x1, Rodrigues
    tvec: np.ndarray            # 3x1, metros
    k: np.ndarray               # 3x3, intrinseca
    residuo_px: float = 0.0
    pontos_usados: int = 0

    @property
    def rotacao(self):
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=float))[0]

    @property
    def posicao(self):
        """Onde a camera esta, em metros no mundo. C = -R^T t."""
        return (-self.rotacao.T @ np.asarray(self.tvec, dtype=float).reshape(3)).ravel()

    @property
    def olhando_para(self):
        """Vetor unitario da direcao de visada, no mundo."""
        return (self.rotacao.T @ np.array([0.0, 0.0, 1.0])).ravel()

    @property
    def altura(self):
        """A que altura do chao esta a camera. Confere com a fita metrica."""
        return float(self.posicao[2])

    @property
    def confiavel(self):
        return (self.pontos_usados >= PONTOS_MINIMOS
                and self.residuo_px <= RESIDUO_MAXIMO_PX)

    def projetar(self, pontos_mundo):
        """Pontos 3D do mundo -> pixels nesta camera."""
        p = np.asarray(pontos_mundo, dtype=float).reshape(-1, 3)
        px, _ = cv2.projectPoints(p, np.asarray(self.rvec, dtype=float),
                                  np.asarray(self.tvec, dtype=float),
                                  self.k, None)
        return px.reshape(-1, 2)

    def raio(self, u, v):
        """A reta que sai da camera e passa por este pixel.

        Devolve (origem, direcao_unitaria) no mundo. E com dois raios de duas
        cameras que se triangula um ponto — e e por isso que a pose precisa
        existir antes da nuvem.
        """
        homog = np.array([float(u), float(v), 1.0])
        direcao = self.rotacao.T @ (np.linalg.inv(self.k) @ homog)
        n = np.linalg.norm(direcao)
        return self.posicao, (direcao / n if n > 0 else direcao)


def resolver_pose(papel, marcados, modelo_3d, tamanho_px, fov_graus=60.0):
    """A pose desta camera, a partir de pontos do gabarito marcados na imagem.

    `marcados`   {nome_do_ponto: (u, v)} — onde cada canto aparece na imagem
    `modelo_3d`  {nome_do_ponto: (x, y, z)} — de `pontos_do_gabarito`

    Devolve `PoseDaCamera`, ou None quando nao ha pontos suficientes ou quando
    a solucao nao fecha.

    O RESIDUO E O PRODUTO PRINCIPAL, e nao um detalhe de diagnostico.

    Resolver a pose sempre devolve alguma resposta — o PnP e um otimizador, e
    otimizador nao se recusa. O que separa uma pose util de uma inventada e o
    erro de reprojecao: pegar a pose encontrada, projetar os pontos 3D de
    volta na imagem e medir quanto eles caem longe de onde foram marcados.

        Um solucionador que sempre responde precisa de alguem que meca a
        resposta. Sem residuo, PnP e um gerador de numeros com seis casas.
    """
    nomes = [n for n in marcados if n in modelo_3d]
    if len(nomes) < PONTOS_MINIMOS:
        return None

    obj = np.array([modelo_3d[n] for n in nomes], dtype=np.float64)
    img = np.array([marcados[n] for n in nomes], dtype=np.float64)
    k = intrinseca_estimada(tamanho_px[0], tamanho_px[1], fov_graus)

    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    pose = PoseDaCamera(papel=papel, rvec=rvec, tvec=tvec, k=k,
                        pontos_usados=len(nomes))
    reprojetado = pose.projetar(obj)
    pose.residuo_px = float(np.sqrt(((reprojetado - img) ** 2).sum(axis=1)).mean())
    return pose


# ------------------------------------------------------------------ a nuvem
def triangular(pose_a, pixel_a, pose_b, pixel_b):
    """Um ponto 3D a partir do mesmo ponto visto em duas cameras com pose.

    Devolve (ponto, erro_m, angulo_graus), ou None quando os dois raios sao
    paralelos demais para a intersecao significar alguma coisa.

    Nao ha intersecao exata entre duas retas no espaco — o ruido garante isso.
    O que se calcula e o ponto mais proximo das duas, e a DISTANCIA entre elas
    ali e o erro. Devolve-lo junto e o que permite jogar fora o ponto ruim
    depois, em vez de descobrir na tela.
    """
    o1, d1 = pose_a.raio(*pixel_a)
    o2, d2 = pose_b.raio(*pixel_b)

    cos = float(np.clip(d1 @ d2, -1.0, 1.0))
    angulo = math.degrees(math.acos(abs(cos)))
    if angulo < ANGULO_MINIMO_GRAUS:
        return None

    # ponto medio do segmento mais curto entre as duas retas
    w0 = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:
        return None
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), angulo


@dataclass
class NuvemDePontos:
    """O ambiente reconstruido, ponto a ponto. Esparsa, e isso e declarado.

    Cada ponto guarda de onde veio: quais cameras o viram, com que angulo
    entre os raios e com que erro. Uma nuvem que so guarda coordenadas nao
    permite separar o que foi bem medido do que foi adivinhado — e numa
    reconstrucao com webcam a maior parte e adivinhada.
    """
    pontos: list = field(default_factory=list)   # (x, y, z)
    erros: list = field(default_factory=list)    # metros
    angulos: list = field(default_factory=list)  # graus
    vistos_por: list = field(default_factory=list)

    def somar(self, ponto, erro, angulo, cameras):
        self.pontos.append(tuple(float(v) for v in ponto))
        self.erros.append(float(erro))
        self.angulos.append(float(angulo))
        self.vistos_por.append(tuple(cameras))

    def __len__(self):
        return len(self.pontos)

    def firmes(self, erro_maximo_m=0.05):
        """So os pontos em que as duas retas quase se encontraram."""
        n = NuvemDePontos()
        for p, e, a, c in zip(self.pontos, self.erros, self.angulos,
                              self.vistos_por):
            if e <= erro_maximo_m:
                n.somar(p, e, a, c)
        return n

    def como_array(self):
        return (np.array(self.pontos, dtype=float).reshape(-1, 3)
                if self.pontos else np.zeros((0, 3)))

    def caixa(self):
        """Os limites do que foi reconstruido, em metros."""
        a = self.como_array()
        if not len(a):
            return None
        return tuple(a.min(axis=0)), tuple(a.max(axis=0))


def nuvem_de(poses, correspondencias, erro_maximo_m=0.05):
    """Triangula tudo que aparece em duas ou mais vistas com pose.

    `correspondencias`  [{papel: (u, v)}] — o mesmo ponto do mundo em cada
                        camera que o enxergou

    Percorre todos os pares de cameras que viram o ponto e fica com a
    triangulacao de MAIOR ANGULO — que e a de melhor condicionamento, nao a
    de menor erro aparente. Erro pequeno com angulo pequeno e o disfarce
    classico: duas retas quase paralelas se encontram bem em qualquer lugar
    ao longo delas.

        Escolher pelo residuo premia a medida que nao tinha como discordar.
    """
    nuvem = NuvemDePontos()
    for visto in correspondencias:
        papeis = [p for p in visto if p in poses and poses[p] is not None]
        melhor = None
        for i, pa in enumerate(papeis):
            for pb in papeis[i + 1:]:
                r = triangular(poses[pa], visto[pa], poses[pb], visto[pb])
                if r is None:
                    continue
                ponto, erro, angulo = r
                if erro > erro_maximo_m:
                    continue
                if melhor is None or angulo > melhor[2]:
                    melhor = (ponto, erro, angulo, (pa, pb))
        if melhor is not None:
            nuvem.somar(*melhor[:3], melhor[3])
    return nuvem


# ------------------------------------------------------------------ o resultado
@dataclass
class Levantamento:
    """Tudo que a leitura de ambiente descobriu. Vira arquivo; o laco so le."""
    poses: dict = field(default_factory=dict)        # papel -> PoseDaCamera
    nuvem: NuvemDePontos = field(default_factory=NuvemDePontos)
    prateleiras_por_camera: dict = field(default_factory=dict)
    medido_em: str = ""

    @property
    def cameras_situadas(self):
        return tuple(sorted(p for p, v in self.poses.items()
                            if v is not None and v.confiavel))

    @property
    def pronto(self):
        """Duas cameras situadas e o minimo para haver fusao de verdade.

        Com uma so, o levantamento vira um monologo: nao ha segunda opiniao
        sobre nada, e nao ha nuvem nenhuma — triangular exige duas.
        """
        return len(self.cameras_situadas) >= 2

    def como_dicionario(self):
        d = {"_medido_em": self.medido_em,
             "_nota": [
                 "ACHADO PELAS CAMERAS contra o gabarito de trena, nao digitado.",
                 "Ver src/mundo/levantamento.py — a estante mede as cameras.",
                 "",
                 "A intrinseca e ESTIMADA por campo de visao, nao calibrada.",
                 "Os angulos saem bons; a escala em profundidade sai enviesada.",
                 "Para dizer quem esta na frente de que, serve.",
             ],
             "cameras": {}, "nuvem": {}}
        for papel, pose in self.poses.items():
            if pose is None:
                d["cameras"][papel] = None
                continue
            d["cameras"][papel] = {
                "posicao_m": [round(float(v), 4) for v in pose.posicao],
                "olhando_para": [round(float(v), 4) for v in pose.olhando_para],
                "altura_m": round(pose.altura, 3),
                "residuo_px": round(pose.residuo_px, 2),
                "pontos_usados": pose.pontos_usados,
                "confiavel": bool(pose.confiavel),
                "rvec": [float(v) for v in np.asarray(pose.rvec).ravel()],
                "tvec": [float(v) for v in np.asarray(pose.tvec).ravel()],
            }
        d["nuvem"] = {
            "pontos": len(self.nuvem),
            "caixa_m": self.nuvem.caixa(),
            "_nota": "esparsa de proposito: quinas e arestas, nao superficie",
        }
        d["prateleiras_por_camera"] = {
            k: list(v) for k, v in self.prateleiras_por_camera.items()}
        return d

    def gravar(self, caminho="loja/levantamento.json"):
        Path(caminho).write_text(
            json.dumps(self.como_dicionario(), ensure_ascii=False, indent=2),
            encoding="utf-8")
