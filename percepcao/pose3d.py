"""
Pose 3D relativa ao quadril, via MediaPipe.

A DECOMPOSICAO, que e a ideia central:

    ONDE a pessoa esta   ->  homografia, ja resolvido, 2 a 5 cm   (bloco 1)
    COMO o corpo esta    ->  este modulo, relativo ao quadril

    somando os dois      ->  esqueleto 3D em coordenadas de mundo

A literatura chama a ambiguidade de profundidade de problema fundamental da
pose monocular. Nos nao a enfrentamos: nao pedimos profundidade ao modelo,
pedimos ao chao. O modelo so precisa dizer o formato do corpo.

PIPELINE
    YOLO detecta e rastreia  ->  recorta a caixa da pessoa
                             ->  MediaPipe roda no recorte
                             ->  landmarks em METROS, origem no quadril

O MediaPipe Pose trata UMA pessoa por vez. Por isso o recorte: cada pessoa
detectada vira uma chamada separada.

CONVERSAO DE EIXOS
    MediaPipe (world):  x direita,  y PARA BAIXO,  z profundidade
    nosso mundo:        x direita,  y para frente, z PARA CIMA

    logo:  nosso_x = mp_x     nosso_y = mp_z     nosso_z = -mp_y
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np

# MediaPipe usa 33 pontos; nosso esqueleto usa os 17 do padrao COCO.
# Esta tabela diz qual ponto do MediaPipe corresponde a cada indice COCO.
MP_PARA_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

RAIZ = Path(__file__).resolve().parent.parent
MODELOS = RAIZ / "modelos"

# O MediaPipe moderno (API tasks) precisa de um arquivo de modelo. Tres
# tamanhos; lite roda bem em CPU e e o suficiente aqui.
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


def _baixar_modelo():
    MODELOS.mkdir(exist_ok=True)
    destino = MODELOS / "pose_landmarker_lite.task"
    if not destino.exists():
        print(f"baixando modelo de pose para {destino} ...")
        urllib.request.urlretrieve(URL_MODELO, destino)
        print("pronto")
    return destino


class Pose3D:
    """Envelope fino em volta do MediaPipe Pose Landmarker (API tasks)."""

    def __init__(self, confianca=0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise SystemExit(f"mediapipe nao instalado ou incompleto: {e}\n"
                             "    pip install mediapipe")

        self.mp = mp
        modelo = _baixar_modelo()

        opcoes = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(modelo)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,                       # uma por recorte
            min_pose_detection_confidence=confianca,
            min_pose_presence_confidence=confianca,
            min_tracking_confidence=confianca,
            output_segmentation_masks=False,
        )
        self.detector = vision.PoseLandmarker.create_from_options(opcoes)
        self._t_ms = 0

    def estimar(self, frame_bgr, caixa=None, margem=0.12, escala_minima=256):
        """Devolve (juntas_coco (17,3) em metros relativas ao quadril, visivel (17,)).

        `caixa` = (x1, y1, x2, y2) em pixels. Recortar antes de estimar melhora
        muito o resultado quando ha mais de uma pessoa, e tambem quando a
        pessoa ocupa pouco do quadro.

        Devolve (None, None) quando nao ha pose confiavel — melhor que chutar.
        """
        img = frame_bgr
        origem_x, origem_y, fator = 0, 0, 1.0

        if caixa is not None:
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = caixa
            mx = int((x2 - x1) * margem)
            my = int((y2 - y1) * margem)
            x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
            x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
            if x2 - x1 < 24 or y2 - y1 < 48:
                return None, None, None
            img = frame_bgr[y1:y2, x1:x2]
            origem_x, origem_y = x1, y1

            # AMPLIA recortes pequenos. O MediaPipe foi treinado com a pessoa
            # ocupando boa parte do quadro; um recorte de 60x140 px de alguem
            # distante rende landmarks ruins. Ampliar nao cria informacao, mas
            # poe a entrada na escala que o modelo espera — e melhora bastante.
            if img.shape[0] < escala_minima:
                fator = escala_minima / img.shape[0]
                img = cv2.resize(img, None, fx=fator, fy=fator,
                                 interpolation=cv2.INTER_LINEAR)

        # A API de video exige carimbo de tempo estritamente crescente.
        self._t_ms += 33
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(img[:, :, ::-1]))
        res = self.detector.detect_for_video(mp_img, self._t_ms)

        if not res.pose_world_landmarks:
            return None, None, None

        lm = res.pose_world_landmarks[0]
        bruto = np.array([[p.x, p.y, p.z] for p in lm])
        vis = np.array([getattr(p, "visibility", 1.0) or 1.0 for p in lm])

        j = bruto[MP_PARA_COCO]
        v = vis[MP_PARA_COCO] > 0.4

        # eixos: mp(x, y_baixo, z_prof) -> nosso(x, y_frente, z_cima)
        juntas = np.stack([j[:, 0], j[:, 2], -j[:, 1]], axis=1)

        # ancora no quadril, para o ponto (0,0,0) ser o centro do quadril
        quadril = juntas[[11, 12]].mean(axis=0)
        juntas = juntas - quadril

        # ---- landmarks 2D, de volta em pixels da imagem inteira ----
        # Assim o MediaPipe tambem fornece os tornozelos na imagem, e o YOLO
        # nao precisa mais calcular pose. Um modelo de pose em vez de dois.
        px2d = None
        if res.pose_landmarks:
            n = res.pose_landmarks[0]
            hh, ww = img.shape[:2]
            p = np.array([[q.x * ww, q.y * hh] for q in n])[MP_PARA_COCO]
            p = p / fator
            p[:, 0] += origem_x
            p[:, 1] += origem_y
            px2d = p

        return juntas, v, px2d

    def fechar(self):
        self.detector.close()


class SuavizadorDeEsqueleto:
    """Media exponencial nas juntas, por rastro.

    Cada quadro e estimado do zero pelo modelo, entao as juntas tremem mesmo
    com a pessoa parada. Isso e RUIDO, nao movimento — e ruido se filtra.

    Uso alfa adaptativo: quanto mais rapido o ponto se move, mais eu confio na
    medicao nova. Assim o esqueleto fica estavel parado, sem ficar borrachudo
    quando a pessoa se mexe de verdade — que e o defeito de uma media simples.

    E o mesmo raciocinio do Kalman, numa versao barata. Kalman de verdade em
    17 juntas x 3 eixos custaria caro, e aqui nao precisamos prever, so alisar.
    """

    def __init__(self, alfa_parado=0.25, alfa_rapido=0.85, limiar_m=0.04):
        self.estado: dict[int, np.ndarray] = {}
        self.a_lento = alfa_parado
        self.a_rapido = alfa_rapido
        self.limiar = limiar_m

    def suavizar(self, tid, juntas):
        j = np.asarray(juntas, dtype=float)
        ant = self.estado.get(tid)
        if ant is None or ant.shape != j.shape:
            self.estado[tid] = j.copy()
            return j

        d = np.linalg.norm(j - ant, axis=1, keepdims=True)
        alfa = np.clip(d / self.limiar, 0, 1) * (self.a_rapido - self.a_lento) + self.a_lento
        novo = ant * (1 - alfa) + j * alfa
        self.estado[tid] = novo
        return novo

    def esquecer(self, vivos):
        for tid in list(self.estado):
            if tid not in vivos:
                del self.estado[tid]


class EstimadorDeInclinacao:
    """Mede sozinho a inclinacao da camera, observando pessoas em pe.

    A IDEIA

    Uma pessoa ANDANDO esta em pe — ninguem caminha inclinado. Entao o vetor
    quadril->ombros dela e vertical no mundo.

    Como o MediaPipe entrega as coordenadas alinhadas com a CAMERA, esse vetor
    aparece girado exatamente pela inclinacao da lente. Basta medir o giro.

    GEOMETRIA (deduzida, nao chutada)

    Camera inclinada θ para baixo. Torso vertical de comprimento L no mundo.
    Nos eixos da camera ele vira (0, −L·cosθ, −L·senθ), e apos a nossa
    conversao de eixos, (0, −L·senθ, L·cosθ).

    Logo  atan2(y, z) = −θ  — o angulo de CORRECAO e o negativo da inclinacao.
    Foi exatamente esse sinal que eu errei no primeiro palpite.

    ROBUSTEZ

    Usa a MEDIANA das ultimas amostras, nao a media. Se a pessoa se abaixar
    para pegar um produto, aquele quadro vira um valor esquisito — e mediana
    ignora valor esquisito, media nao.

    So aceita amostras quando a pessoa esta andando. Parada, ela pode estar
    encostada, curvada sobre o celular, agachada. Andando, esta em pe.
    """

    def __init__(self, memoria=240, vel_minima=0.20, minimo_amostras=20):
        from collections import deque
        self.amostras = deque(maxlen=memoria)
        self.vel_minima = vel_minima
        self.minimo = minimo_amostras
        self.valor = 0.0

    def observar(self, juntas_relativas, velocidade_ms, visivel=None):
        """Alimenta uma amostra. Devolve True se ela foi aceita."""
        if velocidade_ms < self.vel_minima:
            return False

        if visivel is not None and not (visivel[5] and visivel[6]
                                        and visivel[11] and visivel[12]):
            return False

        j = np.asarray(juntas_relativas, dtype=float)
        tronco = j[[5, 6]].mean(axis=0) - j[[11, 12]].mean(axis=0)

        if np.linalg.norm(tronco) < 0.15:      # tronco curto demais: descarta
            return False

        self.amostras.append(float(np.arctan2(tronco[1], tronco[2])))
        if len(self.amostras) >= self.minimo:
            self.valor = float(np.median(self.amostras))
        return True

    @property
    def confiavel(self):
        return len(self.amostras) >= self.minimo

    @property
    def graus(self):
        return float(np.rad2deg(self.valor))

    @property
    def dispersao_graus(self):
        """Quanto as amostras discordam entre si. Alta = algo esta errado."""
        if len(self.amostras) < 4:
            return float("nan")
        a = np.rad2deg(np.array(self.amostras))
        return float(np.percentile(a, 75) - np.percentile(a, 25))


def ancorar_no_chao(juntas_relativas, x_m, y_m, rumo_rad=0.0, inclinacao_rad=0.0):
    """Poe o esqueleto relativo em pe, no ponto do chao.

    juntas_relativas: (17,3), origem no quadril, metros.
    (x_m, y_m): posicao no chao vinda da homografia.
    rumo_rad: para onde a pessoa olha, do vetor velocidade.
    inclinacao_rad: quanto a CAMERA esta inclinada para baixo.

    POR QUE A INCLINACAO E NECESSARIA

    O MediaPipe devolve as coordenadas alinhadas com a CAMERA, nao com a
    gravidade. O "para baixo" dele e o para baixo da imagem, e o eixo de
    profundidade aponta para onde a lente olha.

    Com a camera inclinada olhando o chao, esses eixos estao girados em
    relacao ao mundo — e o esqueleto sai tombado exatamente por esse angulo.
    Aqui desfazemos essa rotacao.

    O valor certo poderia ser extraido da homografia, mas isso exige os
    parametros internos da lente. Por ora e um ajuste, medido olhando.

    A altura do quadril sai do proprio esqueleto: distancia do quadril ate o
    tornozelo mais baixo. Nao chutamos "pessoa tem 1,75 m" — se ela agachar,
    o quadril desce sozinho.
    """
    j = np.asarray(juntas_relativas, dtype=float).copy()

    # 1) desfaz a inclinacao da camera (giro em torno do eixo x = direita)
    if inclinacao_rad:
        c, s = np.cos(inclinacao_rad), np.sin(inclinacao_rad)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        j = (Rx @ j.T).T

    # 2) vira a pessoa para a direcao em que ela anda
    c, s = np.cos(rumo_rad), np.sin(rumo_rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    j = (R @ j.T).T

    tornozelo_z = min(j[15, 2], j[16, 2])
    j[:, 2] -= tornozelo_z            # pe mais baixo encosta em z=0

    j[:, 0] += x_m
    j[:, 1] += y_m
    return j
