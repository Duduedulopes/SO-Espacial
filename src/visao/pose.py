"""
EstimadorDePose — MediaPipe nas vistas frontal e lateral.

POR QUE ESTAS VISTAS, E NAO A DE CIMA

O MediaPipe foi treinado com imagens FRONTAIS de pessoas. Uma camera no teto e
uma vista fora da distribuicao de treino: ele nao erra por ruido, erra por
estar adivinhando.

    Foi essa a causa do esqueleto torto, e nenhuma suavizacao consertava.
    O problema nao estava no codigo — estava na vista.

A camera do alto continua respondendo ONDE (homografia, 2 a 5 cm). As de
frente e de lado respondem COMO o corpo esta. Cada uma no que sabe.

QUADRO INTEIRO, SEM DETECCAO

Estas vistas sao dedicadas a UMA pessoa. Rodar YOLO nelas para achar quem ja
se sabe que esta ali seria trabalho repetido — e o YOLO custa 84 ms contra 26
do MediaPipe.

    Limitacao declarada: com duas pessoas em cena, estas vistas nao sabem
    qual e qual. Resolver exige re-identificacao por aparencia.

CUSTO MEDIDO

    frontal 26 ms   lateral 33 ms   (a lateral e 720p, a frontal 480p)
"""

from src.nucleo.erros import ModeloIndisponivel
from src.visao.observacao import Observacao
from src.visao.trabalhador import Trabalhador


class EstimadorDePose(Trabalhador):
    nome = "pose"

    def __init__(self, papel, a_cada_n=1, confianca=0.5):
        super().__init__(papel, a_cada_n)
        self.confianca = confianca
        self._pose = None

    def iniciar(self):
        # Reaproveita o Pose3D ja escrito e testado: conversao de eixos,
        # ancoragem no quadril e landmarks 2D de volta em pixels da imagem
        # inteira. Muda de lugar no futuro; nao muda de conteudo.
        try:
            from percepcao.pose3d import Pose3D
        except ImportError as e:
            raise ModeloIndisponivel("mediapipe/pose3d indisponivel",
                                     erro=str(e)) from e
        self.log.info("carregando pose", papel=self.papel)
        self._pose = Pose3D(confianca=self.confianca)
        self._aquecer()

    def _aquecer(self):
        """Estreia paga aqui. Medido em 10/08: 102 ms na primeira, 27 ms depois.

        Menor que os 15 s do YOLO, mas pela mesma razao — o delegate XNNPACK
        aloca na primeira chamada. E o mesmo principio: custo de estreia contado
        como custo de regime e diagnostico errado.

        Um quadro preto nao produz pose, e e isso que se quer: exercitar o
        caminho sem inventar observacao.
        """
        import time

        import numpy as np

        t = time.perf_counter()
        self._pose.estimar(np.zeros((240, 320, 3), dtype=np.uint8))
        self.log.info("pose aquecida",
                      ms=round((time.perf_counter() - t) * 1000))

    def _processar(self, frame):
        if self._pose is None:
            raise ModeloIndisponivel("pose nao carregada", papel=self.papel)

        juntas_3d, visivel, juntas_2d = self._pose.estimar(frame.imagem)
        if juntas_3d is None:
            # Nenhuma pose e resultado legitimo, nao falha. Devolver lista
            # vazia deixa a fusao trabalhar com a outra vista.
            return []

        return [Observacao(
            camera_id=frame.camera_id, papel=frame.papel,
            t_mono=frame.t_mono,
            juntas_3d=juntas_3d, juntas_2d=juntas_2d,
            conf_2d=visivel.astype(float) if visivel is not None else None,
            confianca=float(visivel.mean()) if visivel is not None else 0.0,
        )]

    def parar(self):
        if self._pose is not None:
            try:
                self._pose.fechar()
            except Exception:
                pass
            self._pose = None
