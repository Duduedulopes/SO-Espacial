"""
DetectorDePessoas — YOLO com rastreio, na camera do alto.

PAPEL NO SISTEMA

Esta e a unica vista que responde ONDE a pessoa esta. Ela produz caixa,
identidade e — se o modelo for `-pose` — os tornozelos, que dao o ponto do pe
com precisao muito maior que a base da caixa.

    Medido em 07/08: a base da caixa fica ~97 px abaixo dos tornozelos, de
    forma sistematica. Nao e ruido: e vies, e vies nao sai com filtro.

Por isso o padrao e `yolo11n-pose.pt` e nao `yolo11n.pt`, apesar de custar um
pouco mais. Os 17 pontos vem junto com a deteccao, sem uma segunda passada.

CUSTO MEDIDO NESTA MAQUINA

    yolo11n-pose  imgsz=320  ~84 ms/quadro em CPU, e 130 ms em regime (19/08)

E o gargalo do sistema, com folga. Na corrida de 19/08:

    camera do alto entrega          15,0 quadros/s
    o detector consome               7,7 quadros/s
    quadros descartados na fila    721 de 2743

Ou seja: metade do que a camera ve e jogada fora sem ser olhada, e o sistema
inteiro anda no passo deste arquivo.

QUAL MODELO RODA NAO ESTA ESCRITO AQUI

O padrao e `yolo11n-pose.pt` em PyTorch, mas `config/detector.json` pode
apontar para uma versao exportada — ONNX ou OpenVINO — que costuma render 2 a
3x em CPU pela mesma aritmetica e os mesmos pesos.

Qual das tres e a mais rapida DEPENDE DA MAQUINA, e por isso a escolha nao
esta no codigo: ela sai de `ferramentas/acelerar_detector.py`, que mede as
tres nos quadros reais e confere que as tres respondem a mesma coisa.

    Uma constante de desempenho escrita no codigo e uma medicao feita na
    maquina de outra pessoa.
"""

import json
from pathlib import Path

import numpy as np

from src.nucleo.erros import ModeloIndisponivel
from src.nucleo.log import Log
from src.visao.observacao import Observacao
from src.visao.trabalhador import Trabalhador

RAIZ = Path(__file__).resolve().parent.parent.parent
PADRAO = "yolo11n-pose.pt"


def modelo_escolhido(raiz=None):
    """O modelo que `ferramentas/acelerar_detector.py` mediu e escolheu.

    Ausente devolve o padrao em PyTorch, calado — quem nunca rodou a medicao
    nao tem nada errado a corrigir. Presente e ilegivel aparece no log, porque
    ai alguem mediu e o resultado se perdeu.

        arquivo ausente        nunca medi              esperado
        arquivo quebrado       medi e voce perdeu      defeito

    A mesma regra de `_ler_config` em `espacial/motor.py`, e ela existe
    porque `except Exception: return None` custou um dia inteiro em 11/08.

    O caminho e conferido: um `.onnx` que foi apagado nao pode virar um erro
    obscuro do AutoBackend no meio do primeiro quadro.
    """
    raiz = Path(raiz) if raiz else RAIZ
    caminho = raiz / "config" / "detector.json"
    if not caminho.exists():
        return PADRAO
    try:
        nome = json.loads(caminho.read_text(encoding="utf-8"))["modelo"]
    except Exception as e:
        Log("config").aviso(
            f"config/detector.json existe mas nao pode ser lida "
            f"({type(e).__name__}: {e}) — seguindo com {PADRAO}",
            erro=f"{type(e).__name__}: {e}")
        return PADRAO

    alvo = raiz / nome
    if not (alvo.exists() or Path(nome).exists()):
        Log("config").aviso(
            f"config/detector.json aponta para {nome}, que nao existe — "
            f"seguindo com {PADRAO}. Rode: "
            f"python ferramentas/acelerar_detector.py --gravar",
            modelo=nome)
        return PADRAO
    return str(alvo) if alvo.exists() else nome


class DetectorDePessoas(Trabalhador):
    nome = "detector"

    def __init__(self, papel="alto", modelo=None, imgsz=320,
                 conf=0.35, a_cada_n=1):
        super().__init__(papel, a_cada_n)
        self.modelo_nome = modelo if modelo is not None else modelo_escolhido()
        self.imgsz = imgsz
        self.conf = conf
        self._yolo = None

    def iniciar(self):
        """Carga separada do construtor: o sistema monta rapido e paga o
        custo do modelo uma vez, quando o motor sobe."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModeloIndisponivel("ultralytics nao instalado",
                                     erro=str(e)) from e
        self.log.info("carregando modelo", modelo=self.modelo_nome)
        self._yolo = YOLO(self.modelo_nome)
        self._aquecer()

    def _aquecer(self):
        """Paga a PRIMEIRA inferencia aqui, e nao no primeiro quadro real.

        MEDIDO EM 10/08: a primeira chamada custou 15,2 s. As seguintes, 60 ms.
        O Ultralytics faz na estreia coisas que nunca mais repete — checagem de
        AMP (que roda uma inferencia de teste), alocacao dos kernels, montagem
        do rastreador. Carregar o arquivo do modelo nao dispara nada disso.

        Sem aquecer, esses 15 s caem DENTRO do primeiro ciclo medido. O painel
        entao acusou 4,0 fps num sistema que roda a 16 — uma media envenenada
        por uma amostra. E a conclusao natural seria otimizar o que ja estava
        rapido.

            Custo de estreia contado como custo de regime e diagnostico errado.

        Mesma licao do `_aquecer()` da camera, em 08/08: julgar o brilho antes
        do sensor estabilizar fez o diagnostico virar a causa do defeito.

        Usa `predict` e nao `track`: aquecer nao pode criar estado de
        rastreamento, senao o primeiro quadro real ja nasce com historico
        inventado de uma imagem preta.
        """
        import time

        import numpy as np

        t = time.perf_counter()
        vazio = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._yolo.predict(vazio, imgsz=self.imgsz, verbose=False)
        self.log.info("modelo aquecido",
                      ms=round((time.perf_counter() - t) * 1000))

    def _processar(self, frame):
        if self._yolo is None:
            raise ModeloIndisponivel("modelo nao carregado", papel=self.papel)

        # persist=True: o rastreador precisa saber que este quadro continua o
        # anterior. Sem isso cada chamada recomeca e nenhum ID sobrevive.
        r = self._yolo.track(frame.imagem, persist=True, conf=self.conf,
                             classes=[0], imgsz=self.imgsz, verbose=False)[0]

        caixas, poses = r.boxes, r.keypoints
        if caixas is None or len(caixas) == 0:
            return []

        obs = []
        for k in range(len(caixas)):
            b = caixas[k]
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            tid = int(b.id[0]) if b.id is not None else -1

            j2d = c2d = None
            if poses is not None and k < len(poses):
                j2d = poses[k].xy[0].cpu().numpy()
                c2d = (poses[k].conf[0].cpu().numpy()
                       if poses[k].conf is not None else np.ones(17))

            obs.append(Observacao(
                camera_id=frame.camera_id, papel=frame.papel,
                t_mono=frame.t_mono,
                caixa=(x1, y1, x2, y2), id_externo=tid,
                confianca=float(b.conf[0]) if b.conf is not None else 0.0,
                juntas_2d=j2d, conf_2d=c2d,
            ))
        return obs
