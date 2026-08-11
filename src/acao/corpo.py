"""
AnalisadorDeCorpo — a CAMADA DO MEIO entre o que a camera ve e o que o boneco faz.

O BURACO QUE ESTE ARQUIVO FECHA

Ate hoje o fluxo tinha uma metade so:

    cameras -> velocidade do Kalman -> parado / andando / virando / agachado
    cameras -> ??????????????????? -> bracos: DESCONHECIDO fixo no codigo

`classificador.py` ja aceitava `rumo_corpo` e ja tinha lugar para os bracos.
Nunca chegou nada nos dois. Era uma assinatura pronta esperando uma fonte.

    Um parametro opcional que ninguem preenche e um recurso que nao existe.
                                                            — caderno, 10/08

Esta e a fonte.

O QUE ELE LE, E DE ONDE

Do esqueleto RELATIVO AO QUADRIL que o MediaPipe entrega, de UMA camera:

    rumo do corpo     linha dos ombros            -> frente/tras/lado
    estado do braco   pulso contra o ombro        -> ao lado/levantado/estendido
    ALTURA DA MAO     pulso contra o tornozelo    -> metros acima do chao

POR QUE ISTO NAO PRECISA DA FUSAO 3D

O MediaPipe world landmarks ja vem em METROS. Nao sao pixels a converter, nao
sao coordenadas normalizadas: sao metros, com origem no centro do quadril.

Entao a altura da mao acima do chao e uma SUBTRACAO:

    altura = z_do_pulso - z_do_tornozelo

Uma camera. Sem triangulacao, sem homografia, sem a fusao que em 10/08
desenhava o esqueleto deitado. E o ponto inteiro da arquitetura v3:

    o sistema para de depender do elo mais fraco.

A unica coisa que precisa ser desfeita antes e a inclinacao da lente, porque
o MediaPipe entrega os eixos alinhados com a CAMERA e nao com a gravidade.
Isso ja tem dono: `pose3d.desfazer_inclinacao`, a MESMA funcao que o
`ancorar_no_chao` usa. Nao ha segunda copia.

O QUE ELE SE RECUSA A RESPONDER

O MediaPipe SEMPRE devolve as 17 juntas, inclusive as que estao fora do
quadro. Quando a webcam pega do peito para cima, ele entrega tornozelos
extrapolados — numeros com a mesma cara dos medidos e nenhuma relacao com o
corpo real. Foi isso que produziu o amontoado de juntas de 10/08.

    Nao responder o que nao foi visto. Uma altura sem tornozelo visivel e
    mentira com aparencia de dado.

Cada resposta aqui pode ser `None` ou `DESCONHECIDO`, e sai assim sempre que
a junta que a sustenta nao foi vista.

CUSTO

Aritmetica sobre tres dezenas de numeros: dois produtos escalares por braco e
uma subtracao. Contra os 90 a 156 ms do detector, e ruido de medicao. Nao e
esta camada que deixa o sistema lento, e tira-la nao o deixaria rapido.
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from percepcao.pose3d import desfazer_inclinacao        # noqa: E402
from src.acao.angulos import (                          # noqa: E402
    concentracao, diferenca_angular, media_circular,
)
from src.acao.vocabulario import Braco                  # noqa: E402

# Indices do padrao COCO-17, que e o que `pose3d.MP_PARA_COCO` produz.
OMBRO_ESQ, OMBRO_DIR = 5, 6
COTOVELO_ESQ, COTOVELO_DIR = 7, 8
PULSO_ESQ, PULSO_DIR = 9, 10
QUADRIL_ESQ, QUADRIL_DIR = 11, 12
TORNOZELO_ESQ, TORNOZELO_DIR = 15, 16


def _visivel(marcas, *indices):
    """True so se TODAS as juntas pedidas foram realmente vistas.

    `marcas` chega ora como booleano, ora como float 0/1 (`conf_2d`), ora como
    None. None significa "sem informacao de visibilidade" — e sem informacao,
    a resposta honesta e nao afirmar que viu.
    """
    if marcas is None:
        return False
    m = np.asarray(marcas)
    return all(bool(m[i] > 0.5) for i in indices)


@dataclass
class LeituraDoCorpo:
    """O que UMA vista conseguiu dizer do corpo, neste instante.

    Todo campo e opcional de proposito. `None` aqui nao e falta de
    implementacao: e a recusa de responder sem base, que e a unica coisa que
    separa este sistema de um que inventa numeros bonitos.
    """

    rumo_corpo: float | None = None          # radianos, NO MUNDO
    rumo_corpo_camera: float | None = None   # radianos, no referencial da lente

    braco_esquerdo: str = Braco.DESCONHECIDO
    braco_direito: str = Braco.DESCONHECIDO

    # O NUMERO QUE INTERESSA AO NEGOCIO: metros acima do chao.
    # E com ele que "a mao entrou na prateleira entre 1,10 m e 1,35 m" vira
    # uma comparacao, e o produto sai do CADASTRO em vez de sair da imagem.
    altura_mao_esq: float | None = None
    altura_mao_dir: float | None = None

    altura_quadril: float | None = None
    motivo: str = ""

    def para_dicionario(self):
        def m(v):
            return None if v is None else round(v, 3)

        return {
            "rumo_corpo": m(self.rumo_corpo),
            "braco_esquerdo": self.braco_esquerdo,
            "braco_direito": self.braco_direito,
            "altura_mao_esq": m(self.altura_mao_esq),
            "altura_mao_dir": m(self.altura_mao_dir),
            "altura_quadril": m(self.altura_quadril),
        }


class EstimadorDeAzimute:
    """Descobre sozinho para que lado a camera esta virada.

    O PROBLEMA QUE ELE RESOLVE

    A linha dos ombros diz para onde o corpo aponta — mas no referencial da
    CAMERA. O Kalman diz para onde a pessoa anda — no referencial do MUNDO.
    Comparar os dois direto e comparar coisas medidas em sistemas diferentes,
    e o resultado seria um "andando de lado" que muda conforme a camera for
    reposicionada.

    Faltava uma constante: quanto a camera esta girada em relacao ao mundo.

    A IDEIA, QUE E A MESMA DO ESTIMADOR DE INCLINACAO

    Quem ANDA, anda para onde o corpo aponta. Ninguem caminha de lado por
    quinze metros. Entao, para quem esta andando:

        rumo_do_mundo (Kalman)  -  rumo_dos_ombros (camera)  =  giro da camera

    Isso e uma constante da montagem. Basta observar alguem andando e medir a
    diferenca. Nenhuma calibracao manual, nenhum transferidor, nenhum campo
    novo no JSON — e, quando voce mexer a camera de lugar, ele reaprende.

    POR QUE MEDIA CIRCULAR E NAO MEDIANA

    O `EstimadorDeInclinacao` usa mediana simples e esta certo: inclinacao de
    lente fica perto de zero e nunca da a volta. Azimute da: uma camera pode
    estar montada a 179 graus, e a mediana aritmetica de +179 e -179 devolve
    ZERO — a direcao exatamente oposta a verdadeira.

    ELE SE ABSTEM, COMO O FILTRO DE ALTURA

    Se as amostras nao concordam entre si, ele nao responde. E a licao de
    10/08: um filtro que nao consegue ajustar o proprio modelo deve se abster,
    nao chutar. Enquanto abstido, `rumo_corpo` sai `None` e a locomocao
    responde `andando` — que e a resposta honesta e a que o sistema ja dava.
    """

    def __init__(self, memoria=240, vel_minima=0.25, minimo_amostras=20,
                 concentracao_minima=0.75, desvio_maximo_rad=np.pi / 3):
        self.amostras = deque(maxlen=memoria)
        self.vel_minima = vel_minima
        self.minimo = minimo_amostras
        self.concentracao_minima = concentracao_minima
        self.desvio_maximo = desvio_maximo_rad
        self.valor = 0.0
        self.descartadas = 0

    def observar(self, rumo_ombros_camera, rumo_mundo, velocidade):
        """Alimenta uma amostra. Devolve True se ela foi aceita.

        So aceita quem anda ACIMA do limiar. Parado, o vetor velocidade do
        Kalman e quase todo ruido e sua DIRECAO gira loucamente enquanto o
        modulo mal se mexe — um vetor curto tem angulo mal definido. Aprender
        com ele seria aprender ruido e chamar de calibracao.
        """
        if (velocidade < self.vel_minima or rumo_ombros_camera is None
                or rumo_mundo is None):
            return False

        oferta = diferenca_angular(rumo_mundo, rumo_ombros_camera)

        # Depois de convergir, recusa amostra absurda. Sem isto, um unico
        # quadro em que o MediaPipe troca os ombros de lado (acontece quando a
        # pessoa esta de costas) entra na conta como se fosse medida boa.
        if self.confiavel and abs(diferenca_angular(oferta, self.valor)) > self.desvio_maximo:
            self.descartadas += 1
            return False

        self.amostras.append(oferta)
        if len(self.amostras) >= self.minimo:
            self.valor = media_circular(self.amostras)
        return True

    @property
    def concentracao(self):
        return concentracao(self.amostras)

    @property
    def confiavel(self):
        return (len(self.amostras) >= self.minimo
                and self.concentracao >= self.concentracao_minima)

    def para_o_mundo(self, rumo_camera):
        """Converte um rumo do referencial da lente para o do mundo."""
        if rumo_camera is None or not self.confiavel:
            return None
        return diferenca_angular(rumo_camera + self.valor, 0.0)

    @property
    def diagnostico(self):
        n = len(self.amostras)
        if n < self.minimo:
            return f"azimute aprendendo ({n}/{self.minimo})"
        estado = "ativo" if self.confiavel else "ABSTIDO"
        return (f"azimute {np.degrees(self.valor):+.0f}deg "
                f"conc={self.concentracao:.0%} {estado} ({n} amostras)")


class AnalisadorDeCorpo:
    """De juntas relativas para rumo do corpo, bracos e altura das maos.

    UMA VISTA POR VEZ, DE PROPOSITO

    Nao funde nada. Recebe o esqueleto de UMA camera e responde o que aquela
    vista consegue sustentar. Fundir era o caminho antigo, e foi ele que
    produziu o amontoado de meio metro em 10/08 — nao porque a matematica da
    fusao estivesse errada (ela foi testada com entrada limpa e acertou ao
    centimetro), mas porque cada vista inventava uma parte diferente e a fusao
    somava as duas invencoes.

    MEMORIA POR PESSOA

    A altura do quadril acima do chao e aprendida por rastro e guardada. Ela e
    o que permite responder a altura da mao nos quadros em que o tornozelo
    saiu do enquadramento — sem inventar, porque foi medida quando ele estava
    la. Some junto com o rastro.
    """

    def __init__(self,
                 levantado_acima=0.10,
                 estendido_alem=0.25,
                 tronco_minimo=0.15,
                 quadril_min=0.40, quadril_max=1.30,
                 altura_maxima=2.50,
                 memoria_quadril=120,
                 **kw_azimute):
        self.levantado_acima = levantado_acima
        self.estendido_alem = estendido_alem
        self.tronco_minimo = tronco_minimo
        self.quadril_min = quadril_min
        self.quadril_max = quadril_max
        self.altura_maxima = altura_maxima
        self.memoria_quadril = memoria_quadril

        self.azimute = EstimadorDeAzimute(**kw_azimute)
        self._quadris: dict[int, deque] = {}

    # ------------------------------------------------------------ principal
    def ler(self, pessoa_id, juntas_3d, visivel, inclinacao_rad=0.0,
            rumo_mundo=None, velocidade=0.0):
        """Devolve uma `LeituraDoCorpo` a partir de UMA vista.

        `juntas_3d`   (17,3) em metros, origem no quadril, eixos da CAMERA
        `visivel`     (17,) — quais juntas foram de fato vistas
        `rumo_mundo`  rumo do Kalman, para o azimute aprender
        """
        if juntas_3d is None:
            return LeituraDoCorpo(motivo="sem pose")

        j = desfazer_inclinacao(juntas_3d, inclinacao_rad)

        rumo_cam, lateral, frente = self._rumo_dos_ombros(j, visivel)
        if rumo_cam is not None:
            self.azimute.observar(rumo_cam, rumo_mundo, velocidade)

        altura_quadril = self._altura_do_quadril(pessoa_id, j, visivel)

        leitura = LeituraDoCorpo(
            rumo_corpo=self.azimute.para_o_mundo(rumo_cam),
            rumo_corpo_camera=rumo_cam,
            altura_quadril=altura_quadril,
        )

        if rumo_cam is None:
            leitura.motivo = "ombros nao vistos"
            return leitura
        if leitura.rumo_corpo is None:
            leitura.motivo = ("azimute abstido" if self.azimute.amostras
                              else "azimute sem amostras")

        leitura.braco_esquerdo, leitura.altura_mao_esq = self._ler_braco(
            j, visivel, OMBRO_ESQ, PULSO_ESQ, lateral, frente, altura_quadril)
        leitura.braco_direito, leitura.altura_mao_dir = self._ler_braco(
            j, visivel, OMBRO_DIR, PULSO_DIR, lateral, frente, altura_quadril)

        return leitura

    # ------------------------------------------------------------ rumo
    def _rumo_dos_ombros(self, j, visivel):
        """Para onde o CORPO aponta, no referencial da camera.

        POR QUE OS OMBROS E NAO A VELOCIDADE

        O rumo que o sistema conhecia ate hoje vinha de `arctan2(vy, vx)` — a
        direcao do DESLOCAMENTO. Por construcao, rumo e deslocamento eram a
        mesma coisa, e "andar de lado" ficava indistinguivel de "andar para
        frente". A linha dos ombros e independente de para onde a pessoa anda,
        e e por isso que ela separa os dois.

        A GEOMETRIA

        Eixos ja corrigidos pela gravidade: x direita da imagem, y para longe
        da lente, z para cima. O corpo aponta perpendicular a linha dos
        ombros, no plano horizontal — girar o vetor dos ombros em -90 graus:

            (dx, dy)  ->  (dy, -dx)

        Conferindo com o caso obvio: pessoa DE FRENTE para a lente tem o ombro
        esquerdo dela na direita da imagem, entao dx > 0 e dy = 0; o giro da
        (0, -dx), que aponta para a lente. Correto.

        Nao ha ambiguidade de 180 graus porque o MediaPipe rotula os ombros
        anatomicamente: ele sabe qual e o esquerdo mesmo vendo a pessoa de
        costas. Quando ele erra, quem filtra e o limiar de desvio do azimute.
        """
        if not _visivel(visivel, OMBRO_ESQ, OMBRO_DIR):
            return None, None, None

        v = j[OMBRO_ESQ] - j[OMBRO_DIR]
        dx, dy = float(v[0]), float(v[1])
        largura = np.hypot(dx, dy)

        # Ombros quase alinhados com a profundidade: a projecao horizontal
        # some e o angulo passa a ser definido por ruido. E o mesmo motivo
        # geometrico pelo qual o giro exige velocidade — vetor curto tem
        # angulo mal definido.
        if largura < self.tronco_minimo:
            return None, None, None

        lateral = np.array([dx / largura, dy / largura, 0.0])
        frente = np.array([dy / largura, -dx / largura, 0.0])
        return float(np.arctan2(-dx, dy)), lateral, frente

    # ------------------------------------------------------------ quadril
    def _altura_do_quadril(self, pessoa_id, j, visivel):
        """Quanto o quadril desta pessoa esta acima do chao, em metros.

        O quadril e a origem do esqueleto (z=0 por construcao), entao a altura
        dele e menos o z do tornozelo mais baixo.

        POR QUE ISTO E GUARDADO E NAO RECALCULADO SEMPRE

        Quando as pernas saem do quadro, o MediaPipe segue entregando
        tornozelos — extrapolados. Recalcular naquele quadro daria um numero
        com a mesma cara de medido e nenhuma relacao com o corpo.

        Entao mede-se enquanto o tornozelo esta VISIVEL, guarda-se a mediana,
        e nos quadros sem perna usa-se o que foi medido antes. Se nunca houve
        um quadro com perna a vista, a resposta e None e a altura da mao nao e
        respondida. Nao ha invencao em lugar nenhum deste caminho.

        A mediana tambem protege do agachamento: agachar aproxima o quadril do
        chao por alguns quadros, e mediana ignora valor esquisito.
        """
        historico = self._quadris.setdefault(
            pessoa_id, deque(maxlen=self.memoria_quadril))

        if _visivel(visivel, TORNOZELO_ESQ, TORNOZELO_DIR):
            z = -float(min(j[TORNOZELO_ESQ][2], j[TORNOZELO_DIR][2]))
            # Guarda de proporcao humana: quadril de adulto fica perto de
            # 0,95 m e de crianca perto de 0,60. Fora da faixa e reconstrucao
            # ruim, nao pessoa incomum.
            if self.quadril_min <= z <= self.quadril_max:
                historico.append(z)

        if not historico:
            return None
        return float(np.median(historico))

    # ------------------------------------------------------------ bracos
    def _ler_braco(self, j, visivel, i_ombro, i_pulso, lateral, frente,
                   altura_quadril):
        """Devolve (estado do vocabulario, altura da mao em metros ou None).

        A CLASSIFICACAO, NO REFERENCIAL DO CORPO

        Medir o pulso contra o ombro DELA, e nao contra a imagem, e o que faz
        a resposta valer igual com a pessoa virada para qualquer lado.

            levantado   pulso acima do ombro          -> prateleira alta
            estendido   pulso bem a frente do ombro   -> alcancando
            ao lado     nenhum dos dois

        `levantado` ganha de `estendido` quando os dois valem, porque e a
        altura que responde a pergunta comercial — qual prateleira.

        O QUE O VOCABULARIO NAO TEM, DE PROPOSITO

        Nao ha "abaixado". Pegar de uma prateleira baixa aparece como
        `ao_lado` com altura de mao pequena — e e a ALTURA que responde,
        nao o rotulo. Acrescentar estados ao vocabulario fechado e decisao de
        arquitetura, nao conveniencia de implementacao.
        """
        if not _visivel(visivel, i_ombro, i_pulso):
            return Braco.DESCONHECIDO, None

        d = j[i_pulso] - j[i_ombro]
        subida = float(d[2])
        avanco = float(np.dot(d, frente))

        if subida > self.levantado_acima:
            estado = Braco.LEVANTADO
        elif avanco > self.estendido_alem:
            estado = Braco.ESTENDIDO
        else:
            estado = Braco.AO_LADO

        return estado, self._altura_da_mao(j, i_pulso, altura_quadril)

    def _altura_da_mao(self, j, i_pulso, altura_quadril):
        """Metros do chao ate o pulso. `None` quando nao ha base para dizer.

        O quadril e a origem, entao a altura do pulso e a altura do quadril
        mais o quanto o pulso esta acima dele. Se a altura do quadril nunca
        foi medida, nao ha resposta — e o correto e dizer isso.
        """
        if altura_quadril is None:
            return None

        altura = altura_quadril + float(j[i_pulso][2])

        # Guarda final. Uma mao a 3 metros do chao ou 40 cm abaixo dele nao e
        # uma pessoa incomum: e reconstrucao ruim que passou por todos os
        # filtros. Melhor nao responder que responder um numero impossivel.
        if not -0.15 <= altura <= self.altura_maxima:
            return None
        return max(0.0, altura)

    # ------------------------------------------------------------ ciclo
    def esquecer(self, vivos):
        for pid in list(self._quadris):
            if pid not in vivos:
                del self._quadris[pid]

    @property
    def diagnostico(self):
        return self.azimute.diagnostico
