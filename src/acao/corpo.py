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
    concentracao, diferenca_angular, media_circular, moda_circular,
)
from src.acao.vocabulario import Braco                  # noqa: E402

# Indices do padrao COCO-17, que e o que `pose3d.MP_PARA_COCO` produz.
OMBRO_ESQ, OMBRO_DIR = 5, 6
COTOVELO_ESQ, COTOVELO_DIR = 7, 8
PULSO_ESQ, PULSO_DIR = 9, 10
QUADRIL_ESQ, QUADRIL_DIR = 11, 12
JOELHO_ESQ, JOELHO_DIR = 13, 14
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

    # ALTURA DO QUADRIL AGORA, sem mediana. Duas perguntas diferentes.
    #
    # `altura_quadril` e a mediana das ultimas amostras: quanto esta pessoa
    # mede EM PE. Ela e a referencia para a altura da mao, e por isso precisa
    # ignorar os quadros em que a pessoa agachou — mediana existe ali para
    # jogar fora valor esquisito.
    #
    # Mas "a pessoa agachou" e exatamente a informacao que a mediana apaga.
    # Perguntar o agachamento ao numero que foi construido para nao senti-lo
    # nao daria resposta nenhuma.
    #
    #     O mesmo dado, com dois usos, precisa de duas leituras. Reaproveitar
    #     a suavizacao de um no outro apaga justamente o sinal procurado.
    altura_quadril_agora: float | None = None

    # A ALTURA DA MAO FOI MEDIDA, OU ESTIMADA?
    #
    # MEDIDO EM 11/08: com a area de 1,4 m, NENHUMA camera enxerga os pes —
    # frontal 0%, lateral 0%. E nao e ajuste de enquadramento: uma webcam de
    # ~60 graus a 1,4 m cobre cerca de 1,6 m de altura, e uma pessoa em pe tem
    # 1,75 m. Nao cabe. So da para escolher entre ver os ombros ou ver os pes.
    #
    # Sem tornozelo nao ha chao no referencial do corpo, e a altura da mao —
    # o numero que decide qual prateleira — ficaria sem resposta para sempre
    # naquela sala.
    #
    # A saida NAO e inventar um chao. E usar outra medida que existe: o
    # MediaPipe entrega o tronco em METROS, e a proporcao quadril/tronco do
    # corpo humano e estavel. Isso e um MODELO, nao uma medicao — e a diferenca
    # tem que viajar junto com o numero.
    #
    #     Responder por modelo e legitimo. Responder por modelo sem dizer que
    #     e modelo e o defeito que este projeto inteiro combate.
    altura_medida: bool = False

    motivo: str = ""

    # VERTICALIDADE DA COXA: quanto o vetor quadril->joelho aponta para baixo.
    #
    # 1.0 = coxa vertical (em pe).  ~0.2 = coxa horizontal (agachado).
    #
    # POR QUE ESTE SINAL GANHOU DOS OUTROS DOIS
    #
    # MEDIDO EM 11/08, com o roteiro travado: `agachar` NUNCA foi reconhecido,
    # com 36% dos quadros sem leitura nenhuma. As duas fontes anteriores
    # falharam por motivos diferentes, e os dois sao estruturais:
    #
    #     altura da CAIXA (camera do alto)
    #         de cima quase nao se ve mudanca de estatura — a caixa e
    #         dominada pela pegada da pessoa no chao.
    #
    #     altura do QUADRIL acima do chao (camera frontal)
    #         precisa do tornozelo visivel para saber onde e o chao. E quem
    #         agacha tira as pernas do enquadramento justamente ali.
    #
    # A coxa nao depende de nenhum dos dois. Ela e uma razao entre duas juntas
    # vizinhas, e vizinhas costumam ser vistas juntas: se o joelho aparece, o
    # quadril aparece.
    #
    #     Um sinal que precisa do chao morre quando o chao sai do quadro. A
    #     geometria interna do corpo nao tem esse problema.
    #
    # E ele e adimensional: nao precisa de metros, de calibracao, nem de saber
    # a altura da pessoa. Funciona igual para adulto e para crianca.
    verticalidade_coxa: float | None = None

    @property
    def encolhimento(self):
        """Quanto a pessoa encolheu. 1.0 = em pe, ~0.2 = agachada.

        Prefere a coxa; cai na altura do quadril quando o joelho nao aparece.
        As duas medem a mesma coisa por caminhos independentes, e a segunda
        existe porque nenhuma vista ve tudo sempre.
        """
        if self.verticalidade_coxa is not None:
            return self.verticalidade_coxa
        if not self.altura_quadril or self.altura_quadril_agora is None:
            return None
        return self.altura_quadril_agora / self.altura_quadril

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
                 concentracao_minima=0.75, desvio_maximo_rad=np.pi / 3,
                 maioria_minima=0.55):
        self.amostras = deque(maxlen=memoria)
        self.vel_minima = vel_minima
        self.minimo = minimo_amostras
        self.concentracao_minima = concentracao_minima
        self.desvio_maximo = desvio_maximo_rad
        # MAIORIA, E NAO UNANIMIDADE.
        #
        # Num roteiro real a pessoa anda de re e de lado em alguns passos, e
        # essas amostras ficam 180 ou 90 graus fora. Exigir concentracao alta
        # de TODAS as amostras seria exigir um roteiro que so tem caminhada
        # para frente — e nenhum roteiro util e assim.
        #
        # 55% e o suficiente para haver um grupo dominante claro; abaixo disso
        # nao ha maioria e nao ha resposta.
        self.maioria_minima = maioria_minima
        self.recalcular_a_cada = 10
        self._desde_o_calculo = 0
        self.valor = 0.0
        self.maioria = 0.0
        self.descartadas = 0

        # OFFSET CALIBRADO: MEDIDO DE PROPOSITO, MANDA NO APRENDIDO.
        #
        # MEDIDO EM 11/08: com deslocamento no lugar de velocidade, o estimador
        # finalmente convergiu — e convergiu para o grupo ERRADO. As direcoes
        # sairam sistematicamente giradas: `andar_frente` lido como
        # `andando_tras`, `andar_esquerda` como `andando_tras`.
        #
        # A causa nao e o algoritmo. E que a hipotese de que ele inteiro
        # depende — "quem anda, olha para onde vai" — E FALSA nesta sala. Numa
        # area de 1,4 m diante de um computador, a pessoa se desloca olhando
        # para a tela, para a camera, para onde a voz esta falando. O corpo
        # quase nunca aponta para onde os pes vao.
        #
        # Antes ele se abstinha, porque as amostras ficavam espalhadas. Ao
        # aceitar mais amostras, o grupo majoritario passou a ser o errado.
        #
        #     Aumentar a amostra de uma hipotese falsa nao a torna verdadeira;
        #     torna o erro confiante.
        #
        # O calibrado tem prioridade ABSOLUTA sobre o aprendido, e nao a media
        # dos dois. Misturar uma medida honesta com um aprendizado viciado
        # produz um terceiro numero pior que a medida — e ainda esconde qual
        # dos dois estava errado.
        self.calibrado = None

    def observar(self, rumo_ombros_camera, rumo_mundo, velocidade):
        """Alimenta uma amostra. Devolve True se ela foi aceita.

        `velocidade` aqui e uma MEDIDA DE BASE, nao necessariamente a
        velocidade instantanea — ver `DirecaoPorDeslocamento`. O que ela
        precisa garantir e que o vetor cujo angulo se vai usar seja longo o
        bastante para ter angulo bem definido.

        MEDIDO EM 11/08: com velocidade instantanea, a caminhada real do
        Eduardo teve mediana de 0,23 m/s — ABAIXO do limiar de 0,25. O azimute
        recusou quase tudo e aprendeu dos poucos picos que passaram; picos sao
        ruido. E baixar o limiar nao resolvia: `parado` teve pico de 0,23 m/s
        so de tremor do Kalman.

            Naquele espaco, caminhada e ruido se sobrepoem em VELOCIDADE. Em
            DESLOCAMENTO nao se sobrepoem: andar deu 0,68 m, tremer deu 0,04.
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
        self._desde_o_calculo += 1

        # A MODA CUSTA O(n^2) E NAO PRECISA SER RECALCULADA A CADA AMOSTRA.
        #
        # Com 240 amostras sao ~57 mil comparacoes. Rodando a cada quadro isso
        # sozinho multiplicou por seis o custo desta camada — que existe
        # justamente por nao custar nada.
        #
        # Uma amostra nova entre 240 nao move a moda. Recalcular a cada 10
        # mantem a resposta praticamente igual por um decimo do custo, e o
        # atraso maximo e de 10 quadros, ou cerca de um segundo — irrelevante
        # para uma constante de montagem que nao muda enquanto ninguem mexe
        # na camera.
        pronto = len(self.amostras) >= self.minimo
        if pronto and (self._desde_o_calculo >= self.recalcular_a_cada
                       or not self.maioria):
            self.valor, self.maioria = moda_circular(self.amostras)
            self._desde_o_calculo = 0
        return True

    @property
    def concentracao(self):
        return concentracao(self.amostras)

    @property
    def confiavel(self):
        if self.calibrado is not None:
            return True
        return (len(self.amostras) >= self.minimo
                and self.maioria >= self.maioria_minima)

    @property
    def offset(self):
        """O giro em uso. Calibrado ganha do aprendido, sempre."""
        return self.calibrado if self.calibrado is not None else self.valor

    def para_o_mundo(self, rumo_camera):
        """Converte um rumo do referencial da lente para o do mundo."""
        if rumo_camera is None or not self.confiavel:
            return None
        return diferenca_angular(rumo_camera + self.offset, 0.0)

    @property
    def bimodal(self):
        """As amostras estao em DOIS grupos opostos? Entao ha uma causa unica.

        MEDIDO EM 11/08: tres execucoes seguidas, mesma camera, sem ninguem
        mexer nela — o azimute deu +7, +85 e -123 graus, com concentracao entre
        35% e 83%. Tres respostas diferentes para uma montagem que nao mudou.

        Nao era ruido aleatorio: era ANDAR DE RE. Quem vai e volta no mesmo
        eixo sem virar o corpo produz metade das amostras dizendo "o corpo
        aponta para X e ela anda para +X" e a outra metade dizendo "o corpo
        aponta para X e ela anda para -X". Os dois grupos ficam 180 graus
        separados, e a media circular de dois grupos opostos e indefinida —
        ela cai onde o acaso da contagem mandar.

        O estimador ja se abstinha, e estava certo. O que faltava era DIZER
        POR QUE, porque "ABSTIDO" sozinho manda procurar defeito no codigo, e
        o conserto era virar o corpo ao voltar.

            Abster-se sem explicar transfere o problema para quem le.

        A conta: se dobrar todos os angulos concentra o que estava espalhado,
        os dados sao bimodais a 180 graus. E o teste classico de eixo contra
        direcao em estatistica circular — dobrar leva +X e -X para o mesmo
        lugar.
        """
        if len(self.amostras) < self.minimo:
            return False
        dobrados = [2 * a for a in self.amostras]
        return concentracao(dobrados) > self.concentracao_minima \
            and self.concentracao < self.concentracao_minima

    @property
    def diagnostico(self):
        n = len(self.amostras)
        if self.calibrado is not None:
            texto = (f"azimute {np.degrees(self.calibrado):+.0f}deg "
                     f"CALIBRADO")
            # Mostrar tambem o que ele TERIA aprendido nao e curiosidade: se
            # os dois discordarem muito, ou a calibracao envelheceu (alguem
            # mexeu na camera) ou o ambiente mudou. E o unico jeito de a
            # calibracao ser questionada depois de gravada.
            if n >= self.minimo:
                erro = np.degrees(abs(diferenca_angular(
                    self.valor, self.calibrado)))
                texto += (f"   (aprenderia {np.degrees(self.valor):+.0f}deg, "
                          f"{erro:.0f}deg de diferenca)")
                if erro > 45 and self.maioria >= self.maioria_minima:
                    texto += "  [DISCORDAM: a camera foi movida?]"
            return texto
        if n < self.minimo:
            return f"azimute aprendendo ({n}/{self.minimo})"
        estado = "ativo" if self.confiavel else "ABSTIDO"
        texto = (f"azimute {np.degrees(self.valor):+.0f}deg "
                 f"maioria={self.maioria:.0%} {estado} ({n} amostras)")
        if not self.confiavel and self.bimodal:
            texto += "  [dois grupos opostos e nenhum e maioria]"
        elif not self.confiavel:
            texto += "  [amostras espalhadas: pouca caminhada util]"
        return texto


def rumo_do_alto(juntas_2d, conf, para_metros, H, largura_minima=0.15):
    """Rumo do corpo NO MUNDO, direto da camera de cima. Sem azimute.

    A SIMPLIFICACAO QUE APAGA O PROBLEMA INTEIRO

    Ate 11/08 o rumo do corpo saia da linha dos ombros do MediaPipe, medida no
    referencial da LENTE frontal. Para virar rumo de mundo faltava uma
    constante — o azimute — e ela nunca convergiu:

        aprendida sozinha    convergiu para o grupo errado, 180 graus fora
        calibrada a mao      tres travessias discordaram em 105 e 148 graus

    A causa da segunda e geometrica. No referencial da camera, a linha dos
    ombros de quem esta DE PERFIL deita sobre o eixo de PROFUNDIDADE — e a
    profundidade e o eixo mais fraco do MediaPipe, o mesmo que em 10/08
    desenhou o esqueleto deitado. O rumo virava ruido justamente quando a
    pessoa andava de lado, que e o caso que se queria medir.

    VISTA DE CIMA, ESSE PROBLEMA NAO EXISTE

    A camera do alto ja roda `yolo11n-pose.pt` e entrega os dois ombros em
    pixels. De cima, a linha dos ombros esta DEITADA NO PLANO DO CHAO — e o
    plano do chao e exatamente o que a homografia sabe converter em metros.

        Nao ha eixo de profundidade envolvido. Nao ha constante a aprender.
        Nao ha calibracao a fazer.

    O rumo sai em coordenadas de mundo direto, do mesmo jeito que a POSICAO já
    saía desde o bloco 1.

        Quando uma constante nao converge por dois caminhos diferentes, vale
        perguntar se ela precisa existir.

    O QUE SE PAGA, DECLARADO

    Os ombros ficam cerca de 1,4 m ACIMA do chao, e a homografia mapeia o
    CHAO. Cada ombro sai deslocado radialmente a partir do ponto sob a camera.
    Como os dois se deslocam quase igual, a DIRECAO do segmento sobrevive; o
    erro cresce conforme a pessoa se afasta do centro do quadro.

    Numa area de 1,4 m sob a camera, isso e pequeno. Numa loja grande, teria
    que ser conferido — e a forma de conferir e a que ja existe: comparar com
    o rumo do deslocamento de quem anda em linha reta.
    """
    if juntas_2d is None or not _visivel(conf, OMBRO_ESQ, OMBRO_DIR):
        return None

    esq = para_metros(H, *juntas_2d[OMBRO_ESQ])
    dir_ = para_metros(H, *juntas_2d[OMBRO_DIR])
    dx, dy = esq[0] - dir_[0], esq[1] - dir_[1]

    # Ombros colados sao reconstrucao ruim: a largura no chao nao pode
    # encolher, so girar. Mesma guarda do caminho antigo, mesma razao — vetor
    # curto tem angulo mal definido.
    if float(np.hypot(dx, dy)) < largura_minima:
        return None

    # O corpo aponta perpendicular a linha dos ombros. Conferindo com o caso
    # obvio: pessoa virada para +x tem o ombro esquerdo em +y, entao
    # (dx, dy) = (0, +1) e o giro de -90 graus devolve (+1, 0). Correto.
    return float(np.arctan2(-dx, dy))


class DirecaoPorDeslocamento:
    """Para onde a pessoa ANDOU, medido por onde ela chegou.

    O PROBLEMA, MEDIDO EM 11/08 NUM ESPACO DE 1,4 m

        caminhada real do Eduardo   v mediana 0,23 m/s
        ruido do Kalman parado      v maxima  0,23 m/s

    Os dois se sobrepoem. Nao ha limiar de velocidade que separe um do outro
    naquela sala — subir recusa a caminhada, descer aceita o tremor. O azimute
    ficou com 71 amostras e 29% de maioria, e se absteve.

    A SAIDA E TROCAR A GRANDEZA, NAO O LIMIAR

        andar 4 segundos    deslocamento liquido 0,68 m
        tremer no lugar     deslocamento liquido 0,04 m

    Dezessete vezes de diferenca, onde a velocidade dava empate. E a razao e
    geometrica, nao empirica: o ruido do Kalman e centrado em zero e se CANCELA
    ao longo de uma janela; o deslocamento de quem anda se ACUMULA.

        Ruido nao vai a lugar nenhum. Caminhada vai.

    E o angulo tambem melhora de graca: a direcao de um vetor de 0,68 m e bem
    definida; a de um vetor de 0,02 m por quadro e quase toda ruido. Mesmo
    motivo pelo qual o giro exige velocidade alta desde 10/08 — vetor curto tem
    angulo mal definido.

    O QUE SE PAGA

    Meia janela de atraso. Para uma constante de montagem que so muda quando
    alguem mexe na camera, isso nao custa nada.
    """

    # A JANELA E O QUE DEFINE O LIMIAR DE VERDADE, E ELA PRECISA SER LONGA.
    #
    # Exigir 0,25 m numa janela de 1 s e exigir 0,25 m/s — exatamente o limiar
    # que este objeto existe para contornar. A primeira versao fez isso e os
    # testes reprovaram na hora, com a caminhada de 0,23 m/s recusada de novo.
    #
    #     Trocar a grandeza sem trocar a janela nao troca nada.
    #
    # Com 2 s, o piso efetivo cai para 0,125 m/s — abaixo dos 0,23 medidos — e
    # o ruido tem o dobro de tempo para se cancelar: medido, ele acumula cerca
    # de 4 cm em dois segundos contra os 46 cm de quem anda.
    def __init__(self, janela_s=2.0, deslocamento_minimo=0.25):
        self.janela_s = janela_s
        self.minimo = deslocamento_minimo
        self._trilhas = {}

    def observar(self, pessoa_id, x, y, t):
        """Devolve (rumo, deslocamento) da janela, ou (None, 0.0)."""
        trilha = self._trilhas.setdefault(pessoa_id, deque(maxlen=120))
        trilha.append((t, float(x), float(y)))

        while len(trilha) > 2 and t - trilha[0][0] > self.janela_s:
            trilha.popleft()

        if len(trilha) < 3:
            return None, 0.0

        _, x0, y0 = trilha[0]
        dx, dy = x - x0, y - y0
        d = float(np.hypot(dx, dy))
        if d < self.minimo:
            return None, d
        return float(np.arctan2(dy, dx)), d

    def esquecer(self, vivos):
        for pid in list(self._trilhas):
            if pid not in vivos:
                del self._trilhas[pid]


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
                 tronco_minimo=0.15, coxa_minima=0.15,
                 quadril_por_tronco=1.83, tronco_vertical_minimo=0.25,
                 quadril_min=0.40, quadril_max=1.30,
                 altura_maxima=2.50,
                 memoria_quadril=120,
                 **kw_azimute):
        self.levantado_acima = levantado_acima
        self.estendido_alem = estendido_alem
        self.tronco_minimo = tronco_minimo
        self.coxa_minima = coxa_minima
        self.quadril_por_tronco = quadril_por_tronco
        self.tronco_vertical_minimo = tronco_vertical_minimo
        self.quadril_min = quadril_min
        self.quadril_max = quadril_max
        self.altura_maxima = altura_maxima
        self.memoria_quadril = memoria_quadril

        self.azimute = EstimadorDeAzimute(**kw_azimute)
        self._quadris: dict[int, deque] = {}

    # ------------------------------------------------------------ principal
    def ler(self, pessoa_id, juntas_3d, visivel, inclinacao_rad=0.0,
            rumo_mundo=None, velocidade=0.0):
        """Atalho para UMA vista. Ver `ler_varias` para o caminho completo."""
        r = self._ler_uma(pessoa_id, juntas_3d, visivel, inclinacao_rad,
                          rumo_mundo, velocidade)
        return r if r is not None else LeituraDoCorpo(motivo="sem pose")

    def _ler_uma(self, pessoa_id, juntas_3d, visivel, inclinacao_rad=0.0,
                 rumo_mundo=None, velocidade=0.0):
        """Devolve uma `LeituraDoCorpo` a partir de UMA vista.

        `juntas_3d`   (17,3) em metros, origem no quadril, eixos da CAMERA
        `visivel`     (17,) — quais juntas foram de fato vistas
        `rumo_mundo`  rumo do Kalman, para o azimute aprender
        """
        if juntas_3d is None:
            return None

        j = desfazer_inclinacao(juntas_3d, inclinacao_rad)

        rumo_cam, lateral, frente = self._rumo_dos_ombros(j, visivel)
        if rumo_cam is not None:
            self.azimute.observar(rumo_cam, rumo_mundo, velocidade)

        altura_quadril, quadril_agora = self._altura_do_quadril(
            pessoa_id, j, visivel)
        medida = bool(self._quadris.get(pessoa_id))

        leitura = LeituraDoCorpo(
            rumo_corpo=self.azimute.para_o_mundo(rumo_cam),
            rumo_corpo_camera=rumo_cam,
            altura_quadril=altura_quadril,
            altura_quadril_agora=quadril_agora,
            altura_medida=medida,
            verticalidade_coxa=self._verticalidade_da_coxa(j, visivel),
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

        agora = None
        if _visivel(visivel, TORNOZELO_ESQ, TORNOZELO_DIR):
            z = -float(min(j[TORNOZELO_ESQ][2], j[TORNOZELO_DIR][2]))
            # Guarda de proporcao humana: quadril de adulto fica perto de
            # 0,95 m e de crianca perto de 0,60. Fora da faixa e reconstrucao
            # ruim, nao pessoa incomum.
            #
            # A faixa e generosa no piso justamente para deixar o agachamento
            # passar: quem agacha poe o quadril perto de 0,45 m, e recusar
            # isso apagaria o sinal antes de ele ser lido.
            if self.quadril_min <= z <= self.quadril_max:
                agora = z
                historico.append(z)

        if not historico:
            # SEM PE NENHUM: estima pelo tronco, e diz que estimou.
            return self._quadril_pelo_tronco(j, visivel), agora

        # A MEDIANA E O PADRAO EM PE, E A MEMORIA E LONGA DE PROPOSITO.
        #
        # Com 120 amostras a ~10 fps, sao 12 s de historia. Um agachamento de
        # 6 s move a mediana pouco; ficar agachado meio minuto acabaria
        # movendo. E o preco aceito: a alternativa seria aprender so quando o
        # classificador dissesse "em pe", e ai o estimador dependeria do
        # resultado que ele mesmo alimenta.
        return float(np.median(historico)), agora

    def _quadril_pelo_tronco(self, j, visivel):
        """Altura do quadril a partir do comprimento do TRONCO, em metros.

        A CONTA, E DE ONDE ELA VEM

        Antropometria de adultos, medidas classicas em fracao da estatura:

            quadril (trocanter)   0,53
            ombro (acromio)       0,82
            tronco = 0,82 - 0,53 = 0,29

        Logo `altura_do_quadril = (0,53 / 0,29) x tronco = 1,83 x tronco`.

        O ERRO QUE ISTO CARREGA, DITO ANTES DE ALGUEM PERGUNTAR

        A razao varia cerca de 8% entre adultos — pernas mais longas ou mais
        curtas para o mesmo tronco. Num quadril de 0,95 m isso da +-8 cm.

        Para separar prateleiras de 25 cm, serve. Para dizer que a mao estava a
        1,18 e nao a 1,26, nao serve. E por isso que o resultado sai marcado:
        quem consumir decide se aquele erro cabe na decisao dele.

        A MEDICAO CONTINUA GANHANDO SEMPRE QUE EXISTE

        Isto so entra quando NENHUM quadro teve tornozelo visivel. Um unico
        quadro com pe a vista ja produz a mediana medida, e ela e melhor.
        """
        if not _visivel(visivel, OMBRO_ESQ, OMBRO_DIR,
                        QUADRIL_ESQ, QUADRIL_DIR):
            return None

        ombros = (j[OMBRO_ESQ] + j[OMBRO_DIR]) / 2
        quadris = (j[QUADRIL_ESQ] + j[QUADRIL_DIR]) / 2
        tronco = float(ombros[2] - quadris[2])

        # Tronco encurtado e pessoa curvada ou reconstrucao ruim, e nos dois
        # casos a proporcao nao vale. Melhor nao responder.
        if tronco < self.tronco_vertical_minimo:
            return None

        altura = self.quadril_por_tronco * tronco
        if not self.quadril_min <= altura <= self.quadril_max:
            return None
        return altura

    # ------------------------------------------------------------ coxa
    def _verticalidade_da_coxa(self, j, visivel):
        """Quanto a coxa aponta para baixo. 1 = em pe, ~0.2 = agachado.

        A GEOMETRIA, EM UMA LINHA

            em pe       o joelho fica quase 45 cm ABAIXO do quadril
            agachado    o joelho sobe e avanca; a coxa fica quase horizontal

        Dividir a queda vertical pelo comprimento da coxa da uma razao entre 0
        e 1 que nao depende do tamanho da pessoa, da distancia a camera, nem de
        haver chao no quadro.

        USA OS DOIS LADOS, E FICA COM O MAIOR

        Quem agacha costuma pousar um joelho antes do outro, e quem se abaixa
        para pegar algo no chao frequentemente estende uma perna. A perna mais
        ESTICADA e a que descreve melhor a postura do tronco — e ficar com o
        menor faria qualquer passada larga parecer agachamento.
        """
        razoes = []
        for quadril, joelho in ((QUADRIL_ESQ, JOELHO_ESQ),
                                (QUADRIL_DIR, JOELHO_DIR)):
            if not _visivel(visivel, quadril, joelho):
                continue
            coxa = j[quadril] - j[joelho]
            comprimento = float(np.linalg.norm(coxa))
            # Coxa curta demais e reconstrucao ruim, nao anatomia: o segmento
            # nao pode encolher: ele so pode GIRAR.
            if comprimento < self.coxa_minima:
                continue
            razoes.append(float(coxa[2]) / comprimento)

        if not razoes:
            return None
        return max(0.0, min(1.0, max(razoes)))

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

    # ------------------------------------------------------------ varias vistas
    def ler_varias(self, pessoa_id, vistas, inclinacao_rad=0.0,
                   rumo_mundo=None, velocidade=0.0, quadril_do_alto=None,
                   rumo_do_alto=None):
        """Le TODAS as vistas e combina cada resposta com quem conseguiu da-la.

        O PROBLEMA MEDIDO EM 11/08

        Com a frontal sozinha, levantar o braco levava 9 a 10 s para ser
        reconhecido e lia `ao_lado` em 65 a 87% dos quadros. BAIXAR levava 2 s.

        Essa assimetria e a assinatura de mao saindo do quadro: a webcam do
        notebook pega do peito para cima, o pulso levantado sobe alem da borda,
        e o MediaPipe entrega um pulso extrapolado — para baixo, que e o
        palpite mais provavel dele. Baixando, a mao volta ao quadro e a leitura
        acerta na hora.

        E a camera lateral estava entregando 100% de pose e nao era consultada,
        porque a escolha da vista era uma ORDEM FIXA: frontal, senao lateral.

            Preferencia fixa escolhe a vista antes de saber o que se quer ver.
            A pergunta certa nao e "qual camera e melhor", e "qual delas viu
            ESTA junta".

        COMO COMBINA

        Cada braco vem da vista que enxergou aquele pulso. O rumo do corpo e a
        postura vem da primeira que respondeu. Nada e mediado entre vistas —
        media de duas leituras discordantes produz uma terceira que nao
        corresponde a nenhuma, que foi o amontoado de juntas de 10/08.

        O AZIMUTE APRENDE DE UMA VISTA SO

        Ele mede o giro DAQUELA lente. Alimenta-lo com duas cameras diferentes
        misturaria duas constantes distintas — exatamente o caso bimodal que
        ele existe para recusar. A primeira vista com ombros visiveis e a dona.
        """
        leituras = []
        for i, (juntas, visivel) in enumerate(vistas):
            leituras.append(self._ler_uma(
                pessoa_id, juntas, visivel, inclinacao_rad,
                rumo_mundo if i == 0 else None,
                velocidade if i == 0 else 0.0))

        leituras = [x for x in leituras if x is not None]
        if not leituras:
            return LeituraDoCorpo(motivo="sem pose")
        final = leituras[0] if len(leituras) == 1 else self._combinar(leituras)

        # A CAMERA DO ALTO MANDA NO RUMO DO CORPO.
        #
        # O caminho pela frontal precisa do azimute, e o azimute nao convergiu
        # por dois caminhos independentes em 11/08 — nem aprendido nem
        # calibrado a mao. De cima nao ha constante a descobrir: a linha dos
        # ombros ja esta no plano que a homografia converte.
        #
        # O caminho antigo continua como reserva, para quando a camera de cima
        # perder os ombros.
        if rumo_do_alto is not None:
            final.rumo_corpo = rumo_do_alto
        return self._aplicar_escala(final, quadril_do_alto)

    def _aplicar_escala(self, leitura, quadril_do_alto):
        """A camera do ALTO manda no quadril quando nenhuma pose viu o pe.

        TRES FONTES, E A ORDEM E POR QUALIDADE DA EVIDENCIA:

            1. tornozelo VISTO numa vista de pose   medido, ~2 cm
            2. estatura medida pela camera do alto  proporcao sobre MEDIDA, ~3 cm
            3. proporcao sobre o tronco             proporcao sobre proporcao, ~8 cm

        A 2 so entra quando a 1 falha, e a 3 so quando as duas falham. Nenhuma
        substitui uma melhor que exista — e so a 1 sai sem o aviso de estimada.

            Cada camera responde o que enxerga. Nenhuma precisa enxergar tudo.
                                                        — Eduardo, 11/08
        """
        if leitura.altura_medida or quadril_do_alto is None:
            return leitura

        antes = leitura.altura_quadril
        leitura.altura_quadril = quadril_do_alto

        # As alturas de mao ja calculadas usaram a referencia pior. Refaz o
        # deslocamento em vez de recalcular do zero: o que muda e so o ponto
        # de apoio, e a distancia pulso-quadril continua medida.
        if antes:
            desvio = quadril_do_alto - antes
            for campo in ("altura_mao_esq", "altura_mao_dir"):
                v = getattr(leitura, campo)
                if v is not None:
                    setattr(leitura, campo, v + desvio)
        return leitura

    @staticmethod
    def _combinar(leituras):
        """Primeira resposta que existe, campo a campo. Sem media."""
        final = leituras[0]

        def primeiro(pega, invalido=None):
            for x in leituras:
                v = pega(x)
                if v is not invalido and v != invalido:
                    return v
            return invalido

        # O braco vem junto com a altura DAQUELA MESMA vista. Separar os dois
        # deixaria o estado vindo de uma camera e a altura de outra, e a altura
        # e o que sustenta o estado.
        for lado, campo_altura in (("braco_esquerdo", "altura_mao_esq"),
                                   ("braco_direito", "altura_mao_dir")):
            for x in leituras:
                if getattr(x, lado) != Braco.DESCONHECIDO:
                    setattr(final, lado, getattr(x, lado))
                    setattr(final, campo_altura, getattr(x, campo_altura))
                    break

        if final.verticalidade_coxa is None:
            final.verticalidade_coxa = primeiro(
                lambda x: x.verticalidade_coxa)
        if final.rumo_corpo is None:
            final.rumo_corpo = primeiro(lambda x: x.rumo_corpo)
        if final.altura_quadril is None:
            final.altura_quadril = primeiro(lambda x: x.altura_quadril)
        return final

    # ------------------------------------------------------------ ciclo
    def esquecer(self, vivos):
        for pid in list(self._quadris):
            if pid not in vivos:
                del self._quadris[pid]

    @property
    def diagnostico(self):
        return self.azimute.diagnostico
