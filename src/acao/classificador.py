"""
ClassificadorDeAcao — de numeros medidos para o vocabulario fechado.

O QUE ELE USA, E POR QUE SO ISSO

    velocidade do Kalman     parado / andando / para onde
    variacao do rumo         virando / meia-volta
    razao de altura da caixa em pe / agachado

Nada aqui depende da fusao 3D das juntas. Os dois primeiros nao dependem de
pose nenhuma. E esse e o ponto central da arquitetura v3:

    o sistema para de depender do elo mais fraco.

Medido em 10/08: posicao no chao 2 a 5 cm e 99,6% de sobrevivencia do rastro,
contra uma fusao 3D que desenhava o esqueleto deitado. Construir a descricao
sobre o que e solido nao e conservadorismo — e usar o que foi medido.

A POSTURA REAPROVEITA UM MODELO QUE JA EXISTE

O `FiltroDePlausibilidade` aprende `k = altura_px / distancia_ao_horizonte`
para uma pessoa EM PE. Ele usa isso para recusar mobilia.

A mesma constante, invertida, responde outra pergunta: se a razao observada
cai bem abaixo de `k`, a pessoa encolheu — agachou. Nenhum modelo novo,
nenhum custo, e um numero que ja e calculado.

    Quando o filtro esta ABSTIDO, `k` nao vale, e a postura sai DESCONHECIDA.
    Sem base, nao se opina.

CUSTO

Aritmetica sobre uma dezena de numeros. Microssegundos. Em 10/08 o gemeo
inteiro custava 0,2 ms por quadro contra 156 ms do detector — a descricao nao
e o que deixa o sistema lento, e tira-la nao o deixaria rapido.
"""

import math

from src.acao.angulos import diferenca_angular as _diferenca_angular
from src.acao.corpo import LeituraDoCorpo
from src.acao.vocabulario import Acao, Braco, Estavel, Locomocao, Postura

_LEITURA_VAZIA = LeituraDoCorpo()


class ClassificadorDeAcao:
    """Um por pessoa. Guarda historia, porque virar so existe no tempo.

    LOCOMOCAO RELATIVA AO CORPO — RESOLVIDO NA ETAPA B

    O rumo do Kalman vem de `arctan2(vy, vx)`: a direcao do DESLOCAMENTO. Por
    construcao, rumo e deslocamento eram a mesma coisa, e "andar de lado"
    ficava indistinguivel de "andar para frente".

    `AnalisadorDeCorpo` agora fornece o rumo do CORPO, tirado da linha dos
    ombros, que independe de para onde a pessoa anda. Com ele chegando,
    frente/tras/esquerda/direita passam a ser respondidos.

    O caminho antigo continua vivo e nao por acaso: quando o azimute da camera
    ainda nao convergiu, ou os ombros nao foram vistos, `rumo_corpo` chega
    `None` e a resposta volta a ser `ANDANDO`. Degradar para a resposta mais
    pobre e melhor que responder com base que nao existe.
    """

    def __init__(self, parar_abaixo_de=0.15, andar_acima_de=0.25,
                 girar_acima_de=45.0, meia_volta_graus=150.0,
                 janela_giro_s=1.2, agachado_abaixo_de=0.78,
                 # Limite fisico da rotacao humana, em rad/s. Uma volta
                 # completa por segundo ja e um giro brusco; o dobro
                 # disso e folga para nao recusar gesto real.
                 giro_maximo=4 * math.pi,
                 estabilidade_s=0.35, estabilidade_braco_s=0.20):
        # HISTERESE: dois limiares, nao um.
        #
        # Com limiar unico, alguem parado com 0,20 m/s de ruido no Kalman
        # alterna parado/andando dezenas de vezes por minuto. Sobe em 0,25 e
        # so desce em 0,15 — a faixa entre os dois e zona morta.
        self.parar_abaixo = parar_abaixo_de
        self.andar_acima = andar_acima_de
        self.girar_acima = math.radians(girar_acima_de)
        self.meia_volta = math.radians(meia_volta_graus)
        self.janela_giro = janela_giro_s
        self.giro_maximo = giro_maximo
        self.agachado_abaixo = agachado_abaixo_de

        self.locomocao = Estavel(Locomocao.DESCONHECIDA,
                                 minimo_s=estabilidade_s)
        self.postura = Estavel(Postura.DESCONHECIDA, minimo_s=estabilidade_s)

        # BRACO TAMBEM PRECISA DE ESTABILIDADE, E POR UM MOTIVO PROPRIO.
        #
        # Locomocao muda devagar; braco muda rapido. A tentacao e deixar o
        # braco responder na hora — e seria errado pelo mesmo motivo de sempre:
        # o pulso fica exatamente NA altura do ombro durante a subida, e nesse
        # instante a classificacao alterna entre `ao_lado` e `levantado` a cada
        # quadro. Sem histerese temporal, um unico gesto de pegar um produto
        # geraria uma dezena de BRACO_MUDOU.
        #
        # O limiar e menor que o da locomocao porque o gesto e mais curto:
        # esperar 0,35 s para reconhecer um braco que subiu perderia o proprio
        # gesto que interessa medir.
        self.braco_esquerdo = Estavel(Braco.DESCONHECIDO,
                                      minimo_s=estabilidade_braco_s)
        self.braco_direito = Estavel(Braco.DESCONHECIDO,
                                     minimo_s=estabilidade_braco_s)

        self._rumo_anterior = None
        self._giro_acumulado = 0.0
        self._tempo_na_janela = 0.0
        self._andando = False           # lado "quente" da histerese

    # ------------------------------------------------------------ principal
    def classificar(self, pessoa, dt, leitura=None, razao_altura=None,
                    k_referencia=None):
        """Devolve (Acao, mudancas) — `mudancas` e um dict de eixo -> bool.

        `pessoa`   EstadoDePessoa, com vx, vy e rumo ja calculados
        `leitura`  LeituraDoCorpo do AnalisadorDeCorpo: rumo do corpo, bracos
                   e altura das maos. `None` = nenhuma vista respondeu, e a
                   classificacao degrada para o que a velocidade sozinha diz.
        `razao_altura`, `k_referencia`  do FiltroDePlausibilidade
        """
        leitura = leitura or _LEITURA_VAZIA
        rumo_corpo = leitura.rumo_corpo
        v = math.hypot(pessoa.vx, pessoa.vy)
        giro = self._medir_giro(pessoa, dt, v)

        proposta, confianca, motivo = self._propor_locomocao(
            pessoa, v, giro, rumo_corpo)

        # A INCERTEZA CHEGA ANTES DA DECISAO, E NAO DEPOIS DELA.
        #
        # Ate 19/08 esta multiplicacao acontecia LOGO ABAIXO do `propor`. O
        # classificador sabia que a evidencia era um palpite do Kalman,
        # comprometia-se assim mesmo, e so entao rotulava o compromisso como
        # fraco. Na corrida real saiu `andando_tras conf 30%` e um giro de
        # +276 graus/s — impossivel para um corpo humano.
        #
        #     Reduzir a confianca depois de decidir nao e prudencia: e um
        #     rotulo de aviso colado numa decisao que ja foi tomada.
        #
        # Prever onde alguem deveria estar nao e o mesmo que ve-lo ali.
        if pessoa.prevendo:
            confianca *= 0.4

        mudou_loc = self.locomocao.propor(proposta, dt, confianca)

        # O painel mostra o estado COMPROMETIDO. Se o motivo vier da proposta,
        # aparece `parado ... [-61 graus/s]` — o estado diz uma coisa e a
        # justificativa ao lado diz outra, e quem le nao sabe em qual acreditar.
        #
        # Esta penalidade fica DEPOIS de proposito: ela nao mede a qualidade
        # da evidencia, mede a discordancia com o que ja estava comprometido.
        # Usa-la como porteiro reprovaria justamente as mudancas legitimas.
        if proposta != self.locomocao.valor:
            motivo = f"propondo {proposta} ({motivo})"
            confianca *= 0.7

        prop_postura, razao = self._propor_postura(
            razao_altura, k_referencia, leitura.encolhimento)
        # Postura tambem nao se decide em cima de previsao: `razao_altura` sai
        # da caixa do detector, e quando o Kalman esta prevendo nao houve caixa.
        mudou_pos = self.postura.propor(
            prop_postura, dt, 0.0 if pessoa.prevendo else 1.0)

        mudou_bre = self.braco_esquerdo.propor(leitura.braco_esquerdo, dt)
        mudou_brd = self.braco_direito.propor(leitura.braco_direito, dt)

        acao = Acao(
            locomocao=self.locomocao.valor,
            postura=self.postura.valor,
            braco_esquerdo=self.braco_esquerdo.valor,
            braco_direito=self.braco_direito.valor,
            altura_mao_esq=leitura.altura_mao_esq,
            altura_mao_dir=leitura.altura_mao_dir,
            altura_medida=leitura.altura_medida,
            confianca=round(confianca, 3),
            velocidade_ms=v,
            giro_graus_s=math.degrees(giro),
            razao_altura=razao,
            motivo=motivo,
        )
        return acao, {"locomocao": mudou_loc, "postura": mudou_pos,
                      "braco_esquerdo": mudou_bre, "braco_direito": mudou_brd}

    # ------------------------------------------------------------ giro
    def _medir_giro(self, pessoa, dt, velocidade):
        """Velocidade angular, e acumulo para detectar meia-volta.

        SO MEDE QUEM ANDA DE VERDADE, E O LIMIAR NAO E O DE PARAR.

        MEDIDO EM 10/08: o painel mostrou `andando 0,23 m/s -193 graus/s`.
        Meia volta por segundo, com a pessoa quase parada. Nao era giro: era
        ruido.

        O `rumo` vem de `arctan2(vy, vx)`. Perto de velocidade zero, o vetor
        velocidade e quase todo ruido do Kalman, e sua DIRECAO gira loucamente
        enquanto o modulo mal se mexe. Um vetor curto tem angulo mal definido —
        e essa e a razao geometrica, nao um ajuste empirico.

        Por isso o giro exige `andar_acima`, o limiar alto da histerese, e nao
        `parar_abaixo`. Entre os dois ha movimento, mas nao ha direcao
        confiavel — e uma direcao nao confiavel nao pode virar um "virando".

        Consequencia aceita: giro em velocidade baixa nao e reportado. Melhor
        perder um giro lento que anunciar dezenas que nao aconteceram.
        """
        if velocidade < self.andar_acima or dt <= 0:
            self._rumo_anterior = pessoa.rumo
            self._giro_acumulado = 0.0
            self._tempo_na_janela = 0.0
            return 0.0

        if self._rumo_anterior is None:
            self._rumo_anterior = pessoa.rumo
            return 0.0

        d = _diferenca_angular(pessoa.rumo, self._rumo_anterior)
        self._rumo_anterior = pessoa.rumo

        # GIRO QUE A FISICA PROIBE NAO E GIRO: E O RUMO PULANDO.
        #
        # MEDIDO EM 12/08, no painel ao vivo:
        #
        #     #1  parado  em_pe  0.41 m/s  +1077 graus/s  [propondo meia_volta]
        #
        # Tres voltas por segundo. Ninguem gira assim, e o pior nao e o numero
        # absurdo: e que ele virou ACAO. O `meia_volta` foi proposto por causa
        # de ruido do rumo, e chegou ao vocabulario fechado com cara de gesto.
        #
        # Uma pessoa girando depressa faz cerca de 360 graus/s. Acima disso a
        # medicao esta descrevendo o estimador, nao a pessoa.
        #
        #     Descartar o impossivel na ENTRADA e mais barato que explicar o
        #     absurdo na saida — e impede que ele contamine o acumulado, que e
        #     o que decide `meia_volta`.
        if abs(d) > self.giro_maximo * dt:
            return 0.0

        self._giro_acumulado += d
        self._tempo_na_janela += dt
        if self._tempo_na_janela > self.janela_giro:
            self._giro_acumulado = d
            self._tempo_na_janela = dt

        return d / dt

    # ------------------------------------------------------------ locomocao
    def _propor_locomocao(self, pessoa, v, giro, rumo_corpo):
        if abs(self._giro_acumulado) >= self.meia_volta:
            return Locomocao.MEIA_VOLTA, 0.9, "giro acumulado"

        if abs(giro) >= self.girar_acima and v >= self.andar_acima:
            lado = (Locomocao.VIRANDO_ESQ if giro > 0
                    else Locomocao.VIRANDO_DIR)
            return lado, 0.8, f"{math.degrees(giro):.0f} graus/s"

        # histerese
        if self._andando:
            if v < self.parar_abaixo:
                self._andando = False
        else:
            if v > self.andar_acima:
                self._andando = True

        if not self._andando:
            return Locomocao.PARADO, 0.9, f"{v:.2f} m/s"

        if rumo_corpo is None:
            # Honesto: anda, mas nao sabemos em relacao a que.
            return Locomocao.ANDANDO, 0.6, "sem rumo do corpo"

        rel = _diferenca_angular(pessoa.rumo, rumo_corpo)
        g = abs(math.degrees(rel))
        if g <= 45:
            return Locomocao.FRENTE, 0.85, f"{g:.0f} graus do corpo"
        if g >= 135:
            return Locomocao.TRAS, 0.75, f"{g:.0f} graus do corpo"
        return ((Locomocao.ESQUERDA if rel > 0 else Locomocao.DIREITA),
                0.7, f"{g:.0f} graus do corpo")

    # ------------------------------------------------------------ postura
    def _propor_postura(self, razao_altura, k, encolhimento=None):
        """Encolheu em relacao ao que esta pessoa mede EM PE? Entao agachou.

        DUAS FONTES, E A ORDEM NAO E ARBITRARIA

        1. altura do QUADRIL em metros, da camera frontal
        2. altura da CAIXA em pixels, da camera do alto

        MEDIDO EM 11/08: com a fonte 2 sozinha, `agachar` foi lido como
        `em_pe` em 100% dos quadros. Uma camera olhando de cima quase nao ve
        mudanca de estatura — a caixa naquela vista e dominada pela pegada da
        pessoa no chao. O sinal estava errado na ORIGEM, e nenhum ajuste de
        limiar teria consertado; mexer no limiar teria sido a terceira rodada
        de ajuste as cegas deste projeto.

        A fonte 1 vem da camera FRONTAL, que ve a pessoa de lado — a vista em
        que agachar e obvio. Ela e preferida sempre que existe.

        A fonte 2 continua valendo como reserva: sem pose frontal nao ha
        quadril, e a caixa e melhor que nada. Ela nao foi removida porque nao
        esta errada — esta apenas cega para este movimento em UMA montagem de
        camera. Numa camera lateral ou frontal ela funcionaria.
        """
        # ATENCAO, DEFEITO CONHECIDO E NAO CORRIGIDO — 19/08.
        #
        # Na corrida real saiu `POSTURA_MUDOU estado=agachado proporcao=0.0`.
        # Tentei barrar isso com um piso anatomico e ERREI: `encolhimento`
        # nao e uma grandeza, sao DUAS, sob o mesmo nome (ver
        # `corpo.LeituraDoCorpo.encolhimento`):
        #
        #     verticalidade_coxa    um COSSENO. 1,0 de pe, 0,0 coxa
        #                           horizontal. Zero e legitimo.
        #     altura_quadril razao  um COMPRIMENTO. 1,0 de pe, ~0,2 agachado.
        #                           Zero e impossivel.
        #
        # O docstring de la afirma que "as duas medem a mesma coisa por
        # caminhos independentes". Medem coisas parecidas com FAIXAS
        # DIFERENTES, e um piso que serve para uma reprova a outra — foi
        # exatamente o que os testes pegaram.
        #
        #     Duas grandezas sob o mesmo nome nao sao redundancia: sao um
        #     limiar que nao pode existir.
        #
        # Nao da para saber, so pelo log, qual das duas produziu o 0,0. Fica
        # anotado para ser medido antes de qualquer guarda ser escrita.
        if encolhimento is not None:
            if encolhimento < self.agachado_abaixo:
                return Postura.AGACHADO, encolhimento
            return Postura.EM_PE, encolhimento

        if not razao_altura or not k:
            return Postura.DESCONHECIDA, 0.0
        proporcao = razao_altura / k
        if proporcao < self.agachado_abaixo:
            return Postura.AGACHADO, proporcao
        return Postura.EM_PE, proporcao


class Descritor:
    """Um classificador por pessoa, criado e esquecido junto com o rastro.

    Guardar historia por pessoa e obrigatorio: virar so existe comparando com
    o rumo anterior DAQUELA pessoa. Um classificador global misturaria os
    rumos de duas pessoas e anunciaria giros que ninguem fez.
    """

    def __init__(self, **kw):
        self._por_pessoa = {}
        self._kw = kw

    def atualizar(self, pessoas, dt, leituras=None,
                  razoes=None, k_referencia=None):
        """Devolve {id: (Acao, mudancas)}.

        `leituras` = {id: LeituraDoCorpo}, do AnalisadorDeCorpo.
        """
        leituras = leituras or {}
        razoes = razoes or {}
        vivos = set()
        saida = {}

        for p in pessoas:
            vivos.add(p.id)
            c = self._por_pessoa.get(p.id)
            if c is None:
                c = self._por_pessoa[p.id] = ClassificadorDeAcao(**self._kw)
            saida[p.id] = c.classificar(
                p, dt,
                leitura=leituras.get(p.id),
                razao_altura=razoes.get(p.id),
                k_referencia=k_referencia)

        for pid in list(self._por_pessoa):
            if pid not in vivos:
                del self._por_pessoa[pid]
        return saida
