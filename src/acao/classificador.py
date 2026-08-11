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

from src.acao.vocabulario import Acao, Braco, Estavel, Locomocao, Postura


def _diferenca_angular(a, b):
    """Menor angulo entre dois rumos, com sinal, em radianos.

    Sem isto, passar de +179 para -179 graus vira um giro de 358 graus em vez
    de 2 — e o sistema anunciaria uma meia-volta que nunca aconteceu.
    """
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class ClassificadorDeAcao:
    """Um por pessoa. Guarda historia, porque virar so existe no tempo.

    LIMITE DECLARADO — LOCOMOCAO RELATIVA AO CORPO

    O rumo que o sistema conhece hoje vem da DIRECAO DO MOVIMENTO: ele e
    calculado como `arctan2(vy, vx)`. Ou seja, por construcao, rumo e direcao
    de deslocamento sao a mesma coisa — e com isso "andar de lado" e
    indistinguivel de "andar para frente".

    Para separar os dois e preciso saber para onde o CORPO aponta,
    independentemente de para onde ele anda. Isso vem da linha dos ombros, que
    e material da etapa B.

    Enquanto nao houver, a resposta honesta e `ANDANDO`: sabemos que anda, nao
    sabemos em que direcao relativa ao proprio corpo. Quando `rumo_corpo` for
    fornecido, frente/tras/esquerda/direita passam a ser respondidos.
    """

    def __init__(self, parar_abaixo_de=0.15, andar_acima_de=0.25,
                 girar_acima_de=45.0, meia_volta_graus=150.0,
                 janela_giro_s=1.2, agachado_abaixo_de=0.78,
                 estabilidade_s=0.35):
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
        self.agachado_abaixo = agachado_abaixo_de

        self.locomocao = Estavel(Locomocao.DESCONHECIDA,
                                 minimo_s=estabilidade_s)
        self.postura = Estavel(Postura.DESCONHECIDA, minimo_s=estabilidade_s)

        self._rumo_anterior = None
        self._giro_acumulado = 0.0
        self._tempo_na_janela = 0.0
        self._andando = False           # lado "quente" da histerese

    # ------------------------------------------------------------ principal
    def classificar(self, pessoa, dt, rumo_corpo=None, razao_altura=None,
                    k_referencia=None):
        """Devolve (Acao, mudou_locomocao, mudou_postura).

        `pessoa`      EstadoDePessoa, com vx, vy e rumo ja calculados
        `rumo_corpo`  para onde o CORPO aponta, se souber (etapa B)
        `razao_altura`, `k_referencia`  do FiltroDePlausibilidade
        """
        v = math.hypot(pessoa.vx, pessoa.vy)
        giro = self._medir_giro(pessoa, dt, v)

        proposta, confianca, motivo = self._propor_locomocao(
            pessoa, v, giro, rumo_corpo)
        mudou_loc = self.locomocao.propor(proposta, dt)

        # O painel mostra o estado COMPROMETIDO. Se o motivo vier da proposta,
        # aparece `parado ... [-61 graus/s]` — o estado diz uma coisa e a
        # justificativa ao lado diz outra, e quem le nao sabe em qual acreditar.
        if proposta != self.locomocao.valor:
            motivo = f"propondo {proposta} ({motivo})"
            confianca *= 0.7

        prop_postura, razao = self._propor_postura(razao_altura, k_referencia)
        mudou_pos = self.postura.propor(prop_postura, dt)

        # A confianca cai quando a posicao e previsao do Kalman e nao medida.
        # Prever onde alguem deveria estar nao e o mesmo que ve-lo ali.
        if pessoa.prevendo:
            confianca *= 0.4

        acao = Acao(
            locomocao=self.locomocao.valor,
            postura=self.postura.valor,
            braco_esquerdo=Braco.DESCONHECIDO,     # etapa B
            braco_direito=Braco.DESCONHECIDO,
            confianca=round(confianca, 3),
            velocidade_ms=v,
            giro_graus_s=math.degrees(giro),
            razao_altura=razao,
            motivo=motivo,
        )
        return acao, mudou_loc, mudou_pos

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
    def _propor_postura(self, razao_altura, k):
        """Encolheu em relacao ao que uma pessoa em pe mede ALI? Entao agachou.

        `k` vem do filtro de plausibilidade e ja embute a perspectiva: uma
        pessoa longe tem menos pixels de altura, e a razao corrige isso pela
        distancia ao horizonte. Comparar altura em pixels crua nao funcionaria.
        """
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

    def atualizar(self, pessoas, dt, rumos_do_corpo=None,
                  razoes=None, k_referencia=None):
        """Devolve {id: (Acao, mudou_locomocao, mudou_postura)}."""
        rumos_do_corpo = rumos_do_corpo or {}
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
                rumo_corpo=rumos_do_corpo.get(p.id),
                razao_altura=razoes.get(p.id),
                k_referencia=k_referencia)

        for pid in list(self._por_pessoa):
            if pid not in vivos:
                del self._por_pessoa[pid]
        return saida
