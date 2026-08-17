"""
Rastros com filtro de Kalman, em metros no chao.

Duas peças:

    Kalman2D              filtra uma pessoa: suaviza o ruido e preve na ausencia
    GerenciadorDeRastros  cuida de todas, e RECOSTURA identidades partidas


POR QUE FILTRAR EM METROS E NAO EM PIXELS

O modelo e "velocidade constante". Em metros isso e fisica: uma pessoa anda a
1,4 m/s perto ou longe da camera, e a mesma coisa.

Em pixels nao. A mesma pessoa andando na mesma velocidade cobre muitos pixels
perto da camera e poucos ao longe. Um modelo de velocidade constante em pixels
estaria errado em toda parte, e o erro mudaria com a distancia.

Filtrar depois da homografia e o que torna o modelo honesto. E a razao pela
qual o bloco 1 tinha que vir antes.


O QUE ISTO RESOLVE, medido em 07/08

    tremor de ~4 cm por quadro com a pessoa parada
    fragmentacao: ID 2 (33 s) -> ID 3 (28 s), a mesma pessoa, apos 1,6 s fora
"""

import numpy as np


class Kalman2D:
    """Velocidade constante em duas dimensoes, no plano do chao.

    ESTADO:  [x, y, vx, vy]  — posicao em metros, velocidade em m/s

    Guardar velocidade e o que permite continuar prevendo quando a deteccao
    some. Um filtro so de posicao travaria no lugar.
    """

    def __init__(self, x, y, ruido_medicao=0.04, ruido_processo=0.6):
        self.x = np.array([[x], [y], [0.0], [0.0]])

        # sei onde esta (acabei de medir), nao sei a velocidade
        self.P = np.diag([ruido_medicao**2, ruido_medicao**2, 1.0, 1.0])

        self.H = np.array([[1.0, 0, 0, 0],
                           [0, 1.0, 0, 0]])
        self.R = np.eye(2) * ruido_medicao**2
        self.sigma_a = ruido_processo          # aceleracao tipica, m/s^2
        self.I = np.eye(4)

    def prever(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)

        a = self.sigma_a**2
        q = np.array([[dt**4/4, 0, dt**3/2, 0],
                      [0, dt**4/4, 0, dt**3/2],
                      [dt**3/2, 0, dt**2, 0],
                      [0, dt**3/2, 0, dt**2]]) * a

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q

    def corrigir(self, x, y):
        z = np.array([[x], [y]])
        inov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ inov
        self.P = (self.I - K @ self.H) @ self.P

    @property
    def pos(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def vel(self):
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def velocidade(self):
        vx, vy = self.vel
        return float(np.hypot(vx, vy))

    @property
    def incerteza(self):
        """Desvio padrao da posicao, em metros. Cresce sem medicao."""
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))


class Rastro:
    def __init__(self, meu_id, x, y, ruido_medicao, ruido_processo):
        self.id = meu_id
        self.kf = Kalman2D(x, y, ruido_medicao, ruido_processo)
        self.historico = [(x, y)]
        self.sem_medicao = 0
        self.quadros = 1
        self.ids_externos = set()

        # Distancia total percorrida, em metros. E o unico sinal que um movel
        # nao consegue falsificar: cadeira nao anda. Usado para decidir de quem
        # o sistema pode APRENDER.
        self.percorrido = 0.0
        self._ultima_pos = (x, y)

    @property
    def pos(self):
        return self.kf.pos

    @property
    def coasting(self):
        """True quando esta so prevendo, sem medicao ha pelo menos um quadro."""
        return self.sem_medicao > 0


class GerenciadorDeRastros:
    """Cuida dos rastros e RECOSTURA identidades que o rastreador partiu.

    O PROBLEMA MEDIDO: o ByteTrack descarta um rastro quando a pessoa some por
    alguns quadros. Ao voltar, ela ganha ID novo — e o carrinho dela fica para
    tras.

    A SOLUCAO AQUI: manter o rastro vivo em modo previsao (coasting) por alguns
    segundos. Se surgir um ID novo perto de onde o Kalman previu que a pessoa
    estaria, ADOTA aquele ID no rastro antigo em vez de criar outro.

    O raio de aceitacao cresce com o tempo de ausencia, porque a incerteza
    cresce. Quanto mais tempo sem ver, mais longe ela pode estar.

    LIMITE HONESTO: isto associa por PROXIMIDADE, nao por aparencia. Duas
    pessoas que se cruzam podem trocar de identidade e o sistema nao percebe.
    Resolver isso de verdade e re-identificacao por aparencia — bloco 5.
    """

    def __init__(self, ruido_medicao=0.04, ruido_processo=0.6,
                 max_coasting_s=3.0, vel_max=2.0,
                 max_limbo_s=20.0, delta_estatura=0.08):
        self.rastros: dict[int, Rastro] = {}
        self.de_externo: dict[int, int] = {}   # id do ByteTrack -> id nosso
        self.proximo_id = 1
        self.ruido_medicao = ruido_medicao
        self.ruido_processo = ruido_processo
        self.max_coasting_s = max_coasting_s
        self.vel_max = vel_max                 # m/s, para dimensionar o raio
        self.recosturas = 0
        self.dt = 1 / 30                       # atualizado a cada quadro

        # ---- O LIMBO ----
        # Consertado em 14/08. A tela de 12/08 mostrou tres identidades para
        # uma pessoa so numa sessao:
        #
        #     #2  {'p1': 3, 'p3': 6, 'p4': 3, 'p5': 5}
        #     #3  {'p1': 1}
        #     #4  {'p2': 2, 'p3': 1, 'p4': 2}
        #
        # A recostura acima ja existia e funcionava — enquanto o rastro
        # estivesse vivo. Passados `max_coasting_s`, ele era APAGADO, e com
        # ele sumia a unica coisa capaz de reconhecer a pessoa depois: onde
        # ela estava e quanto ela media.
        #
        #     Descartar o rastro perdido para economizar memoria e jogar fora
        #     justamente a prova de que ele era o mesmo.
        #
        # Entao o rastro morto nao some: vira uma ficha barata — posicao,
        # instante e estatura — e fica no limbo por vinte segundos. Um id novo
        # que apareca perto e com a mesma estatura RECEBE O ID ANTIGO, e o
        # carrinho, a estatura fechada e a contagem de unidades continuam.
        #
        # LIMITE DECLARADO, e ele e serio: num quarto de 1,65 x 1,32 m
        # qualquer ponto esta perto de qualquer outro, entao a POSICAO quase
        # nao discrimina — quem discrimina e a estatura. Duas pessoas de
        # altura parecida podem ser fundidas numa so, e o sistema nao vai
        # perceber. Isso e aceitavel enquanto o arranjo de teste tem uma
        # pessoa; deixa de ser no dia em que tiver duas.
        self.limbo: dict[int, dict] = {}
        self.max_limbo_s = max_limbo_s
        self.delta_estatura = delta_estatura
        self.estaturas: dict[int, float] = {}
        self.readocoes = 0
        self.relogio = 0.0                     # segundos acumulados

    def atualizar(self, deteccoes, dt):
        """deteccoes: lista de (id_externo, x_m, y_m). Devolve os rastros vivos."""

        self.dt = dt      # a recostura precisa do dt REAL, nao de um chute
        self.relogio += dt
        self._limpar_limbo()

        for r in self.rastros.values():
            r.kf.prever(dt)

        vistos = set()

        for id_ext, x, y in deteccoes:
            meu = self.de_externo.get(id_ext)

            if meu is None or meu not in self.rastros:
                meu = self._tentar_recosturar(id_ext, x, y)

            if meu is None:
                meu = self.proximo_id
                self.proximo_id += 1
                self.rastros[meu] = Rastro(meu, x, y, self.ruido_medicao,
                                           self.ruido_processo)
                self.de_externo[id_ext] = meu

            r = self.rastros[meu]
            r.kf.corrigir(x, y)
            r.sem_medicao = 0
            r.quadros += 1
            r.ids_externos.add(id_ext)

            px, py = r.pos
            ux, uy = r._ultima_pos
            r.percorrido += float(np.hypot(px - ux, py - uy))
            r._ultima_pos = (px, py)

            r.historico.append(r.pos)
            r.historico = r.historico[-300:]
            vistos.add(meu)

        # quem nao recebeu medicao segue so prevendo
        for meu, r in list(self.rastros.items()):
            if meu in vistos:
                continue
            r.sem_medicao += 1
            r.historico.append(r.pos)
            r.historico = r.historico[-300:]
            if r.sem_medicao * dt > self.max_coasting_s:
                # Morre para o laco, nao para a memoria.
                #
                # `t` e o instante da ULTIMA MEDICAO, e nao o da morte. Sao
                # coisas diferentes: entre uma e outra o rastro passou
                # `max_coasting_s` so prevendo, e a pessoa esteve o tempo todo
                # sem ser vista. Datar a ficha pela morte encolheria o raio de
                # busca em exatamente o intervalo em que ela mais andou.
                self.limbo[meu] = {"pos": r.pos,
                                   "t": self.relogio - r.sem_medicao * dt,
                                   "estatura": self.estaturas.get(meu)}
                del self.rastros[meu]
                for e, m in list(self.de_externo.items()):
                    if m == meu:
                        del self.de_externo[e]

        return self.rastros

    def informar_estatura(self, meu_id, estatura):
        """A estatura ja medida desta pessoa, em metros.

        Quem mede e `src/acao/escala.py`, que fecha o valor depois de 45
        amostras e nao mexe mais. Aqui ela serve de assinatura: e a unica
        propriedade estavel da pessoa que este sistema ja calcula, e sai de
        graca.

            A medida que ja existe para outro fim e a mais barata de todas.
        """
        if estatura is not None:
            self.estaturas[meu_id] = float(estatura)

    def _limpar_limbo(self):
        for meu, ficha in list(self.limbo.items()):
            if self.relogio - ficha["t"] > self.max_limbo_s:
                del self.limbo[meu]
                self.estaturas.pop(meu, None)

    def _combina_estatura(self, meu, ficha):
        """A estatura do limbo bate com a de quem esta chegando?

        Quando uma das duas nao existe, isto devolve True — e nao e descuido:
        recusar por falta de medida faria o conserto depender de a escala ja
        ter fechado, que e justamente o que se perde quando o rastro quebra
        cedo. Sem estatura, decide a proximidade sozinha, como antes.
        """
        antiga = ficha.get("estatura")
        nova = self.estaturas.get(meu)
        if antiga is None or nova is None:
            return True
        return abs(antiga - nova) <= self.delta_estatura

    def _tentar_recosturar(self, id_ext, x, y):
        """Procura um rastro em coasting perto o bastante para ser a mesma pessoa."""
        melhor, menor = None, None

        for meu, r in self.rastros.items():
            if not r.coasting:
                continue

            # BUG corrigido em 08/08: aqui estava 1/30 fixo. Rodando a 4 fps
            # (dt real de 0,25 s), a ausencia era subestimada em 7 vezes, o
            # raio saia pequeno demais e a recostura quase nunca acontecia.
            # Qualquer constante de tempo escondida no codigo mente quando a
            # taxa de quadros muda.
            ausencia_s = r.sem_medicao * self.dt
            # Raio: quanto ela poderia ter andado, mais a incerteza do filtro.
            raio = self.vel_max * ausencia_s + 2 * r.kf.incerteza + 0.30

            px, py = r.pos
            d = float(np.hypot(x - px, y - py))

            if d <= raio and (menor is None or d < menor):
                melhor, menor = meu, d

        if melhor is not None:
            self.de_externo[id_ext] = melhor
            self.recosturas += 1
            return melhor

        return self._tentar_readotar(id_ext, x, y)

    def _tentar_readotar(self, id_ext, x, y):
        """Ninguem vivo serve. Alguem do limbo serve?

        A ordem importa: rastro vivo ganha de ficha morta sempre. Readotar
        alguem que ainda esta na sala seria fundir duas pessoas por causa de
        uma que saiu.
        """
        melhor, menor = None, None
        for meu, ficha in self.limbo.items():
            if not self._combina_estatura(meu, ficha):
                continue
            ausencia_s = self.relogio - ficha["t"]
            raio = self.vel_max * ausencia_s + 0.30
            px, py = ficha["pos"]
            d = float(np.hypot(x - px, y - py))
            if d <= raio and (menor is None or d < menor):
                melhor, menor = meu, d

        if melhor is None:
            return None

        # Renasce com o ID ANTIGO — que e o ponto: o carrinho, a estatura
        # fechada e a contagem de unidades estao guardados sob ele.
        del self.limbo[melhor]
        self.rastros[melhor] = Rastro(melhor, x, y, self.ruido_medicao,
                                      self.ruido_processo)
        self.de_externo[id_ext] = melhor
        self.readocoes += 1
        return melhor
