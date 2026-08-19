"""
Nucleo compartilhado: do pixel ao chao.

POR QUE ESTE ARQUIVO EXISTE

Ate 08/08, tudo isto morava dentro de `percepcao/mapa.py` — que era um
PROGRAMA e uma BIBLIOTECA ao mesmo tempo. O `gemeo3d.py` importava classes de
dentro de um executavel, o que significa que rodar o gemeo carregava o codigo
de desenho do mapa, e mexer num quebrava o outro.

Programa e biblioteca sao coisas diferentes. Biblioteca nao tem `main()`.

Aqui mora tudo que responde "onde esta o chao e quem esta em pe nele":

    carregar_homografia   le a calibracao
    para_metros           pixel -> metros
    EstimadorDePe         ponto do pe, sem teletransporte
    FiltroDeTornozelo     o rastro ja provou ser gente?
    FiltroDePlausibilidade a caixa tem o tamanho de uma pessoa ali?
"""

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
CALIB = RAIZ / "calibracao" / "homografia.json"

TORNOZELO_ESQ, TORNOZELO_DIR = 15, 16


def carregar_homografia(caminho=None):
    p = Path(caminho) if caminho else CALIB
    if not p.exists():
        raise SystemExit(
            f"nao achei {p}\n"
            "Rode antes:  python calibracao/homografia.py --largura-m X --altura-m Y\n"
            "e aperte 's' para salvar."
        )
    d = json.loads(p.read_text(encoding="utf-8"))
    return np.array(d["H"]), d


def para_metros(H, x, y):
    v = H @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


# Quanto chao um pixel pode valer e a medida ainda significar alguma coisa.
#
# Longe da camera um pixel cobre cada vez mais piso. Nao ha uma fronteira
# subita: ha um ponto a partir do qual dizer "a pessoa esta AQUI" deixa de
# querer dizer algo. Cinco centimetros e menos que um pe.
#
#     O alcance de um sensor nao acaba onde ele para de ver. Acaba onde a
#     resolucao dele deixa de responder a pergunta.
#
# Na camera do alto deste projeto o pior pixel da imagem vale 0,83 cm — nada
# e cortado, e o limite existe para o dia em que a camera for outra.
CM_POR_PIXEL_MAXIMO = 5.0


def pegada_no_chao(H, largura_px, altura_px, cm_por_pixel_maximo=None,
                   passos=48):
    """O contorno, em METROS, do chao que esta camera enxerga. Convexo.

    POR QUE ISTO EXISTE, E QUAL ERRO ELE DESFAZ

    Ate 19/08 o tamanho do comodo saia da NUVEM reconstruida — a extensao dos
    pontos que o DUSt3R conseguiu casar entre as tres vistas. Medido:

        quarto desenhado pela nuvem      2,1 m2
        chao que a camera do alto mede   8,4 m2

    Quatro vezes menor, e nao por falta de camera: por deixar o instrumento
    errado responder a pergunta. A reconstrucao diz o que TEM na sala. Quem
    sabe o TAMANHO dela e o campo de visao de quem mede posicao.

        Perguntar as dimensoes do comodo a quem so sabe reconhecer objetos
        devolve o tamanho da coleçao de objetos, nao o do comodo.

    COMO SE ACHA O CONTORNO

    A homografia mapeia o plano do chao inteiro, nao so o retangulo que foi
    clicado na calibracao — o retangulo era o alcance da TRENA, nunca o da
    medida. Entao a resposta e a imagem inteira levada ao chao.

    Duas coisas podem estragar isso, e as duas estao tratadas:

    1. A LINHA DO HORIZONTE. Onde `w = h31*u + h32*v + 1` chega a zero, o
       plano some no infinito. Pixels alem dela caem ATRAS da camera e
       voltariam com sinal trocado — um chao fantasma do outro lado do mundo.
       Ficam de fora pelo teste `w > 0`.

    2. A RESOLUCAO. Chegando perto do horizonte, um pixel passa a valer
       metros. O ponto continua sendo chao de verdade e ja nao serve para
       dizer onde alguem esta. Fica de fora por `cm_por_pixel_maximo`, medido
       de verdade em cada amostra pela diferenca finita — nao estimado.

    O que sobra e convexo por construcao (retangulo da imagem cortado por uma
    reta, levado por uma projetividade), entao o fecho convexo das amostras e
    a resposta exata e nao uma aproximacao.

    Devolve um array (N, 2) em metros, no sentido anti-horario, ou None se
    nada da imagem servir.
    """
    import cv2

    h = np.asarray(H, dtype=float)
    limite_m = (cm_por_pixel_maximo if cm_por_pixel_maximo is not None
                else CM_POR_PIXEL_MAXIMO) / 100.0

    us = np.linspace(0, largura_px - 1, passos)
    vs = np.linspace(0, altura_px - 1, passos)
    grade = np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 2)

    def leva(pontos):
        p = np.column_stack([pontos, np.ones(len(pontos))]) @ h.T
        w = p[:, 2]
        seguro = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return p[:, :2] / seguro[:, None], w

    metros, w = leva(grade)
    # o tamanho do pixel medido, e nao suposto: um passo em u, um passo em v
    du, _ = leva(grade + [1.0, 0.0])
    dv, _ = leva(grade + [0.0, 1.0])
    tamanho = np.maximum(np.linalg.norm(du - metros, axis=1),
                         np.linalg.norm(dv - metros, axis=1))

    bons = (w > 1e-9) & (tamanho <= limite_m)
    if bons.sum() < 3:
        return None

    fecho = cv2.convexHull(metros[bons].astype(np.float32))
    # O fecho de uma grade de amostras traz vertices quase colineares — catorze
    # pontos para descrever um quadrilatero. Um centimetro de tolerancia deixa
    # so as quinas de verdade, e um centimetro e menos que o erro da propria
    # homografia longe do retangulo calibrado (medido: 2 a 5 cm).
    simples = cv2.approxPolyDP(fecho, 0.01, True)
    if len(simples) >= 3:
        fecho = simples
    return fecho.reshape(-1, 2).astype(float)


def caixa_do_contorno(contorno):
    """(xmin, xmax, ymin, ymax) de um contorno. None vira None."""
    if contorno is None or not len(contorno):
        return None
    c = np.asarray(contorno, dtype=float).reshape(-1, 2)
    return (float(c[:, 0].min()), float(c[:, 0].max()),
            float(c[:, 1].min()), float(c[:, 1].max()))


class EstimadorDePe:
    """Combina tornozelo e base da caixa SEM criar teletransporte.

    MEDIDO EM 07/08 (mapa_2026-08-07_235041.jsonl):

        tornozelo -> tornozelo   mediana  3,7 cm por quadro
        caixa     -> caixa       mediana  0,9 cm
        TROCA entre os dois      mediana ~28   cm

        100% dos saltos acima de 50 cm aconteceram numa troca de estimador.

    Causa: a base da caixa fica ~97 px abaixo dos tornozelos, sistematicamente.
    Ao alternar, a pessoa "anda" 28 cm sem sair do lugar.

    CORRECAO: enquanto os dois existem, mede-se o desvio entre eles. Quando o
    tornozelo some, aplica-se esse desvio na caixa. A troca fica invisivel.
    O desvio e por ID e suavizado, porque muda com a distancia a camera.

    Resultado apos a correcao: saltos > 50 cm cairam de 5 para 0.
    """

    def __init__(self, suavizacao=0.15):
        self.desvio: dict[int, np.ndarray] = {}
        self.alfa = suavizacao

    def estimar(self, tid, caixa, kp_xy, kp_conf, minimo=0.5, quadro=None,
                margem_px=6):
        """Devolve ((x, y), origem) ou (None, motivo).

        `quadro` = (largura, altura) da imagem. Sem ele a checagem de corte
        nao acontece, e o comportamento e o de antes.
        """
        x1, y1, x2, y2 = caixa
        pe_caixa = np.array([(x1 + x2) / 2.0, float(y2)])

        pe_tornozelo = None
        if kp_xy is not None:
            validos = [kp_xy[i] for i in (TORNOZELO_ESQ, TORNOZELO_DIR)
                       if kp_conf[i] >= minimo]
            if validos:
                pe_tornozelo = np.mean(validos, axis=0)

        if pe_tornozelo is not None:
            d = pe_tornozelo - pe_caixa
            if tid in self.desvio:
                self.desvio[tid] = (1 - self.alfa) * self.desvio[tid] + self.alfa * d
            else:
                self.desvio[tid] = d
            return (int(pe_tornozelo[0]), int(pe_tornozelo[1])), "tornozelo"

        # CAIXA ENCOSTADA NA BORDA NAO TEM BASE: TEM RECORTE.
        #
        # ISTO CONGELOU O BONECO NA CORRIDA DE 19/08, E O VIDEO MOSTROU.
        #
        # De 28 s em diante o Eduardo andou ate a beirada do campo da camera
        # do teto. Ele continuava aparecendo — mas os PES saiam do quadro. A
        # caixa do detector para de terminar no pe e passa a terminar na
        # BORDA DA IMAGEM, que e uma linha de pixels fixa.
        #
        # A homografia converte linha fixa em posicao fixa. O boneco travou
        # num ponto, o mapa de calor virou uma mancha unica, e a velocidade
        # ficou em 0,00 a 0,06 m/s enquanto ele atravessava a sala.
        #
        #     Uma caixa cortada pela borda nao mede a pessoa: mede onde a
        #     imagem acabou. E a borda nao se move.
        #
        # Pior que o erro: ele nao parece erro. Nada falha, o rastro
        # sobrevive, o funil marca 90% de medidas — e todas dizem a mesma
        # coisa errada com confianca alta.
        #
        # Sem tornozelo E com a caixa cortada, a resposta honesta e nao ter
        # resposta. O Kalman passa a prever SABENDO que preve, e a acao cai
        # para `desconhecida` em vez de inventar.
        if quadro is not None and self._cortada(caixa, quadro, margem_px):
            return None, "caixa cortada pela borda"

        if tid in self.desvio:
            p = pe_caixa + self.desvio[tid]
            return (int(p[0]), int(p[1])), "caixa+correcao"

        return (int(pe_caixa[0]), int(pe_caixa[1])), "caixa"

    @staticmethod
    def _cortada(caixa, quadro, margem_px=6):
        """A caixa encosta em alguma borda da imagem?

        Basta encostar em UMA. Uma pessoa cortada de lado tem o pe do outro
        lado ainda dentro, mas a caixa ja nao a contem — e a base dela deixa
        de ser o ponto mais baixo do corpo para ser o ponto mais baixo do que
        sobrou.
        """
        x1, y1, x2, y2 = caixa
        larg, alt = quadro
        return bool(x1 <= margem_px or y1 <= margem_px
                    or x2 >= larg - 1 - margem_px
                    or y2 >= alt - 1 - margem_px)

    def esquecer(self, vivos):
        for tid in list(self.desvio):
            if tid not in vivos:
                del self.desvio[tid]


class FiltroDeTornozelo:
    """Um rastro so e gente depois de mostrar tornozelo o bastante.

    EVIDENCIA (07 e 08/08): um objeto fantasma teve 0 tornozelos em 79 quadros
    somados, enquanto a pessoa real teve 96%.

    Exige contagem minima E proporcao. So contagem nao basta: em 08/08 um
    objeto arranhou 3 tornozelos em 55 quadros (5%) e ganhou passe vitalicio.

    Regra e "ja mostrou o bastante", nao "esta mostrando agora": quem esta
    atras de uma gondola tem os pes ocultos e continua sendo cliente.

    LIMITE CONHECIDO: o MediaPipe SEMPRE devolve um esqueleto completo quando
    acha que ha gente, inclusive tornozelos inventados. Por isso este filtro
    sozinho nao basta — ver FiltroDePlausibilidade.
    """

    def __init__(self, minimo=3, proporcao=0.20):
        self.minimo = minimo
        self.proporcao = proporcao
        self.com: dict[int, int] = {}
        self.total: dict[int, int] = {}

    def ver(self, tid, origem):
        if tid < 0:
            return False
        self.total[tid] = self.total.get(tid, 0) + 1
        if origem == "tornozelo":
            self.com[tid] = self.com.get(tid, 0) + 1

        com = self.com.get(tid, 0)
        if com < self.minimo:
            return False
        if self.total[tid] < 15:
            return True
        return com / self.total[tid] >= self.proporcao

    def esquecer(self, vivos):
        for d in (self.com, self.total):
            for tid in list(d):
                if tid not in vivos:
                    del d[tid]


class FiltroDePlausibilidade:
    """A caixa tem o tamanho de uma pessoa NAQUELE lugar do chao?

    O PROBLEMA QUE ISTO RESOLVE

    Uma cadeira com roupas e detectada como pessoa com confianca alta, e o
    MediaPipe obedientemente desenha um esqueleto nela. Nem a confianca do
    detector nem a presenca de tornozelos denunciam a fraude.

    Mas a GEOMETRIA denuncia.

    A IDEIA, e ela vem do bloco 1

    Numa imagem de um plano, existe uma LINHA DO HORIZONTE: onde o plano some
    no infinito. A homografia ja carrega essa linha — sao os coeficientes da
    terceira linha da matriz, os que produzem a divisao por w:

        h31*u + h32*v + h33 = 0

    E ha um fato classico de geometria projetiva: para objetos de mesma altura
    apoiados no plano, a altura APARENTE em pixels e proporcional a distancia
    vertical entre o pe e o horizonte.

        altura_px = k * (v_pe - v_horizonte)

    Ou seja: sabendo onde os pes estao, da para PREVER quantos pixels de altura
    uma pessoa deveria ter ali. Quem foge muito da previsao nao e pessoa.

    AUTOCALIBRACAO — E O ERRO QUE ELA QUASE CAUSOU

    A constante k depende da altura tipica das pessoas e da lente. Em vez de
    chutar, aprendemos: cada pessoa contribui com uma amostra de
    altura_px / (v_pe - v_horizonte), e usamos a MEDIANA.

    A PRIMEIRA VERSAO APRENDIA COM QUEM ELA MESMA ACEITAVA. Medido em 08/08:
    uma cadeira com roupas foi aceita, virou amostra, e ensinou o filtro que
    "pessoa aqui tem esse tamanho". Resultado: k=0,207 com dispersao de 61%.

        Um filtro que aprende com o que ele aceita valida os proprios erros.

    A CORRECAO exige um sinal que um movel nao consiga falsificar:
    MOVIMENTO. Cadeira nao anda. Agora `observar()` so aceita amostras de
    rastros que ja percorreram uma distancia minima de verdade.

    Isso torna o aprendizado imune a mobilia — ao custo de precisar que alguem
    caminhe pela cena antes de o filtro entrar em acao.

    LIMITES HONESTOS

    - Nao funciona com a camera quase perpendicular ao chao: sem perspectiva,
      nao ha horizonte utilizavel. Detectamos isso e desligamos o filtro.
    - Uma crianca ou alguem agachado tem altura menor de verdade. Por isso a
      tolerancia e larga (fator 1,7 por padrao) — o objetivo e cortar o
      absurdo, nao afinar.
    - Precisa de amostras antes de julgar.
    """

    def __init__(self, H, tolerancia=1.7, minimo_amostras=25, memoria=400,
                 percurso_minimo_m=1.0, dispersao_maxima=0.30):
        from collections import deque

        self.tol = tolerancia
        self.minimo = minimo_amostras
        self.percurso_minimo = percurso_minimo_m
        self.dispersao_maxima = dispersao_maxima
        self.amostras = deque(maxlen=memoria)
        self.k = None
        self.recusadas_por_imobilidade = 0
        self.desistiu = False

        # Terceira linha da homografia: e ela que define o horizonte.
        self.h31, self.h32, self.h33 = (float(H[2, 0]), float(H[2, 1]),
                                        float(H[2, 2]))

        # Sem inclinacao suficiente, h32 tende a zero e o horizonte vai para o
        # infinito. Nesse caso o filtro nao tem base e fica desligado.
        self.utilizavel = abs(self.h32) > 1e-7

    def v_horizonte(self, u):
        """Altura (em pixels) da linha do horizonte na coluna u. `None` sem base.

        SEGUNDA VEZ QUE ESTE MESMO DEFEITO APARECE NESTE ARQUIVO.

        Em 11/08 foi `razao()`: ela dividia por `h32` sem checar `utilizavel`,
        e so nao explodia porque `plausivel()` checava antes. Quando a escala
        vertical passou a chamar `razao()` direto, a divisao por zero apareceu.
        A guarda foi movida para dentro do objeto.

        `v_horizonte` ficou de fora — era chamada so por `razao()`, que ja
        estava protegida. Em 12/08 a altura do quadril passou a chama-la
        direto, e o ZeroDivisionError voltou, no mesmo arquivo, pelo mesmo
        motivo.

            Guardar a invariante em UM caminho protege aquele caminho. Guardar
            dentro do objeto protege todos os que ainda nao existem.

        Devolver `None` em vez de levantar e deliberado: horizonte no infinito
        e uma configuracao legitima de camera (lente sem inclinacao), nao um
        erro. Quem chama decide se consegue seguir sem ele.
        """
        if not self.utilizavel:
            return None
        return -(self.h31 * u + self.h33) / self.h32

    def razao(self, caixa):
        """altura_px / distancia_ao_horizonte. Constante para pessoas reais.

        SEM HORIZONTE, SEM RAZAO — e a guarda mora aqui, nao em quem chama.
        A versao anterior dependia de `plausivel()` checar `utilizavel` antes;
        quando a escala vertical passou a chamar `razao()` direto, a divisao
        por `h32 = 0` levantou ZeroDivisionError numa camera perpendicular.

            Invariante do objeto se defende dentro do objeto. Guarda que vive
            em quem chama protege so o primeiro chamador.
        """
        if not self.utilizavel:
            return None

        x1, y1, x2, y2 = caixa
        u = (x1 + x2) / 2.0
        d = y2 - self.v_horizonte(u)
        if d <= 1e-6:
            return None            # pe acima do horizonte: impossivel
        return (y2 - y1) / d

    def observar(self, caixa, percorrido_m):
        """Alimenta uma amostra — SO de quem ja andou de verdade.

        `percorrido_m` e a distancia total que o rastro percorreu. Mobilia
        nunca alcanca o limiar, entao nunca entra no aprendizado.
        """
        if not self.utilizavel:
            return False
        if percorrido_m < self.percurso_minimo:
            self.recusadas_por_imobilidade += 1
            return False

        r = self.razao(caixa)
        if r is not None and r > 0:
            self.amostras.append(r)
            if len(self.amostras) >= self.minimo:
                self.k = float(np.median(self.amostras))
                self._julgar_o_proprio_ajuste()
            return True
        return False

    def _julgar_o_proprio_ajuste(self):
        """O modelo descreve ESTA cena? Se nao, o filtro se cala.

        MEDIDO EM 10/08. O filtro aprendeu k=0,149 com dispersao de 48% e
        passou a recusar 358 de 650 deteccoes de uma pessoa real. O rastro
        durou 3 s em 60. Com o filtro desligado, a mesma pessoa foi seguida a
        execucao inteira, sem uma unica perda.

            Nao era o limiar. Era o modelo.

        `altura_px = k * (v_pe - v_horizonte)` vale para gente em pe, vista de
        longe, com horizonte na imagem. Numa camera inclinada sobre uma area
        pequena, a caixa e cortada pela borda do quadro e a altura aparente
        deixa de seguir a distancia ao horizonte. Dispersao de 48% e o modelo
        avisando que nao serve ali.

        O guarda que existia — `abs(h32) > 1e-7` — nunca pegou isso, porque
        testa se ha horizonte, nao se o horizonte EXPLICA as medidas. Um
        numero diferente de zero nao e um ajuste bom.

            Um filtro que nao consegue ajustar o proprio modelo deve se
            ABSTER, nao recusar. Recusar sem base derruba o que era para
            proteger.

        Continua aprendendo: se a cena melhorar (camera mais longe, pessoa
        inteira no quadro), a dispersao cai e ele volta sozinho.
        """
        a = np.array(self.amostras)
        disp = (np.percentile(a, 75) - np.percentile(a, 25)) / self.k
        ruim = disp > self.dispersao_maxima
        if ruim != self.desistiu:
            self.desistiu = ruim
        return not ruim

    def plausivel(self, caixa):
        """(aceita, motivo). Aceita por padrao enquanto nao houver base.

        `desistiu` entra aqui com o mesmo peso de `k is None`: sem modelo que
        sirva, nao ha julgamento a fazer.
        """
        if not self.utilizavel or self.k is None or self.desistiu:
            return True, ""

        r = self.razao(caixa)
        if r is None:
            return False, "acima do horizonte"

        if r > self.k * self.tol:
            return False, f"alta demais ({r/self.k:.1f}x)"
        if r < self.k / self.tol:
            return False, f"baixa demais ({self.k/r:.1f}x)"
        return True, ""

    @property
    def pronto(self):
        return self.utilizavel and self.k is not None and not self.desistiu

    def diagnostico(self):
        if not self.utilizavel:
            return "sem horizonte (camera perpendicular)"
        if self.k is None:
            return f"aprendendo ({len(self.amostras)}/{self.minimo})"
        a = np.array(self.amostras)
        disp = (np.percentile(a, 75) - np.percentile(a, 25)) / self.k
        estado = "ABSTIDO" if self.desistiu else "ativo"
        return f"k={self.k:.3f} disp={disp*100:.0f}% {estado}"
