"""As tres cameras num sistema de coordenadas so, e o numero que prova isso.

    todas as cameras deveriam ajudar a dizer aonde esta a pessoa e unir todas
    as informacoes para apenas 1 movimento   — Eduardo, 19/08

A MATEMATICA, EM UMA LINHA

    s * [u, v, 1]^T  =  K * [R | t] * [X, Y, Z, 1]^T

`K` e a INTRINSECA: leva metros no referencial da camera para pixels. Sai da
calibracao de Zhang, com tabuleiro (`calibracao/intrinseca.py`).

`[R | t]` e a EXTRINSECA: leva metros do MUNDO para o referencial da camera.
Sai da homografia do chao, que veio da trena (`calibracao/homografia.py`).

`P = K [R | t]` e a matriz de projecao, 3x4. Com ela, qualquer ponto 3D do
mundo vira pixel em qualquer camera — e e isso que permite triangular.

DUAS PERGUNTAS DIFERENTES, DUAS CONTAS DIFERENTES

    ponto SOBRE o chao      uma camera basta      homografia
    ponto FORA do chao      duas ou mais          triangulacao

A primeira e a que quase todo mundo esquece, e e a que faz este projeto
funcionar com cameras que nao se sobrepoem. Pessoa em pe e estante em pe
estao sobre o chao, e o chao e um plano CONHECIDO — a restricao do plano
substitui a segunda vista.

    Uma incognita a menos vale mais que uma camera a mais.

A segunda so entra para o que sai do chao: a mao a 1,20 m. Ai nao ha plano
que ajude, e sao precisas duas vistas do MESMO ponto.

POR QUE NAO HA CASAMENTO DE CARACTERISTICAS AQUI

O caminho classico seria ORB ou SIFT achando os mesmos pontos nas tres
imagens. Nao funciona neste arranjo, e isso foi MEDIDO: dois dias de DUSt3R
— que casa vistas muito melhor que ORB — devolveram um comodo de 2,1 m2 com
duas cameras empilhadas no mesmo ponto. As tres quase nao veem as mesmas
superficies, e o piso e ceramica repetitiva, em que todo ladrilho parece
todo ladrilho.

    Casamento entre vistas precisa de vistas que se cruzem. Sem isso o
    metodo nao erra pouco: ele responde outra coisa.

Aqui a correspondencia vem de outro lugar: as tres cameras olham a MESMA
PESSOA, e o detector diz qual pixel e o tornozelo dela em cada uma. Isso e
uma correspondencia semantica, e ela existe mesmo sem sobreposicao.

O NUMERO QUE VALIDA TUDO: ERRO DE REPROJECAO

Projete um ponto 3D conhecido nas tres imagens e meca a distancia, em
pixels, entre onde ele caiu e onde ele deveria cair. Abaixo de 2 px a
calibracao presta; acima, nao adianta melhorar nada depois.

    Um sistema de calibracao que nao mede o proprio erro nao esta
    calibrado: esta configurado.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent.parent

# Quantos pixels de erro tem a localizacao de um ponto na imagem.
#
# Nao e a precisao do detector em si: e quanto o tornozelo dele oscila entre
# quadros com a pessoa parada. Medido neste projeto em 07/08: 3,7 cm por
# quadro no chao, que na resolucao da camera do alto sao cerca de 3 px.
RUIDO_DE_PIXEL = 3.0


@dataclass
class Camara:
    """Uma camera com tudo o que ela precisa para falar de metros."""
    papel: str
    K: np.ndarray                       # 3x3, intrinseca
    R: np.ndarray                       # 3x3, mundo -> camera
    t: np.ndarray                       # (3,), mundo -> camera
    tamanho: tuple                      # (largura_px, altura_px)
    dist: np.ndarray | None = None      # coeficientes de distorcao
    homografia: np.ndarray | None = None   # pixel -> metro, no plano z=0
    origem_da_focal: str = ""           # tabuleiro | trena | deducao

    @property
    def P(self):
        """A matriz de projecao 3x4. `P = K [R | t]`."""
        return self.K @ np.column_stack([self.R, self.t])

    @property
    def posicao(self):
        """Onde a camera esta, em metros no mundo. `C = -R^T t`."""
        return -self.R.T @ self.t

    def projetar(self, pontos_mundo):
        """(N,3) em metros -> (N,2) em pixels. Devolve NaN atras da camera."""
        p = np.asarray(pontos_mundo, dtype=float).reshape(-1, 3)
        homog = np.column_stack([p, np.ones(len(p))])
        q = homog @ self.P.T
        z = q[:, 2]
        fora = np.abs(z) < 1e-9
        z = np.where(fora, 1.0, z)
        uv = q[:, :2] / z[:, None]
        uv[fora | (q[:, 2] <= 0)] = np.nan
        return uv

    def sem_distorcao(self, pontos_px):
        """Desfaz o barril da lente. Sem coeficientes, devolve como veio.

        A C920 tem distorcao visivel nas bordas — e a borda e justamente onde
        a pessoa fica quando anda para o lado. Sem corrigir, um ponto ali erra
        dezenas de pixels, e a homografia converte isso em centimetros.
        """
        p = np.asarray(pontos_px, dtype=float).reshape(-1, 1, 2)
        if self.dist is None:
            return p.reshape(-1, 2)
        import cv2
        limpo = cv2.undistortPoints(p.astype(np.float32), self.K,
                                    np.asarray(self.dist, dtype=float),
                                    P=self.K)
        return limpo.reshape(-1, 2)

    # ------------------------------------------------------- sobre o chao
    def no_chao(self, u, v):
        """Pixel -> metro, para um ponto que ESTA no chao. Ou None.

        Uma camera basta porque o plano z=0 ja e a segunda equacao.
        """
        if self.homografia is None:
            return None
        p = self.homografia @ np.array([float(u), float(v), 1.0])
        if abs(p[2]) < 1e-12 or p[2] <= 0:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def incerteza_no_chao(self, u, v, ruido_px=RUIDO_DE_PIXEL):
        """Quantos metros vale o erro de um pixel AQUI. Ou None.

        E o peso da fusao, e ele nao e constante: um pixel perto da camera
        cobre meio centimetro de piso; perto do horizonte, cobre metros. Uma
        camera que ve a sala inteira nao mede a sala inteira igual.

            Um sensor sem mapa da propria incerteza so pode ser usado inteiro
            ou descartado inteiro.

        Primeira ordem, pela diferenca finita da homografia. Basta: o que
        decide a fusao e a RAZAO entre as incertezas, e ela e dominada por
        este termo.
        """
        aqui = self.no_chao(u, v)
        if aqui is None:
            return None
        du = self.no_chao(u + 1, v)
        dv = self.no_chao(u, v + 1)
        if du is None or dv is None:
            return None
        passo = max(float(np.hypot(du[0] - aqui[0], du[1] - aqui[1])),
                    float(np.hypot(dv[0] - aqui[0], dv[1] - aqui[1])))
        return ruido_px * passo


# --------------------------------------------------------------- o conjunto
@dataclass
class Camaras:
    """As cameras calibradas, por papel. Todas no MESMO sistema de coordenadas.

    Elas caem no mesmo sistema porque cada homografia foi medida sobre um
    retangulo referido a MESMA marca de fita no chao (ver `--origem` em
    `calibracao/homografia.py`). Nao ha etapa de alinhamento depois, e por
    isso nao ha etapa de alinhamento para falhar.
    """
    por_papel: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.por_papel)

    def __getitem__(self, papel):
        return self.por_papel[papel]

    def __contains__(self, papel):
        return papel in self.por_papel

    @property
    def papeis(self):
        return sorted(self.por_papel)

    @property
    def com_chao(self):
        """Quais sabem converter pixel em metro no piso."""
        return [p for p, c in sorted(self.por_papel.items())
                if c.homografia is not None]

    @staticmethod
    def carregar(raiz=None, papeis=("alto", "frontal", "lateral"),
                 altura_da_camera=None):
        """Le o que houver em `calibracao/` e monta o que der.

        Uma camera sem homografia nao entra: sem ela nao ha como amarrar a
        extrinseca ao mundo, e uma camera sem lugar no mundo nao ajuda a dizer
        onde alguem esta.

        `altura_da_camera` = {papel: metros de trena}. Quando a intrinseca
        medida com tabuleiro nao existe, a altura determina a focal — foi
        assim que a do teto saiu, e o resultado bateu com a folha de dados da
        C920 (ver `profundidade.focal_pela_altura`).
        """
        from src.mundo.profundidade import (camera_da_homografia,
                                            focal_pela_altura,
                                            intrinseca_medida)

        raiz = Path(raiz) if raiz else RAIZ
        calib = raiz / "calibracao"
        alturas = altura_da_camera or {}
        achadas = {}

        for papel in papeis:
            homog, tam = _ler_homografia(calib, papel)
            if homog is None:
                continue
            larg, alt = tam

            K, dist, origem = _ler_intrinseca(calib, papel, larg, alt,
                                              intrinseca_medida)
            if K is None and alturas.get(papel):
                K = focal_pela_altura(homog, larg, alt, alturas[papel])
                origem = f"trena ({alturas[papel]:.2f} m)"
            if K is None:
                origem = "deducao"

            cam = camera_da_homografia(homog, larg, alt, K=K)
            if cam is None:
                continue
            achadas[papel] = Camara(
                papel=papel, K=cam.K, R=cam.R, t=cam.t, tamanho=(larg, alt),
                dist=dist, homografia=homog, origem_da_focal=origem)

        return Camaras(achadas)


    # ------------------------------------------------- FORA do chao: DLT
    def triangular(self, vistas, minimo=2):
        """Pixels em 2+ cameras -> um ponto 3D em metros. Ou None.

        `vistas` = {papel: (u, v)}

        A MATEMATICA, QUE E MAIS SIMPLES DO QUE PARECE

        Cada camera diz `s [u v 1]^T = P [X Y Z 1]^T`. Escrevendo as tres
        linhas e eliminando o `s` desconhecido, sobram DUAS equacoes lineares
        por camera, sem incognita nenhuma alem do ponto:

            u * P3 - P1 = 0
            v * P3 - P2 = 0

        onde P1, P2, P3 sao as linhas de P. Com duas cameras sao 4 equacoes
        para 4 incognitas homogeneas — resolvidas pelo menor valor singular,
        que e a solucao de minimos quadrados quando os raios nao se cruzam
        exatamente. E eles nunca se cruzam exatamente.

            Dois raios no espaco quase nunca se encontram. Triangular nao e
            achar o cruzamento: e achar o ponto que menos desagrada aos dois.

        Cada camera a mais acrescenta duas equacoes, e o sistema so melhora.
        Por isso `vistas` e um dicionario e nao um par.
        """
        linhas = []
        for papel, uv in (vistas or {}).items():
            cam = self.por_papel.get(papel)
            if cam is None or uv is None:
                continue
            u, v = self._limpo(cam, uv)
            P = cam.P
            linhas.append(u * P[2] - P[0])
            linhas.append(v * P[2] - P[1])

        if len(linhas) < 2 * minimo:
            return None
        A = np.array(linhas, dtype=float)
        _u, _s, vt = np.linalg.svd(A)
        X = vt[-1]
        if abs(X[3]) < 1e-12:
            return None                    # ponto no infinito: raios paralelos
        return X[:3] / X[3]

    @staticmethod
    def _limpo(cam, uv):
        p = cam.sem_distorcao([uv])[0]
        return float(p[0]), float(p[1])

    def erro_de_reprojecao(self, ponto_mundo, vistas):
        """Quantos pixels cada camera erra sobre um ponto 3D. {papel: px}.

        ESTE E O NUMERO QUE DIZ SE A CALIBRACAO PRESTA, e ele e o unico
        honesto: projete um ponto 3D conhecido e meca onde ele caiu contra
        onde ele deveria cair.

        Abaixo de 2 px, presta. Acima, nao adianta melhorar nada depois — a
        fusao, o Kalman e o desenho vao herdar o erro inteiro e cada um vai
        parecer o culpado.

            Um sistema de calibracao que nao mede o proprio erro nao esta
            calibrado: esta configurado.
        """
        fora = {}
        for papel, uv in (vistas or {}).items():
            cam = self.por_papel.get(papel)
            if cam is None or uv is None:
                continue
            prevista = cam.projetar([ponto_mundo])[0]
            if not np.isfinite(prevista).all():
                continue
            u, v = self._limpo(cam, uv)
            fora[papel] = float(np.hypot(prevista[0] - u, prevista[1] - v))
        return fora

    # --------------------------------------------- SOBRE o chao: fusao
    def no_chao(self, vistas, ruido_px=RUIDO_DE_PIXEL):
        """Onde a pessoa esta, fundindo o que cada camera diz. Ou None.

        `vistas` = {papel: (u, v)} — o PE em cada imagem.
        Devolve (x, y, sigma_m, quantas) em metros.

        ESCOLHER A MELHOR JOGA INFORMACAO FORA.

        Duas medidas independentes da mesma coisa combinam em algo melhor que
        qualquer uma das duas — media ponderada pelo INVERSO DA VARIANCIA:

            x = sum(x_i / s_i^2) / sum(1 / s_i^2)
            1/s^2 = sum(1 / s_i^2)

        Com duas cameras de 5 cm cada, a combinacao da 3,5. Escolher a melhor
        daria 5. E uma camera ruim nunca piora o resultado: o peso dela cai
        sozinho, porque ele e 1/s^2.

            Escolher a melhor devolve a melhor. Combinar devolve melhor que a
            melhor, e de graca.

        O peso vem de `incerteza_no_chao`, que nao e constante por camera: um
        pixel perto da lente cobre meio centimetro de piso e perto do
        horizonte cobre metros. Entao a mesma camera pesa muito de um lado da
        sala e pouco do outro — que e exatamente o que se quer quando uma
        cobre a beirada em que a outra cega.
        """
        medidas = []
        for papel, uv in (vistas or {}).items():
            cam = self.por_papel.get(papel)
            if cam is None or uv is None or cam.homografia is None:
                continue
            u, v = self._limpo(cam, uv)
            ponto = cam.no_chao(u, v)
            sigma = cam.incerteza_no_chao(u, v, ruido_px)
            if ponto is None or not sigma or sigma <= 0:
                continue
            medidas.append((np.array(ponto, dtype=float), float(sigma)))

        if not medidas:
            return None
        if len(medidas) > 2:
            medidas = _sem_disparates(medidas)
        elif len(medidas) == 2:
            medidas = _duas_que_discordam(medidas)

        soma_peso = sum(1.0 / (s * s) for _p, s in medidas)
        soma = sum(p / (s * s) for p, s in medidas)
        if soma_peso <= 0:
            return None
        x, y = soma / soma_peso
        return float(x), float(y), float(1.0 / np.sqrt(soma_peso)), len(medidas)


# A FUSAO PRECISA DE PORTEIRO, E ISSO NAO E OBVIO.
#
# Media ponderada pelo inverso da variancia e OTIMA — sob uma hipotese que
# ninguem enuncia: a de que cada medida erra dentro do proprio sigma. Ela
# protege contra IMPRECISAO declarada, e nao contra ERRO GROSSEIRO.
#
# MEDIDO EM 20/08: uma camera 40 px fora, com sigma otimista, piorou a fusao
# em 11 cm em relacao a usar so a boa. O peso dela era alto porque ela DIZIA
# ser precisa — e ela estava errada, nao imprecisa.
#
#     Ponderar por incerteza declarada supoe que quem se declara preciso
#     esta certo. Um erro grosseiro chega com sigma pequeno e voto grande.
#
# E o caso e real neste projeto: um reflexo no piso, outra pessoa entrando no
# quadro, uma caixa truncada. Nenhum desses avisa que errou.
DESVIOS_PARA_DISPARATE = 3.0


def _sem_disparates(medidas, limite=DESVIOS_PARA_DISPARATE):
    """Com 3+ medidas, a MEDIANA arbitra e quem se afasta demais sai.

    Mediana e nao media: com tres medidas e uma errada, a media ja esta
    contaminada e nao serve para julgar quem contaminou.

        Um arbitro que a parte suspeita ajuda a escolher nao arbitra nada.
    """
    centro = np.median(np.array([p for p, _s in medidas]), axis=0)
    ficam = [(p, s) for p, s in medidas
             if float(np.linalg.norm(p - centro)) <= limite * s]
    return ficam or medidas


def _duas_que_discordam(medidas, limite=DESVIOS_PARA_DISPARATE):
    """Com DUAS, nao ha arbitro. Entao nao se escolhe: acredita-se menos.

    Se as duas discordam alem do que os sigmas admitem, uma delas esta
    errada e nao ha como saber qual. Descartar a "pior" seria escolher pelo
    sigma — que e justamente o numero em que nao se pode confiar quando ha
    disparate.

        Com duas testemunhas que se contradizem, a resposta nao e escolher
        uma: e registrar que a duvida cresceu.

    A saida honesta e manter as duas e INFLAR o sigma pela discordancia. A
    posicao sai no meio, e quem consome sabe que ela vale menos.
    """
    (p1, s1), (p2, s2) = medidas
    separacao = float(np.linalg.norm(p1 - p2))
    admissivel = limite * float(np.hypot(s1, s2))
    if separacao <= admissivel or separacao <= 0:
        return medidas
    inflado = separacao / (2.0 * limite)
    return [(p1, max(s1, inflado)), (p2, max(s2, inflado))]


def _ler_homografia(calib, papel):
    """(H, (largura, altura)) do papel, ou (None, None).

    O arquivo da camera do alto vive em `homografia.json` desde o comeco, e
    continua la: `rodar.py`, `mapear.py` e o `--mono` leem de la. Os outros
    papeis usam `homografia-<papel>.json`.
    """
    candidatos = [calib / f"homografia-{papel}.json"]
    if papel == "alto":
        candidatos.append(calib / "homografia.json")
    caminho = next((c for c in candidatos if c.exists()), None)
    if caminho is None:
        return None, None
    try:
        d = json.loads(caminho.read_text(encoding="utf-8"))
        H = np.array(d["H"], dtype=float)
        larg, alt = d.get("resolucao", (640, 480))
        return H, (int(larg), int(alt))
    except Exception:
        return None, None


def _ler_intrinseca(calib, papel, larg, alt, leitor):
    """(K, dist, origem) do tabuleiro, ou (None, None, "").

    A DISTORCAO VEM JUNTO, E ELA E METADE DO MOTIVO DE RODAR O TABULEIRO.

    Defeito achado em 20/08, antes da primeira corrida: `carregar` recebia
    `dist` de `_ler_homografia`, que nunca teve essa informacao e devolvia
    None sempre. O `intrinseca-<papel>.json` GRAVA os coeficientes desde o
    primeiro dia — e ninguem lia.

    O resultado seria mudo: `sem_distorcao` viraria uma funcao que nao faz
    nada, o barril da C920 continuaria torcendo as bordas, e o erro
    apareceria como centimetros no chao sem nada apontando para a lente.

        Um dado gravado que ninguem le e pior que um dado ausente: o ausente
        aparece na primeira execucao.
    """
    caminho = calib / f"intrinseca-{papel}.json"
    if not caminho.exists():
        return None, None, ""
    K = leitor(caminho, largura_px=larg, altura_px=alt)
    if K is None:
        return None, None, ""
    dist = None
    try:
        d = json.loads(caminho.read_text(encoding="utf-8"))
        bruto = d.get("dist")
        if bruto:
            # Os coeficientes sao adimensionais, em coordenadas normalizadas.
            # Ao contrario de K, eles NAO se reescalam com a resolucao.
            dist = np.asarray(bruto, dtype=float).ravel()
    except Exception:
        dist = None
    return K, dist, "tabuleiro"
