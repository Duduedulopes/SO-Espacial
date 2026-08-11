"""
Fusao de vistas: cada camera contribui com o eixo que ela realmente enxerga.

O PROBLEMA QUE ISTO RESOLVE

Ate 08/08 o esqueleto saia torto e nenhuma suavizacao consertava. A causa nao
era o codigo: era a VISTA.

O MediaPipe foi treinado com imagens frontais de pessoas. Uma camera no teto
olhando para baixo e uma vista fora da distribuicao de treino — ele nao esta
errando por ruido, esta adivinhando o tempo todo.

A IDEIA

Nao forcar as tres cameras a concordarem num sistema comum de coordenadas
(isso exigiria tabuleiro, calibracao intrinseca e extrinseca de cada uma).
Em vez disso, usar cada uma no que ela e naturalmente boa.

    camera do ALTO    ->  ONDE a pessoa esta no chao       (homografia, 2-5 cm)
    camera de FRENTE  ->  largura e altura do corpo
    camera de LADO    ->  a profundidade que a frontal nao ve

POR QUE OS EIXOS SE COMPLEMENTAM

O MediaPipe devolve (x, y, z) nos eixos da camera:
    x = horizontal na imagem   y = vertical na imagem   z = profundidade

E a profundidade e SEMPRE o eixo mais fraco — e o unico que a camera nao mede,
so estima.

    De frente:  x = esquerda-direita da pessoa   CONFIAVEL
                y = cima-baixo                   CONFIAVEL
                z = frente-tras                  fraco

    De lado:    x = frente-tras da pessoa        CONFIAVEL
                y = cima-baixo                   CONFIAVEL
                z = esquerda-direita             fraco

O eixo fraco de uma e o eixo forte da outra. Pegamos de cada uma so o que ela
mede de verdade, e descartamos o que ela chuta.

Isso nao e triangulacao rigorosa — nao ha geometria comum entre as cameras.
E fusao por competencia: cada sensor responde a pergunta que sabe responder.
Custa zero calibracao, e ataca a causa real do esqueleto torto.

SISTEMA DE COORDENADAS DA PESSOA (antes de ir para o mundo)
    +x  para a direita da pessoa
    +y  para a frente da pessoa
    +z  para cima
"""

import numpy as np

# COCO-17, mesma ordem do resto do projeto
OMBRO_E, OMBRO_D = 5, 6
QUADRIL_E, QUADRIL_D = 11, 12
TORNOZELO_E, TORNOZELO_D = 15, 16


def _escala_do_tronco(j):
    """Distancia quadril->ombros. Serve de regua para igualar as vistas.

    Duas cameras podem devolver o mesmo corpo em escalas ligeiramente
    diferentes. Antes de misturar eixos, e preciso que 1 metro numa signifique
    1 metro na outra — senao o corpo sai desproporcional.
    """
    if j is None:
        return None
    ombros = j[[OMBRO_E, OMBRO_D]].mean(axis=0)
    quadris = j[[QUADRIL_E, QUADRIL_D]].mean(axis=0)
    d = float(np.linalg.norm(ombros - quadris))
    return d if d > 0.05 else None


def _normalizar(j):
    """Ancora no quadril e devolve (juntas, escala)."""
    if j is None:
        return None, None
    j = np.asarray(j, dtype=float).copy()
    j -= j[[QUADRIL_E, QUADRIL_D]].mean(axis=0)
    return j, _escala_do_tronco(j)


def fundir(frontal, lateral, lado="direita", peso_altura=0.5,
           vis_frontal=None, vis_lateral=None):
    """Monta um esqueleto usando o eixo forte de cada vista.

    Devolve (juntas (17,3), visivel (17,)) no referencial da PESSOA:
    x direita, y frente, z cima.

    VISIBILIDADE NAO E ENFEITE: E O QUE SEPARA MEDIDA DE INVENCAO

    O MediaPipe SEMPRE devolve as 17 juntas. Quando as pernas estao fora do
    quadro — como na webcam de notebook, que pega do peito para cima — ele
    entrega tornozelos assim mesmo, extrapolados. Sao numeros com a mesma cara
    dos medidos e nenhuma relacao com o corpo real.

    MEDIDO EM 10/08: fundir com esses valores produziu um amontoado de juntas
    de meio metro, ora em pe ora deitado. A matematica da fusao estava certa —
    provada com entrada sintetica limpa, largura 0,60 e altura 1,62 exatas.
    O que entrava e que era lixo.

        Nao desenhar o que nao foi visto. Um esqueleto sem pernas e honesto;
        um esqueleto com pernas inventadas e mentira com aparencia de dado.

    Quando so uma vista ve a junta, usamos o que ela oferece e o eixo que
    falta fica no plano do quadril. Meio esqueleto medido vale mais que um
    inteiro adivinhado.
    """
    f, esc_f = _normalizar(frontal)
    l, esc_l = _normalizar(lateral)

    if f is None and l is None:
        return None, None

    n = len(f) if f is not None else len(l)
    vf = _mascara(vis_frontal, n, f is not None)
    vl = _mascara(vis_lateral, n, l is not None)
    visivel = vf | vl
    if not visivel.any():
        return None, None

    # ---- so uma vista disponivel ----
    if l is None:
        # frontal: x=direita, y_img=baixo -> z=cima, z_mp=profundidade -> y
        return np.stack([f[:, 0], f[:, 2], -f[:, 1]], axis=1), visivel
    if f is None:
        s = -1.0 if lado == "direita" else 1.0
        return np.stack([l[:, 2], s * l[:, 0], -l[:, 1]], axis=1), visivel

    # ---- as duas: iguala a escala antes de misturar ----
    if esc_f and esc_l:
        l = l * (esc_f / esc_l)

    s = -1.0 if lado == "direita" else 1.0

    # Cada eixo vem de quem o mede bem — E SO de quem realmente viu a junta.
    direita = np.where(vf, f[:, 0], l[:, 2])
    frente = np.where(vl, s * l[:, 0], f[:, 2])

    # A altura as duas medem. Media so onde ambas viram; senao, quem viu.
    ambas = vf & vl
    cima = np.where(
        ambas,
        -(peso_altura * f[:, 1] + (1 - peso_altura) * l[:, 1]),
        np.where(vf, -f[:, 1], -l[:, 1]))

    return np.stack([direita, frente, cima], axis=1), visivel


def _mascara(vis, n, ha_vista):
    """Converte confianca em mascara booleana. Sem vista, nada e visivel."""
    if not ha_vista:
        return np.zeros(n, dtype=bool)
    if vis is None:
        return np.ones(n, dtype=bool)
    v = np.asarray(vis)
    return (v > 0.5) if v.dtype != bool else v


class Fusor:
    """Guarda a ultima pose de cada vista e funde quando ha material.

    As cameras nao sao sincronizadas. Guardamos a leitura mais recente de cada
    uma e fundimos com o que houver. Uma vista atrasada em 100 ms e melhor que
    vista nenhuma — e para POSTURA esse atraso importa muito menos que para
    posicao, porque a posicao vem da camera do alto, que e sincrona consigo
    mesma.
    """

    def __init__(self, lado_lateral="direita", validade_s=0.5):
        self.lado = lado_lateral
        self.validade = validade_s
        self.frontal = None
        self.lateral = None
        self.vis_frontal = None
        self.vis_lateral = None
        self.t_frontal = 0.0
        self.t_lateral = 0.0

    def ver_frontal(self, juntas, t, visivel=None):
        if juntas is not None:
            self.frontal, self.t_frontal = juntas, t
            self.vis_frontal = visivel

    def ver_lateral(self, juntas, t, visivel=None):
        if juntas is not None:
            self.lateral, self.t_lateral = juntas, t
            self.vis_lateral = visivel

    def esqueleto(self, agora):
        """Devolve (juntas, visivel). `visivel` nunca deve ser ignorado:
        e ele que impede o desenho de juntas extrapoladas."""
        vale_f = (agora - self.t_frontal) < self.validade
        vale_l = (agora - self.t_lateral) < self.validade
        return fundir(self.frontal if vale_f else None,
                      self.lateral if vale_l else None,
                      self.lado,
                      vis_frontal=self.vis_frontal if vale_f else None,
                      vis_lateral=self.vis_lateral if vale_l else None)

    @property
    def diagnostico(self):
        return f"F{'+' if self.frontal is not None else '-'}" \
               f"L{'+' if self.lateral is not None else '-'}"


def para_o_mundo(juntas_pessoa, x_m, y_m, rumo_rad):
    """Poe o esqueleto em pe no ponto do chao, virado para onde anda.

    juntas_pessoa: (17,3) no referencial da pessoa (x direita, y frente, z cima)
    (x_m, y_m): posicao vinda da homografia da camera do alto
    rumo_rad: direcao do movimento

    A altura vem do proprio esqueleto — do quadril ao tornozelo mais baixo.
    Nada de supor "pessoa tem 1,75 m": se ela agacha, o quadril desce sozinho.
    """
    j = np.asarray(juntas_pessoa, dtype=float).copy()

    c, s = np.cos(rumo_rad), np.sin(rumo_rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    j = (R @ j.T).T

    j[:, 2] -= min(j[TORNOZELO_E, 2], j[TORNOZELO_D, 2])
    j[:, 0] += x_m
    j[:, 1] += y_m
    return j
