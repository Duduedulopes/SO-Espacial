"""
Monitor ao vivo — ver o que a CAMERA enxerga, no instante em que a acao acontece.

POR QUE ISTO PRECISOU EXISTIR

O boletim diz `braco_dir_levantado leu ao_lado em 73% dos quadros`. Verdadeiro
e insuficiente: nao diz se o MediaPipe perdeu o pulso, se achou o pulso no
lugar errado, ou se achou certo e o limiar recusou.

As tres pedem consertos diferentes — e nenhuma delas aparece num numero.

    Quando a leitura discorda da realidade, o proximo passo nao e outro
    numero: e olhar o que entrou.

O QUE ELE DESENHA, E O QUE CADA COR SIGNIFICA

    verde     junta MEDIDA (visibilidade acima do limiar)
    vermelho  junta que o MediaPipe devolveu mas nao viu — extrapolada

Essa distincao e a coisa mais importante da tela. O MediaPipe SEMPRE devolve
as 17 juntas, inclusive as que estao fora do quadro, e as inventadas tem a
mesma cara das medidas. Em 10/08 isso produziu um esqueleto com pernas
imaginarias que ninguem percebeu por uma sessao inteira.

Um pulso VERMELHO acima do ombro nao e um braco levantado: e um palpite.

CUSTO, E POR QUE ELE NAO E ESCONDIDO

Desenhar e mostrar custa caro, e mais ainda com gravacao de tela. MEDIDO EM
10/08: o mesmo sistema entregou 10,7 fps sem gravacao e 4,2 com. O ato de
medir altera a medida.

Por isso a janela e OPCIONAL e o painel avisa quando ela esta ligada. Comparar
um boletim com janela contra um sem janela compararia duas maquinas
diferentes.
"""

import cv2
import numpy as np

VERDE = (90, 220, 120)
VERMELHO = (70, 70, 240)
CINZA = (150, 150, 150)
BRANCO = (245, 245, 245)
PRETO = (20, 20, 20)
AMARELO = (80, 220, 240)

# Ossos do padrao COCO-17. So os do tronco e membros: olhos e orelhas poluem
# a tela sem informar nada sobre a acao.
OSSOS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # ombros e bracos
    (5, 11), (6, 12), (11, 12),                      # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),          # pernas
]

PULSO_ESQ, PULSO_DIR = 9, 10
OMBRO_ESQ, OMBRO_DIR = 5, 6


def _viu(conf, i):
    return conf is not None and float(np.asarray(conf)[i]) > 0.5


def desenhar_pose(imagem, juntas_2d, conf=None, escala=1.0):
    """Esqueleto sobre o quadro. Verde = medida, vermelho = extrapolada."""
    if juntas_2d is None:
        return imagem

    p = np.asarray(juntas_2d, dtype=float) * escala

    for a, b in OSSOS:
        # Osso so e desenhado quando as DUAS pontas foram vistas. Ligar uma
        # junta medida a uma inventada desenharia um membro que nao existe, e
        # com aparencia de medido.
        if _viu(conf, a) and _viu(conf, b):
            cv2.line(imagem, tuple(p[a].astype(int)), tuple(p[b].astype(int)),
                     VERDE, 2, cv2.LINE_AA)

    for i in range(17):
        cor = VERDE if _viu(conf, i) else VERMELHO
        raio = 5 if i in (PULSO_ESQ, PULSO_DIR, OMBRO_ESQ, OMBRO_DIR) else 3
        cv2.circle(imagem, tuple(p[i].astype(int)), raio, cor, -1, cv2.LINE_AA)

    # A LINHA DO OMBRO E A REFERENCIA QUE O CLASSIFICADOR USA.
    #
    # `levantado` exige o pulso ACIMA do ombro. Sem ver essa linha na tela, um
    # braco erguido na horizontal parece levantado para quem olha e nao para o
    # sistema — e a discordancia fica sem explicacao.
    for ombro, pulso in ((OMBRO_ESQ, PULSO_ESQ), (OMBRO_DIR, PULSO_DIR)):
        if not (_viu(conf, ombro) and _viu(conf, pulso)):
            continue
        y = int(p[ombro][1])
        cv2.line(imagem, (0, y), (imagem.shape[1], y), AMARELO, 1, cv2.LINE_AA)
        acima = p[pulso][1] < p[ombro][1]
        cv2.circle(imagem, tuple(p[pulso].astype(int)), 9,
                   AMARELO if acima else CINZA, 2, cv2.LINE_AA)
    return imagem


def desenhar_caixa(imagem, caixa, id_externo=None, escala=1.0):
    if caixa is None:
        return imagem
    x1, y1, x2, y2 = (int(v * escala) for v in caixa)
    cv2.rectangle(imagem, (x1, y1), (x2, y2), VERDE, 2)
    if id_externo is not None:
        cv2.putText(imagem, f"#{id_externo}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, VERDE, 1, cv2.LINE_AA)
    return imagem


def _rotular(imagem, texto, cor=BRANCO):
    cv2.rectangle(imagem, (0, 0), (imagem.shape[1], 22), PRETO, -1)
    cv2.putText(imagem, texto, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                cor, 1, cv2.LINE_AA)
    return imagem


def mosaico(quadros, poses=None, caixas=None, largura=420):
    """Uma faixa horizontal com todas as cameras lado a lado.

    Lado a lado e nao em janelas separadas de proposito: com tres janelas, o
    Windows empilha, esconde e reordena — e a gravacao de tela perde a
    correspondencia entre o que a pessoa fez e o que cada camera viu naquele
    instante. Uma imagem so, um instante so.
    """
    poses = poses or {}
    caixas = caixas or {}
    paineis = []

    for papel, quadro in quadros.items():
        img = quadro.imagem.copy()
        escala = largura / img.shape[1]
        img = cv2.resize(img, (largura, int(img.shape[0] * escala)))

        obs = poses.get(papel)
        if obs is not None:
            desenhar_pose(img, obs.juntas_2d, obs.conf_2d, escala)
        cx = caixas.get(papel)
        if cx is not None:
            desenhar_caixa(img, cx[0], cx[1], escala)

        _rotular(img, f"{papel}   {quadro.imagem.shape[1]}x"
                      f"{quadro.imagem.shape[0]}")
        paineis.append(img)

    if not paineis:
        return None

    altura = max(p.shape[0] for p in paineis)
    iguais = [cv2.copyMakeBorder(p, 0, altura - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=PRETO)
              for p in paineis]
    return np.hstack(iguais)


def faixa_de_leitura(largura, passo, acao, certo, acumulado, confirmar_s,
                     leitura=None, altura=132):
    """O que o sistema esta lendo AGORA, embaixo das imagens.

    Fica na mesma tela que as cameras porque a pergunta e sempre a mesma: o
    que ele leu, no instante em que EU fiz aquilo. Em telas separadas, essa
    correspondencia depende da memoria de quem assiste.
    """
    faixa = np.full((altura, largura, 3), 28, np.uint8)

    def escrever(texto, y, cor=BRANCO, tamanho=0.5):
        cv2.putText(faixa, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    tamanho, cor, 1, cv2.LINE_AA)

    escrever(f"PEDIDO: {passo.instrucao}", 22, AMARELO, 0.55)

    if acao is None:
        escrever("NINGUEM DETECTADO", 50, VERMELHO, 0.6)
        return faixa

    esperado = " ou ".join(passo.certo) if passo.certo else "-"
    atual = getattr(acao, passo.eixo, "?") if passo.eixo else "-"
    escrever(f"esperado: {esperado}", 46, CINZA)
    escrever(f"lendo:    {atual}", 66, VERDE if certo else VERMELHO, 0.55)

    braços = (f"braco E {acao.braco_esquerdo:12} "
              f"{'--' if acao.altura_mao_esq is None else f'{acao.altura_mao_esq:.2f}m'}"
              f"   |   braco D {acao.braco_direito:12} "
              f"{'--' if acao.altura_mao_dir is None else f'{acao.altura_mao_dir:.2f}m'}")
    escrever(braços, 88, CINZA, 0.45)

    extra = f"{acao.locomocao} / {acao.postura}   {acao.velocidade_ms:.2f} m/s"
    if leitura is not None and leitura.encolhimento is not None:
        extra += f"   coxa {leitura.encolhimento:.2f}"
    escrever(extra, 108, CINZA, 0.45)

    if certo:
        largura_barra = int((largura - 20) * min(1.0, acumulado / confirmar_s))
        cv2.rectangle(faixa, (10, altura - 12),
                      (10 + largura_barra, altura - 4), VERDE, -1)
    return faixa
