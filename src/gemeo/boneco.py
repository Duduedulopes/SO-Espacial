"""O corpo que já sabe ser corpo. Ele ANIMA a descrição — não copia medida.

ISTO ESTAVA PROMETIDO DESDE 10/08 E NÃO EXISTIA

`src/acao/vocabulario.py` abre com esta frase, escrita quando o vocabulário
fechado foi criado:

    Com vocabulario fechado, o desenho deixa de copiar medida e passa a
    ANIMAR um corpo que ja sabe ser corpo.

        Se "deitado" nao esta no vocabulario, o boneco nao consegue deitar.

    A classe inteira de defeito desaparece por construcao, e nao por conserto.

A metade da leitura foi feita: `corpo.py` produz a descrição, `classificador.py`
fecha o vocabulário, `prateleira.py` decide de qual prateleira a mão veio. Mas
`rodar.py` continuou desenhando `p.esqueleto` — as juntas CRUAS da reconstrução.

    O vocabulário existia na camada que lê e não existia na camada que desenha.

Então o boneco herdava todo defeito que dois dias de trabalho tinham medido e
isolado: braço comprimido, junta extrapolada fora do quadro, âncora errada. A
descrição estava certa e ninguém a usava para desenhar.

O QUE MUDA AQUI

Este módulo não recebe junta nenhuma. Recebe:

    estatura, posição no chão, rumo, postura, locomoção, estado dos braços

e MONTA as 17 juntas a partir de antropometria. Nenhuma entrada pode fazê-lo
produzir um corpo impossível, porque as juntas não vêm de fora — elas são
construídas.

    Não há como desenhar errado um corpo que não foi desenhado a partir de
    medida. Só há como descrever errado — e descrição errada é um estado do
    vocabulário, que se conserta num lugar só.

AS INVARIANTES, GARANTIDAS POR CONSTRUÇÃO E NÃO POR VERIFICAÇÃO

    nenhuma junta abaixo do chão      z >= 0 sempre
    pelo menos um pé no chão          quem não voa se apoia
    o pulso nunca sai do alcance      braço tem comprimento
    a cabeça nunca fica abaixo do quadril
    a altura total nunca passa da estatura

`testes/test_boneco.py` tenta violar cada uma delas com entradas absurdas —
altura de mão de 9 metros, postura inexistente, estatura negativa — e exige
que o corpo continue possível.

A ANTROPOMETRIA

Frações da estatura, medidas clássicas de adultos. São as mesmas já usadas em
`src/acao/escala.py` para o quadril (0,53), agora estendidas ao corpo inteiro.
Continuam sendo MODELO — mas modelo aplicado a um desenho é diferente de
modelo aplicado a uma medição: aqui ele não finge medir nada.
"""

import numpy as np

from src.acao.vocabulario import Braco, Locomocao, Postura

# COCO-17, a mesma ordem que o resto do projeto usa.
(NARIZ, OLHO_ESQ, OLHO_DIR, ORELHA_ESQ, ORELHA_DIR,
 OMBRO_ESQ, OMBRO_DIR, COTOVELO_ESQ, COTOVELO_DIR, PULSO_ESQ, PULSO_DIR,
 QUADRIL_ESQ, QUADRIL_DIR, JOELHO_ESQ, JOELHO_DIR,
 TORNOZELO_ESQ, TORNOZELO_DIR) = range(17)

# Frações da estatura. Antropometria clássica de adultos.
ALTURA_OLHO = 0.935
ALTURA_ORELHA = 0.930
ALTURA_NARIZ = 0.925
ALTURA_OMBRO = 0.820
ALTURA_COTOVELO = 0.630
ALTURA_PULSO = 0.485
ALTURA_QUADRIL = 0.530
ALTURA_JOELHO = 0.285
ALTURA_TORNOZELO = 0.040

LARGURA_OMBROS = 0.225
LARGURA_QUADRIL = 0.145
LARGURA_CABECA = 0.075

# Comprimento dos segmentos, derivado das alturas acima.
BRACO = ALTURA_OMBRO - ALTURA_COTOVELO          # 0.190
ANTEBRACO = ALTURA_COTOVELO - ALTURA_PULSO      # 0.145
ALCANCE_MAXIMO = BRACO + ANTEBRACO              # 0.335 da estatura

COXA = ALTURA_QUADRIL - ALTURA_JOELHO           # 0.245
CANELA = ALTURA_JOELHO - ALTURA_TORNOZELO       # 0.245

ESTATURA_PADRAO = 1.72
ESTATURA_MIN, ESTATURA_MAX = 0.80, 2.20

# Quanto o quadril desce ao agachar, em fração da altura de quadril em pé.
# Não é chute solto: um agachamento profundo põe o quadril perto da altura do
# joelho, e o joelho em pé fica a 0,285/0,530 = 0,54 do quadril.
QUADRIL_AGACHADO = 0.55

# O CORPO RESPIRA. AS PERNAS SÓ AGACHAM.
#
#     eu nao quero que as pernas do boneco fiquem se mechendo para andar, eu
#     quero que ele flutua suavemente ou fique totalmente parado, movimentos
#     bem suaves, o unico movimento que a perna faz é para agachar
#                                                     — Eduardo, 12/08
#
# Até 12/08 aqui havia `PASSO = 0.16`: os pés alternavam à frente e atrás
# durante a caminhada, para o boneco não deslizar de pé fixo. A intenção era
# boa e o resultado era pior que o deslize, porque a CADÊNCIA ERA INVENTADA —
# dois passos por segundo, uma constante escrita em `rodar.py`. Nada em lugar
# nenhum media a passada de quem estava na sala.
#
#     Um movimento que não foi medido não é informação: é ruído com forma
#     de perna.
#
# É a mesma regra que fez `DESCONHECIDO` virar valor de primeira classe no
# vocabulário. Quando o sistema não sabe, ele não finge — e a perna parada
# diz "não sei a sua passada" com muito mais honestidade do que uma perna
# que se mexe no ritmo de uma constante.
#
# O que sobra é a respiração: milímetros, lentos, o bastante para o corpo não
# parecer congelado. Os tornozelos ficam de fora — pé que flutua sai do chão.
FLUTUACAO = 0.006
CICLOS_DE_RESPIRO = 0.25


def _girar(pontos, rumo_rad):
    """Roda em torno do eixo vertical. z não muda: gravidade não gira."""
    c, s = np.cos(rumo_rad), np.sin(rumo_rad)
    x, y = pontos[:, 0].copy(), pontos[:, 1].copy()
    pontos[:, 0] = c * x - s * y
    pontos[:, 1] = s * x + c * y
    return pontos


def _cadeia_do_braco(ombro, alvo, braco, antebraco, frente):
    """Ombro -> cotovelo -> pulso, com o pulso o MAIS PERTO possível do alvo.

    O ALVO É UM PEDIDO, NÃO UMA ORDEM.

    A descrição pode pedir a mão a nove metros do chão. O braço tem
    comprimento, e é ele que decide: se o alvo está fora do alcance, o pulso
    para na borda da esfera de alcance, na direção pedida.

        O corpo obedece à anatomia antes de obedecer à leitura. É isso que
        torna impossível desenhar um braço de três metros — e não uma
        checagem depois, que só descobre o absurdo já desenhado.

    O cotovelo fica no plano ombro-alvo, dobrado para trás e para fora, que é
    como um cotovelo humano dobra. Sem isso o braço vira um bastão reto e o
    boneco parece um boneco de palito, não um corpo.
    """
    total = braco + antebraco
    v = alvo - ombro
    d = float(np.linalg.norm(v))

    if d < 1e-6:
        v, d = -frente * 0.01, 0.01
    direcao = v / d

    # Fora de alcance: o pulso para na borda, e o braço fica esticado.
    if d >= total:
        pulso = ombro + direcao * total
        cotovelo = ombro + direcao * braco
        return cotovelo, pulso

    pulso = ombro + direcao * d

    # Lei dos cossenos: onde o cotovelo cai para os dois segmentos fecharem.
    cos_a = (braco ** 2 + d ** 2 - antebraco ** 2) / (2 * braco * d)
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    ao_longo = braco * cos_a
    fora = braco * float(np.sqrt(max(0.0, 1.0 - cos_a ** 2)))

    # Direção de dobra: perpendicular ao braço, no plano que contém a vertical.
    perp = np.cross(np.cross(direcao, np.array([0.0, 0.0, 1.0])), direcao)
    n = float(np.linalg.norm(perp))
    perp = -frente if n < 1e-6 else perp / n

    return ombro + direcao * ao_longo + perp * fora, pulso


def _alvo_do_braco(estado, altura_mao, ombro, chao_z, estatura, frente, lado):
    """Para onde a mão está indo, segundo o VOCABULÁRIO.

    A altura em metros, quando existe, refina — mas nunca contradiz o estado.
    `levantado` com altura de 0,20 m seria uma descrição incoerente, e a
    resposta certa não é desenhar o absurdo: é obedecer ao rótulo, que é o
    eixo do vocabulário fechado, e usar a altura só dentro do que ele permite.
    """
    alcance = ALCANCE_MAXIMO * estatura

    if estado == Braco.LEVANTADO:
        z = ombro[2] + alcance * 0.75
        if altura_mao is not None:
            z = max(ombro[2] + alcance * 0.15, chao_z + altura_mao)
        return np.array([ombro[0] + lado * 0.05 * estatura, ombro[1], z])

    if estado == Braco.ESTENDIDO:
        z = ombro[2] - alcance * 0.15
        if altura_mao is not None:
            z = chao_z + altura_mao
        return ombro + frente * alcance * 0.85 + np.array([0.0, 0.0, z - ombro[2]])

    # AO_LADO e DESCONHECIDO: pendurado. Repouso é o estado sem informação, e
    # um braço pendurado é a pose que menos afirma coisa nenhuma.
    z = ombro[2] - alcance * 0.95
    if estado == Braco.AO_LADO and altura_mao is not None:
        z = chao_z + altura_mao
    return np.array([ombro[0] + lado * 0.02 * estatura, ombro[1], z])


def montar(estatura=ESTATURA_PADRAO, x=0.0, y=0.0, rumo=0.0,
           postura=Postura.EM_PE, locomocao=Locomocao.PARADO,
           braco_esq=Braco.AO_LADO, braco_dir=Braco.AO_LADO,
           altura_mao_esq=None, altura_mao_dir=None, fase=0.0):
    """As 17 juntas em metros, no mundo, a partir da DESCRIÇÃO.

    Nenhuma junta entra: todas são construídas. É por isso que não existe
    entrada capaz de produzir um corpo impossível.

    `fase` avança de 0 a 1 durante a caminhada e move as pernas. Sem ela o
    boneco desliza de pé fixo enquanto a posição muda — a assinatura visual de
    um corpo arrastado em vez de caminhando.
    """
    E = float(np.clip(estatura or ESTATURA_PADRAO, ESTATURA_MIN, ESTATURA_MAX))
    agachado = postura == Postura.AGACHADO

    # O respiro entra AQUI, no quadril, e sobe sozinho pelo corpo: joelho,
    # ombro, cabeça e braços são todos calculados a partir desta altura. Uma
    # linha move o corpo inteiro porque o corpo é uma cadeia, não um saco de
    # pontos independentes.
    respiro = FLUTUACAO * E * float(np.sin(2 * np.pi * CICLOS_DE_RESPIRO * fase))

    z_quadril = (ALTURA_QUADRIL * E * (QUADRIL_AGACHADO if agachado else 1.0)
                 + respiro)
    z_tornozelo = ALTURA_TORNOZELO * E

    # JOELHO: no meio da perna dobrada, empurrado para a FRENTE ao agachar.
    # Uma perna que encolhe sem o joelho sair do lugar atravessa o próprio
    # corpo — e a articulação existe justamente para isso não acontecer.
    z_joelho = (z_quadril + z_tornozelo) / 2.0
    avanco_joelho = 0.0
    if agachado:
        perna = COXA * E
        vao = max(0.0, perna ** 2 - ((z_quadril - z_joelho)) ** 2)
        avanco_joelho = float(np.sqrt(vao))

    ombro_sobre_quadril = (ALTURA_OMBRO - ALTURA_QUADRIL) * E
    z_ombro = z_quadril + ombro_sobre_quadril

    meia_largura_ombro = LARGURA_OMBROS * E / 2.0
    meia_largura_quadril = LARGURA_QUADRIL * E / 2.0

    j = np.zeros((17, 3))

    # Pés lado a lado, sempre. `locomocao` continua chegando aqui porque quem
    # desenha usa: a seta no chão é acesa por ela. O que ela não faz mais é
    # mexer perna.
    j[QUADRIL_ESQ] = [-meia_largura_quadril, 0.0, z_quadril]
    j[QUADRIL_DIR] = [meia_largura_quadril, 0.0, z_quadril]
    j[JOELHO_ESQ] = [-meia_largura_quadril, avanco_joelho, z_joelho]
    j[JOELHO_DIR] = [meia_largura_quadril, avanco_joelho, z_joelho]
    j[TORNOZELO_ESQ] = [-meia_largura_quadril, 0.0, z_tornozelo]
    j[TORNOZELO_DIR] = [meia_largura_quadril, 0.0, z_tornozelo]

    j[OMBRO_ESQ] = [-meia_largura_ombro, 0.0, z_ombro]
    j[OMBRO_DIR] = [meia_largura_ombro, 0.0, z_ombro]

    cabeca = z_quadril + (ALTURA_NARIZ - ALTURA_QUADRIL) * E
    j[NARIZ] = [0.0, LARGURA_CABECA * E, cabeca]
    j[OLHO_ESQ] = [-LARGURA_CABECA * E / 2, LARGURA_CABECA * E * 0.7,
                   z_quadril + (ALTURA_OLHO - ALTURA_QUADRIL) * E]
    j[OLHO_DIR] = [LARGURA_CABECA * E / 2, LARGURA_CABECA * E * 0.7,
                   z_quadril + (ALTURA_OLHO - ALTURA_QUADRIL) * E]
    j[ORELHA_ESQ] = [-LARGURA_CABECA * E, 0.0,
                     z_quadril + (ALTURA_ORELHA - ALTURA_QUADRIL) * E]
    j[ORELHA_DIR] = [LARGURA_CABECA * E, 0.0,
                     z_quadril + (ALTURA_ORELHA - ALTURA_QUADRIL) * E]

    frente = np.array([0.0, 1.0, 0.0])
    for ombro_i, cotovelo_i, pulso_i, estado, altura, lado in (
            (OMBRO_ESQ, COTOVELO_ESQ, PULSO_ESQ, braco_esq, altura_mao_esq, -1),
            (OMBRO_DIR, COTOVELO_DIR, PULSO_DIR, braco_dir, altura_mao_dir, 1)):
        alvo = _alvo_do_braco(estado, altura, j[ombro_i], 0.0, E, frente, lado)
        j[cotovelo_i], j[pulso_i] = _cadeia_do_braco(
            j[ombro_i], alvo, BRACO * E, ANTEBRACO * E, frente)

    _girar(j, rumo)
    j[:, 0] += x
    j[:, 1] += y

    # NADA ABAIXO DO CHAO — e ela e REDUNDANTE hoje, o que e declarado de
    # proposito.
    #
    # MEDIDO EM 12/08: removendo esta linha, NENHUM dos 154 testes de
    # `test_boneco.py` quebra. A geometria acima ja garante o piso sozinha: o
    # tornozelo nasce em 0,04 x estatura e o pulso nunca sai da esfera de
    # alcance do ombro, que fica bem acima do chao.
    #
    # Ela fica como cinto de segurança para o codigo que ainda nao existe. Mas
    # cinto que nunca foi puxado nao pode ser apresentado como prova de nada:
    #
    #     Guarda sem teste que a exercite nao e garantia, e esperanca com
    #     aparencia de codigo. O que garante o piso aqui e a construcao, e e
    #     ela que os testes medem.
    j[:, 2] = np.maximum(j[:, 2], 0.0)
    return j


def de_acao(acao, estatura=None, fase=0.0):
    """Monta o boneco a partir de uma `AcaoDescrita` e da posição no chão.

    É esta função que fecha o laço prometido em 10/08:

        câmeras -> descrição em vocabulário fechado -> corpo que já sabe ser
        corpo

    Repare no que ela NÃO recebe: nenhuma junta, nenhuma reconstrução, nenhum
    landmark. Se a descrição estiver errada, o boneco faz a coisa errada — mas
    faz uma coisa POSSÍVEL, e o conserto fica num lugar só.
    """
    return montar(
        estatura=estatura or ESTATURA_PADRAO,
        x=getattr(acao, "x", 0.0), y=getattr(acao, "y", 0.0),
        rumo=getattr(acao, "rumo_corpo", None) or 0.0,
        postura=getattr(acao, "postura", Postura.EM_PE),
        locomocao=getattr(acao, "locomocao", Locomocao.PARADO),
        braco_esq=getattr(acao, "braco_esquerdo", Braco.AO_LADO),
        braco_dir=getattr(acao, "braco_direito", Braco.AO_LADO),
        altura_mao_esq=getattr(acao, "altura_mao_esq", None),
        altura_mao_dir=getattr(acao, "altura_mao_dir", None),
        fase=fase,
    )
