"""
Gabarito — comparar o que o sistema LEU com o que de fato ACONTECEU.

O PROBLEMA QUE ISTO RESOLVE, E ELE JA CUSTOU UM DIA

Em 10/08 foram tres rodadas mexendo em limiares de estabilidade e de giro. A
contagem de mudancas de locomocao foi 12 -> 16 -> 17. E entao a percepcao do
obvio, escrita no caderno daquele dia:

    Eu nao sei se 17 esta certo. Nao tenho registro do que o Eduardo realmente
    fez. Estive afinando um numero que nao consigo avaliar.

    Sem registro do que aconteceu, nao ha como julgar o que o sistema disse
    que aconteceu.

O sistema publica `andando_frente`. Ninguem consegue dizer se a pessoa andou
para frente. Sem essa comparacao, toda mudanca de parametro e feita as cegas —
e ajustar as cegas foi exatamente o que aconteceu duas vezes: em 08/08
otimizando inferencia enquanto metade do tempo ia para o desenho, e em 10/08
mexendo em limiares sem saber o alvo.

A SAIDA E BARATA: DECLARAR ANTES

A pessoa declara o que vai fazer, faz, e o sistema anota o que leu naquela
janela. A comparacao vira nota. Nao ha modelo novo, nao ha anotacao manual de
video, nao ha custo nenhum alem de seguir um roteiro por dois minutos.

QUATRO RESULTADOS, NAO DOIS

Certo e errado nao bastam, porque escondem os dois casos mais informativos:

    certo        leu exatamente o que aconteceu
    pobre        leu algo mais generico, mas nao contraditorio.
                 `andando` quando a pessoa andou para frente e POBRE: o
                 sistema se absteve de dizer a direcao porque o azimute nao
                 convergiu. Ele nao errou — respondeu menos.
    errado       leu outra coisa. E o unico que exige conserto no codigo.
    sem leitura  nao havia pessoa nenhuma. Falha de DETECCAO, nao de
                 classificacao, e o conserto fica em outro lugar.

Somar `pobre` com `errado` faria uma abstencao honesta parecer um defeito, e
empurraria para "consertar" um sistema que estava se comportando exatamente
como projetado. Somar `sem leitura` com `errado` mandaria procurar defeito no
classificador quando o problema esta na camera.

    Nota que mistura causas manda consertar o lugar errado.
"""

from collections import Counter
from dataclasses import dataclass, field

CERTO = "certo"
POBRE = "pobre"
ERRADO = "errado"
SEM_LEITURA = "sem_leitura"


@dataclass
class Passo:
    """Uma instrucao do roteiro, e o que o sistema deveria responder.

    `eixo` e UM so de proposito. Um passo que cobra locomocao, postura e os
    dois bracos ao mesmo tempo produz uma nota que nao diz o que falhou — e o
    numero existe justamente para apontar onde mexer.
    """

    acao: str
    instrucao: str
    eixo: str
    certo: tuple
    pobre: tuple = ()
    segundos: float = 6.0
    instrucao_extra: str = ""

    # ESTE PASSO TIRA A PESSOA DO LUGAR?
    #
    # Sem retorno entre eles, cada deslocamento empurra a pessoa mais para
    # longe do centro, e depois de dois ou tres passos ela esta fora do
    # enquadramento. Os passos seguintes entao nao medem o classificador:
    # medem uma pessoa que nao esta mais no quadro, e a nota culpa o codigo
    # por um defeito do roteiro.
    #
    #     Um roteiro que expulsa o sujeito da cena mede a saida dele.
    desloca: bool = False

    # DOIS PASSOS SEM NOTA, COM PAPEIS DIFERENTES.
    #
    # Ambos tem `eixo=None`, mas nao sao intercambiaveis:
    #
    #     aquecimento    obrigatorio sempre — o azimute e a inclinacao so
    #                    aprendem ali. Tirar significa reprovar o sistema por
    #                    falta de materia-prima.
    #     reposicionar   necessario so onde o passo anterior tirou a pessoa do
    #                    lugar. Num recorte de `--acao`, os pares que se
    #                    desfaziam somem e o retorno precisa ser recolocado —
    #                    mas colocar dois seguidos so faz a pessoa esperar.
    #
    # Distinguir pelo `eixo` trataria os dois igual e foi o que produziu
    # `['aquecer', 'andar_frente', 'voltar', 'voltar']`.
    reposiciona: bool = False

    # PARTE DO PASSO QUE NAO CONTA, E POR QUE ELA PRECISA EXISTIR.
    #
    # No comeco de cada passo a pessoa ainda esta comecando o movimento, E o
    # `Estavel` do classificador ainda nao se comprometeu — ele exige 0,35 s de
    # concordancia antes de mudar de estado, por decisao tomada em 10/08.
    #
    # Sem esta margem, todo passo seria penalizado pela propria transicao, e a
    # nota mediria tempo de reacao em vez de acerto. Pior: a penalidade seria
    # maior nos passos curtos, o que faria a nota depender da duracao escolhida
    # no roteiro em vez de depender do sistema.
    acomodacao_s: float = 1.2


def ir_para_a_borda(qual="DE TRAS", segundos=5.0):
    """Reposiciona na BORDA, e nao no meio. Dobra o espaco de caminhada.

    MEDIDO EM 11/08: a area util e de 140 x 140 cm. Comecando no meio sobram
    70 cm para cada lado; comecando na borda, ha 140 cm inteiros para
    atravessar. O mesmo espaco fisico, o dobro de caminhada.

    E os 70 cm eram o problema. Em 6 segundos eles dao 0,12 m/s — abaixo do
    limiar de 0,25 que separa `parado` de `andando`. O sistema respondia
    `parado` e estava CERTO: a pessoa estava praticamente parada.

        A area nao era pequena demais para medir. O roteiro e que estava
        usando metade dela.
    """
    return Passo("posicionar", f"VA ATE A BORDA {qual} DA AREA",
                 eixo=None, certo=(), segundos=segundos, reposiciona=True,
                 instrucao_extra="so reposiciona, nao vale nota")


def voltar_ao_meio(segundos=6.0):
    """Reposiciona sem valer nota. Existe por causa do enquadramento.

    Poderia ser resolvido com "ande menos", e seria pior: andar pouco e o que
    mantem a velocidade abaixo do limiar de 0,25 m/s e faz tudo sair `parado`.
    O roteiro precisa de caminhada LARGA e de pessoa CENTRADA, e as duas so
    convivem com um retorno explicito.

    `eixo=None` faz o placar ignorar estes quadros — o que a pessoa faz aqui
    nao e uma acao declarada e nao pode contar a favor nem contra.
    """
    return Passo("voltar", "VOLTE PARA O MEIO, de frente para a camera",
                 eixo=None, certo=(), segundos=segundos, reposiciona=True,
                 instrucao_extra="so reposiciona, nao vale nota")


def roteiro_padrao():
    """As acoes que o gemeo precisa saber ler, uma por passo.

    A ORDEM NAO E ARBITRARIA

    Andar vem antes de tudo que depende de rumo do corpo, porque o
    `EstimadorDeAzimute` so aprende com quem anda: ele precisa comparar a
    linha dos ombros com a direcao do deslocamento. Pedir "vire para a
    direita" antes de qualquer caminhada mediria um estimador que ainda nao
    teve materia-prima — e reprovaria o sistema por uma coisa que o roteiro
    causou.

    Foi a mesma licao de 10/08 com o estimador de inclinacao: ele apareceu com
    0 amostras porque a sessao nao teve caminhada suficiente. Nao estava
    quebrado, estava sem materia-prima.

        Um componente que exige movimento nao pode ser julgado por uma
        sessao sem movimento.
    """
    from src.acao.vocabulario import Braco, Locomocao, Postura

    def andar(nome, instrucao, esperado, segundos=4.0):
        return Passo(nome, instrucao, eixo="locomocao", certo=(esperado,),
                     pobre=(Locomocao.ANDANDO,), segundos=segundos,
                     acomodacao_s=0.8, desloca=True)

    def ir(instrucao, segundos=4.0):
        return Passo("posicionar", instrucao, eixo=None, certo=(),
                     segundos=segundos, reposiciona=True)

    return [
        # ROTEIRO ESCRITO PELO EDUARDO, 11/08. Substitui o meu.
        #
        # A diferenca nao e de gosto, e de como gente funciona: CADA ACAO
        # VOLTA AO NEUTRO ANTES DA SEGUINTE. Ninguem emenda "ande de lado" em
        # "agache" sem passar por ficar em pe no meio — e quando o roteiro
        # pede isso, a pessoa improvisa a transicao e a transicao entra na
        # medicao.
        #
        #     Roteiro que emenda movimentos mede as emendas.
        #
        # E ha um ganho que eu nao tinha visto: OS PROPRIOS PASSOS DE IR ATE A
        # BORDA SAO O AQUECIMENTO. Sao cinco trechos em que a pessoa anda
        # olhando para onde vai, que e exatamente a hipotese do
        # `EstimadorDeAzimute`. O passo dedicado de 14 s existia porque eu nao
        # tinha reparado que o roteiro ja produzia as amostras.
        #
        # Isso so funciona porque o azimute passou a usar a MODA e nao a
        # media: `VOLTE DE RE` e `ANDE DE LADO` produzem amostras a 180 e 90
        # graus, e a moda as descarta como minoria em vez de cair no meio.
        ir("VA ATE A BORDA DE TRAS"),
        andar("andar_frente", "ANDE PARA FRENTE ATRAVESSANDO A AREA",
              Locomocao.FRENTE),
        andar("andar_tras", "VOLTE DE RE ate a borda", Locomocao.TRAS),

        ir("VA PARA O MEIO", 3.0),
        ir("VA ATE A BORDA DA DIREITA"),
        ir("VOLTE PARA O MEIO", 3.0),
        andar("andar_esquerda", "ANDE DE LADO PARA A SUA ESQUERDA",
              Locomocao.ESQUERDA, segundos=3.0),
        ir("VOLTE PARA O MEIO", 3.0),

        Passo("parado", "FIQUE PARADO", eixo="locomocao",
              certo=(Locomocao.PARADO,), segundos=5),

        # POSTURA: agachar e levantar. O retorno nao e simetria — um estado
        # que so entra e nunca sai nao e estado, e armadilha.
        Passo("agachar", "AGACHE", eixo="postura",
              certo=(Postura.AGACHADO,), pobre=(Postura.DESCONHECIDA,),
              segundos=4),
        Passo("levantar", "LEVANTE", eixo="postura",
              certo=(Postura.EM_PE,), segundos=4),

        # BRACOS: cada levantada tem a sua baixada, e cada uma vale nota.
        #
        # Na minha versao a baixada vinha embutida na instrucao seguinte
        # ("BAIXE O DIREITO e LEVANTE O ESQUERDO") — dois movimentos num passo
        # so, com uma nota so. Se o sistema perdesse a descida do direito,
        # nada acusaria. Separados, cada um responde por si.
        Passo("braco_dir_levantado", "LEVANTE O BRACO DIREITO",
              eixo="braco_direito", certo=(Braco.LEVANTADO,), segundos=4),
        Passo("braco_dir_baixado", "BAIXE O BRACO DIREITO",
              eixo="braco_direito", certo=(Braco.AO_LADO,), segundos=4),
        Passo("braco_esq_levantado", "LEVANTE O BRACO ESQUERDO",
              eixo="braco_esquerdo", certo=(Braco.LEVANTADO,), segundos=4),
        Passo("braco_esq_baixado", "BAIXE O BRACO ESQUERDO",
              eixo="braco_esquerdo", certo=(Braco.AO_LADO,), segundos=4),

        # `PARADO` NO FIM RESPONDE OUTRA PERGUNTA QUE `PARADO` NO MEIO.
        #
        # No meio ele mede um retrato. No fim, depois de andar, agachar e
        # levantar os dois bracos, ele pergunta: o sistema VOLTA a parado, ou
        # grudou no ultimo estado?
        Passo("parado_fim", "FIQUE PARADO", eixo="locomocao",
              certo=(Locomocao.PARADO,), segundos=5,
              instrucao_extra="ultimo passo"),
    ]


@dataclass
class Contagem:
    """O que o sistema respondeu durante UM passo."""

    passo: Passo
    veredictos: Counter = field(default_factory=Counter)
    respostas: Counter = field(default_factory=Counter)
    alturas: list = field(default_factory=list)
    ids: set = field(default_factory=set)

    # OS NUMEROS CRUS QUE PRODUZIRAM A NOTA.
    #
    # Sem eles, `parado` em 66% dos quadros de alguem que andou tem duas
    # explicacoes indistinguiveis: a pessoa nao andou, ou o sistema mediu a
    # caminhada dela como um quinto do que foi. As duas mandam consertar
    # lugares opostos — uma pede outra sessao, a outra pede recalibrar a
    # homografia.
    #
    #     Nota sem o numero cru nao aponta conserto.
    velocidades: list = field(default_factory=list)
    posicoes: list = field(default_factory=list)

    @property
    def velocidade_mediana(self):
        if not self.velocidades:
            return None
        return float(sorted(self.velocidades)[len(self.velocidades) // 2])

    @property
    def deslocamento(self):
        """Distancia entre o ponto mais distante e o mais proximo do inicio.

        Nao e a soma dos passinhos: ruido de medicao infla soma de trechos e
        faria uma pessoa parada "percorrer" metros. Extremos nao inflam.
        """
        if len(self.posicoes) < 2:
            return None
        x0, y0 = self.posicoes[0]
        return max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
                   for x, y in self.posicoes)

    @property
    def total(self):
        return sum(self.veredictos.values())

    @property
    def nota(self):
        """Fracao de quadros lidos exatamente certo. 0 a 1."""
        if not self.total:
            return 0.0
        return self.veredictos[CERTO] / self.total

    @property
    def aproveitamento(self):
        """Certo + pobre. Quanto o sistema NAO se contradisse.

        Serve para separar dois diagnosticos muito diferentes: nota 30% com
        aproveitamento 95% e um sistema que se absteve quase sempre — falta
        materia-prima, nao ha bug. Nota 30% com aproveitamento 35% e um
        sistema que esta afirmando outra coisa, e ai ha bug.
        """
        if not self.total:
            return 0.0
        return (self.veredictos[CERTO] + self.veredictos[POBRE]) / self.total

    @property
    def pior_confusao(self):
        """O que ele mais disse quando errou, e quanto.

        Numero de erro sozinho nao aponta conserto. Em 10/08 o painel disse
        `rejeitadas plausibilidade 358` e a pergunta que importava — 358 de
        quantas, e em favor de que? — nao tinha resposta na tela.
        """
        erradas = [(r, n) for r, n in self.respostas.items()
                   if r not in self.passo.certo and r not in self.passo.pobre]
        if not erradas or not self.total:
            return None, 0.0
        r, n = max(erradas, key=lambda rn: rn[1])
        return r, n / self.total


class Placar:
    """Acumula os veredictos e monta o boletim.

    NAO CONTA QUADRO PREVISTO, PELA MESMA REGRA DO MAPA DE CALOR

    Quando o Kalman esta so prevendo, o sistema esta dizendo onde a pessoa
    DEVERIA estar, nao onde ela foi vista. Pontuar essa resposta misturaria
    erro de classificacao com falta de medicao — e o mapa de calor ja aprendeu
    essa licao em 10/08:

        O mapa de calor so acumula posicao MEDIDA. Sem isso, o sistema
        registraria permanencia num lugar onde ninguem foi visto.
    """

    def __init__(self, contar_previstos=False):
        self.contar_previstos = contar_previstos
        self.contagens = {}
        self.previstos_ignorados = 0

    def anotar(self, passo, acoes, decorrido_s, pessoas=None):
        """Anota um quadro.

        `acoes`    {id: (Acao, mudancas)} do SpatialEngine
        `pessoas`  {id: EstadoDePessoa} do gemeo — posicao e `prevendo`.
                   Fica FORA do `Acao` de proposito: `Acao` e vocabulario
                   fechado, e nem "onde ela esta" nem "ha quantos quadros
                   ninguem a ve" sao acoes dela.
        """
        if passo.eixo is None:
            return
        if decorrido_s < passo.acomodacao_s:
            return

        pessoas = pessoas or {}
        c = self.contagens.setdefault(passo.acao, Contagem(passo))

        if not acoes:
            c.veredictos[SEM_LEITURA] += 1
            return

        # Uma pessoa por vez, que e o limite declarado do sistema hoje: sem
        # re-identificacao por aparencia, com duas pessoas em cena nao se sabe
        # qual pose pertence a qual corpo. Pontuar a segunda seria pontuar um
        # palpite.
        pid = sorted(acoes)[0]
        acao = acoes[pid][0]
        c.ids.add(pid)

        p = pessoas.get(pid)
        if p is not None and getattr(p, "prevendo", 0) \
                and not self.contar_previstos:
            self.previstos_ignorados += 1
            return

        # Os numeros crus entram ANTES do veredicto, porque valem mesmo quando
        # a classificacao erra — e principalmente quando ela erra.
        c.velocidades.append(float(getattr(acao, "velocidade_ms", 0.0)))
        if p is not None:
            c.posicoes.append((float(p.x), float(p.y)))

        resposta = getattr(acao, passo.eixo, None)
        c.respostas[resposta] += 1

        if resposta in passo.certo:
            c.veredictos[CERTO] += 1
        elif resposta in passo.pobre:
            c.veredictos[POBRE] += 1
        else:
            c.veredictos[ERRADO] += 1

        # A altura coletada e a do braco QUE O PASSO COBRA.
        #
        # A primeira versao guardava os dois lados, e a mediana saia no meio do
        # caminho entre a mao levantada e a mao pendurada — um numero que nao
        # correspondia a nenhuma das duas. Um teste com a mao a 1,42 m recebeu
        # 1,16 m, que e exatamente a media entre 1,42 e 0,90.
        #
        #     Misturar dois lados num numero so produz um terceiro numero que
        #     nao descreve nenhum deles.
        campo = {"braco_direito": "altura_mao_dir",
                 "braco_esquerdo": "altura_mao_esq"}.get(passo.eixo)
        if campo:
            v = getattr(acao, campo, None)
            if v is not None:
                c.alturas.append(v)

    # ------------------------------------------------------------- boletim
    def linhas(self, reprovar_abaixo_de=0.70):
        if not self.contagens:
            return ["Nenhum passo pontuado."]

        linhas = ["ACAO DECLARADA          QUADROS  CERTO  POBRE  ERRADO  "
                  "S/LEITURA   PIOR CONFUSAO",
                  "-" * 92]

        for acao, c in self.contagens.items():
            t = max(1, c.total)
            v = c.veredictos
            confusao, frac = c.pior_confusao
            texto = f"{confusao} ({frac:.0%})" if confusao else "-"
            marca = "  <- FALHOU" if c.nota < reprovar_abaixo_de else ""
            linhas.append(
                f"{acao:22} {c.total:7}  {v[CERTO] / t:5.0%}  "
                f"{v[POBRE] / t:5.0%}  {v[ERRADO] / t:6.0%}  "
                f"{v[SEM_LEITURA] / t:9.0%}   {texto}{marca}")

        linhas += self._linhas_de_movimento()
        linhas += ["", self._resumo(reprovar_abaixo_de)]

        alturas = [a for c in self.contagens.values() for a in c.alturas]
        if alturas:
            linhas.append(
                f"ALTURA DA MAO   medidas {len(alturas)}   "
                f"faixa {min(alturas):.2f} a {max(alturas):.2f} m")
            linhas.append(
                "    CONFIRA COM FITA METRICA. O sistema nao tem como saber "
                "se este numero esta certo.")

        if self.previstos_ignorados:
            linhas.append(f"\n{self.previstos_ignorados} quadros ignorados: "
                          f"posicao prevista pelo Kalman, nao medida.")
        return linhas

    # LIMIAR DO CLASSIFICADOR. Repetido aqui de proposito e nao importado:
    # esta secao existe para JULGAR aquele numero, e se ela o importasse, os
    # dois mudariam juntos e a comparacao perderia o sentido.
    ANDAR_ACIMA = 0.25

    def _linhas_de_movimento(self):
        """Quanto o sistema MEDIU de caminhada. A fita metrica do chao.

        POR QUE ESTA SECAO EXISTE

        MEDIDO EM 11/08: `andar_frente` leu `parado` em 66% dos quadros de
        alguem que estava andando. Duas explicacoes, indistinguiveis pela nota:

            a pessoa mal se deslocou
            a homografia encolhe a distancia, e 1 m real vira 0,2 m medido

        A primeira pede outra sessao. A segunda pede recalibrar. Escolher entre
        elas no chute seria a terceira rodada de ajuste as cegas deste projeto
        — depois de 08/08 (otimizar inferencia enquanto o custo estava no
        desenho) e de 10/08 (mexer em limiares sem gabarito).

        A saida e a mesma das outras duas vezes: MEDIR. O sistema diz quantos
        metros viu; quem andou sabe quantos andou. A discordancia entre os dois
        e a resposta, e ela nao precisa de mais nenhuma execucao.
        """
        andantes = [(a, c) for a, c in self.contagens.items()
                    if c.passo.eixo == "locomocao" and c.velocidades]
        if not andantes:
            return []

        linhas = ["", "MOVIMENTO MEDIDO   (compare com o que voce fez)",
                  f"{'ACAO':22} {'v mediana':>10} {'v maxima':>10} "
                  f"{'deslocamento':>13}"]

        suspeitos = []
        for acao, c in andantes:
            desloc = c.deslocamento
            texto = "     --" if desloc is None else f"{desloc:9.2f} m"
            linhas.append(
                f"{acao:22} {c.velocidade_mediana:8.2f} m/s "
                f"{max(c.velocidades):8.2f} m/s {texto}")

            esperava_andar = c.passo.acao != "parado"
            if esperava_andar and c.velocidade_mediana < self.ANDAR_ACIMA:
                suspeitos.append((acao, c))

        if suspeitos:
            linhas += [
                "",
                f"  {len(suspeitos)} passo(s) de caminhada com velocidade abaixo",
                f"  de {self.ANDAR_ACIMA} m/s — que e o limiar de 'andando'. Por isso",
                "  sairam `parado`. Duas explicacoes, e o deslocamento decide:",
                "",
                "    voce andou MENOS que o deslocamento acima",
                "      -> o roteiro precisa de mais espaco, nao ha bug",
                "",
                "    voce andou MAIS que o deslocamento acima",
                "      -> a homografia esta encolhendo a distancia.",
                "         Recalibrar: python calibracao/homografia.py",
                "         Nenhum limiar deve ser tocado antes disso."]
        return linhas

    def _resumo(self, limite):
        notas = [c.nota for c in self.contagens.values()]
        geral = sum(notas) / len(notas)
        reprovados = [a for a, c in self.contagens.items() if c.nota < limite]
        aprov = sum(c.aproveitamento for c in self.contagens.values()) / len(notas)

        texto = (f"NOTA GERAL {geral:.0%}   "
                 f"(sem contradicao {aprov:.0%})   "
                 f"{len(reprovados)} acoes abaixo de {limite:.0%}")
        if reprovados:
            texto += "\n  reprovadas: " + ", ".join(reprovados)
        return texto

    def para_dicionario(self):
        return {
            "acoes": {
                acao: {
                    "eixo": c.passo.eixo,
                    "esperado": list(c.passo.certo),
                    "quadros": c.total,
                    "nota": round(c.nota, 4),
                    "aproveitamento": round(c.aproveitamento, 4),
                    "veredictos": dict(c.veredictos),
                    "respostas": dict(c.respostas),
                    "ids": sorted(c.ids),
                    "alturas_m": [round(a, 3) for a in c.alturas],
                    "velocidade_mediana": (
                        None if c.velocidade_mediana is None
                        else round(c.velocidade_mediana, 3)),
                    "velocidade_maxima": (round(max(c.velocidades), 3)
                                          if c.velocidades else None),
                    "deslocamento_m": (None if c.deslocamento is None
                                       else round(c.deslocamento, 3)),
                }
                for acao, c in self.contagens.items()
            },
            "previstos_ignorados": self.previstos_ignorados,
        }
