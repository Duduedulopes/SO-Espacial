"""De qual prateleira a mao veio — por EVIDENCIA CONJUNTA, nao por regua.

A CORRECAO DO EDUARDO, 12/08, E ELA REORGANIZA O PROBLEMA

    as cameras nao estao trabalhando juntas E PRECISO DE UM METODO QUE AS
    INFORMACOES SE COMPLETEM (...) o cliente so tem essas opcao, pegar algo
    da prateleira 1,2,3,4,5... NAO EXISTE OUTRA OPCAO

E, sobre a obsessao com o centimetro:

    um cliente pode agachar para pegar um produto, ou pode simplesmente
    esticar o braco para baixo para pegar algo... NAO E UMA CIENCIA EXATA

Ele esta certo, e o erro era de ENQUADRAMENTO DO PROBLEMA, nao de conta.

Eu passei dois dias tentando medir uma altura continua em metros para depois
comparar com faixas. Mas a resposta nao e um numero: e UMA DE CINCO. E o
caminho da regua exigia, ao mesmo tempo, o chao, a escala e a postura — tres
coisas frageis, que quebraram uma a uma:

    10/08  sem tornozelo visivel nao ha chao
    11/08  a ancora vinha da mesma vista do pulso e herdava o erro dela
    12/08  a fracao da caixa nao vale a fracao do corpo numa camera de teto

Tres consertos, tres vezes o mesmo resultado: numero plausivel e errado.

    Um bit sobrevive ao ruido que destroi um angulo.

Essa frase ja estava escrita neste projeto, sobre o rumo do corpo. Vale igual
aqui: CLASSIFICAR entre cinco opcoes aguenta um erro que MEDIR nao aguenta.

O QUE MUDA NA PRATICA

    antes   altura em metros -> compara com faixa -> acerta ou erra
    agora   varias evidencias fracas -> pontua as cinco -> a mais provavel

Nenhuma evidencia precisa ser metrica. Nenhuma decide sozinha. Nenhuma camera
precisa enxergar tudo — que e o criterio de projeto desde 11/08:

    nenhuma delas vai captar 100% de tudo, as 3 ja existem ao mesmo tempo
    para uma complementar a outra                          — Eduardo, 11/08

E DUAS POSTURAS PODEM APONTAR PARA A MESMA PRATELEIRA

Agachar e esticar o braco para baixo sao gestos diferentes com a mesma
intencao. Numa regua isso e impossivel de expressar — as duas dao alturas de
mao parecidas por caminhos incompativeis. Numa tabela de evidencias e trivial:
duas assinaturas, mesma saida.

    Vocabulario fechado nao e limitacao: e o que permite responder "a mais
    provavel" em vez de "nao sei".

NUNCA SEM RESPOSTA

    Se nenhuma delas corresponder ao banco, escolher a mais provavel — a
    acao tem que ter logica.                               — Eduardo, 09/08

Este modulo nunca devolve `None` quando ha alguem alcancando. Devolve a
prateleira mais provavel E a margem sobre a segunda colocada. Margem pequena
significa "provavel, nao certo" — e quem consome decide o que fazer com isso.
O que ele nao faz e calar.
"""

from collections import deque
from dataclasses import dataclass, field

NENHUMA = "nenhuma"


@dataclass
class Evidencia:
    """O que as tres cameras disseram sobre UM alcance, neste quadro.

    Todo campo e opcional: cada camera responde o que enxerga. Um campo `None`
    nao pesa contra nem a favor de prateleira nenhuma — ele simplesmente nao
    vota. E assim que a complementaridade acontece sem ninguem coordenar.

    NENHUM CAMPO E METRICO, E ISSO E DELIBERADO.
    """

    # (frontal ou lateral) pulso no corpo, em fracao de tronco.
    #     0 = altura do quadril      1 = altura do ombro
    # Razao entre duas medidas da MESMA vista: a escala se cancela.
    alcance: float | None = None

    # (frontal ou lateral) verticalidade da coxa. 1 = em pe, ~0.2 = agachado.
    # Nao depende do chao — e por isso sobreviveu quando o resto caiu.
    coxa: float | None = None

    # (frontal ou lateral) o rotulo do vocabulario fechado.
    braco: str | None = None

    # (alto) quanto a caixa encolheu em relacao ao padrao em pe daquela
    # pessoa. Agachar encurta a caixa vista de cima.
    encolhimento: float | None = None

    # (frontal ou lateral) o pulso FOI VISTO nesta vista?
    #
    # Falta de leitura tambem e evidencia. Uma webcam de mesa perde o pulso
    # que sobe demais ou desce demais, e QUAL delas perdeu diz de que lado da
    # faixa a mao estava. Ate 12/08 isso era jogado fora como falha.
    viu_frontal: bool | None = None
    viu_lateral: bool | None = None

    def vazia(self):
        return all(getattr(self, c) is None for c in
                   ("alcance", "coxa", "braco", "encolhimento",
                    "viu_frontal", "viu_lateral"))


@dataclass
class Assinatura:
    """Como as evidencias se comportam quando a mao esta NESTA prateleira.

    Aprendida com a pessoa fazendo o gesto, uma vez por prateleira — e o
    `ferramentas/conferir_altura.py` ja percorre exatamente esse roteiro.

    A MUDANCA DE PAPEL DO GABARITO

    Ele existia para calcular erro em centimetros. Os mesmos dados dizem algo
    muito mais util:

        na prateleira 5 o pulso fica acima do ombro, o corpo ereto, e a
        frontal costuma PERDER o pulso pela borda de cima

        na 1 a coxa sai da vertical, a caixa encolhe no alto, e a frontal
        perde o pulso pela borda de baixo

    Nenhum centimetro nisso, e as cinco se separam.

        A regua media o quanto erramos. A assinatura mede o que acontece.
        A segunda responde a pergunta que a loja faz.
    """

    id: str
    nome: str = ""
    altura: float | None = None        # so para relatorio humano

    alcance: tuple | None = None       # (centro, tolerancia)
    coxa: tuple | None = None
    encolhimento: tuple | None = None
    bracos: dict = field(default_factory=dict)      # rotulo -> fracao
    visto_frontal: float | None = None              # fracao de quadros
    visto_lateral: float | None = None

    amostras: int = 0

    def pontuar(self, ev):
        """Quanto esta evidencia parece com esta prateleira. 0 a 1 por sinal.

        Cada sinal contribui de forma independente e a soma e normalizada pelo
        que efetivamente votou. Assim uma camera cega nao penaliza ninguem —
        ela so nao participa daquele quadro.

        Devolve (pontos, quantos_sinais_votaram).
        """
        pontos, votos = 0.0, 0

        for valor, faixa, peso in ((ev.alcance, self.alcance, 2.0),
                                   (ev.coxa, self.coxa, 1.0),
                                   (ev.encolhimento, self.encolhimento, 1.0)):
            if valor is None or faixa is None:
                continue
            centro, tolerancia = faixa
            if tolerancia <= 1e-9:
                continue
            # Queda linear ate zero a duas tolerancias: o sinal deixa de
            # apoiar sem virar prova CONTRA. Penalizar o distante faria uma
            # camera ruim derrubar prateleiras que outra camera sustentava.
            d = abs(valor - centro) / (2.0 * tolerancia)
            pontos += peso * max(0.0, 1.0 - d)
            votos += peso

        if ev.braco is not None and self.bracos:
            pontos += 1.5 * self.bracos.get(ev.braco, 0.0)
            votos += 1.5

        # O QUE NAO FOI VISTO TAMBEM VOTA.
        for visto, esperado in ((ev.viu_frontal, self.visto_frontal),
                                (ev.viu_lateral, self.visto_lateral)):
            if visto is None or esperado is None:
                continue
            p = esperado if visto else (1.0 - esperado)
            pontos += 0.5 * p
            votos += 0.5

        return (pontos / votos if votos else 0.0), votos


@dataclass
class Palpite:
    """A resposta. Sempre existe quando houve alcance."""

    prateleira: str
    confianca: float          # 0 a 1: o quanto a evidencia apoiou a vencedora
    margem: float             # distancia para a segunda colocada
    quadros: int
    ranking: list = field(default_factory=list)

    @property
    def firme(self):
        """Da para agir sem perguntar? Precisa das DUAS coisas.

            margem     a vencedora esta bem a frente da segunda
            confianca  a evidencia disponivel APOIOU a vencedora

        Margem sozinha nao basta, e o caso que ensinou isso apareceu no teste:
        um alcance de 2.5 troncos — anatomicamente impossivel — com a coxa
        vertical e o braco levantado. A margem ficou folgada (0,32), porque
        nenhuma outra prateleira explica braco levantado. Mas a confianca
        caiu para 0,54: um dos tres sinais nao apoiou ninguem.

            Ficar na frente das outras nao e o mesmo que estar certa. Quando
            um sinal nao apoia NENHUMA hipotese, ele nao esta escolhendo — ele
            esta avisando que a leitura daquele quadro nao presta.

        Por isso o piso e 0,55: mais da metade da evidencia disponivel tem que
        apontar ativamente para a vencedora. Um gesto limpo passa de 0,85.
        """
        return self.margem >= 0.15 and self.confianca >= 0.55

    def __str__(self):
        estado = "" if self.firme else "  (provavel, nao certo)"
        return (f"{self.prateleira}  conf {self.confianca:.0%}  "
                f"margem {self.margem:.2f}{estado}")


class ClassificadorDePrateleira:
    """Junta a evidencia das tres cameras e diz de qual prateleira foi.

    O ACUMULO E POR GESTO, NAO POR QUADRO

    Um quadro isolado e ruim: o pulso treme, a coxa oscila, uma camera perde
    a junta por um instante. O gesto inteiro — dois, tres segundos de mao
    parada na prateleira — e estavel.

    Entao as pontuacoes se acumulam numa janela deslizante, e a decisao sai da
    soma. E a mesma logica do `Estavel` do classificador de acao: compromisso
    temporal antes de mudar de estado.

        Decidir por quadro produz um sistema que gagueja. Decidir por gesto
        produz um sistema que responde.
    """

    def __init__(self, assinaturas=None, janela=20):
        self.assinaturas = list(assinaturas or [])
        self.janela = janela
        self._historico = {}

    def declarar(self, assinatura):
        self.assinaturas = [a for a in self.assinaturas
                            if a.id != assinatura.id] + [assinatura]

    @property
    def pronto(self):
        return bool(self.assinaturas)

    def observar(self, pessoa_id, evidencia):
        """Alimenta um quadro. Devolve o `Palpite` atual, ou None.

        `None` aqui significa apenas "ainda nao ha evidencia nenhuma" — nao e
        recusa de responder. Assim que qualquer sinal chega, ha palpite.
        """
        if not self.assinaturas or evidencia is None or evidencia.vazia():
            return self.palpite(pessoa_id)

        marcador = {}
        for a in self.assinaturas:
            p, votos = a.pontuar(evidencia)
            if votos:
                marcador[a.id] = p

        if marcador:
            self._historico.setdefault(
                pessoa_id, deque(maxlen=self.janela)).append(marcador)
        return self.palpite(pessoa_id)

    def palpite(self, pessoa_id):
        h = self._historico.get(pessoa_id)
        if not h:
            return None

        soma = {}
        for marcador in h:
            for k, v in marcador.items():
                soma[k] = soma.get(k, 0.0) + v

        ranking = sorted(((v / len(h), k) for k, v in soma.items()),
                         reverse=True)
        if not ranking:
            return None

        melhor, id_melhor = ranking[0]
        segunda = ranking[1][0] if len(ranking) > 1 else 0.0

        return Palpite(prateleira=id_melhor, confianca=melhor,
                       margem=melhor - segunda, quadros=len(h),
                       ranking=[(k, round(v, 3)) for v, k in ranking])

    def esquecer(self, vivos):
        for pid in list(self._historico):
            if pid not in vivos:
                del self._historico[pid]

    def reiniciar(self, pessoa_id):
        """Fim de um gesto: o proximo alcance comeca do zero.

        Sem isto, a prateleira anterior continuaria pesando na proxima —
        e um cliente que pega da 5 e depois da 1 teria a segunda resposta
        contaminada pela primeira.
        """
        self._historico.pop(pessoa_id, None)


def evidencia_de(leitura, lado="direita", encolhimento=None):
    """Extrai a `Evidencia` de uma `LeituraDoCorpo` ja combinada.

    A leitura combinada JA e o resultado das tres cameras: `_combinar` escolheu
    por merito qual vista respondeu cada campo, e `fonte_braco_*` registra
    quem foi. Aqui so se traduz para a linguagem da evidencia.

        A complementaridade nao acontece aqui. Ela ja aconteceu, e este modulo
        colhe o resultado.
    """
    if leitura is None:
        return None

    dir_ = lado == "direita"
    fonte = leitura.fonte_braco_dir if dir_ else leitura.fonte_braco_esq
    braco = leitura.braco_direito if dir_ else leitura.braco_esquerdo

    return Evidencia(
        alcance=leitura.alcance_dir if dir_ else leitura.alcance_esq,
        coxa=leitura.verticalidade_coxa,
        braco=braco if braco and braco != "desconhecido" else None,
        encolhimento=encolhimento,
        viu_frontal=(fonte == "frontal") if fonte else None,
        viu_lateral=(fonte == "lateral") if fonte else None,
    )
