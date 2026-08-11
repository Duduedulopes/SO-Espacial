"""
O vocabulario fechado. Este arquivo E o contrato.

POR QUE UM VOCABULARIO FECHADO

Ate 10/08 o sistema transmitia coordenadas de 17 juntas e mandava desenhar
exatamente aquilo. O desenho herdava todo erro da reconstrucao — e produziu um
esqueleto deitado no chao enquanto a pessoa andava em pe. Nao havia como
consertar o desenho: ele estava mostrando fielmente dados ruins.

Com vocabulario fechado, o desenho deixa de copiar medida e passa a ANIMAR um
corpo que ja sabe ser corpo.

    Se "deitado" nao esta no vocabulario, o boneco nao consegue deitar.

A classe inteira de defeito desaparece por construcao, e nao por conserto.

TRES EIXOS, NAO UMA LISTA

Locomocao, postura e bracos sao INDEPENDENTES e se combinam. "Andando para a
frente + agachado + braco direito estendido" e valido sem nenhuma entrada
nova. Uma lista unica precisaria de um item por combinacao e explodiria.

DESCONHECIDO E UM VALOR DE PRIMEIRA CLASSE

Todo eixo tem `DESCONHECIDA`. Sempre existe um padrao mais proximo, inclusive
quando nenhum serve — e escolher o menos ruim em silencio e exatamente o
defeito do filtro de plausibilidade que recusou 55% das deteccoes reais na
manha do mesmo dia.

    Abster-se em vez de forcar.
"""

from dataclasses import dataclass, field


class Locomocao:
    """Como a pessoa se desloca. Relativa ao CORPO quando ha rumo do corpo."""

    PARADO = "parado"
    FRENTE = "andando_frente"
    TRAS = "andando_tras"
    ESQUERDA = "andando_esquerda"        # de lado
    DIREITA = "andando_direita"          # de lado
    VIRANDO_ESQ = "virando_esquerda"
    VIRANDO_DIR = "virando_direita"
    MEIA_VOLTA = "meia_volta"
    DESCONHECIDA = "desconhecida"

    # Quando nao ha rumo do corpo, so sabemos que anda — nao para onde em
    # relacao a ele. `ANDANDO` e a resposta honesta nesse caso.
    ANDANDO = "andando"

    TODAS = (PARADO, FRENTE, TRAS, ESQUERDA, DIREITA, ANDANDO,
             VIRANDO_ESQ, VIRANDO_DIR, MEIA_VOLTA, DESCONHECIDA)


class Postura:
    EM_PE = "em_pe"
    AGACHADO = "agachado"
    DESCONHECIDA = "desconhecida"

    TODAS = (EM_PE, AGACHADO, DESCONHECIDA)


class Braco:
    AO_LADO = "ao_lado"
    LEVANTADO = "levantado"
    ESTENDIDO = "estendido"              # a frente, alcancando
    DESCONHECIDO = "desconhecido"

    TODOS = (AO_LADO, LEVANTADO, ESTENDIDO, DESCONHECIDO)


@dataclass
class Acao:
    """O que uma pessoa esta fazendo, num instante.

    E isto que o renderizador consome. Nenhuma coordenada de junta atravessa
    esta fronteira — de proposito.

    `confianca` viaja junto com o estado e nao e enfeite: estado incerto nao
    pode chegar ao desenho com a mesma aparencia de estado medido. COMO isso
    aparece na tela e decisao de quem desenha; QUE apareca nao e negociavel.
    """

    locomocao: str = Locomocao.DESCONHECIDA
    postura: str = Postura.DESCONHECIDA
    braco_esquerdo: str = Braco.DESCONHECIDO
    braco_direito: str = Braco.DESCONHECIDO

    # ALTURA DA MAO ACIMA DO CHAO, EM METROS.
    #
    # E o unico campo deste dataclass que nao e um rotulo, e ha uma razao para
    # abrir a excecao: e ele que responde a pergunta comercial. A planta
    # declara que na gondola A, entre 1,10 m e 1,35 m, esta o produto X; a
    # visao so precisa dizer se o pulso entrou naquela faixa e voltou.
    #
    #     O produto sai do CADASTRO, nao da imagem.
    #
    # Isso converte visao computacional dificil — reconhecer embalagem sob
    # oclusao e iluminacao variavel — em comparacao de numeros.
    #
    # `None` quando o tornozelo nunca foi visto e a altura do quadril nao pode
    # ser aprendida. Sem base, nao se responde.
    altura_mao_esq: float | None = None
    altura_mao_dir: float | None = None

    # A altura veio de tornozelo VISTO, ou da proporcao do tronco?
    #
    # Sem este campo, um numero estimado com +-8 cm de erro chega a quem decide
    # com a mesma cara de um medido com +-2 cm. Quem consome tem que poder
    # escolher se aquele erro cabe na decisao dele.
    altura_medida: bool = False

    # 0 a 1. Confianca do eixo de locomocao, que e o mais usado pelo desenho.
    confianca: float = 0.0

    # Numeros crus que produziram a classificacao. Ficam aqui para o painel
    # poder explicar POR QUE o sistema decidiu o que decidiu — sem isso, um
    # estado errado nao tem como ser investigado.
    velocidade_ms: float = 0.0
    giro_graus_s: float = 0.0
    razao_altura: float = 0.0
    motivo: str = ""

    def para_dicionario(self):
        d = {
            "locomocao": self.locomocao,
            "postura": self.postura,
            "braco_esquerdo": self.braco_esquerdo,
            "braco_direito": self.braco_direito,
            "confianca": round(self.confianca, 2),
            "velocidade_ms": round(self.velocidade_ms, 2),
            "giro_graus_s": round(self.giro_graus_s, 1),
        }
        # So publica altura que foi MEDIDA. Uma chave com `null` e uma chave
        # ausente dizem a mesma coisa, mas a ausente nao tenta ser lida por
        # engano por quem consome o JSON sem checar.
        if self.altura_mao_esq is not None:
            d["altura_mao_esq"] = round(self.altura_mao_esq, 3)
        if self.altura_mao_dir is not None:
            d["altura_mao_dir"] = round(self.altura_mao_dir, 3)
        if self.altura_mao_esq is not None or self.altura_mao_dir is not None:
            d["altura_medida"] = self.altura_medida
        return d

    def __repr__(self):
        b = ""
        for lado, estado, altura in (
                ("E", self.braco_esquerdo, self.altura_mao_esq),
                ("D", self.braco_direito, self.altura_mao_dir)):
            if estado in (Braco.AO_LADO, Braco.DESCONHECIDO):
                continue
            b += f" {lado}:{estado}"
            if altura is not None:
                b += f"@{altura:.2f}m" + ("" if self.altura_medida else "~")
        return f"{self.locomocao}/{self.postura}{b} ({self.confianca:.0%})"


@dataclass
class Estavel:
    """Compromisso so depois de o candidato insistir por TEMPO suficiente.

    POR QUE ISTO PRECISA EXISTIR

    Um classificador que muda de opiniao a cada quadro produz um fluxo de
    eventos em estrobo e um boneco epileptico. E a mesma licao dos eventos de
    zona: entrada e saida saem UMA vez por travessia, nao uma por quadro.

    POR QUE EM SEGUNDOS, E NAO EM QUADROS

    MEDIDO EM 10/08: com 3 quadros de confirmacao a 14 fps, o limiar valia
    0,21 s — e a execucao real produziu 12 mudancas de locomocao em 45 s, seis
    delas em 4 segundos. Estado humano de locomocao dura cerca de um segundo;
    0,2 s so filtra ruido de amostra.

    E ha um problema pior que a quantidade: em quadros, o mesmo codigo se
    comporta diferente conforme a maquina. A 30 fps aquele mesmo `3` valeria
    0,1 s; num computador lento, 0,5 s. E a mesma armadilha do `1/30` cravado
    no raio de recostura, que quebrou a 4 fps em 08/08.

        Limiar temporal se declara em tempo. Quadro nao e unidade de tempo.
    """

    valor: str
    minimo_s: float = 0.35
    _candidato: str = field(default="", repr=False)
    _acumulado: float = field(default=0.0, repr=False)

    def propor(self, novo, dt):
        """Devolve True quando o valor MUDOU de fato."""
        if novo == self.valor:
            self._candidato, self._acumulado = "", 0.0
            return False

        if novo == self._candidato:
            self._acumulado += dt
        else:
            self._candidato, self._acumulado = novo, dt

        if self._acumulado >= self.minimo_s:
            self.valor = novo
            self._candidato, self._acumulado = "", 0.0
            return True
        return False
