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

    return [
        Passo("aquecer", "ANDE DE UM LADO PARA O OUTRO, natural",
              eixo=None, certo=(), segundos=10,
              instrucao_extra="o sistema esta aprendendo o angulo da camera"),

        Passo("parado", "FIQUE PARADO, em pe, bracos ao lado do corpo",
              eixo="locomocao", certo=(Locomocao.PARADO,), segundos=6),

        Passo("andar_frente", "ANDE PARA FRENTE, alguns passos",
              eixo="locomocao", certo=(Locomocao.FRENTE,),
              pobre=(Locomocao.ANDANDO,), segundos=6),

        Passo("andar_tras", "ANDE PARA TRAS, sem virar o corpo",
              eixo="locomocao", certo=(Locomocao.TRAS,),
              pobre=(Locomocao.ANDANDO,), segundos=6),

        Passo("andar_lado", "ANDE DE LADO, sem virar o corpo",
              eixo="locomocao",
              certo=(Locomocao.ESQUERDA, Locomocao.DIREITA),
              pobre=(Locomocao.ANDANDO,), segundos=6),

        Passo("virar_direita", "GIRE PARA A DIREITA, andando devagar",
              eixo="locomocao",
              certo=(Locomocao.VIRANDO_DIR, Locomocao.MEIA_VOLTA),
              pobre=(Locomocao.ANDANDO,), segundos=6),

        Passo("virar_esquerda", "GIRE PARA A ESQUERDA, andando devagar",
              eixo="locomocao",
              certo=(Locomocao.VIRANDO_ESQ, Locomocao.MEIA_VOLTA),
              pobre=(Locomocao.ANDANDO,), segundos=6),

        Passo("em_pe", "FIQUE EM PE, parado",
              eixo="postura", certo=(Postura.EM_PE,), segundos=5),

        Passo("agachar", "AGACHE e fique agachado",
              eixo="postura", certo=(Postura.AGACHADO,),
              pobre=(Postura.DESCONHECIDA,), segundos=6),

        Passo("braco_dir_levantado", "LEVANTE O BRACO DIREITO e segure",
              eixo="braco_direito", certo=(Braco.LEVANTADO,), segundos=6),

        Passo("braco_esq_levantado", "LEVANTE O BRACO ESQUERDO e segure",
              eixo="braco_esquerdo", certo=(Braco.LEVANTADO,), segundos=6),

        Passo("braco_dir_estendido",
              "ESTENDA O BRACO DIREITO A FRENTE, como quem pega um produto",
              eixo="braco_direito", certo=(Braco.ESTENDIDO,),
              pobre=(Braco.LEVANTADO,), segundos=6),

        Passo("bracos_ao_lado", "BAIXE OS DOIS BRACOS",
              eixo="braco_direito", certo=(Braco.AO_LADO,), segundos=5),
    ]


@dataclass
class Contagem:
    """O que o sistema respondeu durante UM passo."""

    passo: Passo
    veredictos: Counter = field(default_factory=Counter)
    respostas: Counter = field(default_factory=Counter)
    alturas: list = field(default_factory=list)
    ids: set = field(default_factory=set)

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

    def anotar(self, passo, acoes, decorrido_s, prevendo=None):
        """Anota um quadro.

        `acoes`     {id: (Acao, mudancas)} do SpatialEngine
        `prevendo`  {id: quadros_sem_medicao}, do gemeo. Fica FORA do `Acao`
                    de proposito: `Acao` e vocabulario fechado, e "ha quantos
                    quadros ninguem ve esta pessoa" nao e uma acao dela.
        """
        if passo.eixo is None:
            return
        if decorrido_s < passo.acomodacao_s:
            return

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

        if (prevendo or {}).get(pid) and not self.contar_previstos:
            self.previstos_ignorados += 1
            return

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
                }
                for acao, c in self.contagens.items()
            },
            "previstos_ignorados": self.previstos_ignorados,
        }
