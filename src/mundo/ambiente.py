"""O AMBIENTE E RECONHECIDO ANTES DA PESSOA.

    crie um sistema que antes de detectar o gemeo digital ele detecta o
    ambiente, e assim ele vai ser aonde o gemeo digital esta em relacao a
    prateleira... use as cameras combinadas a uma memoria de como e a
    prateleira                                      — Eduardo, 13/08

INVERSAO DE ORDEM, E ELA MUDA O SIGNIFICADO DE TUDO.

Ate aqui o sistema comecava pela pessoa: detectava, rastreava, media a mao, e
so entao tentava adivinhar de qual prateleira ela pegou — com a estante
existindo apenas como cinco alturas soltas num arquivo. A posicao da pessoa era
absoluta e a prateleira era um palpite.

Reconhecendo o ambiente primeiro, a relacao se inverte: a pessoa passa a ser
medida CONTRA UM MOVEL QUE JA ESTA NO MUNDO. "A mao esta a 1,35 m" vira "a mao
esta na quarta prateleira daquela estante, a 40 cm da face". A segunda frase
responde a pergunta do negocio; a primeira nao.

    Sem o ambiente, a posicao da pessoa e um par de numeros.
    Com o ambiente, e uma relacao — e relacao e o que decide a venda.

REGISTRO DE MODELO, NAO DETECCAO LIVRE

O sistema NAO pergunta "que objeto e aquele?". Pergunta:

    onde encaixa a estante que eu JA CONHECO?

E ele conhece: 0,92 x 0,30 x 1,90 m, com prateleiras a 0,15 / 0,55 / 0,95 /
1,35 / 1,90 — medidas com trena em 11/08, guardadas em `loja/estante.json`.

Essa diferenca e o que dispensa rede neural. Detectar "uma estante qualquer"
exige um modelo treinado em milhares de estantes. Encontrar UM RETANGULO DE
DIMENSOES CONHECIDAS num plano calibrado e geometria: mede-se cada candidato em
METROS — pela homografia — e sobra o que tem o tamanho certo. Os falsos
positivos morrem sozinhos, sem ninguem precisar ensina-los a morrer.

    A medida conhecida nao e um detalhe do problema. E o filtro que o resolve.

AS TRES CAMERAS, CADA UMA NO QUE ELA E BOA

    ALTO      posicao (x, y) e orientacao no chao — a unica com homografia
    FRONTAL   largura da face e as alturas das prateleiras, vistas de frente
    LATERAL   profundidade do movel e as alturas, vistas de perfil

Nenhuma delas resolve sozinha, e nenhuma precisa. E a mesma regra que vale para
o corpo desde 10/08 — e a razao pela qual este arquivo aceita evidencia PARCIAL
e diz o quanto do movel foi realmente visto, em vez de exigir as tres.

O QUE ESTE ARQUIVO NAO FAZ

Nao processa imagem. Recebe as evidencias que cada camera extraiu e as funde
com o gabarito. Essa separacao existe para que a fusao — que e onde mora a
decisao dificil — possa ser testada sem camera nenhuma, com verdade conhecida.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Quanto uma dimensao medida pode divergir do gabarito e ainda ser aceita.
#
# 25% e generoso de proposito. A largura vista de cima por uma webcam a 2,5 m
# de altura, depois da homografia, carrega o erro da calibracao mais o da
# deteccao de borda. Apertar isso rejeitaria a estante certa; afrouxar demais
# aceitaria a mesa. Entre 0,92 e 0,30 ha um fator de 3 — folga de sobra para
# distinguir a face da lateral mesmo com um quarto de erro.
TOLERANCIA_DIMENSAO = 0.25

# Minimo de evidencia para o ambiente valer. Uma camera basta para arriscar,
# duas para confiar.
CAMERAS_PARA_CONFIAR = 2


@dataclass
class Gabarito:
    """O que o sistema JA SABE sobre a estante. Vem de `loja/estante.json`."""
    id: str
    largura: float
    profundidade: float
    altura: float
    prateleiras: list          # [(id, altura_m)]

    @classmethod
    def de_arquivo(cls, caminho="loja/estante.json"):
        d = json.loads(Path(caminho).read_text(encoding="utf-8"))
        dim = d["dimensoes"]
        return cls(id=d["id"], largura=dim["largura"],
                   profundidade=dim["profundidade"], altura=dim["altura"],
                   prateleiras=[(p["id"], p["altura"])
                                for p in d["prateleiras"]])

    def cabe(self, medida, nominal):
        """Esta medida pode ser esta dimensao do movel?"""
        if medida is None or nominal <= 0:
            return False
        return abs(medida - nominal) / nominal <= TOLERANCIA_DIMENSAO


@dataclass
class VistaDoAlto:
    """O que a camera cenital extraiu: um retangulo no chao, em METROS.

    Ja convertido pela homografia — e por isso que so esta camera pode dar
    posicao. As outras duas veem forma, nao lugar.
    """
    centro: tuple               # (x, y) em metros
    lado_maior: float           # comprimento do lado maior, em metros
    lado_menor: float
    angulo: float               # radianos: direcao do lado MAIOR


@dataclass
class VistaDeFrente:
    """O que a frontal ou a lateral extraiu: alturas de linhas horizontais.

    Em metros acima do chao, ja convertidas pela escala vertical. Sao as
    prateleiras vistas de canto — o que numa imagem aparece como um feixe de
    linhas paralelas.
    """
    alturas: list
    largura_aparente: float | None = None


@dataclass
class Ambiente:
    """A estante posta no mundo. E o que o gemeo passa a ter como referencia."""
    x: float
    y: float
    rumo_da_face: float         # radianos: para onde a face OLHA
    largura: float
    profundidade: float
    altura: float
    prateleiras: list
    cameras: tuple = ()         # quais confirmaram
    alturas_conferidas: int = 0

    @property
    def confiavel(self):
        return len(self.cameras) >= CAMERAS_PARA_CONFIAR

    @property
    def normal(self):
        return np.array([-math.sin(self.rumo_da_face),
                         math.cos(self.rumo_da_face)])

    def relacao(self, x, y):
        """Onde esta esta pessoa EM RELACAO a estante. Em metros.

        Devolve (adiante, lateral): quanto ela esta a frente da face, e quanto
        esta deslocada do centro dela. E este par que transforma uma posicao
        absoluta numa relacao — que e o ponto do arquivo inteiro.
        """
        d = np.array([x - self.x, y - self.y])
        ao_longo = np.array([math.cos(self.rumo_da_face),
                             math.sin(self.rumo_da_face)])
        return float(d @ self.normal), float(d @ ao_longo)

    def de_frente(self, x, y, alcance=0.85):
        adiante, lateral = self.relacao(x, y)
        return (0.0 <= adiante <= alcance
                and abs(lateral) <= self.largura / 2 + 0.25)

    def prateleira_na_altura(self, altura_mao, tolerancia=0.15):
        """Qual prateleira esta altura alcanca. None quando fica no meio do vao.

        A tolerancia PRECISA ser menor que metade do vao de 40 cm. Com 0,20 —
        exatamente meio vao — toda altura cairia em alguma prateleira, e a
        funcao perderia a capacidade de dizer "nao sei".

            Um limiar que nunca recusa nao esta classificando: esta
            arredondando.
        """
        if altura_mao is None or not self.prateleiras:
            return None
        pid, alvo = min(self.prateleiras, key=lambda p: abs(p[1] - altura_mao))
        return pid if abs(alvo - altura_mao) < tolerancia else None

    def de_qual_prateleira(self, x, y, altura_mao):
        """A resposta que o negocio quer, e as DUAS condicoes que ela exige.

        Estar de frente e a mao estar na faixa. Alguem com o braco a 0,95 m do
        outro lado da sala nao esta pegando da p3 — esta cocando a cabeca.
        """
        if not self.de_frente(x, y):
            return None
        return self.prateleira_na_altura(altura_mao)


def _casar_lados(vista, gab):
    """O lado maior visto e a LARGURA da estante ou a profundidade dela?

    Vista de cima, a estante e um retangulo. Qual lado e a face de onde se pega
    nao esta na imagem — esta no gabarito. Como 0,92 e 0,30 diferem por um
    fator de 3, a resposta e inequivoca mesmo com erro grosseiro.

    DEVOLVE APENAS O RUMO, e essa e a correcao de 18/08.

    Antes esta funcao devolvia tambem `lado_maior` e `lado_menor` — os valores
    MEDIDOS — e `reconhecer` os gravava como as dimensoes da estante. Em 18/08
    isso escreveu no `quarto.json` uma estante de 1,01 x 0,23 m, quando a
    estante mede 0,92 x 0,30 e isso esta escrito com trena em
    `loja/estante.json` desde 11/08.

        A camera nao foi chamada para medir o movel. Ela foi chamada para
        dizer ONDE ele esta. Medir o que ja foi medido com trena e trocar uma
        certeza por uma estimativa.

    O tamanho visto continua servindo — para RECONHECER, que e comparar e
    decidir se aquilo pode ser ela, e para saber qual lado e qual. Depois
    disso ele nao tem mais utilidade e nao deve sobreviver a esta funcao.

    Devolve `rumo_da_face` em radianos, ou None se o retangulo nao for ela.
    """
    # Hipotese A: o lado maior e a largura (o caso normal).
    if vista.lado_maior >= vista.lado_menor:
        maior_e_largura = (gab.cabe(vista.lado_maior, gab.largura)
                           and gab.cabe(vista.lado_menor, gab.profundidade))
        if maior_e_largura:
            # A NORMAL DA FACE E PERPENDICULAR AO LADO DA LARGURA.
            #
            # Com a convencao do projeto — normal = (-sin r, cos r) — um lado
            # de largura na direcao `a` tem perpendicular (-sin a, cos a), que
            # ja E a normal para r = a. Somar 90 graus aqui foi o defeito que
            # os testes de relacao pegaram: punha a face virada para o lado,
            # e toda distancia "a frente" dava zero.
            return vista.angulo
    # Hipotese B: o maior e a profundidade — estante de lado para a camera.
    if (gab.cabe(vista.lado_maior, gab.profundidade)
            and gab.cabe(vista.lado_menor, gab.largura)):
        # Aqui a largura esta no lado MENOR, girado 90 graus do maior.
        return vista.angulo + math.pi / 2
    return None


def _conferir_alturas(vista, gab, tolerancia=0.12):
    """Quantas prateleiras do gabarito aparecem nas alturas vistas.

    Nao exige todas. Uma estante com produto em cima tem prateleiras que a
    camera nao ve, e cobrar as cinco seria rejeitar a estante certa por estar
    cheia — o que e o uso normal dela.
    """
    if not vista or not vista.alturas:
        return 0
    achadas = 0
    for _, alvo in gab.prateleiras:
        if any(abs(h - alvo) <= tolerancia for h in vista.alturas):
            achadas += 1
    return achadas


def reconhecer(gabarito, do_alto=None, da_frente=None, da_lateral=None):
    """Funde as evidencias com o gabarito. Devolve o Ambiente, ou None.

    A CAMERA DO ALTO E OBRIGATORIA, e nao por preferencia: ela e a unica com
    homografia, e sem homografia nao existe POSICAO — existe forma. Uma estante
    reconhecida na frontal e na lateral, sem a do alto, e uma estante que o
    sistema sabe que existe e nao sabe onde esta. Isso nao serve para medir
    relacao nenhuma, que e a unica coisa que este modulo produz.

    As outras duas CONFIRMAM: cada uma que reconhece as alturas do gabarito
    soma confianca. Duas cameras concordando e o limiar para `confiavel`.
    """
    if do_alto is None:
        return None

    rumo = _casar_lados(do_alto, gabarito)
    if rumo is None:
        return None                 # o retangulo visto nao tem o tamanho dela

    cameras = ["alto"]
    conferidas = 0
    for nome, vista in (("frontal", da_frente), ("lateral", da_lateral)):
        n = _conferir_alturas(vista, gabarito)
        if n >= 2:                  # duas prateleiras reconhecidas ja e padrao
            cameras.append(nome)
            conferidas = max(conferidas, n)

    # A NORMAL APONTA PARA O LADO DE ONDE SE PEGA.
    #
    # Vista de cima, as duas faces sao iguais. Quem desempata e o arranjo
    # registrado em `loja/quarto.json`: a estante e o LIMITE do campo util, e
    # atras dela ha parede. Entao a face voltada para DENTRO da area calibrada
    # e a que se usa — e "dentro" e o lado da origem, onde as pessoas andam.
    n = np.array([-math.sin(rumo), math.cos(rumo)])
    para_dentro = np.array([0.0, 0.0]) - np.array(do_alto.centro)
    if float(n @ para_dentro) < 0:
        rumo = rumo + math.pi

    # AS DIMENSOES SAO AS DA TRENA. SEMPRE.
    #
    # Nenhum numero de camera entra aqui. Largura, profundidade, altura e as
    # cinco prateleiras vem inteiras de `loja/estante.json`, medidas a mao em
    # 11/08. O que a camera acrescenta sao tres numeros e mais nada:
    #
    #     x, y            onde ela esta no chao
    #     rumo_da_face    para que lado a face olha
    #
    # Em 18/08 este retorno gravou 1,01 x 0,23 m — a leitura da camera — numa
    # estante que mede 0,92 x 0,30. O `achar_ambiente._gravar` ja dizia, na
    # propria nota que escreve no arquivo, que as dimensoes vinham do
    # gabarito. Dizia e nao fazia.
    #
    #     Documentacao que descreve a intencao em vez do codigo e pior que
    #     nenhuma: ela faz o leitor parar de conferir.
    return Ambiente(x=float(do_alto.centro[0]), y=float(do_alto.centro[1]),
                    rumo_da_face=float(math.atan2(math.sin(rumo),
                                                  math.cos(rumo))),
                    largura=gabarito.largura,
                    profundidade=gabarito.profundidade,
                    altura=gabarito.altura, prateleiras=gabarito.prateleiras,
                    cameras=tuple(cameras), alturas_conferidas=conferidas)


@dataclass
class MemoriaDoAmbiente:
    """Guarda o ambiente reconhecido e so troca quando ha motivo.

    O movel nao se mexe — entao a resposta de ontem continua valendo hoje, e
    reconhecer a cada quadro seria gastar CPU para confirmar o obvio. Mas ele
    PODE ser movido, e por isso a memoria aceita substituicao quando um
    reconhecimento novo e melhor que o guardado.

        O que nao muda deve ser medido uma vez. O que pode mudar deve ser
        medido de novo — mas so quando ha razao para duvidar.
    """

    atual: Ambiente | None = None
    _rejeitados: int = field(default=0)

    def registrar(self, ambiente):
        """Aceita o reconhecimento novo se ele for pelo menos tao bom."""
        if ambiente is None:
            self._rejeitados += 1
            return self.atual
        if self.atual is None or len(ambiente.cameras) >= len(self.atual.cameras):
            self.atual = ambiente
        return self.atual

    @property
    def pronto(self):
        return self.atual is not None

    def resumo(self):
        a = self.atual
        if a is None:
            return f"ambiente NAO reconhecido ({self._rejeitados} tentativas)"
        return (f"estante em ({a.x:.2f}, {a.y:.2f}) "
                f"face {math.degrees(a.rumo_da_face):+.0f} graus  "
                f"{a.largura:.2f}x{a.profundidade:.2f} m  "
                f"{'+'.join(a.cameras)}"
                f"{'' if a.confiavel else '  (uma camera so)'}")
