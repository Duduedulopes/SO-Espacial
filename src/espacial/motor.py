"""
SpatialEngine — de observacoes para posicoes e esqueletos no mundo.

O QUE ELE FAZ

Recebe Observacoes (hipoteses do que cada camera viu) e devolve
EstadoDePessoa (o que o sistema conclui). Entre uma coisa e outra estao todos
os filtros, a geometria e o Kalman.

REAPROVEITAMENTO

Nada aqui e novo. Cada peca ja foi escrita, medida e registrada no caderno:

    chao.EstimadorDePe            saltos > 50 cm: 5 -> 0
    chao.FiltroDeTornozelo        0 tornozelos em 79 quadros de mobilia
    chao.FiltroDePlausibilidade   rejeita cadeira e poste, aceita crianca
    rastreio.GerenciadorDeRastros recostura testada a 30 e 4 fps
    fusao.Fusor                   33 cm -> 1,3 cm em simulacao
    pose3d.SuavizadorDeEsqueleto  corta metade do tremor, acompanha 170/200 mm

O que muda e a ORGANIZACAO: antes tudo isso vivia solto dentro de um laco de
317 linhas que tambem abria camera e desenhava. Aqui esta um objeto com uma
entrada e uma saida.

A ORDEM IMPORTA, E ESTA E A RAZAO DE CADA POSICAO

    1. plausibilidade   e GRATIS. Vem antes de gastar 30 ms de pose numa cadeira.
    2. ponto do pe      tornozelo quando ha; caixa corrigida pelo vies quando nao.
    3. homografia       pixel -> metros.
    4. tornozelo        o rastro ja provou ser gente?
    5. Kalman           suaviza o ruido e sobrevive a sumicos.
    6. fusao            frontal da largura e altura; lateral da profundidade.
    7. ancoragem        esqueleto em pe no ponto do chao, virado para o rumo.

NOTA DE MIGRACAO

Os modulos importados ainda vivem em `percepcao/` e `estado/`. Sao os mesmos
arquivos que os programas antigos usam. Movê-los agora quebraria o
`gemeo_multi.py`, que continua sendo a unica versao que roda de ponta a ponta.
A mudanca de lugar acontece quando o orquestrador novo substituir o antigo.
"""

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from estado.rastreio import GerenciadorDeRastros            # noqa: E402
from percepcao.chao import (                                # noqa: E402
    EstimadorDePe, FiltroDePlausibilidade, FiltroDeTornozelo, para_metros,
)
from percepcao.fusao import Fusor                          # noqa: E402
from percepcao.pose3d import (                              # noqa: E402
    EstimadorDeInclinacao, SuavizadorDeEsqueleto, ancorar_no_chao,
)
from src.acao.classificador import Descritor                 # noqa: E402
from src.acao.corpo import AnalisadorDeCorpo                 # noqa: E402
from src.acao.escala import EscalaVertical                   # noqa: E402
from src.espacial.estado import EstadoDePessoa              # noqa: E402
from src.nucleo.log import Log                              # noqa: E402


def _altura_da_camera():
    """Le `config/escala.json`. Ausente significa nao calibrado, e tudo bem:
    a altura da mao sai estimada pelo tronco e marcada como tal."""
    import json

    caminho = RAIZ / "config" / "escala.json"
    if not caminho.exists():
        return None
    try:
        return float(json.loads(caminho.read_text(encoding="utf-8"))
                     ["altura_camera_m"])
    except Exception:
        return None


class SpatialEngine:
    def __init__(self, H, resolucao_calibracao=None, resolucao_captura=None,
                 papel_chao="alto", lado_lateral="direita",
                 min_tornozelo=3, usar_plausibilidade=True,
                 ruido_processo=0.6, max_coasting_s=3.0):
        # o log vem ANTES de qualquer coisa que possa registrar algo
        self.log = Log("espacial")

        # Guardamos a homografia ORIGINAL e a resolucao em que ela foi
        # medida. A resolucao de captura configurada e um PEDIDO; a camera
        # responde o que quiser. Quando a real chegar, reajustamos a partir
        # do original — nunca compondo escala sobre escala.
        self.H_original = H
        self.resolucao_calibracao = resolucao_calibracao
        self.H = self._ajustar_escala(H, resolucao_calibracao,
                                      resolucao_captura)
        self.papel_chao = papel_chao
        self.usar_plausibilidade = usar_plausibilidade

        self.estimador_pe = EstimadorDePe()
        self.filtro_tornozelo = FiltroDeTornozelo(minimo=min_tornozelo)
        self.plausibilidade = FiltroDePlausibilidade(self.H)
        self.rastros = GerenciadorDeRastros(ruido_processo=ruido_processo,
                                            max_coasting_s=max_coasting_s)
        self.suavizador = SuavizadorDeEsqueleto()
        self.fusor = Fusor(lado_lateral=lado_lateral)

        # A INCLINACAO DA CAMERA NAO E DETALHE: E O QUE MANTEM O BONECO EM PE.
        #
        # O MediaPipe devolve coordenadas alinhadas com a CAMERA, nao com a
        # gravidade. Com a lente inclinada olhando o chao, o esqueleto sai
        # tombado exatamente por esse angulo.
        #
        # REGRESSAO DE MIGRACAO, 10/08: este estimador ja existia e ja tinha
        # medido -42 graus sozinho. Ao montar o SpatialEngine eu chamei
        # `para_o_mundo`, que faz tudo o que `ancorar_no_chao` faz MENOS
        # desfazer a inclinacao. Duas funcoes quase iguais, e eu peguei a
        # errada — o resultado foi um boneco deitado por uma sessao inteira.
        #
        #     Codigo duplicado nao e so feio: ele deixa escolher a versao
        #     incompleta sem perceber.
        self.inclinacao = EstimadorDeInclinacao()

        # CAMADA DE ACAO (arquitetura v3, etapa A).
        #
        # Roda DENTRO do ciclo, a cada quadro, sobre numeros que ja estao
        # calculados. Custa aritmetica — microssegundos contra os 156 ms do
        # detector. Nao e ela que deixa o sistema lento.
        self.descritor = Descritor()
        self.acoes = {}                # id -> Acao, para o painel e o desenho

        # ETAPA B: a camada que le o CORPO, e nao so o deslocamento.
        #
        # Ela consome o mesmo esqueleto relativo que ja chega das vistas de
        # pose — nao pede quadro novo, nao roda modelo novo, nao abre camera.
        # E aritmetica sobre numeros que ja estavam sendo calculados e
        # jogados fora depois da fusao.
        self.corpo = AnalisadorDeCorpo()
        self.leituras = {}             # id -> LeituraDoCorpo, para o painel

        # A CAMERA DO ALTO E A UNICA QUE VE OS PES, E E ELA QUE DA A ESCALA.
        #
        # Ideia do Eduardo, 11/08: as tres cameras existem para se
        # complementar, e nenhuma precisa ver tudo. A frontal e a lateral
        # ficam sobre a mesa e nunca verao um tornozelo — medido: 0% nas duas.
        # A do alto ve, e o `FiltroDePlausibilidade` ja calculava a razao
        # geometrica que, multiplicada pela altura da camera, E a estatura.
        #
        #     O dado que faltava ja estava sendo calculado para outra
        #     finalidade: recusar movel.
        self.escala = EscalaVertical(altura_camera_m=_altura_da_camera())

        self.log = Log("espacial")
        self._rumos = {}
        self._caixas = {}
        # Ultima pose relativa de cada papel, guardada CRUA. O fusor guarda a
        # dele ja combinada; a camada de corpo precisa de UMA vista por vez,
        # justamente para nao herdar o erro da combinacao.
        self._poses_cruas = {}
        self._confiaveis = set()
        # FUNIL, nao so rejeicoes.
        #
        # Em 10/08 o painel disse `rejeitadas plausibilidade 358`. Numero solto
        # nao responde a pergunta que importa: 358 de quantas? De 400 e um
        # filtro quebrado; de 100 mil e ruido. Sem o total, a unica saida seria
        # adivinhar — e adivinhar com trabalho ja custou caro duas vezes.
        self._dt_atual = 1 / 30
        self.funil = {"observadas": 0, "sem_id": 0, "plausibilidade": 0,
                      "tornozelo": 0, "medidas": 0}
        self.rejeitadas = {"plausibilidade": 0, "tornozelo": 0}

    # ------------------------------------------------------------ geometria
    def _ajustar_escala(self, H, calib, captura):
        """A homografia vale para UMA resolucao.

        Se a captura mudar, os pixels mudam de escala e a calibracao deixa de
        valer. Em vez de exigir recalibracao, compomos com a mudanca de escala
        — que e exatamente o que uma matriz sabe fazer.
        """
        if not calib or not captura or tuple(calib) == tuple(captura):
            return H
        cw, ch = calib
        pw, ph = captura
        S = np.array([[cw / pw, 0, 0], [0, ch / ph, 0], [0, 0, 1.0]])
        self.log.info("homografia reescalada",
                      de=f"{cw}x{ch}", para=f"{pw}x{ph}")
        return H @ S

    def ajustar_para_resolucao(self, largura, altura):
        """Reajusta a homografia para a resolucao que a camera DE FATO entregou.

        Pedir 1280x720 e receber 640x480 acontece — a camera decide. Se o
        sistema seguir usando a resolucao pedida, cada pixel vale o dobro do
        que deveria e a posicao sai com o dobro do erro, sem nenhum sintoma
        alem de numeros errados.

        Recalcula sempre a partir de `H_original`. Compor escala sobre escala
        ja ajustada acumularia erro a cada chamada.
        """
        nova = self._ajustar_escala(self.H_original, self.resolucao_calibracao,
                                    (largura, altura))
        mudou = not np.allclose(nova, self.H)
        self.H = nova
        self.plausibilidade = FiltroDePlausibilidade(self.H)
        return mudou

    # ------------------------------------------------------------ ciclo
    def atualizar(self, observacoes, dt):
        self._dt_atual = dt
        chao = [o for o in observacoes if o.papel == self.papel_chao
                and o.tem_caixa]
        poses = [o for o in observacoes if o.tem_pose]

        self._confiaveis = self._ids_ja_provados()
        medidas = self._medir_no_chao(chao)
        self._alimentar_fusor(poses)

        rastros = self.rastros.atualizar(medidas, dt)
        self._observar_inclinacao(
            poses, [float(np.hypot(*r.kf.vel)) for r in rastros.values()])
        return self._montar_estados(rastros)

    @property
    def poses_por_papel(self):
        """Ultima pose CRUA de cada vista. Para a janela desenhar o que entrou.

        Publica porque a janela precisa mostrar exatamente os landmarks que o
        analisador consumiu — nao uma segunda estimativa. Ver a entrada de
        verdade e a unica forma de separar "o modelo perdeu a junta" de "o
        limiar recusou a junta".
        """
        return dict(self._poses_cruas)

    @property
    def caixas_por_id(self):
        return dict(self._caixas)

    def _ids_ja_provados(self, percorrido_minimo=0.8):
        """Ids do rastreador que pertencem a um rastro que JA ANDOU.

        Mobilia nao anda. Oitenta centimetros e curto o bastante para uma
        pessoa cobrir em dois passos e longo o bastante para nenhuma cadeira
        acumular por ruido de deteccao.

        A leitura e do quadro ANTERIOR, porque os rastros deste ainda nao
        foram atualizados. E isso e o correto: a prova tem que vir de antes,
        nao da caixa que esta sendo julgada agora.
        """
        provados = set()
        for r in self.rastros.rastros.values():
            if r.percorrido >= percorrido_minimo:
                provados |= set(r.ids_externos)
        return provados

    # ------------------------------------------------------------ 1 a 4
    def _medir_no_chao(self, observacoes):
        """Observacoes da camera do alto -> (id_externo, x_m, y_m)."""
        medidas = []
        vivos = set()

        for o in observacoes:
            self.funil["observadas"] += 1
            tid = o.id_externo
            if tid < 0:
                self.funil["sem_id"] += 1
                continue
            vivos.add(tid)

            # 1. plausibilidade — geometria, gratis, antes de tudo.
            #
            # MAS NAO CONTRA QUEM JA PROVOU SER GENTE.
            #
            # MEDIDO EM 11/08: `agachar` nunca foi reconhecido, com 36% dos
            # quadros sem leitura nenhuma e 198 caixas recusadas por
            # plausibilidade. O motivo e aritmetico: o filtro recusa abaixo de
            # 1/1,7 = 0,59x da altura de uma pessoa em pe, e agachar da cerca
            # de 0,62x. Com ruido, metade dos quadros cai abaixo da linha.
            #
            #     O filtro que existe para recusar movel estava recusando uma
            #     pessoa agachada.
            #
            # A regra que resolve ja estava escrita no projeto, aplicada do
            # outro lado: o filtro so APRENDE com quem andou, porque "cadeira
            # nao anda". Falta usar a mesma prova para a RECUSA. Um rastro que
            # ja percorreu distancia demonstrou ser gente; uma caixa baixa dele
            # e agachamento, nao mobilia.
            #
            # O que se paga: se uma pessoa desaparecer e uma cadeira herdar o
            # id do rastreador, a cadeira fica isenta ate o rastro morrer. E um
            # risco menor que o de nao enxergar ninguem agachado — que e a
            # postura de quem pega produto na prateleira de baixo.
            if self.usar_plausibilidade and tid not in self._confiaveis:
                ok, _motivo = self.plausibilidade.plausivel(o.caixa)
                if not ok:
                    self.rejeitadas["plausibilidade"] += 1
                    self.funil["plausibilidade"] += 1
                    continue

            # 2. ponto do pe
            pe, origem = self.estimador_pe.estimar(
                tid, o.caixa, o.juntas_2d, o.conf_2d)

            # 4. o rastro ja provou ser gente?
            if not self.filtro_tornozelo.ver(tid, origem):
                self.rejeitadas["tornozelo"] += 1
                self.funil["tornozelo"] += 1
                continue

            # 3. homografia
            mx, my = para_metros(self.H, *pe)
            medidas.append((tid, mx, my))
            self.funil["medidas"] += 1
            self._caixas[tid] = o.caixa

        self.estimador_pe.esquecer(vivos)
        self.filtro_tornozelo.esquecer(vivos)
        for tid in list(self._caixas):
            if tid not in vivos:
                del self._caixas[tid]
        return medidas

    # ------------------------------------------------------------ 6
    def _observar_inclinacao(self, poses, velocidades):
        """So aprende com quem esta ANDANDO — parado, o corpo mente.

        Ninguem caminha inclinado. O vetor quadril->ombros de quem anda e
        vertical no mundo, entao o giro que ele mostra na imagem E a
        inclinacao da lente.
        """
        if not velocidades:
            return
        v = max(velocidades)
        for o in poses:
            if o.juntas_3d is not None:
                self.inclinacao.observar(o.juntas_3d, v, o.conf_2d)

    def _alimentar_fusor(self, poses):
        """Guarda a ultima pose de cada vista.

        LIMITE DECLARADO: um fusor so. Com duas pessoas em cena, o sistema nao
        sabe qual pose da frontal pertence a qual pessoa do alto. Resolver
        exige re-identificacao por aparencia — bloco 5 do plano de estudo.
        """
        for o in poses:
            self._poses_cruas[o.papel] = o
            if o.papel == "frontal":
                self.fusor.ver_frontal(o.juntas_3d, o.t_mono, o.conf_2d)
            elif o.papel == "lateral":
                self.fusor.ver_lateral(o.juntas_3d, o.t_mono, o.conf_2d)

    # ------------------------------------------------------------ 5 e 7
    def _montar_estados(self, rastros):
        agora = max((r.kf and 0 for r in rastros.values()), default=0)
        juntas_pessoa, visiveis = self.fusor.esqueleto(self._agora())
        varias = len(rastros) > 1

        estados = []
        for meu, r in rastros.items():
            vx, vy = r.kf.vel
            if np.hypot(vx, vy) > 0.15:
                self._rumos[meu] = float(np.arctan2(vy, vx))
            rumo = self._rumos.get(meu, -np.pi / 2)

            # o filtro de altura so aprende com quem ANDOU. Mobilia nao anda.
            ext = [e for e in r.ids_externos if e in self._caixas]
            if ext and self.usar_plausibilidade:
                self.plausibilidade.observar(self._caixas[ext[-1]],
                                             r.percorrido)

            esqueleto = None
            if juntas_pessoa is not None and not varias:
                esqueleto = ancorar_no_chao(
                    juntas_pessoa, r.pos[0], r.pos[1], rumo,
                    inclinacao_rad=self.inclinacao.valor)
                esqueleto = self.suavizador.suavizar(meu, esqueleto)

            estados.append(EstadoDePessoa(
                id=meu, x=r.pos[0], y=r.pos[1], vx=vx, vy=vy,
                incerteza=r.kf.incerteza, rumo=rumo,
                esqueleto=esqueleto,
                juntas_visiveis=visiveis if esqueleto is not None else None,
                prevendo=r.sem_medicao,
                percorrido=r.percorrido, quadros=r.quadros,
                visto_por=self._vistas_ativas(),
                associacao_confiavel=not varias,
                t_mono=self._agora(),
            ))

        self._descrever(estados, rastros)

        self.suavizador.esquecer(set(rastros))
        for meu in list(self._rumos):
            if meu not in rastros:
                del self._rumos[meu]
        return estados

    def _descrever(self, estados, rastros):
        """Traduz numeros medidos em vocabulario fechado, e pendura na pessoa.

        A postura reaproveita o `k` do filtro de altura: ele ja aprendeu
        quanto mede uma pessoa EM PE naquele ponto do chao, corrigido pela
        perspectiva. Quem encolhe em relacao a isso, agachou.

        Quando o filtro esta ABSTIDO, `k` nao vale e a postura sai
        DESCONHECIDA — sem base, nao se opina.
        """
        k = self.plausibilidade.k if self.plausibilidade.pronto else None

        razoes = {}
        if k:
            for e in estados:
                r = rastros.get(e.id)
                ext = [x for x in (r.ids_externos if r else ()) 
                       if x in self._caixas]
                if ext:
                    razoes[e.id] = self.plausibilidade.razao(
                        self._caixas[ext[-1]])

        self._medir_estaturas(estados, rastros)
        self.leituras = self._ler_corpos(estados)

        resultado = self.descritor.atualizar(
            estados, self._dt_atual, leituras=self.leituras,
            razoes=razoes, k_referencia=k)

        self.acoes = {}
        for e in estados:
            acao, mudancas = resultado[e.id]
            e.acao = acao
            self.acoes[e.id] = (acao, mudancas)

        self.corpo.esquecer({e.id for e in estados})

    def _medir_estaturas(self, estados, rastros):
        """Estatura em metros, da camera do alto, para quem esta EM PE.

        `em_pe` vem do classificador do quadro ANTERIOR, que decidiu a postura
        pela coxa — um caminho independente da caixa. Sem esse cuidado, quem
        agacha alimentaria a escala com uma caixa menor e seria registrado como
        uma pessoa de 1,10 m; a mediana por pessoa levaria minutos para se
        recuperar.
        """
        from src.acao.vocabulario import Postura

        for e in estados:
            r = rastros.get(e.id)
            ext = [x for x in (r.ids_externos if r else ()) if x in self._caixas]
            if not ext:
                continue
            anterior = self.acoes.get(e.id)
            em_pe = (anterior is None
                     or anterior[0].postura != Postura.AGACHADO)
            self.escala.observar(
                e.id, self.plausibilidade.razao(self._caixas[ext[-1]]),
                em_pe=em_pe)
        self.escala.esquecer({e.id for e in estados})

    def _ler_corpos(self, estados, validade_s=0.5):
        """Le o CORPO de cada pessoa a partir de UMA vista de pose.

        POR QUE UMA VISTA E NAO A FUSAO

        A fusao existe e funciona — testada com entrada limpa, acertou a
        largura e a altura ao centimetro. Mas em 10/08 ela produziu um
        amontoado de meio metro, porque cada vista entregava juntas
        extrapoladas de partes do corpo que estavam fora do quadro, e somar
        duas invencoes diferentes da uma terceira invencao.

        A leitura do corpo nao precisa de profundidade: rumo dos ombros,
        pulso contra ombro e pulso contra tornozelo sao todos medidas que UMA
        camera responde. Pedir menos e o que a torna confiavel.

        TODAS AS VISTAS, E NAO A PRIMEIRA DA FILA

        Ate 11/08 a escolha era uma ordem fixa: frontal, senao lateral. E a
        frontal quase sempre respondia, entao a lateral nunca era consultada —
        mesmo entregando 100% de pose.

        MEDIDO NAQUELE DIA: levantar o braco levava 9 a 10 s para ser
        reconhecido e lia `ao_lado` em 65 a 87% dos quadros; BAIXAR levava 2 s.
        A assimetria e assinatura de mao saindo do quadro — a webcam do
        notebook pega do peito para cima, e o MediaPipe extrapola o pulso que
        subiu alem da borda.

            Preferencia fixa escolhe a vista antes de saber o que se quer ver.
            A pergunta certa nao e "qual camera e melhor", e "qual delas viu
            ESTA junta".

        Agora as duas sao lidas e cada braco vem de quem enxergou aquele pulso.
        A frontal continua vindo primeiro, e isso ainda importa: e dela que o
        azimute aprende, porque o giro medido e o DAQUELA lente.

        JANELA DE VALIDADE

        Uma pose de 3 segundos atras descreve um corpo que ja nao esta ali. Se
        a frontal cair, o dicionario segue com o ultimo valor dela para
        sempre — e o sistema passaria a descrever um braco levantado que
        baixou faz tempo. Pose velha e descartada, e a resposta vira
        DESCONHECIDO, que e o que ela de fato e.
        """
        if not estados:
            return {}

        agora = self._agora()
        vistas = []
        for papel in ("frontal", "lateral"):
            candidata = self._poses_cruas.get(papel)
            if candidata is None or candidata.juntas_3d is None:
                continue
            if agora - candidata.t_mono > validade_s:
                continue
            vistas.append((candidata.juntas_3d, candidata.conf_2d))

        if not vistas:
            return {}

        # LIMITE HERDADO, E DECLARADO OUTRA VEZ AQUI: com mais de uma pessoa
        # em cena, nao se sabe qual pose pertence a qual corpo. Atribuir a
        # leitura a alguem seria escolher no chute — e a acao escolhida no
        # chute chegaria ao desenho com a mesma aparencia de medida.
        if len(estados) > 1:
            return {}

        pessoa = estados[0]
        return {pessoa.id: self.corpo.ler_varias(
            pessoa.id, vistas,
            inclinacao_rad=self.inclinacao.valor,
            rumo_mundo=pessoa.rumo,
            velocidade=pessoa.velocidade,
            quadril_do_alto=self.escala.altura_do_quadril(pessoa.id))}

    def _vistas_ativas(self):
        v = {self.papel_chao}
        if self.fusor.frontal is not None:
            v.add("frontal")
        if self.fusor.lateral is not None:
            v.add("lateral")
        return v

    @staticmethod
    def _agora():
        import time
        return time.monotonic()

    # ------------------------------------------------------------ metricas
    def resumo(self):
        return {
            "funil": dict(self.funil),
            "rastros": len(self.rastros.rastros),
            "recosturas": self.rastros.recosturas,
            "rejeitadas": dict(self.rejeitadas),
            "altura": self.plausibilidade.diagnostico(),
            "vistas": self.fusor.diagnostico,
            "inclinacao": (f"{np.degrees(self.inclinacao.valor):+.0f}deg "
                           f"({len(self.inclinacao.amostras)} amostras"
                           f"{'' if self.inclinacao.confiavel else ', aprendendo'})"),
            "corpo": self.corpo.diagnostico,
            "escala": self.escala.diagnostico,
        }
