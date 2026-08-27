"""
Agrupa quadros de fontes diferentes em um Instante.

O PROBLEMA

Tres cameras sem sincronia de hardware. Cada uma entrega quando pode. A fusao
de eixos — que pega largura da frontal e profundidade da lateral — assume que
as duas vistas mostram O MESMO INSTANTE. Se uma estiver 200 ms atrasada, a
1,4 m/s a pessoa ja andou 28 cm, e o esqueleto sai esticado.

A REGRA QUE DECIDE TUDO AQUI

    O sincronizador NAO ESPERA.

Se a lateral esta atrasada, o Instante sai sem ela e a fusao trabalha com uma
vista so. Esperar acumularia atraso em todas as outras — e para visao em tempo
real, meio dado agora vale mais que dado completo tarde.

TOLERANCIA DE 120 ms, E DE ONDE VEM

A 1,4 m/s (caminhada normal), 120 ms sao 17 cm no pior caso. E aceitavel para
POSTURA, que e o que a fusao usa.

E nao afeta a POSICAO: ela vem so da camera do alto, que e sincrona consigo
mesma. Essa separacao e o que torna a tolerancia larga aceitavel — o numero
preciso do sistema (2 a 5 cm) nao passa por aqui.

`defasagem_ms` vai junto no Instante para quem consome saber quanta
simultaneidade esta assumindo.
"""


from src.fluxo.quadro import Instante


class Sincronizador:
    def __init__(self, tolerancia_ms=120.0, papel_obrigatorio=None):
        """
        papel_obrigatorio: se definido, so monta Instante quando esse papel
        estiver presente. Usado para "alto": sem ele nao ha posicao no chao,
        e um Instante sem posicao nao serve para nada.
        """
        self.tolerancia = tolerancia_ms / 1000.0
        self.obrigatorio = papel_obrigatorio
        self.montados = 0
        self.rejeitados = 0
        self.fora_de_tolerancia = 0

    def montar(self, buffers):
        """buffers: dict[papel -> FrameBuffer]. Devolve Instante ou None.

        Consome apenas os quadros que entram no grupo. Os que ficaram fora da
        tolerancia permanecem na fila para a proxima rodada — podem estar
        adiantados, e nesse caso servirao daqui a pouco.
        """
        candidatos = {}
        for papel, buf in buffers.items():
            f = buf.espiar()
            if f is not None:
                candidatos[papel] = f

        if not candidatos:
            return None

        if self.obrigatorio and self.obrigatorio not in candidatos:
            self.rejeitados += 1
            return None

        # A referencia e o MAIS ANTIGO do grupo. Assim o Instante representa um
        # momento que todos ja viveram, em vez de um futuro que so um alcancou.
        t_ref = min(f.t_mono for f in candidatos.values())

        grupo, descartados_por_tempo = {}, 0
        for papel, f in candidatos.items():
            if abs(f.t_mono - t_ref) <= self.tolerancia:
                grupo[papel] = f
            else:
                descartados_por_tempo += 1

        if self.obrigatorio and self.obrigatorio not in grupo:
            self.rejeitados += 1
            return None

        # consome so o que entrou
        for papel in grupo:
            buffers[papel].pegar()

        if descartados_por_tempo:
            self.fora_de_tolerancia += descartados_por_tempo

        tempos = [f.t_mono for f in grupo.values()]
        defasagem = (max(tempos) - min(tempos)) * 1000.0

        self.montados += 1
        return Instante(t_ref=t_ref, quadros=grupo, defasagem_ms=defasagem)

    def resumo(self):
        return {
            "montados": self.montados,
            "rejeitados": self.rejeitados,
            "fora_de_tolerancia": self.fora_de_tolerancia,
        }
