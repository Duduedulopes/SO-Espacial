"""
Taxonomia de erros.

POR QUE UMA HIERARQUIA, E NAO `Exception` GENERICA

Cada erro destes exige um tratamento DIFERENTE, e sem tipos distintos o codigo
que trata acaba escrevendo `except Exception` e agindo igual para todos.

    CameraNaoEncontrada   -> a configuracao esta desatualizada; avise o usuario
    CameraNaoAbriu        -> outro processo segura o dispositivo; tente depois
    CameraSemImagem       -> driver travado; tente recuperar
    CameraImagemInvalida  -> abriu e entrega preto; problema de exposicao
    ConexaoPerdida        -> rede; reconecte com recuo exponencial

Confundir os quatro primeiros custou horas em 08/08: o sintoma "0 pessoas
detectadas" mandava procurar no detector, quando a causa era imagem preta.

Todo erro carrega `dados` — o contexto que o log estruturado vai registrar.
"""


class ErroDoSistema(Exception):
    """Raiz. Tudo que o sistema lanca de proposito herda daqui."""

    def __init__(self, mensagem, **dados):
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.dados = dados

    def __str__(self):
        if not self.dados:
            return self.mensagem
        extra = "  ".join(f"{k}={v}" for k, v in self.dados.items())
        return f"{self.mensagem}  [{extra}]"


# ---------------------------------------------------------------- cameras
class ErroDeCamera(ErroDoSistema):
    pass


class CameraNaoEncontrada(ErroDeCamera):
    """O nome configurado nao esta entre os dispositivos presentes."""
    sugestao = "rode: python ferramentas/identificar.py"


class CameraNaoAbriu(ErroDeCamera):
    """Presente no sistema, mas recusou abrir.

    Quase sempre outro processo segurando o dispositivo — inclusive um
    python.exe travado de uma execucao anterior.
    """
    sugestao = "feche outros programas de camera e processos python travados"


class CameraSemImagem(ErroDeCamera):
    """Abriu, mas nao entrega quadro.

    Na C920 isto foi resultado de excesso de `set()` numa camera que ainda
    estava inicializando. Aquecer antes de julgar evita causar o defeito.
    """
    sugestao = "desconecte o cabo por 10s e reconecte"


class CameraImagemInvalida(ErroDeCamera):
    """Entrega quadro, mas preto ou corrompido."""
    sugestao = "exposicao travada no driver; a fonte tentara recuperar"


# ---------------------------------------------------------------- streams
class ErroDeStream(ErroDoSistema):
    pass


class ConexaoPerdida(ErroDeStream):
    sugestao = "verifique a rede e o aplicativo no celular"


class TempoEsgotado(ErroDeStream):
    pass


# ---------------------------------------------------------------- visao
class ErroDeVisao(ErroDoSistema):
    pass


class ModeloIndisponivel(ErroDeVisao):
    pass


class FalhaNaInferencia(ErroDeVisao):
    pass


# ---------------------------------------------------------------- calibracao
class ErroDeCalibracao(ErroDoSistema):
    pass


class HomografiaAusente(ErroDeCalibracao):
    sugestao = "rode: python ferramentas/calibrar.py"


class ResolucaoIncompativel(ErroDeCalibracao):
    """A homografia foi calibrada noutra resolucao.

    Nao e fatal: a matriz pode ser reescalada. Vira aviso, nao excecao, quando
    a diferenca e so de escala.
    """
    pass
