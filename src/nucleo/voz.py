"""
Voz e apitos — para o teste nao exigir que a pessoa OLHE a tela.

O DEFEITO QUE ISTO CONSERTA E DE PROJETO, NAO DE CODIGO

MEDIDO EM 11/08: a primeira execucao do roteiro completo tirou 28%, com dez
acoes reprovadas. Quase todas leram `parado`. A causa nao estava no
classificador:

    o teste mandava ANDAR pela sala e ao mesmo tempo LER instrucoes numa tela
    que a pessoa nao enxerga de longe.

Entao a pessoa ficou perto do monitor e mal se deslocou. Sem caminhada, o
azimute e a inclinacao nunca aprenderam, e tudo que depende deles desabou —
dez sintomas de uma causa so, e a causa era o aparato de medicao.

    Um instrumento que atrapalha o fenomeno mede o instrumento.

Voz resolve por construcao: a instrucao chega ao ouvido, a pessoa fica livre
para andar pela sala inteira, e o resultado de cada passo tambem e falado —
entao ela sabe na hora se aquele passo funcionou, sem voltar ao computador.

POR QUE POWERSHELL E NAO UMA BIBLIOTECA

`pyttsx3` e `comtypes` fariam isso com menos linhas. Mas o `requirements.txt`
deste projeto tem cinco linhas e cada uma tem justificativa escrita. Uma
dependencia nova para falar frases numa ferramenta de teste nao paga o proprio
custo — `System.Speech` ja vem no Windows, e o `winsound` ja vem no Python.

    Dependencia que so a ferramenta de teste usa e peso que a producao carrega
    de graca.

NUNCA DERRUBA QUEM CHAMOU

Mesma regra do publicador e do gravador de eventos, aprendida em 08/08 com um
arquivo aberto no editor: canal lateral nao pode interromper o principal. Se a
sintese de voz falhar, o roteiro continua — mudo, e nada mais.
"""

import platform
import shutil
import subprocess
import sys

WINDOWS = platform.system() == "Windows"


class Voz:
    """Fala frases em portugues. Silenciosa e inofensiva fora do Windows."""

    def __init__(self, ligada=True, velocidade=1):
        self.ligada = ligada and WINDOWS and bool(shutil.which("powershell"))
        self.velocidade = max(-10, min(10, velocidade))
        self._processo = None
        self.falhou = False

        if ligada and not self.ligada:
            motivo = ("nao e Windows" if not WINDOWS
                      else "powershell nao encontrado")
            print(f"  (voz desligada: {motivo})", file=sys.stderr)

    def dizer(self, texto, esperar=False):
        """Fala. `esperar=False` devolve na hora e a fala segue em paralelo.

        O padrao e NAO esperar porque a contagem do passo nao pode ficar presa
        atras da sintese — o cronometro do gabarito precisa comecar quando o
        apito toca, nao quando a frase termina.
        """
        if not self.ligada or not texto:
            return

        self.calar()
        # Aspa simples e o escape do PowerShell, e frase com apostrofo quebra
        # o comando inteiro. Dobrar e o suficiente; nada aqui vem do usuario.
        seguro = str(texto).replace("'", "''")
        comando = (
            "Add-Type -AssemblyName System.Speech; "
            "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$v.Rate = {self.velocidade}; "
            f"$v.Speak('{seguro}')")

        try:
            self._processo = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-Command", comando],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if esperar:
                self._processo.wait(timeout=30)
        except Exception:
            # Canal lateral: falar e enfeite, o roteiro e o que importa.
            self.ligada = False
            self.falhou = True

    def calar(self):
        """Interrompe a fala em curso.

        Sem isto, anunciar o passo seguinte enquanto o resultado do anterior
        ainda esta sendo lido empilha vozes sobrepostas — e a pessoa perde as
        duas informacoes em vez de ganhar uma.
        """
        if self._processo and self._processo.poll() is None:
            try:
                self._processo.kill()
            except Exception:
                pass
        self._processo = None


def apitar(frequencia=880, ms=180):
    """Marca o instante EXATO em que a contagem comeca e termina.

    A fala tem duracao variavel e nao serve de marco temporal: quem escuta
    "ande para frente" nao sabe se o cronometro comecou na primeira ou na
    ultima silaba. O apito e instantaneo e inequivoco.

    Fora do Windows nao faz nada, e tudo bem — o roteiro nao depende disso.
    """
    if not WINDOWS:
        return
    try:
        import winsound
        winsound.Beep(int(frequencia), int(ms))
    except Exception:
        pass


def apito_de_inicio():
    apitar(1200, 150)


def apito_de_fim():
    apitar(600, 120)
    apitar(440, 200)
