"""
Teste de ligacao do Orquestrador — o laco inteiro, sem camera e sem modelo.

O QUE ESTE TESTE PEGA QUE OS OUTROS NAO PEGAM

Os testes anteriores provam cada peca sozinha: a fonte reconecta, o
sincronizador nao espera, o motor espacial rejeita a cadeira, o gemeo emite a
entrada em zona uma vez so. Nenhum deles nota se eu ligar a saida do
sincronizador na entrada errada.

Erro de LIGACAO nao aparece em teste de unidade. Aparece quando se roda o
sistema — e ai o diagnostico e caro, porque tudo esta acontecendo ao mesmo
tempo. Aqui o laco roda de ponta a ponta em memoria, e o que chega ao arquivo
publicado tem que bater com o que foi injetado na primeira etapa.

POR QUE O TRABALHADOR DE MENTIRA VIVE AQUI E NAO NO PROGRAMA

Um trabalhador falso dentro do `rodar.py` seria um caminho pelo qual a execucao
normal poderia, por engano, exibir dados inventados como se fossem medidos.
Dentro do teste nao ha esse risco: teste e o unico lugar onde inventar dado e
o comportamento correto. `rodar.py --falsas` continua rodando o YOLO de
verdade, so que sobre quadros sinteticos.

    python testes/test_orquestrador.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.app.orquestrador import Orquestrador          # noqa: E402
from src.eventos.motor import Tipo                     # noqa: E402
from src.nucleo import log as logmod                   # noqa: E402
from src.visao.observacao import Observacao            # noqa: E402
from src.visao.trabalhador import Trabalhador          # noqa: E402

logmod.configurar("ERROR")


# ------------------------------------------------------- trabalhador de teste
class PessoaGuiada(Trabalhador):
    """Devolve uma caixa que caminha por um trajeto conhecido em PIXEIS.

    Nao chama modelo nenhum. A intencao e exercitar o caminho
    observacao -> homografia -> Kalman -> gemeo -> arquivo, com um percurso
    cujo resultado eu consigo prever a mao.
    """

    nome = "guiada"

    def __init__(self, papel="alto", passo_px=14):
        super().__init__(papel)
        self.passo = passo_px
        self.n = 0

    def _processar(self, frame):
        self.n += 1
        cx = 240 + self.n * self.passo
        x1, y1, x2, y2 = cx - 40, 170, cx + 40, 400

        j2d = np.zeros((17, 2))
        j2d[15] = [cx - 8, y2 - 4]          # tornozelos: o filtro exige ver
        j2d[16] = [cx + 8, y2 - 4]          # tornozelo para aceitar o rastro
        c2d = np.zeros(17)
        c2d[15] = c2d[16] = 0.9

        return [Observacao(
            camera_id=frame.camera_id, papel=self.papel, t_mono=frame.t_mono,
            id_externo=1, caixa=(x1, y1, x2, y2), confianca=0.9,
            juntas_2d=j2d, conf_2d=c2d)]


def arquivo_temporario(nome):
    """Caminho descartavel que existe no Windows e no Linux.

    A primeira versao deste arquivo escrevia em `/tmp`, que nao existe no
    Windows. Passou aqui e falhou na maquina do Eduardo. Teste que so roda no
    sistema de quem escreveu nao esta testando o sistema — esta testando a
    sorte.
    """
    return Path(tempfile.gettempdir()) / nome


def caminho_inexistente():
    """Pasta que garantidamente nao existe, em qualquer sistema."""
    return arquivo_temporario("so_espacial_sem_pasta") / "nao" / "existe.json"


def montar(tmp, **kw):
    """Orquestrador com fontes sinteticas e visao substituida pelo guia."""
    app = Orquestrador(captura=(640, 480), com_pose=False, **kw)
    app.montar_cameras_falsas()
    app.espacial.usar_plausibilidade = False   # sem camera real, sem altura
    app.visao.registrar(PessoaGuiada("alto"))
    app.publicador.destino = Path(tmp)
    app.publicador.a_cada = 0.0
    return app


def rodar(app, quadros=25, limite_s=15):
    """Gira o laco ate juntar N quadros processados."""
    t0 = time.monotonic()
    while app.quadros < quadros and time.monotonic() - t0 < limite_s:
        if app.passo() is None:
            time.sleep(0.005)
    return app.quadros


# ------------------------------------------------------------------- testes
def test_laco_completo_produz_pessoa_no_arquivo():
    tmp = arquivo_temporario("so_estado.json")
    app = montar(tmp)
    app.montar_visao = lambda: app        # visao ja montada a mao
    app.iniciar(espera_s=10)
    try:
        n = rodar(app)
        assert n >= 20, f"o laco girou pouco: {n} quadros"

        d = json.loads(Path(tmp).read_text(encoding="utf-8"))
        assert d["pessoas"], "o arquivo publicado saiu sem ninguem dentro"

        pessoa = d["pessoas"][0]
        assert pessoa["quadros"] > 10
        assert pessoa["percorrido"] > 0.05, "andou em pixeis e nao em metros"
        assert d["quadros"] == app.quadros, "o gemeo e o arquivo divergiram"
    finally:
        app.parar()


def test_evento_de_inicio_de_rastro_chega_ao_barramento():
    app = montar(arquivo_temporario("so_ev.json"))
    recebidos = []
    app.eventos.assinar(Tipo.TRACK_STARTED, recebidos.append)
    app.iniciar(espera_s=10)
    try:
        rodar(app, quadros=20)
        assert app.eventos.contagem.get(Tipo.SYSTEM_STARTED) == 1
        assert len(recebidos) == 1, f"esperava 1 inicio, veio {len(recebidos)}"
        assert app.eventos.contagem.get(Tipo.CAMERA_CONNECTED, 0) >= 1
    finally:
        app.parar()


def test_publicado_bate_com_o_instantaneo_do_gemeo():
    """Uma verdade so. Se o arquivo for montado por outro caminho, ele diverge
    do gemeo sem dar erro — e o painel passa a mostrar outra coisa."""
    tmp = arquivo_temporario("so_igual.json")
    app = montar(tmp)
    app.iniciar(espera_s=10)
    try:
        rodar(app, quadros=15)
        arquivo = json.loads(Path(tmp).read_text(encoding="utf-8"))
        memoria = json.loads(json.dumps(app.gemeo.instantaneo()))

        assert arquivo["pessoas"][0]["id"] == memoria["pessoas"][0]["id"]
        assert set(arquivo) == set(memoria), "o arquivo tem outro formato"
        assert arquivo["loja"] == memoria["loja"]
    finally:
        app.parar()


def test_paralelismo_e_medido_mesmo_com_um_trabalhador():
    app = montar(arquivo_temporario("so_par.json"))
    app.iniciar(espera_s=10)
    try:
        rodar(app, quadros=12)
        d = app.visao.diagnostico_paralelismo()
        assert d["real_ms"] >= 0
        assert "\n".join(app.painel()).count("CAMERAS") == 1
    finally:
        app.parar()


def test_parar_encerra_threads_de_camera_e_de_visao():
    """Thread solta segura o processo aberto no fim da sessao."""
    import threading
    antes = threading.active_count()

    app = montar(arquivo_temporario("so_parar.json"))
    app.iniciar(espera_s=10)
    rodar(app, quadros=8)
    app.parar()

    for _ in range(30):
        if threading.active_count() <= antes:
            break
        time.sleep(0.1)
    assert threading.active_count() <= antes, (
        f"sobraram threads: {antes} -> {threading.active_count()}")


def test_falha_de_publicacao_nao_derruba_o_laco():
    """Regra de 08/08: arquivo aberto no editor nao pode matar a sessao."""
    app = montar(arquivo_temporario("so_ok.json"))
    app.publicador.destino = caminho_inexistente()
    app.iniciar(espera_s=10)
    try:
        n = rodar(app, quadros=12)
        assert n >= 10, "o laco parou por causa de um canal lateral"
        assert app.publicador.falhas > 0, "a falha deveria ter sido contada"
    finally:
        app.parar()


def test_config_aceita_resolucao_por_camera():
    """Uma configuracao unica obriga a melhor camera a andar no passo da pior.

    Em 10/08 as tres rodaram a 1280x720 porque `--captura` era global. A C920
    caiu para 1,0 fps e, sendo o papel obrigatorio, impos 1 fps ao sistema
    inteiro enquanto o tablet entregava 30.
    """
    app = Orquestrador(captura=(1280, 720), com_pose=False)

    curta, _ = app._montar_uma("alto", {"fonte": "Camera X",
                                        "captura": "640x480"})
    longa, _ = app._montar_uma("lateral", {"fonte": "Camera Y",
                                           "captura": "1280x720"})
    antiga, _ = app._montar_uma("frontal", "Camera Z")   # forma antiga

    assert (curta.largura, curta.altura) == (640, 480)
    assert (longa.largura, longa.altura) == (1280, 720)
    assert (antiga.largura, antiga.altura) == (1280, 720), "cai no padrao"
    assert app._captura_por_papel["alto"] == (640, 480)


def test_config_aceita_url_e_exposicao_por_camera():
    app = Orquestrador(captura=(640, 480), com_pose=False)

    remota, _ = app._montar_uma("lateral",
                                {"fonte": "http://10.0.0.5:8080/video"})
    escura, _ = app._montar_uma("alto", {"fonte": "Camera X",
                                         "exposicao": -6})

    assert remota.tipo == "remota"
    assert escura.exposicao == -6, "exposicao por camera nao foi respeitada"


# ---------------------------------------------------------------- execucao
if __name__ == "__main__":
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    falhas = 0
    for t in testes:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            falhas += 1
            print(f"  FALHA {t.__name__}: {e}")
        except Exception as e:
            falhas += 1
            print(f"  ERRO  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(testes) - falhas}/{len(testes)} passaram")
    sys.exit(1 if falhas else 0)
