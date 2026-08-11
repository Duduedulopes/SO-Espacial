"""
Testes das fontes, do buffer e do sincronizador — SEM HARDWARE.

Roda em qualquer maquina, em segundos, e reproduz sempre o mesmo caso. E o que
faltava: ate 08/08 toda verificacao era manual, com a pessoa andando na frente
da camera, e por isso as regressoes so apareciam quando ja tinham custado tempo.

    python -m pytest testes/ -v
ou
    python testes/test_fontes.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cameras.falsa import FonteFalsa                      # noqa: E402
from src.cameras.fonte import Estado                          # noqa: E402
from src.cameras.gerenciador import GerenciadorDeCameras      # noqa: E402
from src.fluxo.buffer import FrameBuffer                      # noqa: E402
from src.fluxo.quadro import Frame, agora_iso                 # noqa: E402
from src.fluxo.sincronizador import Sincronizador             # noqa: E402
from src.nucleo.metricas import MetricasDeFonte               # noqa: E402


def quadro(papel, t, seq=1):
    return Frame(camera_id=f"c-{papel}", papel=papel, seq=seq, t_mono=t,
                 t_wall=agora_iso(), imagem=np.zeros((48, 64, 3), np.uint8))


def esperar(cond, timeout=8.0, passo=0.05):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(passo)
    return False


# ---------------------------------------------------------------- buffer
def test_buffer_descarta_o_antigo():
    m = MetricasDeFonte()
    b = FrameBuffer(maxlen=2, metricas=m)
    for i in range(5):
        b.colocar(quadro("alto", i, seq=i))
    assert len(b) == 2
    assert m.descartados == 3, f"esperava 3 descartes, veio {m.descartados}"


def test_buffer_devolve_o_mais_recente():
    """Nao e FIFO de proposito: quadro velho ja perdeu a validade."""
    m = MetricasDeFonte()
    b = FrameBuffer(maxlen=3, metricas=m)
    for i in range(3):
        b.colocar(quadro("alto", i, seq=i))
    f = b.pegar()
    assert f.seq == 2, "deveria devolver o ultimo, nao o primeiro"
    assert b.vazio()
    assert m.descartados == 2


# ---------------------------------------------------------------- sincronizador
def test_sincronizador_agrupa_dentro_da_tolerancia():
    bufs = {p: FrameBuffer(2) for p in ("alto", "frontal", "lateral")}
    t = 100.0
    bufs["alto"].colocar(quadro("alto", t))
    bufs["frontal"].colocar(quadro("frontal", t + 0.03))
    bufs["lateral"].colocar(quadro("lateral", t + 0.05))

    inst = Sincronizador(tolerancia_ms=120).montar(bufs)
    assert inst is not None and len(inst) == 3
    assert inst.defasagem_ms < 60


def test_sincronizador_exclui_quem_esta_fora():
    bufs = {p: FrameBuffer(2) for p in ("alto", "frontal")}
    t = 100.0
    bufs["alto"].colocar(quadro("alto", t))
    bufs["frontal"].colocar(quadro("frontal", t + 0.9))     # 900 ms atrasado

    s = Sincronizador(tolerancia_ms=120)
    inst = s.montar(bufs)
    assert "frontal" not in inst, "quadro fora da tolerancia nao pode entrar"
    assert len(bufs["frontal"]) == 1, "o excluido deve continuar na fila"
    assert s.fora_de_tolerancia == 1


def test_sincronizador_exige_o_papel_obrigatorio():
    """Sem a camera do alto nao ha posicao no chao — o Instante nao serve."""
    bufs = {p: FrameBuffer(2) for p in ("alto", "frontal")}
    bufs["frontal"].colocar(quadro("frontal", 100.0))
    s = Sincronizador(papel_obrigatorio="alto")
    assert s.montar(bufs) is None
    assert s.rejeitados == 1


# ---------------------------------------------------------------- fonte
def test_fonte_entrega_quadros():
    f = FonteFalsa("alto", fps=60)
    f.iniciar()
    try:
        assert esperar(lambda: f.estado == Estado.ONLINE), f.estado
        assert esperar(lambda: f.metricas.recebidos > 5)
        assert f.ler() is not None
        # metrica que mente e pior que metrica ausente: uma camera que nunca
        # caiu nao pode reportar reconexao
        assert f.metricas.reconexoes == 0, "primeira conexao nao e reconexao"
    finally:
        f.parar()


def test_reconexao_e_contada_so_quando_e_reconexao():
    f = FonteFalsa("alto", fps=60, cair_apos_s=0.4,
                   silencio_degradada=0.5, silencio_falha=1.0)
    f.iniciar()
    try:
        assert esperar(lambda: f.estado == Estado.ONLINE)
        assert f.metricas.reconexoes == 0
        assert esperar(lambda: f.metricas.reconexoes >= 1, timeout=8), \
            "deveria ter contado ao menos uma reconexao apos a queda"
    finally:
        f.parar()


def test_fonte_preta_nao_fica_online_por_engano():
    """Preto e imagem valida: a fonte fica ONLINE. Quem julga brilho e a
    camera USB, na abertura. Aqui garantimos que o encanamento nao trava."""
    f = FonteFalsa("alto", modo="preta", fps=60)
    f.iniciar()
    try:
        assert esperar(lambda: f.metricas.recebidos > 3)
        assert f.metricas.brilho == 0.0
    finally:
        f.parar()


def test_fonte_que_cai_vai_para_degradada_e_falha():
    """O caso que motivou o estado DEGRADADA: antes, a fonte devolvia o ultimo
    quadro para sempre e o sistema processava imagem congelada."""
    f = FonteFalsa("alto", fps=60, cair_apos_s=0.5,
                   silencio_degradada=0.6, silencio_falha=1.5)
    f.iniciar()
    try:
        assert esperar(lambda: f.estado == Estado.ONLINE)
        assert esperar(lambda: f.estado == Estado.DEGRADADA, timeout=4)
        assert f.ler() is None, "DEGRADADA nao pode entregar imagem velha"
        assert esperar(lambda: f.estado == Estado.FALHA, timeout=6)
    finally:
        f.parar()


def test_fonte_morta_agenda_retentativas_com_recuo():
    f = FonteFalsa("alto", falhar_ao_abrir=True)
    f.iniciar()
    try:
        assert esperar(lambda: f.metricas.reconexoes >= 0 and
                       f.estado == Estado.FALHA, timeout=5)
        assert f.ultimo_erro is not None
    finally:
        f.parar()


# ---------------------------------------------------------------- gerenciador
def test_uma_camera_cai_e_as_outras_seguem():
    """O requisito central: falha parcial nao derruba o sistema."""
    eventos = []
    g = GerenciadorDeCameras(ao_evento=lambda t, d: eventos.append(t))
    g.registrar(FonteFalsa("alto", fps=60))
    g.registrar(FonteFalsa("frontal", fps=60))
    g.registrar(FonteFalsa("lateral", fps=60, cair_apos_s=0.5,
                           silencio_degradada=0.6, silencio_falha=1.5))
    g.iniciar()
    try:
        assert esperar(lambda: len(g.online()) == 3, timeout=6)
        assert esperar(lambda: len(g.online()) == 2, timeout=8), \
            "a lateral deveria ter saido"
        assert g.tem("alto") and g.tem("frontal")
        assert "CAMERA_CONNECTED" in eventos
    finally:
        g.parar()


def test_camera_lenta_nao_atrasa_as_outras():
    g = GerenciadorDeCameras()
    g.registrar(FonteFalsa("alto", fps=60))
    g.registrar(FonteFalsa("lateral", modo="lenta", fps=60))
    g.iniciar()
    try:
        esperar(lambda: g.por_papel("alto").metricas.recebidos > 25, timeout=6)
        rapida = g.por_papel("alto").metricas.recebidos
        lenta = g.por_papel("lateral").metricas.recebidos
        assert rapida > lenta * 3, \
            f"a rapida ({rapida}) deveria adiantar muito a lenta ({lenta})"
    finally:
        g.parar()


def test_pipeline_completo_tres_fontes():
    g = GerenciadorDeCameras()
    for p in ("alto", "frontal", "lateral"):
        g.registrar(FonteFalsa(p, fps=40))
    g.iniciar()
    try:
        assert esperar(lambda: len(g.online()) == 3, timeout=6)
        s = Sincronizador(tolerancia_ms=150, papel_obrigatorio="alto")
        montados = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            if s.montar(g.buffers()) is not None:
                montados += 1
            time.sleep(0.01)
        assert montados > 20, f"so montou {montados} instantes em 2 s"
    finally:
        g.parar()


def test_estreia_falhada_nao_e_desconexao():
    """Nao se cai de um lugar onde nunca se esteve.

    Em 10/08 o painel mostrou quatro CAMERA_DISCONNECTED do tablet ANTES do
    primeiro CAMERA_CONNECTED. As duas situacoes pedem acoes opostas —
    conferir cabo/driver contra conferir rede/contencao — e nao podem dividir
    o mesmo nome.
    """
    from src.cameras.gerenciador import GerenciadorDeCameras

    vistos = []
    g = GerenciadorDeCameras(ao_evento=lambda t, d: vistos.append(t))
    g.registrar(FonteFalsa("alto", falhar_ao_abrir=True))
    g.iniciar()
    time.sleep(1.2)
    g.parar()

    assert "CAMERA_ERROR" in vistos, f"eventos: {vistos}"
    assert "CAMERA_DISCONNECTED" not in vistos, (
        "camera que nunca subiu nao pode 'desconectar'")
    assert "CAMERA_CONNECTED" not in vistos


def test_queda_depois_de_online_e_desconexao():
    """O contraste do teste acima: aqui a queda e real."""
    from src.cameras.gerenciador import GerenciadorDeCameras

    vistos = []
    g = GerenciadorDeCameras(ao_evento=lambda t, d: vistos.append(t))
    # Limiares encolhidos: o teste quer a ORDEM dos eventos, nao os 10 s de
    # silencio que a producao usa para nao chamar de falha um engasgo.
    g.registrar(FonteFalsa("alto", cair_apos_s=0.4,
                           silencio_degradada=0.3, silencio_falha=0.8))
    g.iniciar()
    time.sleep(2.5)
    g.parar()

    assert "CAMERA_CONNECTED" in vistos, f"eventos: {vistos}"
    assert "CAMERA_DISCONNECTED" in vistos, f"eventos: {vistos}"
    assert vistos.index("CAMERA_CONNECTED") < vistos.index("CAMERA_DISCONNECTED")


def test_evento_de_falha_mostra_o_motivo_antes_de_tudo():
    """O painel corta o evento nos tres primeiros campos.

    Em 10/08 tres CAMERA_ERROR do tablet sairam como
    `camera=... papel=lateral de=conectando` — falhou, e o motivo, unico campo
    que dizia o que fazer a respeito, ficou de fora.
    """
    from src.cameras.gerenciador import GerenciadorDeCameras

    vistos = []
    g = GerenciadorDeCameras(ao_evento=lambda t, d: vistos.append((t, d)))
    g.registrar(FonteFalsa("alto", falhar_ao_abrir=True))
    g.iniciar()
    time.sleep(1.2)
    g.parar()

    erros = [d for t, d in vistos if t == "CAMERA_ERROR"]
    assert erros, f"eventos: {vistos}"
    assert list(erros[0])[0] == "erro", (
        f"o motivo tem que vir primeiro: {list(erros[0])}")
    assert erros[0]["erro"], "o motivo veio vazio"


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
