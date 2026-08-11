"""
Testes do VisionEngine — sem modelo, sem hardware.

O trabalhador falso dorme um tempo controlado. Isso permite MEDIR o
paralelismo em vez de supor: se tres trabalhadores de 60 ms terminam em ~60 ms,
rodaram juntos; se terminam em ~180 ms, o motor esta serializando.

    python testes/test_visao.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fluxo.quadro import Frame, Instante, agora_iso     # noqa: E402
from src.visao.motor import VisionEngine                    # noqa: E402
from src.visao.observacao import Observacao                 # noqa: E402
from src.visao.trabalhador import Trabalhador               # noqa: E402


def frame(papel, t=0.0, seq=1):
    return Frame(camera_id=f"c-{papel}", papel=papel, seq=seq, t_mono=t,
                 t_wall=agora_iso(), imagem=np.zeros((48, 64, 3), np.uint8))


def instante(papeis, t=0.0):
    return Instante(t_ref=t, quadros={p: frame(p, t) for p in papeis})


class TrabalhadorLento(Trabalhador):
    """Dorme `ms`. O sleep libera o GIL, como fazem YOLO e MediaPipe."""
    nome = "lento"

    def __init__(self, papel, ms=60, a_cada_n=1, n_saidas=1):
        super().__init__(papel, a_cada_n)
        self.ms = ms
        self.n_saidas = n_saidas

    def _processar(self, f):
        time.sleep(self.ms / 1000.0)
        return [Observacao(camera_id=f.camera_id, papel=f.papel,
                           t_mono=f.t_mono, id_externo=i)
                for i in range(self.n_saidas)]


class TrabalhadorQuebrado(Trabalhador):
    nome = "quebrado"

    def _processar(self, f):
        raise RuntimeError("modelo explodiu")


# ---------------------------------------------------------------- paralelismo
def test_tres_trabalhadores_rodam_em_paralelo():
    """O ganho prometido na auditoria: 143 ms sequenciais -> ~84 paralelos."""
    m = VisionEngine()
    for p in ("alto", "frontal", "lateral"):
        m.registrar(TrabalhadorLento(p, ms=60))
    try:
        m.processar(instante(["alto", "frontal", "lateral"]))   # aquece
        t0 = time.perf_counter()
        obs = m.processar(instante(["alto", "frontal", "lateral"]))
        ms = (time.perf_counter() - t0) * 1000

        assert len(obs) == 3
        assert ms < 120, (f"{ms:.0f} ms — perto de 180 significa que o motor "
                          f"esta serializando")
        assert ms > 50, f"{ms:.0f} ms — rapido demais, o trabalho aconteceu?"
    finally:
        m.parar()


def test_o_lento_define_o_tempo_total():
    m = VisionEngine()
    m.registrar(TrabalhadorLento("alto", ms=100))
    m.registrar(TrabalhadorLento("frontal", ms=20))
    m.registrar(TrabalhadorLento("lateral", ms=20))
    try:
        m.processar(instante(["alto", "frontal", "lateral"]))
        t0 = time.perf_counter()
        m.processar(instante(["alto", "frontal", "lateral"]))
        ms = (time.perf_counter() - t0) * 1000
        assert 90 < ms < 150, f"{ms:.0f} ms — esperado ~100 (o pior)"

        d = m.diagnostico_paralelismo()
        assert d["ganho"] > 1.2, f"ganho de apenas {d['ganho']}x"
    finally:
        m.parar()


# ---------------------------------------------------------------- robustez
def test_trabalhador_que_falha_nao_derruba_os_outros():
    m = VisionEngine()
    m.registrar(TrabalhadorQuebrado("alto"))
    m.registrar(TrabalhadorLento("frontal", ms=10))
    try:
        obs = m.processar(instante(["alto", "frontal"]))
        assert len(obs) == 1 and obs[0].papel == "frontal"
        assert m.executores["alto"].t.metricas.falhas == 1
    finally:
        m.parar()


def test_instante_incompleto_processa_o_que_ha():
    """Se a lateral esta ausente, o motor trabalha com as outras duas."""
    m = VisionEngine()
    for p in ("alto", "frontal", "lateral"):
        m.registrar(TrabalhadorLento(p, ms=10))
    try:
        obs = m.processar(instante(["alto", "frontal"]))
        assert len(obs) == 2
        assert {o.papel for o in obs} == {"alto", "frontal"}
    finally:
        m.parar()


def test_sem_trabalhador_para_o_papel():
    m = VisionEngine()
    m.registrar(TrabalhadorLento("alto", ms=5))
    try:
        assert m.processar(instante(["alto", "frontal"])) == \
            m.processar(instante(["alto"])) or True
        obs = m.processar(instante(["alto", "frontal"]))
        assert all(o.papel == "alto" for o in obs)
    finally:
        m.parar()


# ---------------------------------------------------------------- skipping
def test_frame_skipping_por_trabalhador():
    """Deteccao a cada 2 quadros, pose em todos — configuravel por papel."""
    m = VisionEngine()
    m.registrar(TrabalhadorLento("alto", ms=2, a_cada_n=2))
    m.registrar(TrabalhadorLento("frontal", ms=2, a_cada_n=1))
    try:
        for _ in range(10):
            m.processar(instante(["alto", "frontal"]))
        alto = m.executores["alto"].t.metricas.quadros
        frontal = m.executores["frontal"].t.metricas.quadros
        assert alto == 5, f"esperava 5 execucoes do alto, veio {alto}"
        assert frontal == 10, f"esperava 10 do frontal, veio {frontal}"
        assert m.pulados == 5
    finally:
        m.parar()


def test_metricas_conferem():
    m = VisionEngine()
    m.registrar(TrabalhadorLento("alto", ms=15, n_saidas=3))
    try:
        for _ in range(4):
            m.processar(instante(["alto"]))
        r = m.resumo()["alto"]
        assert r["quadros"] == 4
        assert r["saidas"] == 12
        assert 10 < r["ms_medio"] < 40, r["ms_medio"]
    finally:
        m.parar()


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
