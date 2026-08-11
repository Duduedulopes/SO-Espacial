"""
Filtro de Kalman 1D, escrito do zero — critério de domínio do bloco 2.

Sem filterpy, sem biblioteca de filtragem. So numpy, para a matematica ficar
visivel.

DADOS SINTETICOS DE PROPOSITO: com a verdade conhecida da para MEDIR o erro.
Na vida real voce nunca sabe onde a pessoa realmente estava, entao nunca sabe
se o filtro ajudou. Aqui sabe.

O experimento reproduz os dois problemas medidos em 07/08:

    1. RUIDO      — a posicao treme ~4 cm por quadro
    2. AUSENCIA   — a pessoa some por ~1,6 s e o rastro se parte

Rode:
    python estado/kalman_1d.py
"""

import numpy as np


class Kalman1D:
    """Modelo de velocidade constante.

    ESTADO (o que o filtro carrega entre quadros):

        x = [ posicao ]
            [ velocidade ]

    Guardar velocidade e o que permite prever durante a ausencia. Um filtro
    que so guardasse posicao nao teria como adivinhar para onde a pessoa ia.

    MATRIZES:

        F  como o estado evolui sozinho, em dt segundos:
               posicao_nova = posicao + velocidade * dt
               velocidade_nova = velocidade
           ou seja  F = [[1, dt],
                         [0,  1]]

        H  o que o sensor enxerga do estado. Medimos posicao, nao velocidade:
               H = [[1, 0]]

        P  incerteza sobre o estado, como matriz de covariancia.
           Cresce quando prevemos, encolhe quando corrigimos.

        Q  ruido de processo — o quanto o mundo foge do modelo.
           Pessoa nao anda em velocidade constante: ela acelera, para, vira.
           Q grande = "confio pouco no meu modelo de movimento".

        R  ruido de medicao — o quanto o sensor erra.
           Voce MEDIU isto: ~4 cm de tremor. R = 0.04^2.
    """

    def __init__(self, pos_inicial, dt, ruido_medicao, ruido_processo):
        self.dt = dt
        self.x = np.array([[pos_inicial], [0.0]])   # comeca parado

        # Incerteza inicial: sei bem onde esta (acabei de medir),
        # nao faco ideia da velocidade.
        self.P = np.diag([ruido_medicao ** 2, 1.0])

        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[ruido_medicao ** 2]])

        # Q para modelo de velocidade constante com aceleracao aleatoria.
        # Derivado da integracao do ruido de aceleracao ao longo de dt.
        a = ruido_processo ** 2
        self.Q = a * np.array([[dt**4 / 4, dt**3 / 2],
                               [dt**3 / 2, dt**2]])

        self.I = np.eye(2)

    def prever(self):
        """Onde deveria estar agora. A incerteza CRESCE: o tempo passou."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0])

    def corrigir(self, z):
        """Chegou medicao. A incerteza DIMINUI: acabei de olhar."""
        y = np.array([[z]]) - self.H @ self.x          # inovacao
        S = self.H @ self.P @ self.H.T + self.R        # incerteza da inovacao
        K = self.P @ self.H.T @ np.linalg.inv(S)       # ganho de Kalman

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(K[0, 0])

    @property
    def posicao(self):
        return float(self.x[0, 0])

    @property
    def velocidade(self):
        return float(self.x[1, 0])

    @property
    def incerteza(self):
        """Desvio padrao da posicao, em metros."""
        return float(np.sqrt(self.P[0, 0]))


def main():
    rng = np.random.default_rng(42)

    dt = 1 / 30
    n = 300
    RUIDO_MEDICAO = 0.04       # 4 cm — foi o que voce mediu
    RUIDO_PROCESSO = 0.6       # aceleracao tipica de pedestre, m/s^2

    # ---------- verdade ----------
    # Pessoa andando a 0,6 m/s e parando na metade.
    t = np.arange(n) * dt
    v = np.where(t < 5, 0.6, 0.0)
    verdade = np.cumsum(v) * dt

    # ---------- medicao ----------
    medida = verdade + rng.normal(0, RUIDO_MEDICAO, n)

    # ---------- ausencia ----------
    # 1,6 s sem deteccao, como aconteceu na sua sessao.
    inicio_gap, fim_gap = 150, 150 + int(1.6 / dt)
    disponivel = np.ones(n, dtype=bool)
    disponivel[inicio_gap:fim_gap] = False

    # ---------- filtro ----------
    kf = Kalman1D(medida[0], dt, RUIDO_MEDICAO, RUIDO_PROCESSO)
    filtrado = np.zeros(n)
    incerteza = np.zeros(n)
    ganhos = np.zeros(n)

    for i in range(n):
        kf.prever()
        if disponivel[i]:
            _, k = kf.corrigir(medida[i])
            ganhos[i] = k
        else:
            ganhos[i] = np.nan
        filtrado[i] = kf.posicao
        incerteza[i] = kf.incerteza

    # ---------- avaliacao ----------
    def erro(est, mascara=None):
        m = np.ones(n, dtype=bool) if mascara is None else mascara
        return float(np.sqrt(np.mean((est[m] - verdade[m]) ** 2)))

    tremor_bruto = np.median(np.abs(np.diff(medida))) * 100
    tremor_filt = np.median(np.abs(np.diff(filtrado))) * 100

    print("=" * 60)
    print("ERRO em relacao a verdade (RMSE, cm)")
    print("=" * 60)
    print(f"  medicao crua      : {100*erro(medida):6.2f}")
    print(f"  filtro de Kalman  : {100*erro(filtrado):6.2f}")
    print(f"  reducao           : {100*(1 - erro(filtrado)/erro(medida)):5.0f}%")
    print()
    print("TREMOR entre quadros consecutivos (mediana, cm)")
    print(f"  medicao crua      : {tremor_bruto:6.2f}")
    print(f"  filtrado          : {tremor_filt:6.2f}")
    print()
    print("DURANTE A AUSENCIA (1,6 s sem medicao)")
    g = ~disponivel
    print(f"  erro do filtro    : {100*erro(filtrado, g):6.2f} cm")
    print(f"  incerteza no fim  : {100*incerteza[fim_gap-1]:6.2f} cm")
    print(f"  (a incerteza cresce sozinha — o filtro SABE que esta chutando)")
    print()
    print("GANHO DE KALMAN — o peso dado a medicao")
    print(f"  primeiro quadro   : {ganhos[0]:.3f}")
    print(f"  apos 1 s          : {ganhos[30]:.3f}")
    print(f"  em regime         : {np.nanmedian(ganhos[60:150]):.3f}")
    print("  (comeca alto: nao sei nada. cai: ja tenho historico.)")

    # ---------- grafico ----------
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       height_ratios=[3, 1])

        ax1.plot(t, medida, ".", ms=3, alpha=.4, label="medicao crua")
        ax1.plot(t, verdade, "-", lw=2, label="verdade")
        ax1.plot(t, filtrado, "-", lw=1.8, label="Kalman")
        ax1.fill_between(t, filtrado - 2*incerteza, filtrado + 2*incerteza,
                         alpha=.2, label="incerteza (2 sigma)")
        ax1.axvspan(t[inicio_gap], t[fim_gap], alpha=.15, color="red")
        ax1.text(t[inicio_gap], ax1.get_ylim()[0], " sem medicao", va="bottom", fontsize=9)
        ax1.set_ylabel("posicao (m)")
        ax1.legend(loc="lower right")
        ax1.set_title("Kalman 1D — ruido de 4 cm e 1,6 s de ausencia")
        ax1.grid(alpha=.3)

        ax2.plot(t, ganhos, lw=1.5)
        ax2.axvspan(t[inicio_gap], t[fim_gap], alpha=.15, color="red")
        ax2.set_ylabel("ganho K")
        ax2.set_xlabel("tempo (s)")
        ax2.grid(alpha=.3)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\n(sem grafico: {e})")


if __name__ == "__main__":
    main()
