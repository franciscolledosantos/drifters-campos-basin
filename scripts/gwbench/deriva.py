"""Deriva lagrangiana híbrida física + aprendizado de máquina (gwbench v3.4).

Origem: previsão de trajetórias de derivadores na Bacia de Campos (Global Drifter Program + HYCOM + NCEP).
Funções genéricas, independentes da fonte dos dados:

- ``integrar_rk2``: integra posições (lat, lon) com um campo de velocidade qualquer, passo fixo (s).
- ``separacao_km``: distância de grande círculo entre trajetórias previstas e observadas.
- ``skill_liu_weisberg``: escore de habilidade de Liu & Weisberg (2011), separação cumulativa normalizada.
- ``janela_vento``: coeficiente de arrasto do vento (windage) por mínimos quadrados, por grupo (ex.: drogue).
- ``ResiduoHibrido``: aprende o resíduo de velocidade (observado − modelo físico) com qualquer regressor
  (LightGBM, MLP...), usando como preditores as forçantes no ponto, o resíduo inicial r0 e o horizonte tau.
"""
import numpy as np

R_TERRA = 6371e3


def integrar_rk2(lat0, lon0, t0, vel, n_passos, dt=3600.0):
    """vel(t, lat, lon, k) -> (u, v) em m/s; t em horas. Devolve array (n, n_passos+1, 2)."""
    lat = np.asarray(lat0, float).copy(); lon = np.asarray(lon0, float).copy()
    out = np.zeros((len(lat), n_passos + 1, 2)); out[:, 0, 0] = lat; out[:, 0, 1] = lon
    h = dt / 3600.0
    for k in range(n_passos):
        t = t0 + k * h
        u1, v1 = vel(t, lat, lon, k)
        lam = lat + 0.5 * dt * v1 / R_TERRA * 180 / np.pi
        lom = lon + 0.5 * dt * u1 / (R_TERRA * np.cos(np.deg2rad(lat))) * 180 / np.pi
        u2, v2 = vel(t + 0.5 * h, lam, lom, k + 0.5)
        lat = lat + dt * v2 / R_TERRA * 180 / np.pi
        lon = lon + dt * u2 / (R_TERRA * np.cos(np.deg2rad(lat))) * 180 / np.pi
        out[:, k + 1, 0] = lat; out[:, k + 1, 1] = lon
    return out


def separacao_km(a, b):
    la1, lo1, la2, lo2 = map(np.deg2rad, (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    x = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_TERRA * np.arcsin(np.sqrt(x)) / 1e3


def skill_liu_weisberg(prev, obs, ate=None, n=1.0):
    """s = 1 − c/n se c ≤ n, senão 0; c = Σ d_i / Σ l_oi (Liu & Weisberg 2011)."""
    ate = ate or (obs.shape[1] - 1)
    d = separacao_km(prev[:, 1:ate + 1], obs[:, 1:ate + 1])
    l = np.cumsum(separacao_km(obs[:, 1:ate + 1], obs[:, :ate]), 1)
    c = d.sum(1) / np.maximum(l.sum(1), 1e-9)
    return np.where(c <= n, 1 - c / n, 0.0)


def janela_vento(res_u, res_v, uw, vw, grupo=None):
    """alfa minimizando |r − alfa·W|² (por grupo). Devolve dict grupo -> alfa."""
    grupo = np.zeros(len(res_u), int) if grupo is None else np.asarray(grupo)
    out = {}
    for gr in np.unique(grupo):
        m = grupo == gr
        W = np.r_[uw[m], vw[m]]; r = np.r_[res_u[m], res_v[m]]
        out[gr] = float(W @ r / (W @ W))
    return out


class ResiduoHibrido:
    """Regressor do resíduo de velocidade (2 saídas) a partir de [forçantes, r0, tau]."""

    def __init__(self, fabrica):
        self.fabrica = fabrica

    def fit(self, X, R):
        self.m = [self.fabrica().fit(X, R[:, j]) for j in range(2)]
        return self

    def predict(self, X):
        return np.c_[self.m[0].predict(X), self.m[1].predict(X)]
