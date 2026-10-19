"""Previsão de trajetórias de derivadores na Bacia de Campos: física (HYCOM, HYCOM+vento calibrado), persistência
e híbridos física+ML (resíduo de velocidade por LightGBM, MLP e Transformer), com validação agrupada por derivador
(10 dobras, nenhum derivador em treino e teste) e horizontes de 24/48/72 h."""
import sys, json, time, numpy as np, pandas as pd, torch, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from forcantes import carrega_hycom, carrega_vento, carrega_glorys, carrega_era5, Campo, horas, D

rng = np.random.default_rng(0)
torch.manual_seed(0); torch.set_num_threads(2)
R = 6371e3
HMAX = 72
import os
SMOKE = os.environ.get('SMOKE') == '1'
FORC = os.environ.get('FORC', 'hycom'); VENTO = os.environ.get('VENTO', 'ncep')
CV = os.environ.get('CV', 'drifter')
TAG = ('' if (FORC, VENTO) == ('hycom', 'ncep') else f'_{FORC}_{VENTO}') + ('_cvano' if CV == 'year' else '') + os.environ.get('SUF', '')
NOME = {'hycom': 'HYCOM', 'glorys': 'GLORYS12'}[FORC]

# ---------------- derivadores
g = pd.read_csv(D + 'gdp.csv', skiprows=[1], parse_dates=['time', 'drogue_lost_date'])
g['time'] = g.time.dt.tz_localize(None); g['drogue_lost_date'] = g.drogue_lost_date.dt.tz_localize(None)
g = g.sort_values(['ID', 'time']).reset_index(drop=True)
g['drog'] = ((g.drogue_lost_date.isna()) | (g.time < g.drogue_lost_date)).astype(int)
g['th'] = horas(g.time)
nov = (g.ID != g.ID.shift()) | (g.th.diff() > 1.01)
g['seg'] = nov.cumsum()

h = carrega_glorys() if FORC == 'glorys' else carrega_hycom(); H = Campo(h, 'water_u', 'water_v')
w = carrega_era5() if VENTO == 'era5' else carrega_vento(); V = Campo(w, 'uw', 'vw')
g['uH'], g['vH'] = H(g.th.values, g.latitude.values, g.longitude.values)
g = g[np.isfinite(g.uH) & ~((g.uH == 0) & (g.vH == 0))].copy()        # só onde há HYCOM (fora de terra)
g['uw'], g['vw'] = V(g.th.values, g.latitude.values, g.longitude.values)
g['vort'], g['div'], g['strain'] = H.derivadas(g.th.values, g.latitude.values, g.longitude.values)
g['rU'] = g.ve - g.uH; g['rV'] = g.vn - g.vH
print('obs com forçantes:', len(g), 'derivadores:', g.ID.nunique(), flush=True)

# reindexar segmentos contínuos após filtro
nov = (g.ID != g.ID.shift()) | (g.th.diff() > 1.01)
g['seg'] = nov.cumsum(); g = g.reset_index(drop=True)


def feats_ponto(th, lat, lon, drog):
    uH, vH = H(th, lat, lon); uw, vw = V(th, lat, lon)
    vo, dv, st = H.derivadas(th, lat, lon)
    doy = (pd.Timestamp('2005-01-01') + pd.to_timedelta(th, 'h')).dayofyear.values
    return np.c_[uH, vH, np.hypot(uH, vH), uw, vw, np.hypot(uw, vw), vo * 1e5, dv * 1e5, st * 1e5,
                 lat, lon, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), drog]


NF = 14

# ---------------- inícios de previsão (a cada 24 h, com 72 h contínuas à frente)
starts = []
for s, gs in g.groupby('seg'):
    if len(gs) < HMAX + 25:
        continue
    idx = gs.index.values
    for k in range(24, len(idx) - HMAX, 24):          # 24 h de histórico (para o Transformer) e 72 h à frente
        starts.append(idx[k])
starts = np.array(starts)
print('inícios de previsão:', len(starts), flush=True)

# ---------------- pares de treino (t0, tau) a partir das trajetórias observadas (inícios a cada 6 h)
def pares(ids_ok, passo0=6, passotau=3):
    linhas = []
    for s, gs in g[g.ID.isin(ids_ok)].groupby('seg'):
        if len(gs) < 25:
            continue
        a = gs[['rU', 'rV']].values; X = gs[['th', 'latitude', 'longitude', 'drog']].values
        n = len(gs)
        for k in range(24, n - 1, passo0):
            hist = a[k - 23:k + 1]                       # 24 h de resíduos até t0 (inclusive)
            for tau in range(0, min(HMAX, n - 1 - k) + 1, passotau):
                linhas.append((gs.index[k], gs.index[k + tau], tau))
    return np.array(linhas)


def montar(P):
    i0, it, tau = P[:, 0], P[:, 1], P[:, 2]
    F = g.loc[it, ['uH', 'vH']].values
    Fp = np.c_[g.loc[it, ['uH', 'vH']].values, np.hypot(*g.loc[it, ['uH', 'vH']].values.T),
               g.loc[it, ['uw', 'vw']].values, np.hypot(*g.loc[it, ['uw', 'vw']].values.T),
               g.loc[it, ['vort', 'div', 'strain']].values * 1e5, g.loc[it, ['latitude', 'longitude']].values]
    doy = g.loc[it, 'time'].dt.dayofyear.values
    Fp = np.c_[Fp, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), g.loc[it, 'drog'].values]
    r0 = g.loc[i0, ['rU', 'rV']].values
    Xf = np.c_[Fp, r0, tau]
    Y = g.loc[it, ['rU', 'rV']].values
    return Xf, Y, i0


def resumo_hist(S):
    """Resumo do histórico de 24 h (resíduos) para modelos tabulares: médias das últimas 3, 6, 12 e 24 h e valores em t0−12 e t0−23."""
    r = S[:, :, :2]
    return np.c_[r[:, -3:].mean(1), r[:, -6:].mean(1), r[:, -12:].mean(1), r.mean(1), r[:, -13], r[:, 0]]


def historico(i0):
    """sequência dos 24 resíduos e forçantes até t0 (para o Transformer)."""
    S = np.stack([g.loc[i0 - 23 + j, ['rU', 'rV', 'uH', 'vH', 'uw', 'vw']].values for j in range(24)], 1)
    return S.astype(np.float32)


# ---------------- modelos de resíduo
class TransRes(torch.nn.Module):
    def __init__(self, nf, d=32):
        super().__init__()
        self.emb_seq = torch.nn.Linear(6, d); self.emb_q = torch.nn.Linear(nf, d)
        self.pos = torch.nn.Parameter(torch.zeros(1, 25, d))
        lay = torch.nn.TransformerEncoderLayer(d, 4, 64, dropout=0.1, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(lay, 2)
        self.out = torch.nn.Linear(d, 2)

    def forward(self, seq, q):
        x = torch.cat([self.emb_seq(seq), self.emb_q(q)[:, None]], 1) + self.pos
        return self.out(self.enc(x)[:, -1])


class MLPRes(torch.nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(nf, 128), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 2))

    def forward(self, seq, q):
        return self.net(q)


def treinar_torch(cls, Xtr, Ytr, Str, epocas=12):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    net = cls(Xtr.shape[1]); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Ytr, dtype=torch.float32)
    St = torch.tensor(Str / np.array([0.3, 0.3, 0.5, 0.5, 8, 8], np.float32))
    n = len(Xt); bs = 512
    for ep in range(epocas):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            j = perm[b:b + bs]
            loss = torch.nn.functional.huber_loss(net(St[j], Xt[j]), Yt[j], delta=0.2)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda X, S: net(torch.tensor(S / np.array([0.3, 0.3, 0.5, 0.5, 8, 8], np.float32)),
                            torch.tensor((X - mu) / sd, dtype=torch.float32)).detach().numpy()


# ---------------- integração de trajetórias (RK2, dt = 1 h)
def integrar(i0s, modo, modelo=None, alfa=None):
    lat = g.loc[i0s, 'latitude'].values.astype(float); lon = g.loc[i0s, 'longitude'].values.astype(float)
    th0 = g.loc[i0s, 'th'].values; drog = g.loc[i0s, 'drog'].values.astype(float)
    r0 = g.loc[i0s, ['rU', 'rV']].values; v0 = g.loc[i0s, ['ve', 'vn']].values
    seq = historico(i0s) if modo in ('trans', 'mlp', 'lgbh') else None
    traj = np.zeros((len(i0s), HMAX + 1, 2)); traj[:, 0] = np.c_[lat, lon]

    def vel(t, la, lo, tau):
        if modo == 'persist':
            return v0[:, 0], v0[:, 1]
        uH, vH = H(t, la, lo); uH = np.nan_to_num(uH); vH = np.nan_to_num(vH)
        if modo == 'hycom':
            return uH, vH
        uw, vw = V(t, la, lo)
        if modo == 'vento':
            a = np.where(drog == 1, alfa[1], alfa[0])
            return uH + a * uw, vH + a * vw
        X = feats_ponto(t, la, lo, drog)
        X = np.c_[X, r0, np.full(len(la), tau)]
        r = modelo(X, seq)
        return uH + r[:, 0], vH + r[:, 1]

    for k in range(HMAX):
        t = th0 + k
        u1, v1 = vel(t, lat, lon, k)
        lam = lat + 0.5 * 3600 * v1 / R * 180 / np.pi; lom = lon + 0.5 * 3600 * u1 / (R * np.cos(np.deg2rad(lat))) * 180 / np.pi
        u2, v2 = vel(t + 0.5, lam, lom, k + 0.5)
        lat = lat + 3600 * v2 / R * 180 / np.pi; lon = lon + 3600 * u2 / (R * np.cos(np.deg2rad(lat))) * 180 / np.pi
        traj[:, k + 1] = np.c_[lat, lon]
    return traj


def dist(a, b):
    la1, lo1, la2, lo2 = map(np.deg2rad, (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    x = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x)) / 1e3


def obs_traj(i0s):
    return np.stack([g.loc[i0s + k, ['latitude', 'longitude']].values for k in range(HMAX + 1)], 1).astype(float)


def skill_lw(prev, obs, n=1.0, ate=HMAX):
    d = dist(prev[:, 1:ate + 1], obs[:, 1:ate + 1])
    passos = dist(obs[:, 1:ate + 1], obs[:, :ate])
    l = np.cumsum(passos, 1)
    c = d.sum(1) / np.maximum(l.sum(1), 1e-6)
    return np.where(c <= n, 1 - c / n, 0.0)


# ---------------- validação agrupada por derivador
ids_start = g.loc[starts, 'ID'].values
gkf = GroupKFold(10)
res = []
modelos = ['Persistence', f'{NOME} (OpenDrift-equivalent)', f'{NOME} + calibrated windage', 'Hybrid LightGBM',
           'Hybrid MLP', 'Hybrid Transformer', 'Hybrid LightGBM + history']
CSV = D + 'resultados/' + ('smoke_' if SMOKE else '') + f'resultados_deriva{TAG}.csv'
feitos = set()
if os.path.exists(CSV) and not SMOKE:
    res = pd.read_csv(CSV).to_dict('records'); feitos = {r['fold'] for r in res}
anos_start = g.loc[starts, 'time'].dt.year.values
grupos = anos_start if CV == 'year' else ids_start
for f, (tr_s, te_s) in enumerate(gkf.split(starts, groups=grupos)):
    if SMOKE and f > 0:
        break
    if f in feitos:
        continue
    t0 = time.time()
    if CV == 'year':   # teste: todos os inícios dos anos do fold; treino: derivadores sem nenhum dado nesses anos
        anos_te = np.unique(anos_start[te_s]); ids_te = g[g.time.dt.year.isin(anos_te)].ID.unique()
        ids_tr = np.setdiff1d(g.ID.unique(), ids_te)
    else:
        ids_te = np.unique(ids_start[te_s]); ids_tr = np.setdiff1d(g.ID.unique(), ids_te)
    gtr = g[g.ID.isin(ids_tr)]
    # janela de vento calibrada (mínimos quadrados) por estado do drogue
    alfa = {}
    for dg in (0, 1):
        s = gtr[gtr.drog == dg]
        W = np.r_[s.uw.values, s.vw.values]; Rr = np.r_[s.rU.values, s.rV.values]; okw = np.isfinite(W) & np.isfinite(Rr); W, Rr = W[okw], Rr[okw]
        alfa[dg] = float(W @ Rr / (W @ W))
    P = pares(ids_tr)
    Xtr, Ytr, i0tr = montar(P)
    ok = np.isfinite(Xtr).all(1) & np.isfinite(Ytr).all(1); Xtr, Ytr, i0tr = Xtr[ok], Ytr[ok], i0tr[ok]
    sub = rng.choice(len(Xtr), size=min(len(Xtr), 20000 if SMOKE else 250000), replace=False)
    mods = {}
    lgbs = [lgb.LGBMRegressor(n_estimators=50 if SMOKE else 600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(Xtr[sub], Ytr[sub, j]) for j in range(2)]
    mods['lgb'] = lambda X, S: np.c_[lgbs[0].predict(X), lgbs[1].predict(X)]
    sub2h = sub[:5000 if SMOKE else 120000]; Sh = resumo_hist(historico(i0tr[sub2h])); Xh = np.c_[Xtr[sub2h], Sh]
    okh = np.isfinite(Xh).all(1)
    lgbh = [lgb.LGBMRegressor(n_estimators=50 if SMOKE else 600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(Xh[okh], Ytr[sub2h][okh, j]) for j in range(2)]
    mods['lgbh'] = lambda X, S: (lambda Z: np.c_[lgbh[0].predict(Z), lgbh[1].predict(Z)])(np.c_[X, resumo_hist(S)])
    sub2 = sub[:5000 if SMOKE else 120000]
    Str = historico(i0tr[sub2])
    mods['mlp'] = treinar_torch(MLPRes, Xtr[sub2], Ytr[sub2], Str, epocas=1 if SMOKE else 12)
    mods['trans'] = treinar_torch(TransRes, Xtr[sub2], Ytr[sub2], Str, epocas=1 if SMOKE else 8)
    i0s = starts[te_s]
    obs = obs_traj(i0s)
    prevs = {'Persistence': integrar(i0s, 'persist'), modelos[1]: integrar(i0s, 'hycom'),
             modelos[2]: integrar(i0s, 'vento', alfa=alfa),
             'Hybrid LightGBM': integrar(i0s, 'lgb', mods['lgb']), 'Hybrid LightGBM + history': integrar(i0s, 'lgbh', mods['lgbh']), 'Hybrid MLP': integrar(i0s, 'mlp', mods['mlp']),
             'Hybrid Transformer': integrar(i0s, 'trans', mods['trans'])}
    for m, pv in prevs.items():
        d = dist(pv, obs)
        for j, i0 in enumerate(i0s):
            res.append(dict(fold=f, ID=g.loc[i0, 'ID'], i0=i0, drog=g.loc[i0, 'drog'], year=g.loc[i0, 'time'].year, model=m,
                            d24=d[j, 24], d48=d[j, 48], d72=d[j, 72],
                            ss24=skill_lw(pv[j:j + 1], obs[j:j + 1], ate=24)[0], ss72=skill_lw(pv[j:j + 1], obs[j:j + 1])[0]))
    np.save(D + f'resultados/res_traj{TAG}_fold{f}.npy', np.stack([prevs[m] for m in modelos] + [obs]))
    print(f'fold {f}: {len(i0s)} inícios, alfa={alfa}, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(res).to_csv(CSV, index=False)
r = pd.DataFrame(res)
print(r.groupby('model')[['d24', 'd48', 'd72', 'ss24', 'ss72']].mean().round(3))
print('FIM')
