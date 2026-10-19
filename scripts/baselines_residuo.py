"""Análises adicionais pedidas pelo revisor (17/10/2026), nas mesmas 3.834 previsões e dobras de deriva.py (GLORYS12 + NCEP):
  (a) baseline "corrente + windage + resíduo inicial persistido" (r0), com e sem decaimento exponencial (T ajustado no treino)
      e com a média do resíduo das últimas 24 h persistida;
  (b) ridge linear sobre as mesmas entradas, sem e com o resumo do histórico;
  (c) LightGBM sem histórico treinado com os mesmos 120.000 pares das redes (controle de tamanho de amostra);
  (d) importância por permutação (grupos de variáveis) do LightGBM + histórico, nos pares de teste;
  (e) estatística do resíduo observado e do resíduo previsto (r̂) ao longo da previsão, a partir das trajetórias salvas.
Uso: CV=drifter|year python3 baselines_residuo.py  -> resultados/baselines_residuo_<cv>.csv (+ _T.csv, _importancia.csv)"""
import os, sys, time, glob, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from forcantes import carrega_vento, carrega_glorys, Campo, horas, D

rng = np.random.default_rng(0)
R = 6371e3; HMAX = 72
CV = os.environ.get('CV', 'drifter')
OUT = D + 'resultados/'; os.makedirs(OUT, exist_ok=True)
SUF = '_cvano' if CV == 'year' else ''
RODAR_ML = os.environ.get('ML', '1') == '1'

g = pd.read_csv(D + 'gdp.csv', skiprows=[1], parse_dates=['time', 'drogue_lost_date'])
g['time'] = g.time.dt.tz_localize(None); g['drogue_lost_date'] = g.drogue_lost_date.dt.tz_localize(None)
g = g.sort_values(['ID', 'time']).reset_index(drop=True)
g['drog'] = ((g.drogue_lost_date.isna()) | (g.time < g.drogue_lost_date)).astype(int)
g['th'] = horas(g.time)
from scipy.interpolate import RegularGridInterpolator


class Campo32(Campo):
    """Igual a forcantes.Campo, mas em float32 e sem cópias (memória compartilhada é limitada)."""
    def __init__(self, ds, un, vn, lat='lat', lon='lon'):
        t = (pd.DatetimeIndex(ds.time.values) - pd.Timestamp('2005-01-01')) / pd.Timedelta(hours=1)
        self.t = np.asarray(t, float); la = ds[lat].values.astype(float); lo = ds[lon].values.astype(float)
        U = ds[un].values.astype(np.float32); np.nan_to_num(U, copy=False)
        V_ = ds[vn].values.astype(np.float32); np.nan_to_num(V_, copy=False)
        self.fu = RegularGridInterpolator((self.t, la, lo), U, bounds_error=False, fill_value=np.nan)
        self.fv = RegularGridInterpolator((self.t, la, lo), V_, bounds_error=False, fill_value=np.nan)
        self.dlat = la[1] - la[0]; self.dlon = lo[1] - lo[0]


import gc
def carrega_glorys32():
    """Mesmo resultado de forcantes.carrega_glorys (grade arredondada, carimbo +12 h, tempos duplicados removidos), em float32
    e arquivo por arquivo, para caber na memória compartilhada."""
    import xarray as xr
    ts, us, vs, la, lo = [], [], [], None, None
    for f in sorted(glob.glob(D + 'glorys/glorys_*.nc')):
        d = xr.open_dataset(f).isel(depth=0)
        lat_ = np.round(d.latitude.values.astype(float), 4); lon_ = np.round(d.longitude.values.astype(float) % 360, 4)
        o = np.argsort(lon_)
        if la is None:
            la, lo = lat_, lon_[o]
        assert np.array_equal(la, lat_) and np.array_equal(lo, lon_[o])
        ts.append(d.time.values + np.timedelta64(12, 'h'))
        us.append(d.uo.values.astype(np.float32)[:, :, o]); vs.append(d.vo.values.astype(np.float32)[:, :, o]); d.close(); gc.collect()
    t = np.concatenate(ts); U = np.concatenate(us); del us; gc.collect(); Vv = np.concatenate(vs); del vs; gc.collect()
    k = np.argsort(t, kind='stable'); t, U, Vv = t[k], U[k], Vv[k]
    keep = ~pd.Index(t).duplicated(); t, U, Vv = t[keep], U[keep], Vv[keep]
    return xr.Dataset(dict(water_u=(('time', 'lat', 'lon'), U), water_v=(('time', 'lat', 'lon'), Vv)), coords=dict(time=t, lat=la, lon=lo))


_h = carrega_glorys32(); H = Campo32(_h, 'water_u', 'water_v'); del _h; gc.collect()
_w = carrega_vento(); V = Campo32(_w, 'uw', 'vw'); del _w; gc.collect()
g['uH'], g['vH'] = H(g.th.values, g.latitude.values, g.longitude.values)
g = g[np.isfinite(g.uH) & ~((g.uH == 0) & (g.vH == 0))].copy()
g['uw'], g['vw'] = V(g.th.values, g.latitude.values, g.longitude.values)
g['vort'], g['div'], g['strain'] = H.derivadas(g.th.values, g.latitude.values, g.longitude.values)
g['rU'] = g.ve - g.uH; g['rV'] = g.vn - g.vH
nov = (g.ID != g.ID.shift()) | (g.th.diff() > 1.01)
g['seg'] = nov.cumsum(); g = g.reset_index(drop=True)
print('obs com forçantes:', len(g), 'derivadores:', g.ID.nunique(), flush=True)

SKIP = os.environ.get('SKIP_STATS') == '1'
# ---------- (e1) estatística do resíduo observado
def estatisticas():
    spd_d = np.hypot(g.ve, g.vn); spd_m = np.hypot(g.uH, g.vH); rmag = np.hypot(g.rU, g.rV)
    est = dict(n=len(g), vel_media_derivador=spd_d.mean(), vel_media_modelo=spd_m.mean(),
               res_abs_media=rmag.mean(), res_abs_mediana=np.median(rmag), res_rms=np.sqrt((rmag ** 2).mean()), res_p90=np.percentile(rmag, 90),
               res_rms_u=np.sqrt((g.rU ** 2).mean()), res_rms_v=np.sqrt((g.rV ** 2).mean()),
               var_explicada_u=1 - g.rU.var() / g.ve.var(), var_explicada_v=1 - g.rV.var() / g.vn.var(),
               corr_u=np.corrcoef(g.ve, g.uH)[0, 1], corr_v=np.corrcoef(g.vn, g.vH)[0, 1])
    # fração da variância do resíduo em períodos < 48 h: média móvel de 48 h por segmento (passa-baixa) vs. resto
    lp = g.groupby('seg')[['rU', 'rV']].transform(lambda s: s.rolling(48, center=True, min_periods=24).mean())
    hp = g[['rU', 'rV']].values - lp.values; okl = np.isfinite(hp).all(1)
    est['frac_var_res_periodo_lt48h'] = (hp[okl] ** 2).sum() / ((g[['rU', 'rV']].values[okl] - g[['rU', 'rV']].values[okl].mean(0)) ** 2).sum()
    est['res_rms_lowpass48h'] = np.sqrt(np.nanmean(lp.values[okl] ** 2))
    # autocorrelação do resíduo (lags 3, 6, 12, 24, 48, 72 h) dentro de segmentos
    for lag in (3, 6, 12, 24, 48, 72):
        a = g.groupby('seg')[['rU', 'rV']].shift(-lag); ok = a.rU.notna()
        est[f'autocorr_res_{lag}h'] = 0.5 * (np.corrcoef(g.rU[ok], a.rU[ok])[0, 1] + np.corrcoef(g.rV[ok], a.rV[ok])[0, 1])
    pd.Series(est).to_csv(OUT + 'estatistica_residuo_observado.csv'); print(pd.Series(est).round(3))
if not SKIP:
    estatisticas()

# ---------- inícios (idênticos a deriva.py)
starts = []
for s, gs in g.groupby('seg'):
    if len(gs) < HMAX + 25:
        continue
    idx = gs.index.values
    for k in range(24, len(idx) - HMAX, 24):
        starts.append(idx[k])
starts = np.array(starts); print('inícios:', len(starts), flush=True)
# perda de âncora dentro das 72 h seguintes ao início
t0s = g.loc[starts, 'time']; dl = g.loc[starts, 'drogue_lost_date']
perde = ((g.loc[starts, 'drog'] == 1) & dl.notna() & (dl.values > t0s.values) & (dl.values <= (t0s + pd.Timedelta(hours=72)).values)).sum()
print('inícios com perda de âncora nas 72 h:', int(perde))
# segmentos e derivadores com inícios
print('segmentos com inícios:', g.loc[starts, 'seg'].nunique(), 'derivadores:', g.loc[starts, 'ID'].nunique())
pd.Series(dict(n_starts=len(starts), perda_ancora_72h=int(perde), segmentos=g.loc[starts, 'seg'].nunique(),
               derivadores=g.loc[starts, 'ID'].nunique())).to_csv(OUT + 'contagens_inicios.csv')


def feats_ponto(th, lat, lon, drog):
    uH, vH = H(th, lat, lon); uw, vw = V(th, lat, lon)
    vo, dv, st = H.derivadas(th, lat, lon)
    doy = (pd.Timestamp('2005-01-01') + pd.to_timedelta(th, 'h')).dayofyear.values
    return np.c_[uH, vH, np.hypot(uH, vH), uw, vw, np.hypot(uw, vw), vo * 1e5, dv * 1e5, st * 1e5,
                 lat, lon, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), drog]


def pares(ids_ok, passo0=6, passotau=3):
    linhas = []
    for s, gs in g[g.ID.isin(ids_ok)].groupby('seg'):
        if len(gs) < 25:
            continue
        n = len(gs)
        for k in range(24, n - 1, passo0):
            for tau in range(0, min(HMAX, n - 1 - k) + 1, passotau):
                linhas.append((gs.index[k], gs.index[k + tau], tau))
    return np.array(linhas)


def montar(P):
    i0, it, tau = P[:, 0], P[:, 1], P[:, 2]
    Fp = np.c_[g.loc[it, ['uH', 'vH']].values, np.hypot(*g.loc[it, ['uH', 'vH']].values.T),
               g.loc[it, ['uw', 'vw']].values, np.hypot(*g.loc[it, ['uw', 'vw']].values.T),
               g.loc[it, ['vort', 'div', 'strain']].values * 1e5, g.loc[it, ['latitude', 'longitude']].values]
    doy = g.loc[it, 'time'].dt.dayofyear.values
    Fp = np.c_[Fp, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), g.loc[it, 'drog'].values]
    r0 = g.loc[i0, ['rU', 'rV']].values
    return np.c_[Fp, r0, tau], g.loc[it, ['rU', 'rV']].values, i0


def resumo_hist(S):
    r = S[:, :, :2]
    return np.c_[r[:, -3:].mean(1), r[:, -6:].mean(1), r[:, -12:].mean(1), r.mean(1), r[:, -13], r[:, 0]]


def historico(i0):
    S = np.stack([g.loc[i0 - 23 + j, ['rU', 'rV', 'uH', 'vH', 'uw', 'vw']].values for j in range(24)], 1)
    return S.astype(np.float32)


def integrar(i0s, modo, modelo=None, alfa=None, T=None):
    lat = g.loc[i0s, 'latitude'].values.astype(float); lon = g.loc[i0s, 'longitude'].values.astype(float)
    th0 = g.loc[i0s, 'th'].values; drog = g.loc[i0s, 'drog'].values.astype(float)
    r0 = g.loc[i0s, ['rU', 'rV']].values
    seq = historico(i0s)
    # resíduo após o windage (evita contar o vento duas vezes): r_w = r - alpha_d * u_wind
    a0 = np.where(drog == 1, alfa[1], alfa[0])[:, None]
    r0w = r0 - a0 * g.loc[i0s, ['uw', 'vw']].values
    rbar = (seq[:, :, :2] - a0[:, None, :] * seq[:, :, 4:6]).mean(1)
    traj = np.zeros((len(i0s), HMAX + 1, 2)); traj[:, 0] = np.c_[lat, lon]
    rhat = np.zeros((len(i0s), HMAX))

    def vel(t, la, lo, tau):
        uH, vH = H(t, la, lo); uH = np.nan_to_num(uH); vH = np.nan_to_num(vH)
        uw, vw = V(t, la, lo); a = np.where(drog == 1, alfa[1], alfa[0])
        if modo == 'r0':
            return uH + a * uw + r0w[:, 0], vH + a * vw + r0w[:, 1]
        if modo == 'r0decay':
            f = np.exp(-tau / T)
            return uH + a * uw + f * r0w[:, 0], vH + a * vw + f * r0w[:, 1]
        if modo == 'rbar':
            return uH + a * uw + rbar[:, 0], vH + a * vw + rbar[:, 1]
        X = feats_ponto(t, la, lo, drog); X = np.c_[X, r0, np.full(len(la), tau)]
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
    d = dist(prev[:, 1:ate + 1], obs[:, 1:ate + 1]); passos = dist(obs[:, 1:ate + 1], obs[:, :ate])
    c = d.sum(1) / np.maximum(np.cumsum(passos, 1).sum(1), 1e-6)
    return np.where(c <= n, 1 - c / n, 0.0)


# ---------- (e2) |r̂| previsto ao longo de τ, a partir das trajetórias salvas da rodada principal
TAG = '_glorys_ncep_v2' + SUF
arr = [np.load(f) for f in sorted(glob.glob(D + f'resultados/res_traj{TAG}_fold*.npy'), key=lambda s: int(s.split('fold')[1][:-4]))]
if arr and not SKIP:
    Tj = np.concatenate(arr, 1)   # (7+1, n, 73, 2) na ordem de deriva.py: Persistence, GLORYS, +windage, LGBM, MLP, Trans, LGBM+hist, obs
    nomes = ['Persistence', 'GLORYS12', 'GLORYS12 + windage', 'Hybrid LightGBM', 'Hybrid MLP', 'Hybrid Transformer', 'Hybrid LightGBM + history']
    res_csv = pd.read_csv(D + f'resultados/resultados_deriva{TAG}.csv'); meta = res_csv[res_csv.model == 'Persistence'].reset_index(drop=True)
    i0s = meta.i0.values; th0 = g.loc[i0s, 'th'].values
    linhas = []
    for mi, nome in enumerate(nomes):
        if 'Hybrid' not in nome:
            continue
        P = Tj[mi]  # (n, 73, 2)
        # velocidade da trajetória por diferença centrada entre k e k+1 (posição média), menos corrente modelada no ponto médio
        rh = []
        for k in range(HMAX):
            la = 0.5 * (P[:, k, 0] + P[:, k + 1, 0]); lo = 0.5 * (P[:, k, 1] + P[:, k + 1, 1])
            v = (P[:, k + 1, 0] - P[:, k, 0]) * np.pi / 180 * R / 3600
            u = (P[:, k + 1, 1] - P[:, k, 1]) * np.pi / 180 * R * np.cos(np.deg2rad(la)) / 3600
            uH, vH = H(th0 + k + 0.5, la, lo)
            rh.append(np.hypot(u - uH, v - vH))
        rh = np.array(rh).T   # (n, 72)
        for k in (0, 6, 12, 24, 48, 71):
            linhas.append(dict(model=nome, tau_h=k + 1, rhat_abs_mean=np.nanmean(rh[:, k]), rhat_abs_median=np.nanmedian(rh[:, k]),
                               rhat_rms=np.sqrt(np.nanmean(rh[:, k] ** 2))))
    r0mag = np.hypot(*g.loc[i0s, ['rU', 'rV']].values.T)
    linhas.append(dict(model='observed r0 at start', tau_h=0, rhat_abs_mean=r0mag.mean(), rhat_abs_median=np.median(r0mag), rhat_rms=np.sqrt((r0mag ** 2).mean())))
    pd.DataFrame(linhas).to_csv(OUT + f'rhat_previsto_por_tau{SUF}.csv', index=False); print(pd.DataFrame(linhas).round(3))

# ---------- validação (mesmas dobras de deriva.py)
ids_start = g.loc[starts, 'ID'].values; anos_start = g.loc[starts, 'time'].dt.year.values
grupos = anos_start if CV == 'year' else ids_start
gkf = GroupKFold(10)
CSV = OUT + f'baselines_residuo{SUF}.csv'
res, Ts, imps, comp = [], [], [], []
import pickle
for f, (tr_s, te_s) in enumerate(gkf.split(starts, groups=grupos)):
    t0 = time.time()
    PK = OUT + f'_parcial{SUF}_fold{f}.pkl'
    if os.path.exists(PK):   # retomada após interrupção (memória compartilhada)
        pr = pickle.load(open(PK, 'rb')); res += pr['res']; Ts += pr['Ts']; imps += pr['imps']; comp += pr['comp']
        print(f'fold {f}: retomado', flush=True); continue
    n_res, n_Ts, n_imps, n_comp = len(res), len(Ts), len(imps), len(comp)
    if CV == 'year':
        anos_te = np.unique(anos_start[te_s]); ids_te = g[g.time.dt.year.isin(anos_te)].ID.unique()
        ids_tr = np.setdiff1d(g.ID.unique(), ids_te)
        comp.append(dict(fold=f, anos=' '.join(map(str, anos_te)), n_test=len(te_s), n_drifters_test=len(np.unique(ids_start[te_s])), n_drifters_train=len(ids_tr)))
    else:
        ids_te = np.unique(ids_start[te_s]); ids_tr = np.setdiff1d(g.ID.unique(), ids_te)
    gtr = g[g.ID.isin(ids_tr)]
    alfa = {}
    for dg in (0, 1):
        s = gtr[gtr.drog == dg]
        W = np.r_[s.uw.values, s.vw.values]; Rr = np.r_[s.rU.values, s.rV.values]; okw = np.isfinite(W) & np.isfinite(Rr)
        alfa[dg] = float(W[okw] @ Rr[okw] / (W[okw] @ W[okw]))
    P = pares(ids_tr); Xtr, Ytr, i0tr = montar(P)
    ok = np.isfinite(Xtr).all(1) & np.isfinite(Ytr).all(1); Xtr, Ytr, i0tr = Xtr[ok], Ytr[ok], i0tr[ok]
    sub = rng.choice(len(Xtr), size=min(len(Xtr), 250000), replace=False)   # mesma sequência de deriva.py
    # T do decaimento: mínimos quadrados de r_tau ~ r0 exp(-tau/T), com windage removido do alvo e de r0
    a_tr = np.where(Xtr[:, 13] == 1, alfa[1], alfa[0])[:, None]
    Y_w = Ytr - a_tr * Xtr[:, 3:5]; r0_w = Xtr[:, 14:16] - a_tr * g.loc[i0tr, ['uw', 'vw']].values
    tau = Xtr[:, 16]
    grade = np.r_[np.arange(3, 97, 3), np.inf]
    sse = [((Y_w - np.exp(-tau / T)[:, None] * r0_w) ** 2).sum() for T in grade]
    Tbest = float(grade[int(np.argmin(sse))])
    sse0 = ((Y_w) ** 2).sum()
    Ts.append(dict(fold=f, T_h=Tbest, sse_persist=sse[-1] / sse0, sse_decay=min(sse) / sse0, alfa0=alfa[0], alfa1=alfa[1]))
    i0s = starts[te_s]; obs = obs_traj(i0s)
    prevs = {'GLORYS12 + windage + r0': integrar(i0s, 'r0', alfa=alfa),
             'GLORYS12 + windage + r0 decay': integrar(i0s, 'r0decay', alfa=alfa, T=Tbest),
             'GLORYS12 + windage + mean 24-h residual': integrar(i0s, 'rbar', alfa=alfa)}
    if RODAR_ML:
        mu, sd = Xtr[sub].mean(0), Xtr[sub].std(0) + 1e-9
        rg = [Ridge(alpha=1.0).fit((Xtr[sub] - mu) / sd, Ytr[sub, j]) for j in range(2)]
        prevs['Hybrid ridge'] = integrar(i0s, 'ridge', lambda X, S: np.c_[rg[0].predict((X - mu) / sd), rg[1].predict((X - mu) / sd)], alfa=alfa)
        sub2 = sub[:120000]; Sh = resumo_hist(historico(i0tr[sub2])); Xh = np.c_[Xtr[sub2], Sh]; okh = np.isfinite(Xh).all(1)
        muh, sdh = Xh[okh].mean(0), Xh[okh].std(0) + 1e-9
        rgh = [Ridge(alpha=1.0).fit((Xh[okh] - muh) / sdh, Ytr[sub2][okh, j]) for j in range(2)]
        prevs['Hybrid ridge + history'] = integrar(i0s, 'ridgeh', lambda X, S: (lambda Z: np.c_[rgh[0].predict(Z), rgh[1].predict(Z)])((np.c_[X, resumo_hist(S)] - muh) / sdh), alfa=alfa)
        lg = [lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, verbose=-1).fit(Xtr[sub2], Ytr[sub2, j]) for j in range(2)]
        prevs['Hybrid LightGBM (120k pairs)'] = integrar(i0s, 'lgb', lambda X, S: np.c_[lg[0].predict(X), lg[1].predict(X)], alfa=alfa)
        # (d) importância por permutação do LightGBM + histórico (retreinado como em deriva.py), nos pares de teste
        lgh = [lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, verbose=-1).fit(Xh[okh], Ytr[sub2][okh, j]) for j in range(2)]
        Pte = pares(ids_te); Xte, Yte, i0te = montar(Pte); okt = np.isfinite(Xte).all(1) & np.isfinite(Yte).all(1); Xte, Yte, i0te = Xte[okt], Yte[okt], i0te[okt]
        if len(Xte) > 40000:
            j = rng.choice(len(Xte), 40000, replace=False); Xte, Yte, i0te = Xte[j], Yte[j], i0te[j]
        Xht = np.c_[Xte, resumo_hist(historico(i0te))]; okt2 = np.isfinite(Xht).all(1); Xht, Yte = Xht[okt2], Yte[okt2]
        def mse(Z): return np.mean((np.c_[lgh[0].predict(Z), lgh[1].predict(Z)] - Yte) ** 2)
        base = mse(Xht)
        grupos_var = {'model current (u,v,speed)': [0, 1, 2], 'wind (u,v,speed)': [3, 4, 5], 'vorticity/divergence/strain': [6, 7, 8],
                      'latitude/longitude': [9, 10], 'day of year': [11, 12], 'drogue': [13], 'r0': [14, 15], 'lead time': [16],
                      'history mean 3 h': [17, 18], 'history mean 6 h': [19, 20], 'history mean 12 h': [21, 22], 'history mean 24 h': [23, 24],
                      'history t0-12 h': [25, 26], 'history t0-23 h': [27, 28], 'all history (12 inputs)': list(range(17, 29))}
        for nome, cols in grupos_var.items():
            Zp = Xht.copy(); pi = rng.permutation(len(Zp)); Zp[:, cols] = Zp[pi][:, cols]
            imps.append(dict(fold=f, group=nome, mse_base=base, mse_perm=mse(Zp), increase=mse(Zp) / base - 1))
        # importância por faixa de lead time (τ ≤ 24 h vs > 24 h) para "all history" e "r0"
        for nome, cols in [('all history (12 inputs)', list(range(17, 29))), ('r0', [14, 15])]:
            for lab, sel in [('tau<=24', Xht[:, 16] <= 24), ('tau>24', Xht[:, 16] > 24)]:
                Zp = Xht.copy(); pi = rng.permutation(len(Zp)); Zp[:, cols] = Zp[pi][:, cols]
                b = np.mean((np.c_[lgh[0].predict(Xht[sel]), lgh[1].predict(Xht[sel])] - Yte[sel]) ** 2)
                p = np.mean((np.c_[lgh[0].predict(Zp[sel]), lgh[1].predict(Zp[sel])] - Yte[sel]) ** 2)
                imps.append(dict(fold=f, group=f'{nome} [{lab}]', mse_base=b, mse_perm=p, increase=p / b - 1))
    for m, pv in prevs.items():
        d = dist(pv, obs)
        for j, i0 in enumerate(i0s):
            res.append(dict(fold=f, ID=g.loc[i0, 'ID'], i0=i0, drog=g.loc[i0, 'drog'], year=g.loc[i0, 'time'].year, model=m,
                            d24=d[j, 24], d48=d[j, 48], d72=d[j, 72],
                            ss24=skill_lw(pv[j:j + 1], obs[j:j + 1], ate=24)[0], ss72=skill_lw(pv[j:j + 1], obs[j:j + 1])[0]))
    pd.DataFrame(res).to_csv(CSV, index=False); pd.DataFrame(Ts).to_csv(OUT + f'decaimento_T{SUF}.csv', index=False)
    if imps:
        pd.DataFrame(imps).to_csv(OUT + f'importancia_permutacao_lgbm_hist{SUF}.csv', index=False)
    if comp:
        pd.DataFrame(comp).to_csv(OUT + 'composicao_dobras_ano.csv', index=False)
    pickle.dump(dict(res=res[n_res:], Ts=Ts[n_Ts:], imps=imps[n_imps:], comp=comp[n_comp:]), open(PK, 'wb'))
    print(f'fold {f}: {len(i0s)} inícios, T={Tbest} h, {time.time() - t0:.0f}s', flush=True)
r = pd.DataFrame(res)
print(r.groupby('model')[['d24', 'd48', 'd72', 'ss24', 'ss72']].mean().round(3))
print('FIM')
