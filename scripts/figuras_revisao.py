"""Figuras revisadas (17/10/2026): Fig. 1 com trajetórias quebradas nas lacunas; Fig. 2 com o baseline
"GLORYS12 + windage + r0 decay" e ordem única dos modelos; Figs. 3 e 4 com rótulos de eixo em graus;
Fig. 4 com ticks legíveis e barra de escala. Lê resultados/decaimento_T.csv (alfa e T por dobra) e re-integra o
baseline r0-decay (sem treino). Saída: figuras/Fig{1..4}_main.png (as versões anteriores ficam em figuras/*_v2.png)."""
import glob, numpy as np, pandas as pd, matplotlib, matplotlib.ticker, xarray as xr
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from forcantes import carrega_vento, carrega_glorys, Campo, horas, D
F = D + 'figuras/'; R = 6371.0; HMAX = 72
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
TAG = '_glorys_ncep_v2'
r = pd.read_csv(D + f'resultados/resultados_deriva{TAG}.csv')
fis = 'GLORYS12 (OpenDrift-equivalent)'; vento = 'GLORYS12 + calibrated windage'; R0 = 'GLORYS12 + windage + r0 decay'
mods_npy = ['Persistence', fis, vento, 'Hybrid LightGBM', 'Hybrid MLP', 'Hybrid Transformer', 'Hybrid LightGBM + history']
ORD = ['Persistence', fis, vento, R0, 'Hybrid LightGBM', 'Hybrid MLP', 'Hybrid LightGBM + history', 'Hybrid Transformer']
COR = {'Persistence': '#9e9e9e', fis: '#1b6ca8', vento: '#64a6d8', R0: '#2a9d8f', 'Hybrid LightGBM': '#e36414', 'Hybrid MLP': '#9a7b4f',
       'Hybrid LightGBM + history': '#7b2cbf', 'Hybrid Transformer': '#c1121f'}
LAB = {fis: 'GLORYS12', vento: 'GLORYS12 + windage', R0: 'GLORYS12 + windage + r$_0$ decay', 'Persistence': 'Persistence',
       'Hybrid LightGBM': 'Hybrid LightGBM', 'Hybrid MLP': 'Hybrid MLP', 'Hybrid Transformer': 'Hybrid Transformer', 'Hybrid LightGBM + history': 'Hybrid LightGBM + history'}
CURTO = {'Persistence': 'Persist.', fis: 'GLORYS12', vento: 'GLORYS12+W', R0: 'GLORYS12+W+r$_0$', 'Hybrid LightGBM': 'H-LightGBM', 'Hybrid MLP': 'H-MLP',
         'Hybrid LightGBM + history': 'H-LGBM+hist', 'Hybrid Transformer': 'H-Transformer'}

arr = [np.load(f) for f in sorted(glob.glob(D + f'resultados/res_traj{TAG}_fold*.npy'), key=lambda s: int(s.split('fold')[1][:-4]))]
T = np.concatenate(arr, 1); obs = T[-1]
meta = r[r.model == 'Persistence'].reset_index(drop=True)


def dist(a, b):
    la1, lo1, la2, lo2 = map(np.deg2rad, (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    x = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x))


Dm = {m: dist(T[i], obs) for i, m in enumerate(mods_npy)}
assert np.allclose(Dm['Persistence'][:, 72], meta.d72.values, atol=1e-6)

# ---------- re-integra o baseline r0-decay para a Fig. 2 (mesmas dobras; alfa e T de resultados/decaimento_T.csv)
g = pd.read_csv(D + 'gdp.csv', skiprows=[1], parse_dates=['time', 'drogue_lost_date'])
g['time'] = g.time.dt.tz_localize(None); g['drogue_lost_date'] = g.drogue_lost_date.dt.tz_localize(None)
g = g.sort_values(['ID', 'time']).reset_index(drop=True)
g['drog'] = ((g.drogue_lost_date.isna()) | (g.time < g.drogue_lost_date)).astype(int); g['th'] = horas(g.time)
H = Campo(carrega_glorys(), 'water_u', 'water_v'); V = Campo(carrega_vento(), 'uw', 'vw')
g['uH'], g['vH'] = H(g.th.values, g.latitude.values, g.longitude.values)
g = g[np.isfinite(g.uH) & ~((g.uH == 0) & (g.vH == 0))].copy()
g['uw'], g['vw'] = V(g.th.values, g.latitude.values, g.longitude.values)
g['rU'] = g.ve - g.uH; g['rV'] = g.vn - g.vH
g = g.reset_index(drop=True)
starts = meta.i0.values; ids_start = meta.ID.values
assert (g.loc[starts, 'ID'].values == ids_start).all()
Tdec = pd.read_csv(D + 'resultados/decaimento_T.csv')
traj0 = np.zeros((len(starts), HMAX + 1, 2))
for f, (tr_s, te_s) in enumerate(GroupKFold(10).split(starts, groups=ids_start)):
    row = Tdec[Tdec.fold == f].iloc[0]; alfa = {0: row.alfa0, 1: row.alfa1}; Tb = row.T_h
    i0s = starts[te_s]
    lat = g.loc[i0s, 'latitude'].values.astype(float); lon = g.loc[i0s, 'longitude'].values.astype(float)
    th0 = g.loc[i0s, 'th'].values; drog = g.loc[i0s, 'drog'].values.astype(float)
    a0 = np.where(drog == 1, alfa[1], alfa[0])
    r0w = g.loc[i0s, ['rU', 'rV']].values - a0[:, None] * g.loc[i0s, ['uw', 'vw']].values
    tr = np.zeros((len(i0s), HMAX + 1, 2)); tr[:, 0] = np.c_[lat, lon]

    def vel(t, la, lo, tau):
        uH, vH = H(t, la, lo); uH = np.nan_to_num(uH); vH = np.nan_to_num(vH); uw, vw = V(t, la, lo)
        fct = np.exp(-tau / Tb)
        return uH + a0 * uw + fct * r0w[:, 0], vH + a0 * vw + fct * r0w[:, 1]
    for k in range(HMAX):
        t = th0 + k
        u1, v1 = vel(t, lat, lon, k)
        lam = lat + 0.5 * 3600 * v1 / (R * 1e3) * 180 / np.pi; lom = lon + 0.5 * 3600 * u1 / (R * 1e3 * np.cos(np.deg2rad(lat))) * 180 / np.pi
        u2, v2 = vel(t + 0.5, lam, lom, k + 0.5)
        lat = lat + 3600 * v2 / (R * 1e3) * 180 / np.pi; lon = lon + 3600 * u2 / (R * 1e3 * np.cos(np.deg2rad(lat))) * 180 / np.pi
        tr[:, k + 1] = np.c_[lat, lon]
    traj0[te_s] = tr
Dm[R0] = dist(traj0, obs)
b = pd.read_csv(D + 'resultados/baselines_residuo.csv'); b0 = b[b.model == R0].set_index('i0').loc[starts]
print('check r0 decay d72:', np.nanmean(Dm[R0][:, 72]).round(2), b0.d72.mean().round(2))

# ---------- Fig 2
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
hh = np.arange(73)
for m in ORD:
    axs[0].plot(hh, np.nanmean(Dm[m], 0), color=COR[m], lw=1.6 if 'Transformer' in m else 1.1, ls='--' if m == R0 else '-', label=LAB[m])
axs[0].set_xlabel('Lead time (h)'); axs[0].set_ylabel('Mean separation (km)'); axs[0].set_xticks([0, 24, 48, 72])
axs[0].legend(frameon=False, fontsize=6.5)
pos = np.arange(len(ORD)); w = 0.38
ss = {m: r[r.model == m] for m in mods_npy}
ss[R0] = b[b.model == R0]
for k, (dg, nome, hatch) in enumerate([(0, 'undrogued', ''), (1, 'drogued', '//')]):
    v = [np.nanmean(ss[m][ss[m].drog == dg].ss72) for m in ORD]
    axs[1].bar(pos + (k - 0.5) * w, v, w, color=[COR[m] for m in ORD], hatch=hatch, edgecolor='white', label=nome)
axs[1].set_xticks(pos); axs[1].set_xticklabels([CURTO[m] for m in ORD], rotation=40, ha='right', fontsize=6.5)
axs[1].set_ylabel('Liu–Weisberg skill (72 h)'); axs[1].legend(frameon=False, fontsize=7)
for ax, l in zip(axs, 'ab'): ax.set_title(f'({l})', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig(F + 'Fig2_main.png', dpi=300); plt.close()

# ---------- Fig 1
gl = xr.open_dataset(D + 'glorys/glorys_2005_2021.nc').isel(depth=0)
sp = np.hypot(gl.uo, gl.vo).mean('time').load(); um = gl.uo.mean('time').load(); vm = gl.vo.mean('time').load()
gdp = pd.read_csv(D + 'gdp.csv', skiprows=[1], usecols=['ID', 'time', 'latitude', 'longitude'], parse_dates=['time']).sort_values(['ID', 'time'])
fig, ax = plt.subplots(figsize=(4.6, 4.6))
pc = ax.pcolormesh(sp.longitude, sp.latitude, sp, cmap='Blues', vmin=0, vmax=0.6, shading='auto')
terra = np.isnan(sp.values).astype(float)
ax.pcolormesh(sp.longitude, sp.latitude, np.ma.masked_where(terra < 0.5, terra), cmap=matplotlib.colors.ListedColormap(['#d9d4c7']), shading='auto')
s = 6; ax.quiver(um.longitude[::s], um.latitude[::s], um.values[::s, ::s], vm.values[::s, ::s], scale=6, width=0.003, color='#333')
used = set(meta.ID)
for i, gi in gdp.groupby('ID'):
    if i not in used:
        continue
    lo = gi.longitude.values.astype(float).copy(); la = gi.latitude.values.astype(float).copy()
    gap = np.r_[False, np.diff(gi.time.values).astype('timedelta64[m]').astype(float) > 61]   # quebra a linha nas lacunas (> 1 h)
    lo[gap] = np.nan; la[gap] = np.nan
    ax.plot(lo, la, lw=0.35, color='#c1121f', alpha=0.45)
ax.set_xlim(-43.5, -37.5); ax.set_ylim(-26, -19); ax.set_aspect(1 / np.cos(np.deg2rad(22.5)))
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
for nome, x, y in [('Cabo Frio', -42.0, -23.0), ('Cabo de São Tomé', -41.0, -22.0)]:
    ax.plot(x, y, 'k.', ms=3); ax.text(x - 0.1, y + 0.12, nome, fontsize=7, ha='right')
plt.colorbar(pc, ax=ax, shrink=0.7, label='Mean surface speed, GLORYS12 (m s$^{-1}$)')
fig.tight_layout(); fig.savefig(F + 'Fig1_main.png', dpi=300); plt.close()

# ---------- Fig 3
agg = pd.read_csv(D + f'resultados/mapa_ganho{TAG}.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax, col, cm, vmin, vmax, tit in [(axs[0], 'e', 'magma_r', 0, 80, '(a) GLORYS12 + windage: separation at 72 h (km)'),
                                     (axs[1], 'g', 'RdBu_r', -25, 25, '(b) Hybrid Transformer minus (a) (km)')]:
    ax.pcolormesh(sp.longitude, sp.latitude, np.ma.masked_where(terra < 0.5, terra), cmap=matplotlib.colors.ListedColormap(['#d9d4c7']), shading='auto')
    scx = ax.scatter(agg.lo + 0.25, agg.la + 0.25, c=agg[col], s=np.clip(agg.n, 10, 120) * 1.2, marker='s', cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xlim(-43.5, -37.5); ax.set_ylim(-26, -19); ax.set_aspect(1 / np.cos(np.deg2rad(22.5)))
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    for nome, x, y in [('CF', -42.0, -23.0), ('CST', -41.0, -22.0)]:
        ax.plot(x, y, 'k.', ms=3); ax.text(x - 0.1, y + 0.1, nome, fontsize=6, ha='right')
    ax.set_title(tit, loc='left', fontsize=8); plt.colorbar(scx, ax=ax, shrink=0.75)
fig.tight_layout(); fig.savefig(F + 'Fig3_main.png', dpi=300); plt.close()

# ---------- Fig 4
gan = Dm[vento][:, 72] - Dm['Hybrid Transformer'][:, 72]
ordem = np.argsort(gan); ex = [ordem[int(0.9 * len(ordem))], ordem[len(ordem) // 2], ordem[int(0.1 * len(ordem))]]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, j, t in zip(axs, ex, ['(a) 90th percentile', '(b) median', '(c) 10th percentile']):
    ax.plot(obs[j, :, 1], obs[j, :, 0], 'k-', lw=1.6, label='Observed')
    for m in [fis, vento, 'Hybrid Transformer']:
        ax.plot(T[mods_npy.index(m), j, :, 1], T[mods_npy.index(m), j, :, 0], color=COR[m], lw=1.1, label=LAB[m])
    ax.plot(obs[j, 0, 1], obs[j, 0, 0], 'ko', ms=3)
    ax.set_title(t, fontsize=7.5, loc='left'); ax.tick_params(labelsize=6.5)
    # janela quadrada (em km) comum aos painéis: limites = extensão das curvas, ampliada para um quadrado
    cs = np.cos(np.deg2rad(obs[j, 0, 0]))
    cur = [obs[j]] + [T[mods_npy.index(m), j] for m in [fis, vento, 'Hybrid Transformer']]
    la_all = np.concatenate([c[:, 0] for c in cur]); lo_all = np.concatenate([c[:, 1] for c in cur])
    cy, cx = 0.5 * (la_all.min() + la_all.max()), 0.5 * (lo_all.min() + lo_all.max())
    half = 0.55 * max(la_all.max() - la_all.min(), (lo_all.max() - lo_all.min()) * cs)
    ax.set_ylim(cy - half, cy + half); ax.set_xlim(cx - half / cs, cx + half / cs); ax.set_aspect(1 / cs)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, prune='both')); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune='both'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f')); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel('Longitude (°E)', fontsize=7); ax.set_ylabel('Latitude (°N)', fontsize=7)
    # barra de escala de 10 km
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    dlon = 10 / (111.0 * np.cos(np.deg2rad(obs[j, 0, 0])))
    xs = x0 + 0.06 * (x1 - x0); ys = y0 + 0.05 * (y1 - y0)
    ax.plot([xs, xs + dlon], [ys, ys], 'k-', lw=1.5); ax.text(xs + dlon / 2, ys + 0.02 * (y1 - y0), '10 km', fontsize=6, ha='center', va='bottom')
    print(t, 'start', obs[j, 0], 'end', obs[j, 72], 'gain', gan[j].round(1))
axs[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(F + 'Fig4_main.png', dpi=300); plt.close()
print('ok')
