"""Análise da rodada principal (TAG): bootstrap por derivador, curvas de separação, skill por estado da âncora,
mapa e exemplos. Uso: python3 analise.py _glorys_ncep"""
import sys, glob, numpy as np, pandas as pd, matplotlib, matplotlib.ticker, xarray as xr
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')
TAG = sys.argv[1] if len(sys.argv) > 1 else '_glorys_ncep'
F = D + 'figuras/'; R = 6371.0
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
r = pd.read_csv(D + f'resultados/resultados_deriva{TAG}.csv')
mods = list(r.model.unique()); fis = [m for m in mods if m.endswith('(OpenDrift-equivalent)')][0]; vento = [m for m in mods if 'windage' in m][0]
ORD = ['Persistence', fis, vento, 'Hybrid LightGBM', 'Hybrid MLP', 'Hybrid Transformer'] + (['Hybrid LightGBM + history'] if 'Hybrid LightGBM + history' in mods else [])
COR = {'Hybrid LightGBM + history': '#7b2cbf', 'Persistence': '#9e9e9e', fis: '#1b6ca8', vento: '#64a6d8', 'Hybrid LightGBM': '#e36414', 'Hybrid MLP': '#9a7b4f', 'Hybrid Transformer': '#c1121f'}
LAB = {fis: fis.replace(' (OpenDrift-equivalent)', ''), vento: vento.replace('calibrated ', ''), 'Persistence': 'Persistence',
       'Hybrid LightGBM': 'Hybrid LightGBM', 'Hybrid MLP': 'Hybrid MLP', 'Hybrid Transformer': 'Hybrid Transformer', 'Hybrid LightGBM + history': 'Hybrid LightGBM + history'}

# ---------- trajetórias completas
arr = [np.load(f) for f in sorted(glob.glob(D + f'resultados/res_traj{TAG}_fold*.npy'), key=lambda s: int(s.split('fold')[1][:-4]))]
T = np.concatenate(arr, 1)                 # (7, n, 73, 2): 6 modelos + obs, na ordem do script
mods_npy = ['Persistence', fis, vento, 'Hybrid LightGBM', 'Hybrid MLP', 'Hybrid Transformer'] + (['Hybrid LightGBM + history'] if 'Hybrid LightGBM + history' in mods else [])
meta = r[r.model == 'Persistence'].reset_index(drop=True)
def dist(a, b):
    la1, lo1, la2, lo2 = map(np.deg2rad, (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    x = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x))
obs = T[-1]; Dm = {m: dist(T[i], obs) for i, m in enumerate(mods_npy)}
assert np.allclose(Dm['Persistence'][:, 72], meta.d72.values, atol=1e-6)

# ---------- tabela com IC por derivador
rng = np.random.default_rng(0); ids = meta.ID.values; uid = np.unique(ids)
grp = {u: np.where(ids == u)[0] for u in uid}
B = 2000; boot = {m: {h: [] for h in (24, 48, 72)} for m in ORD}; dif = {m: [] for m in ORD}; dsk = {m: [] for m in ORD}
sk = {m: r[r.model == m].ss72.values for m in ORD}
for b in range(B):
    idx = np.concatenate([grp[u] for u in rng.choice(uid, len(uid))])
    for m in ORD:
        for h in (24, 48, 72):
            boot[m][h].append(np.nanmean(Dm[m][idx, h]))
        dif[m].append(np.nanmean(Dm[m][idx, 72] - Dm[vento][idx, 72]))
        dsk[m].append(np.nanmean(sk[m][idx]))
lin = []
for m in ORD:
    q = lambda x: np.percentile(x, [2.5, 97.5])
    row = dict(model=LAB[m])
    for h in (24, 48, 72):
        lo, hi = q(boot[m][h]); row[f'd{h}'] = f"{np.nanmean(Dm[m][:, h]):.1f} [{lo:.1f}, {hi:.1f}]"
    row['d72_median'] = f"{np.nanmedian(Dm[m][:, 72]):.1f}"; row['n_nan'] = int(np.isnan(Dm[m][:, 72]).sum())
    lo, hi = q(dsk[m]); row['ss72'] = f"{np.nanmean(sk[m]):.2f} [{lo:.2f}, {hi:.2f}]"
    lo, hi = q(dif[m]); row['diff72_vs_wind'] = f"{np.mean(dif[m]):+.1f} [{lo:+.1f}, {hi:+.1f}]" if m != vento else '–'
    row['d72_undrogued'] = f"{np.nanmean(Dm[m][meta.drog.values == 0, 72]):.1f}"; row['d72_drogued'] = f"{np.nanmean(Dm[m][meta.drog.values == 1, 72]):.1f}"
    row['frac_better_than_wind'] = f"{(Dm[m][:, 72] < Dm[vento][:, 72]).mean():.2f}"
    lin.append(row)
tab = pd.DataFrame(lin); tab.to_csv(D + f'resultados/tabela_principal{TAG}.csv', index=False); print(tab.to_string())

# ---------- Fig 2: separação × horizonte
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
hh = np.arange(73)
for m in ORD:
    axs[0].plot(hh, np.nanmean(Dm[m], 0), color=COR[m], lw=1.6 if 'Transformer' in m else 1.1, label=LAB[m])
axs[0].set_xlabel('Lead time (h)'); axs[0].set_ylabel('Mean separation (km)'); axs[0].set_xticks([0, 24, 48, 72])
axs[0].legend(frameon=False, fontsize=7)
# painel b: skill de Liu-Weisberg em 72 h por estado da âncora
pos = np.arange(len(ORD)); w = 0.38
for k, (dg, nome, hatch) in enumerate([(0, 'undrogued', ''), (1, 'drogued', '//')]):
    v = [np.nanmean(r[(r.model == m) & (r.drog == dg)].ss72) for m in ORD]
    axs[1].bar(pos + (k - 0.5) * w, v, w, color=[COR[m] for m in ORD], hatch=hatch, edgecolor='white', label=nome)
axs[1].set_xticks(pos); axs[1].set_xticklabels([LAB[m].replace('Hybrid ', 'H-').replace(' + windage', '+W').replace('LightGBM + history', 'LGBM+hist') for m in ORD], rotation=35, ha='right', fontsize=7)
axs[1].set_ylabel('Liu–Weisberg skill (72 h)'); axs[1].legend(frameon=False, fontsize=7)
for ax, l in zip(axs, 'ab'): ax.set_title(f'({l})', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig(F + f'Fig2_separation{TAG}.png', dpi=300); plt.close()

# ---------- Fig 1: domínio, trajetórias e corrente média
gl = xr.open_dataset(D + 'glorys/glorys_2005_2021.nc').isel(depth=0)
sp = np.hypot(gl.uo, gl.vo).mean('time').load(); um = gl.uo.mean('time').load(); vm = gl.vo.mean('time').load()
gdp = pd.read_csv(D + 'gdp.csv', skiprows=[1], usecols=['ID', 'latitude', 'longitude'])
fig, ax = plt.subplots(figsize=(4.6, 4.6))
pc = ax.pcolormesh(sp.longitude, sp.latitude, sp, cmap='Blues', vmin=0, vmax=0.6, shading='auto')
terra = np.isnan(sp.values).astype(float)
ax.pcolormesh(sp.longitude, sp.latitude, np.ma.masked_where(terra < 0.5, terra), cmap=matplotlib.colors.ListedColormap(['#d9d4c7']), shading='auto')
s = 6; ax.quiver(um.longitude[::s], um.latitude[::s], um.values[::s, ::s], vm.values[::s, ::s], scale=6, width=0.003, color='#333')
used = set(meta.ID)
for i, gi in gdp.groupby('ID'):
    if i in used: ax.plot(gi.longitude, gi.latitude, lw=0.35, color='#c1121f', alpha=0.45)
ax.set_xlim(-43.5, -37.5); ax.set_ylim(-26, -19); ax.set_aspect(1 / np.cos(np.deg2rad(22.5)))
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
for nome, x, y in [('Cabo Frio', -42.0, -23.0), ('Cabo de São Tomé', -41.0, -22.0)]:
    ax.plot(x, y, 'k.', ms=3); ax.text(x - 0.1, y + 0.12, nome, fontsize=7, ha='right')
plt.colorbar(pc, ax=ax, shrink=0.7, label='Mean surface speed, GLORYS12 (m s$^{-1}$)')
fig.tight_layout(); fig.savefig(F + 'Fig1_domain.png', dpi=300); plt.close()

# ---------- Fig 3: mapa do ganho do Transformer sobre o físico+vento, por caixa de 0,5°
lat0 = obs[:, 0, 0]; lon0 = obs[:, 0, 1]
bl = np.floor(lat0 / 0.5) * 0.5; bo = np.floor(lon0 / 0.5) * 0.5
dfm = pd.DataFrame(dict(la=bl, lo=bo, g=Dm['Hybrid Transformer'][:, 72] - Dm[vento][:, 72], e=Dm[vento][:, 72]))
agg = dfm.groupby(['la', 'lo']).agg(g=('g', 'mean'), e=('e', 'mean'), n=('g', 'size')).reset_index(); agg = agg[agg.n >= 30]; agg.to_csv(D + f'resultados/mapa_ganho{TAG}.csv', index=False)
print('caixas', len(agg), 'com ganho', int((agg.g < 0).sum()), 'mediana ganho', round(agg.g.median(), 1), 'min', round(agg.g.min(), 1), 'max', round(agg.g.max(), 1))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, col, cm, vmin, vmax, tit in [(axs[0], 'e', 'magma_r', 0, 80, f'(a) {LAB[vento]}: separation at 72 h (km)'),
                                     (axs[1], 'g', 'RdBu_r', -25, 25, '(b) Hybrid Transformer minus (a) (km)')]:
    ax.pcolormesh(sp.longitude, sp.latitude, np.ma.masked_where(terra < 0.5, terra), cmap=matplotlib.colors.ListedColormap(['#d9d4c7']), shading='auto')
    scx = ax.scatter(agg.lo + 0.25, agg.la + 0.25, c=agg[col], s=np.clip(agg.n, 10, 120) * 1.2, marker='s', cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xlim(-43.5, -37.5); ax.set_ylim(-26, -19); ax.set_aspect(1 / np.cos(np.deg2rad(22.5)))
    ax.set_title(tit, loc='left', fontsize=8); plt.colorbar(scx, ax=ax, shrink=0.75)
fig.tight_layout(); fig.savefig(F + f'Fig3_map_gain{TAG}.png', dpi=300); plt.close()

# ---------- Fig 4: exemplos (melhor, mediano e pior ganho)
gan = Dm[vento][:, 72] - Dm['Hybrid Transformer'][:, 72]
ordem = np.argsort(gan); ex = [ordem[int(0.9 * len(ordem))], ordem[len(ordem) // 2], ordem[int(0.1 * len(ordem))]]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, j, t in zip(axs, ex, ['90th percentile of gain', 'Median gain', '10th percentile of gain']):
    ax.plot(obs[j, :, 1], obs[j, :, 0], 'k-', lw=1.6, label='Observed')
    for m in [fis, vento, 'Hybrid Transformer']:
        ax.plot(T[mods_npy.index(m), j, :, 1], T[mods_npy.index(m), j, :, 0], color=COR[m], lw=1.1, label=LAB[m])
    ax.plot(obs[j, 0, 1], obs[j, 0, 0], 'ko', ms=3)
    ax.set_title(t, fontsize=8); ax.set_aspect(1 / np.cos(np.deg2rad(obs[j, 0, 0]))); ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3)); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axs[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(F + f'Fig4_examples{TAG}.png', dpi=300); plt.close()
print('ok')
