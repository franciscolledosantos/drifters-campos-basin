"""Linhas de tabela (com IC bootstrap por derivador, como analise.py) para os modelos novos de resultados/baselines_residuo*.csv,
diferenças pareadas vs. GLORYS12 + windage e vs. os híbridos da rodada principal, e resumo da importância por permutação.
Uso: python3 tabela_revisao.py [_cvano]"""
import sys, numpy as np, pandas as pd
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')
SUF = sys.argv[1] if len(sys.argv) > 1 else ''
a = pd.read_csv(D + 'resultados/' + ('resultados_deriva_glorys_ncep_cvano_v2.csv' if SUF else 'resultados_deriva_glorys_ncep_v2.csv')); b = pd.read_csv(D + f'resultados/baselines_residuo{SUF}.csv')
r = pd.concat([a, b]); r['model'] = r.model.replace({'GLORYS12 (OpenDrift-equivalent)': 'GLORYS12', 'GLORYS12 + calibrated windage': 'GLORYS12 + windage'})
W = r.pivot(index='i0', columns='model', values='d72'); W24 = r.pivot(index='i0', columns='model', values='d24'); W48 = r.pivot(index='i0', columns='model', values='d48')
S = r.pivot(index='i0', columns='model', values='ss72')
meta = r[r.model == 'Persistence'].set_index('i0').loc[W.index]
ids = meta.ID.values; uid = np.unique(ids); grp = {u: np.where(ids == u)[0] for u in uid}
rng = np.random.default_rng(0); B = 2000
idxs = [np.concatenate([grp[u] for u in rng.choice(uid, len(uid))]) for _ in range(B)]
def ic(x):
    v = np.array([np.nanmean(x[i]) for i in idxs]); return np.nanmean(x), np.percentile(v, 2.5), np.percentile(v, 97.5)
novos = ['GLORYS12 + windage + r0', 'GLORYS12 + windage + r0 decay', 'GLORYS12 + windage + mean 24-h residual', 'Hybrid ridge', 'Hybrid ridge + history', 'Hybrid LightGBM (120k pairs)']
novos = [m for m in novos if m in W.columns]
lin = []
for m in novos:
    row = dict(model=m)
    for h, M in ((24, W24), (48, W48), (72, W)):
        mu, lo, hi = ic(M[m].values); row[f'd{h}'] = f'{mu:.1f} [{lo:.1f}, {hi:.1f}]'
    row['d72_median'] = f'{np.nanmedian(W[m]):.1f}'; row['n_nan'] = int(W[m].isna().sum())
    mu, lo, hi = ic(S[m].values); row['ss72'] = f'{mu:.2f} [{lo:.2f}, {hi:.2f}]'
    mu, lo, hi = ic((W[m] - W['GLORYS12 + windage']).values); row['diff72_vs_wind'] = f'{mu:+.1f} [{lo:+.1f}, {hi:+.1f}]'
    row['d72_undrogued'] = f'{np.nanmean(W[m][meta.drog.values == 0]):.1f}'; row['d72_drogued'] = f'{np.nanmean(W[m][meta.drog.values == 1]):.1f}'
    row['frac_better_than_wind'] = f'{(W[m] < W["GLORYS12 + windage"]).mean():.2f}'
    lin.append(row)
tab = pd.DataFrame(lin); tab.to_csv(D + f'resultados/tabela_baselines_residuo{SUF}.csv', index=False); print(tab.to_string())
# diferenças pareadas de interesse
pares = [('Hybrid Transformer', 'GLORYS12 + windage + r0 decay'), ('Hybrid LightGBM + history', 'GLORYS12 + windage + r0 decay'),
         ('Hybrid LightGBM', 'GLORYS12 + windage + r0 decay'), ('Hybrid MLP', 'GLORYS12 + windage + r0 decay'),
         ('Hybrid ridge', 'GLORYS12 + windage + r0 decay'), ('Hybrid ridge + history', 'GLORYS12 + windage + r0 decay'),
         ('Hybrid LightGBM', 'Hybrid ridge'), ('Hybrid LightGBM + history', 'Hybrid ridge + history'), ('Hybrid Transformer', 'Hybrid ridge + history'),
         ('Hybrid LightGBM (120k pairs)', 'Hybrid LightGBM'), ('Hybrid LightGBM + history', 'Hybrid LightGBM (120k pairs)'),
         ('GLORYS12 + windage + r0 decay', 'GLORYS12 + windage + r0'), ('GLORYS12 + windage + r0 decay', 'GLORYS12 + windage + mean 24-h residual'),
         ('GLORYS12 + windage + r0 decay', 'Persistence')]
dl = []
for m1, m2 in pares:
    if m1 in W.columns and m2 in W.columns:
        for h, M in ((24, W24), (72, W)):
            mu, lo, hi = ic((M[m1] - M[m2]).values); dl.append(dict(model=m1, minus=m2, h=h, diff=round(mu, 2), lo=round(lo, 2), hi=round(hi, 2), frac_better=round((M[m1] < M[m2]).mean(), 2)))
dd = pd.DataFrame(dl); dd.to_csv(D + f'resultados/diferencas_pareadas{SUF}.csv', index=False); print(dd.to_string())
# ganho por estado da âncora, Transformer vs windage e vs r0 decay
for dg in (0, 1):
    sel = meta.drog.values == dg
    print('drog', dg, 'windage', round(np.nanmean(W['GLORYS12 + windage'][sel]), 1), 'r0decay', round(np.nanmean(W['GLORYS12 + windage + r0 decay'][sel]), 1),
          'Transformer', round(np.nanmean(W['Hybrid Transformer'][sel]), 1), 'LGBM', round(np.nanmean(W['Hybrid LightGBM'][sel]), 1))
try:
    imp = pd.read_csv(D + f'resultados/importancia_permutacao_lgbm_hist{SUF}.csv')
    print(imp.groupby('group').increase.agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).round(3))
    print(pd.read_csv(D + f'resultados/decaimento_T{SUF}.csv').round(4))
except FileNotFoundError:
    pass
