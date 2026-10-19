"""Substitui os marcadores {{...}} de manuscrito_OD.md pelos números de resultados/ (revisão de 17/10/2026).
Roda após baselines_residuo.py (CV=drifter e CV=year) e tabela_revisao.py ('' e '_cvano')."""
import re, numpy as np, pandas as pd
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')
md = open(D + 'manuscrito_OD.md', encoding='utf-8').read()
tab = pd.read_csv(D + 'resultados/tabela_baselines_residuo.csv').set_index('model')
dd = pd.read_csv(D + 'resultados/diferencas_pareadas.csv')
Td = pd.read_csv(D + 'resultados/decaimento_T.csv')
imp = pd.read_csv(D + 'resultados/importancia_permutacao_lgbm_hist.csv')
R0D = 'GLORYS12 + windage + r0 decay'
A = pd.read_csv(D + 'resultados/resultados_deriva_glorys_ncep_v2.csv'); B = pd.read_csv(D + 'resultados/baselines_residuo.csv')
m72 = {m: A[A.model == m].d72.mean() for m in A.model.unique()}; m72.update({m: B[B.model == m].d72.mean() for m in B.model.unique()})
m24 = {m: A[A.model == m].d24.mean() for m in A.model.unique()}; m24.update({m: B[B.model == m].d24.mean() for m in B.model.unique()})
m48 = {m: A[A.model == m].d48.mean() for m in A.model.unique()}; m48.update({m: B[B.model == m].d48.mean() for m in B.model.unique()})
W = 'GLORYS12 + calibrated windage'; TR = 'Hybrid Transformer'


def num(x): return f'{x:.1f}'


def dif(m1, m2, h):
    r = dd[(dd.model == m1) & (dd.minus == m2) & (dd.h == h)].iloc[0]; return r


def sdif(m1, m2, h, ci=True):
    r = dif(m1, m2, h); s = f'{r["diff"]:+.1f}'.replace('-', '−')
    return s + (f' (95% CI {r.lo:+.1f} to {r.hi:+.1f})'.replace('-', '−') if ci else '')


def row(m, label):
    t = tab.loc[m]
    cells = [label, t.d24, t.d48, t.d72, t.d72_median, t.ss72, t.diff72_vs_wind, t.d72_undrogued, t.d72_drogued]
    return '| ' + ' | '.join(str(c).replace('-', '−') for c in cells) + ' |'


rep = {}
rep['R0D_72'] = num(m72[R0D]); rep['R0D_24'] = num(m24[R0D]); rep['R0D_48'] = num(m48[R0D])
rep['R0_72'] = num(m72['GLORYS12 + windage + r0']); rep['RB_72'] = num(m72['GLORYS12 + windage + mean 24-h residual'])
rep['RG_72'] = num(m72['Hybrid ridge']); rep['RGH_72'] = num(m72['Hybrid ridge + history']); rep['LG120_72'] = num(m72['Hybrid LightGBM (120k pairs)'])
rep['T_RANGE'] = f'{Td.T_h.min():.0f}–{Td.T_h.max():.0f}' if Td.T_h.min() != Td.T_h.max() else f'{Td.T_h.min():.0f}'
rep['T_TYP'] = f'{Td.T_h.median():.0f}'
rep['R0D_DIFF_W'] = tab.loc[R0D].diff72_vs_wind.split(' ')[0].replace('-', '−')
rep['R0D_SS'] = tab.loc[R0D].ss72.split(' ')[0]
for h, k in ((24, 'TR_R0D_PCT24'), (48, 'TR_R0D_PCT48'), (72, 'TR_R0D_PCT72')):
    base = {24: m24, 48: m48, 72: m72}[h]
    rep[k] = f'{100 * (1 - base[TR] / base[R0D]):.0f}'
rep['TR_R0D_24'] = sdif(TR, R0D, 24, ci=False); rep['TR_R0D_72'] = sdif(TR, R0D, 72, ci=False)
r = dif(TR, R0D, 72); rep['TR_R0D_72_CI'] = f'{r.lo:+.1f} to {r.hi:+.1f}'.replace('-', '−')
rep['LH_R0D_72'] = sdif('Hybrid LightGBM + history', R0D, 72, ci=False)
rep['LG_R0D_72'] = sdif('Hybrid LightGBM', R0D, 72, ci=False); rep['ML_R0D_72'] = sdif('Hybrid MLP', R0D, 72, ci=False)
rl = dif('Hybrid LightGBM', R0D, 72); rm = dif('Hybrid MLP', R0D, 72)
def ciword(r): return 'CI excludes zero' if (r.lo < 0 and r.hi < 0) or (r.lo > 0 and r.hi > 0) else 'CI includes zero'
rep['MLP_R0D_NOTE'] = f'95% CI {rl.lo:+.1f} to {rl.hi:+.1f} and {rm.lo:+.1f} to {rm.hi:+.1f}'.replace('-', '−')
rep['RG_VS_R0D'] = sdif('Hybrid ridge', R0D, 72, ci=False); rep['RGH_VS_R0D'] = sdif('Hybrid ridge + history', R0D, 72, ci=False)
gap = m72[W] - m72[TR]; rep['R0D_SHARE'] = f'{100 * (m72[W] - m72[R0D]) / gap:.0f}'
rg = dif('Hybrid ridge', R0D, 72); rgh = dif('Hybrid ridge + history', R0D, 72); lgr = dif('Hybrid LightGBM', 'Hybrid ridge', 72); trh = dif(TR, 'Hybrid ridge + history', 72)
rep['RG_SENTENCE'] = (f'{"did not improve on" if rg.lo <= 0 <= rg.hi or rg["diff"] > 0 else "improved on"} the decaying residual without history and '
                      f'{"did not improve on it" if rgh.lo <= 0 <= rgh.hi or rgh["diff"] > 0 else "improved on it by " + num(-rgh["diff"]) + " km"} with history; '
                      f'LightGBM was {num(-lgr["diff"])} km better than ridge on the same inputs and the Transformer {num(-trh["diff"])} km better than ridge with history, so the nonlinearity matters')
# importância
g = imp[~imp.group.str.contains(r'\[')].groupby('group').increase.mean().sort_values(ascending=False)
top = g.head(4)
rep['IMP_SENTENCE'] = ('shuffling ' + ', '.join(f'{k} raised the test error by {100 * v:.0f}%' for k, v in top.items()) +
                       f', whereas the local forcing groups raised it by at most {100 * g[[k for k in g.index if k.startswith(("model current", "wind", "vorticity"))]].max():.0f}%' +
                       f' (Table S3); the history mattered more at τ ≤ 24 h ({100 * imp[imp.group == "all history (12 inputs) [tau<=24]"].increase.mean():.0f}%) than at τ > 24 h ({100 * imp[imp.group == "all history (12 inputs) [tau>24]"].increase.mean():.0f}%)')
rep['ROW_R0D'] = row(R0D, 'GLORYS12 + windage + r₀ decay'); rep['ROW_RG'] = row('Hybrid ridge', 'Hybrid ridge'); rep['ROW_RGH'] = row('Hybrid ridge + history', 'Hybrid ridge + history')
# tabela S2
S2 = ['| Model | 24 h | 48 h | 72 h | Median 72 h | Skill 72 h | Δ72 h vs. windage | Undrogued 72 h | Drogued 72 h |', '|---|---|---|---|---|---|---|---|---|']
for m, lab in [('GLORYS12 + windage + r0', 'GLORYS12 + windage + r₀ (no decay)'), ('GLORYS12 + windage + r0 decay', 'GLORYS12 + windage + r₀ decay'),
               ('GLORYS12 + windage + mean 24-h residual', 'GLORYS12 + windage + mean 24-h residual'), ('Hybrid ridge', 'Hybrid ridge'),
               ('Hybrid ridge + history', 'Hybrid ridge + history'), ('Hybrid LightGBM (120k pairs)', 'Hybrid LightGBM (120k pairs)')]:
    S2.append(row(m, lab))
rep['TABLE_S2'] = '\n'.join(S2)
# tabela S3
gi = imp[~imp.group.str.contains(r'\[')].groupby('group').increase.agg(['mean', 'std']).sort_values('mean', ascending=False)
S3 = ['| Input group | Increase in MSE (%) |', '|---|---|'] + [f'| {k} | {100 * v["mean"]:.1f} ({100 * v["std"]:.1f}) |' for k, v in gi.iterrows()]
S3 += [f'| all history, τ ≤ 24 h | {100 * imp[imp.group == "all history (12 inputs) [tau<=24]"].increase.mean():.1f} |', f'| all history, τ > 24 h | {100 * imp[imp.group == "all history (12 inputs) [tau>24]"].increase.mean():.1f} |',
       f'| r₀, τ ≤ 24 h | {100 * imp[imp.group == "r0 [tau<=24]"].increase.mean():.1f} |', f'| r₀, τ > 24 h | {100 * imp[imp.group == "r0 [tau>24]"].increase.mean():.1f} |']
rep['TABLE_S3'] = '\n'.join(S3)
# ano
comp = pd.read_csv(D + 'resultados/composicao_dobras_ano.csv')
S1 = ['| Fold | Test years | Test starts | Test drifters | Training drifters |', '|---|---|---|---|---|'] + [f'| {r.fold} | {r.anos.replace(" ", ", ")} | {r.n_test} | {r.n_drifters_test} | {r.n_drifters_train} |' for r in comp.itertuples()]
rep['TABLE_S1'] = '\n'.join(S1)
try:
    tabY = pd.read_csv(D + 'resultados/tabela_baselines_residuo_cvano.csv').set_index('model'); ddY = pd.read_csv(D + 'resultados/diferencas_pareadas_cvano.csv')
    t = tabY.loc[R0D]; cells = ['GLORYS12 + windage + r₀ decay', t.d24, t.d48, t.d72, t.d72_median, t.ss72, t.diff72_vs_wind, t.d72_undrogued, t.d72_drogued]
    rep['ROW_R0D_CVANO'] = '| ' + ' | '.join(str(c).replace('-', '−') for c in cells) + ' |'
    rY = ddY[(ddY.model == TR) & (ddY.minus == R0D) & (ddY.h == 72)].iloc[0]
    rep['CVANO_R0D_SENTENCE'] = f'The persisted-residual baseline reached {t.d72.split(" ")[0]} km, and the Transformer remained {-rY["diff"]:.1f} km (95% CI {-rY.hi:.1f}–{-rY.lo:.1f}) better than it.'
except FileNotFoundError:
    rep['ROW_R0D_CVANO'] = ''; rep['CVANO_R0D_SENTENCE'] = ''
falt = set(re.findall(r'\{\{(\w+)\}\}', md)) - set(rep)
assert not falt, falt
for k, v in rep.items():
    md = md.replace('{{' + k + '}}', str(v))
md = re.sub(r'\n\n\n+', '\n\n', md)
open(D + 'manuscrito_OD.md', 'w', encoding='utf-8').write(md)
print('ok; valores:'); [print(k, '=', str(v)[:120]) for k, v in rep.items() if not k.startswith(('TABLE', 'ROW'))]
