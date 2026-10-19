"""Ajustes de prosa após preencher_manuscrito.py (18/10/2026): as frases geradas automaticamente para o ridge e para a
importância por permutação não descrevem bem o resultado (o ridge com histórico ficou a 0,4 km do Transformer e o
LightGBM sem histórico ficou pior que o resíduo persistido), por isso são reescritas aqui com os números finais."""
import re, pandas as pd
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')
md = open(D + 'manuscrito_OD.md', encoding='utf-8').read()
assert '{{' not in md
dd = pd.read_csv(D + 'resultados/diferencas_pareadas.csv')


def d(m1, m2, h):
    r = dd[(dd.model == m1) & (dd.minus == m2) & (dd.h == h)].iloc[0]
    return r['diff'], r.lo, r.hi


def km(x): return f'{abs(x):.1f}'


def ci(lo, hi): return f'(95% CI {-hi:+.1f} to {-lo:+.1f})'.replace('-', '−').replace('+', '')


R0D = 'GLORYS12 + windage + r0 decay'
rg = d('Hybrid ridge', R0D, 72); rgh = d('Hybrid ridge + history', R0D, 72); lgr = d('Hybrid LightGBM', 'Hybrid ridge', 72)
trh72 = d('Hybrid Transformer', 'Hybrid ridge + history', 72); trh24 = d('Hybrid Transformer', 'Hybrid ridge + history', 24)
lhr = d('Hybrid LightGBM + history', 'Hybrid ridge + history', 72)
# 1) frase do ridge em 3.2 (gerada por RG_SENTENCE)
m = re.search(r'A linear model on the same inputs .*? so the nonlinearity matters\.', md)
assert m, 'RG_SENTENCE'
novo = (f'A linear model on the same inputs improved on the decaying residual by {km(rg[0])} km {ci(rg[1], rg[2])} without history and by '
        f'{km(rgh[0])} km {ci(rgh[1], rgh[2])} with it; at 72 h, ridge was {km(lgr[0])} km {ci(-lgr[2], -lgr[1])} better than LightGBM on the same inputs, '
        f'LightGBM + history was within {km(lhr[0])} km of ridge + history (CI includes zero), and the Transformer was {km(trh72[0])} km '
        f'{ci(trh72[1], trh72[2])} better than ridge + history at 72 h and {km(trh24[0])} km {ci(trh24[1], trh24[2])} at 24 h. '
        f'Most of the predictable part of the residual is therefore a linear function of the recent residual and the local forcing; the nonlinearity matters mainly in the first day')
md = md[:m.start()] + novo + md[m.end():]
# 2) frase da importância em 3.3 (gerada por IMP_SENTENCE)
imp = pd.read_csv(D + 'resultados/importancia_permutacao_lgbm_hist.csv'); g = imp.groupby('group').increase.mean() * 100
m = re.search(r'Permutation of the input groups of LightGBM \+ history on the test pairs confirms where the information lies: .*?\(Table S3\)[^.]*\.', md)
assert m, 'IMP_SENTENCE'
novo = (f'Permutation of the input groups of LightGBM + history on the test pairs confirms where the information lies: shuffling r₀ raised the test error by '
        f'{g["r0"]:.0f}% and the 12 history inputs together by {g["all history (12 inputs)"]:.0f}%, against {g["wind (u,v,speed)"]:.0f}% for the wind, '
        f'{g["model current (u,v,speed)"]:.0f}% for the model current, {g["lead time"]:.0f}% for the lead time, {g["latitude/longitude"]:.0f}% for latitude and longitude, '
        f'{g["drogue"]:.0f}% for the drogue state and about 1% for the day of the year and the current gradients (Table S3); r₀ and the history mattered far more at '
        f'τ ≤ 24 h ({g["r0 [tau<=24]"]:.0f}% and {g["all history (12 inputs) [tau<=24]"]:.0f}%) than at τ > 24 h ({g["r0 [tau>24]"]:.0f}% and {g["all history (12 inputs) [tau>24]"]:.0f}%).')
md = md[:m.start()] + novo + md[m.end():]
# 3) Discussão, par. 2: o que exige modelo não linear
tr = d('Hybrid Transformer', R0D, 72)
m = re.search(r'The learned models add to this, but the part that requires a nonlinear model with history is [^.]*\.', md)
assert m, 'DISC'
novo = (f'The learned models add to this: {km(tr[0])} km at 72 h for the Transformer, of which {km(rgh[0])} km is obtained by a linear regression with the history summary; '
        f'the part that requires a nonlinear sequence model is {km(trh72[0])} km at 72 h (not distinguishable from zero) and {km(trh24[0])} km at 24 h, far smaller than the headline gain over the windage baseline.')
md = md[:m.start()] + novo + md[m.end():]
open(D + 'manuscrito_OD.md', 'w', encoding='utf-8').write(md); print('ok pos_preencher')
