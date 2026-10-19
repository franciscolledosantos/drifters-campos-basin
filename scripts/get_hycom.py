"""Baixa correntes HYCOM (u, v; 0 e 15 m; 3-horário) via NCSS do tds.hycom.org, mês a mês, só nos meses com boias."""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # run relative to the repository root
import os, subprocess, pandas as pd

d = pd.read_csv('gdp.csv', skiprows=[1], parse_dates=['time'])
dias = pd.DatetimeIndex(d.time.dt.floor('D').unique())
dias = dias.union(dias + pd.Timedelta(days=1)).union(dias + pd.Timedelta(days=2)).union(dias + pd.Timedelta(days=3))
meses = sorted(set((t.year, t.month) for t in dias if t.year <= 2022))
cont = d.groupby([d.time.dt.year, d.time.dt.month]).size().to_dict()
meses = sorted(meses, key=lambda k: -cont.get(k, 0))
os.makedirs('hycom', exist_ok=True)


def exp(t):
    if t < pd.Timestamp('2014-07-01'): return f'GLBv0.08/expt_53.X/data/{t.year}', 316.5, 322.5
    if t < pd.Timestamp('2016-05-01'): return 'GLBv0.08/expt_56.3', 316.5, 322.5
    if t < pd.Timestamp('2017-02-01'): return 'GLBv0.08/expt_57.2', 316.5, 322.5
    if t < pd.Timestamp('2017-06-01'): return 'GLBv0.08/expt_92.8', 316.5, 322.5
    if t < pd.Timestamp('2017-10-01'): return 'GLBv0.08/expt_57.7', 316.5, 322.5
    if t < pd.Timestamp('2018-01-01'): return 'GLBv0.08/expt_92.9', 316.5, 322.5
    if t < pd.Timestamp('2020-02-01'): return 'GLBv0.08/expt_93.0', 316.5, 322.5
    return f'GLBy0.08/expt_93.0/uv3z/{t.year}', 316.5, 322.5


import sys
grupo = sys.argv[1]
for (y, m) in meses:
    if (grupo == 'old') != (pd.Timestamp(y, m, 1) < pd.Timestamp('2014-07-01')):
        continue
    ini = pd.Timestamp(y, m, 1, 12)
    fim = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=12)
    # experimentos que trocam no meio do mês: usar o do início, mas cortar no limite
    path, w, e = exp(ini)
    if path.endswith('/2015') or '53.X' in path:
        fim = min(fim, pd.Timestamp(y, 12, 31, 12))
    for z in (0,):
        out = f'hycom/{y}{m:02d}_z{z}.nc'
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            continue
        url = (f'https://tds.hycom.org/thredds/ncss/{path}?var=water_u&var=water_v&north=-19&west={w}&east={e}'
               f'&south=-26&horizStride=1&time_start={ini:%Y-%m-%dT%H:%M:%SZ}&time_end={fim:%Y-%m-%dT%H:%M:%SZ}'
               f'&vertCoord={z}&timeStride=8&accept=netcdf4')
        for tent in range(3):
            r = subprocess.run(['curl', '-s', '-m', '2400', '-o', out, '-w', '%{http_code}', url], capture_output=True, text=True)
            if r.stdout == '200' and os.path.getsize(out) > 10000:
                break
        print(y, m, z, path, r.stdout, os.path.getsize(out) if os.path.exists(out) else 0, flush=True)
print('FIM')
