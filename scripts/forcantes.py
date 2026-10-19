"""Carrega e interpola as forçantes: HYCOM (u,v a 0 m, diário 12Z) e NCEP/NCAR R1 (vento a 10 m, 6-horário).
Todas as grades HYCOM (GLBv 0,08° e GLBy 0,08°×0,04°) são reamostradas para uma grade comum de 0,08°."""
import glob, numpy as np, pandas as pd, xarray as xr
from scipy.interpolate import RegularGridInterpolator

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')
LAT = np.round(np.arange(-26.0, -18.99, 0.08), 4)
LON = np.round(np.arange(316.56, 322.45, 0.08), 4)


def carrega_hycom():
    arrs = []
    for f in sorted(glob.glob(D + 'hycom/*_z0.nc')):
        d = xr.open_dataset(f)
        d = d.isel(depth=0).drop_vars('depth', errors='ignore')[['water_u', 'water_v']]
        d = d.assign_coords(lon=(d.lon.values % 360)).sortby('lon')
        d = d.interp(lat=LAT, lon=LON)
        arrs.append(d.load())
    h = xr.concat(arrs, 'time').sortby('time')
    h = h.isel(time=~pd.Index(h.time.values).duplicated())
    return h


def carrega_vento():
    us, vs = [], []
    for y in range(2005, 2023):
        u = xr.open_dataset(D + f'wind/uwnd_{y}.nc').uwnd; v = xr.open_dataset(D + f'wind/vwnd_{y}.nc').vwnd
        us.append(u); vs.append(v)
    u = xr.concat(us, 'time'); v = xr.concat(vs, 'time')
    if 'level' in u.dims:
        u = u.isel(level=0); v = v.isel(level=0)
    return xr.Dataset(dict(uw=u, vw=v)).sortby('lat')


class Campo:
    """Interpolador (t, lat, lon) -> (u, v) com tempo em horas desde 2005-01-01."""

    def __init__(self, ds, un, vn, lat='lat', lon='lon'):
        t = (pd.DatetimeIndex(ds.time.values) - pd.Timestamp('2005-01-01')) / pd.Timedelta(hours=1)
        self.t = np.asarray(t, float)
        la = ds[lat].values; lo = ds[lon].values
        U = ds[un].values.astype(float); V = ds[vn].values.astype(float)
        self.mask_nan = np.isnan(U)
        U = np.nan_to_num(U); V = np.nan_to_num(V)
        self.fu = RegularGridInterpolator((self.t, la, lo), U, bounds_error=False, fill_value=np.nan)
        self.fv = RegularGridInterpolator((self.t, la, lo), V, bounds_error=False, fill_value=np.nan)
        self.dlat = la[1] - la[0]; self.dlon = lo[1] - lo[0]

    def __call__(self, th, lat, lon):
        lon360 = np.where(lon < 0, lon + 360, lon)
        th = np.broadcast_to(np.asarray(th, float), np.shape(lat))
        p = np.c_[th, lat, lon360]
        u, v = self.fu(p), self.fv(p)
        # não interpolar através de lacunas de dados (> 36 h entre campos consecutivos)
        k = np.clip(np.searchsorted(self.t, th), 1, len(self.t) - 1)
        lac = (self.t[k] - self.t[k - 1]) > 36
        u[lac] = np.nan; v[lac] = np.nan
        return u, v

    def derivadas(self, th, lat, lon):
        """Vorticidade relativa, divergência e deformação (s^-1) por diferenças centradas."""
        lon360 = np.where(lon < 0, lon + 360, lon)
        dy = self.dlat * 111e3; dx = self.dlon * 111e3 * np.cos(np.deg2rad(lat))
        ue, ve = self(th, lat, lon360 + self.dlon); uw, vw = self(th, lat, lon360 - self.dlon)
        un, vn = self(th, lat + self.dlat, lon360); us, vs = self(th, lat - self.dlat, lon360)
        dudx = (ue - uw) / (2 * dx); dvdx = (ve - vw) / (2 * dx)
        dudy = (un - us) / (2 * dy); dvdy = (vn - vs) / (2 * dy)
        return dvdx - dudy, dudx + dvdy, np.hypot(dudx - dvdy, dvdx + dudy)


def horas(t):
    return np.asarray((pd.DatetimeIndex(t) - pd.Timestamp('2005-01-01')) / pd.Timedelta(hours=1), float)


def carrega_glorys():
    """GLORYS12V1 (médias diárias, ~0,5 m): carimbo às 00 h representa a média do dia -> centrado às 12 h."""
    arrs = []
    for f in sorted(glob.glob(D + 'glorys/glorys_*.nc')):
        d = xr.open_dataset(f).isel(depth=0)[['uo', 'vo']].rename(uo='water_u', vo='water_v',
                                                                   latitude='lat', longitude='lon')
        d = d.drop_vars('depth', errors='ignore')
        # arredonda a grade: arquivos do GLORYS (reanálise e análise) diferem na 6ª casa decimal, e um concat com
        # junção externa intercalaria NaN entre os pontos, zerando metade da grade
        d = d.assign_coords(lat=np.round(d.lat.values.astype(float), 4), lon=np.round(d.lon.values.astype(float) % 360, 4),
                            time=d.time.values + np.timedelta64(12, 'h')).sortby('lon')
        arrs.append(d.load())
    h = xr.concat(arrs, 'time', join='exact').sortby('time')
    return h.isel(time=~pd.Index(h.time.values).duplicated())


def carrega_era5():
    """ERA5 vento a 10 m horário (u10, v10)."""
    arrs = []
    for f in sorted(glob.glob(D + 'era5/era5_vento10m_*.nc')):
        ds = xr.open_dataset(f).drop_vars(['expver', 'number'], errors='ignore')
        tn = 'valid_time' if 'valid_time' in ds.dims else 'time'
        ds = ds.rename({tn: 'time', 'latitude': 'lat', 'longitude': 'lon', 'u10': 'uw', 'v10': 'vw'})[['uw', 'vw']]
        ds = ds.assign_coords(lon=np.round(ds.lon.values.astype(float) % 360, 4), lat=np.round(ds.lat.values.astype(float), 4))
        arrs.append(ds.sortby('lat').sortby('lon').load())
    return xr.concat(arrs, 'time', join='exact').sortby('time')
