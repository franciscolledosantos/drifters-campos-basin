# drifters-campos-basin

Code and result tables for the manuscript *Hybrid physics–machine-learning forecasts of surface drifter trajectories in the Campos Basin, Brazil: learning the residual velocity of modelled ocean currents* (submitted to *Ocean Dynamics*, 2026).

Hourly Global Drifter Program trajectories (146 drifters, 2005–2022, 3,834 forecasts) in the Campos Basin are forecast 24–72 h ahead with GLORYS12 surface currents and NCEP/NCAR winds, a windage calibrated by drogue state, a persisted initial residual with exponential decay, and hybrid models (ridge regression, LightGBM, MLP and a Transformer reading the drifter's last 24 h) that learn the residual velocity (drifter minus modelled current). Validation holds out whole drifters and, as a stricter test, whole years; HYCOM GOFS 3.1 currents are a sensitivity test. The hybrid Transformer reaches 9.2, 19.4 and 30.2 km mean separation at 24, 48 and 72 h, against 37.4 km for GLORYS12 + windage and 32.3 km for the persisted residual; ridge and LightGBM given the same 24-h history come within 0.4–0.9 km, so the gain comes from the recent history of the model error rather than from the architecture.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Forcing loaders (`forcantes.py`), forecast experiments (`deriva.py`), persisted-residual/linear baselines and permutation importance (`baselines_residuo.py`), tables, figures, HYCOM download, manuscript fillers and builder; `gwbench/deriva.py` holds the generic tested routines (RK2 integration, separation, Liu–Weisberg skill, windage, residual model) |
| `resultados/` | Per-forecast separations of every model and fold (`resultados_deriva_glorys_ncep_v2.csv` drifter-held-out, `_cvano_v2` year-held-out, `resultados_deriva.csv` HYCOM), summary tables with bootstrap CIs (`tabela_principal*.csv`, `tabela_baselines_residuo*.csv`), paired differences, decay constants, permutation importance, fold composition, gain maps. Also the working folder for the forecast trajectories `res_traj*_fold*.npy` (not redistributed) |
| `figuras/` | Figs. 1–4 of the manuscript (`Fig*_main.png`); `supplementary/` holds the diagnostic figures of the HYCOM run and of the year-held-out run produced by `analise.py` |

## Pipeline

1. Download the raw data to the repository root (see **Data**): `gdp.csv`, `glorys/glorys_*.nc`, `wind/uwnd_<year>.nc`, `wind/vwnd_<year>.nc` and, for the sensitivity test, `hycom/<yyyymm>_z0.nc` (`python3 scripts/get_hycom.py old` and `... new`).
2. `scripts/forcantes.py` — loads GLORYS12, HYCOM, NCEP/NCAR (and optionally ERA5) fields and interpolates them in time and space (imported by the other scripts).
3. `scripts/deriva.py` (or `scripts/rodadas.sh`) — physical, persistence and hybrid forecasts with 10-fold drifter-grouped (`CV=drifter`) or year-grouped (`CV=year`) validation; `FORC=glorys|hycom`, `VENTO=ncep|era5` → `resultados/resultados_deriva<tag>.csv`, `resultados/res_traj<tag>_fold<k>.npy`.
4. `scripts/analise.py <tag>` — drifter-cluster bootstrap, summary table, gain map and diagnostic figures → `tabela_principal<tag>.csv`, `mapa_ganho<tag>.csv`, `figuras/Fig*_<tag>.png`.
5. `scripts/baselines_residuo.py` (or `scripts/rodar_revisao17.sh drifter|year`) — persisted-residual variants (no decay, exponential decay, 24-h mean), ridge with/without history, LightGBM on 120k pairs, permutation importance → `baselines_residuo*.csv`, `decaimento_T*.csv`, `importancia_permutacao_lgbm_hist*.csv`, `composicao_dobras_ano.csv`.
6. `scripts/tabela_revisao.py [_cvano]` — table rows and paired differences for the new baselines → `tabela_baselines_residuo*.csv`, `diferencas_pareadas*.csv`.
7. `scripts/figuras_revisao.py` — final Figs. 1–4 (`figuras/Fig*_main.png`).
8. `scripts/preencher_manuscrito.py`, `pos_preencher.py`, `build_od_docx.py` — fill the numbers into the manuscript source and build the submission .docx (manuscript source not included).

Run the scripts from any working directory: every path is resolved relative to the script's own location (`os.path.dirname(__file__)`).

Code comments and some file names are in Portuguese (`resultados` = results, `figuras` = figures, `deriva` = drift, `forcantes` = forcings, `cvano` = year-held-out validation).

## Data

- **Drifters**: NOAA/AOML Global Drifter Program, quality-controlled hourly product (Elipot et al. 2016; https://www.aoml.noaa.gov/phod/gdp/), e.g. through the AOML ERDDAP server (dataset `drifter_hourly_qc`), box 25–20°S, 42.5–38.5°W, 2005–2022, exported as CSV with columns `ID,time,latitude,longitude,ve,vn,err_lat,err_lon,drogue_lost_date` (ERDDAP units row kept as line 2) and saved as `gdp.csv` (≈14 MB, not included). NOAA data are freely available; the file is omitted because of its size and to keep the authoritative source.
- **GLORYS12V1**: Copernicus Marine Service, `GLOBAL_MULTIYEAR_PHY_001_030` (daily means, `uo`/`vo`, surface level, 2005–2021) and the operational analysis of the same system, `GLOBAL_ANALYSISFORECAST_PHY_001_024`, for June–December 2022 (https://marine.copernicus.eu; free registration). Subset to 26–19°S, 43.5–37.5°W and save as `glorys/glorys_2005_2021.nc`, `glorys/glorys_2021_2022.nc`.
- **HYCOM GOFS 3.1** (sensitivity test): https://www.hycom.org, retrieved with `scripts/get_hycom.py` from the THREDDS NCSS of tds.hycom.org into `hycom/`.
- **Winds**: NCEP/NCAR Reanalysis 1, 10-m `uwnd`/`vwnd`, 6-hourly Gaussian grid (Kalnay et al. 1996; NOAA PSL, https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html), one file per year as `wind/uwnd_<year>.nc`, `wind/vwnd_<year>.nc`, 2005–2022. Optional ERA5 10-m wind (Copernicus Climate Data Store) as `era5/era5_vento10m_<year>.nc` for `VENTO=era5`.

Raw input files are **not** redistributed in this repository; they are large and/or belong to third parties. Download them from the sources above and place them as described, or start from the derived tables in `resultados/`.

## Citation

See `CITATION.cff` (co-authors to be added). Licence: MIT.
