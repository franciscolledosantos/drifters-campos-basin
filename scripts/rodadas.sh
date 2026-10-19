#!/bin/bash
# Main runs (GLORYS12 + NCEP/NCAR): drifter-held-out and year-held-out validation
cd "$(dirname "$0")/.." || exit 1
FORC=glorys SUF=_v2 python3 scripts/deriva.py > deriva_glorys_ncep_v2.log 2>&1
FORC=glorys CV=year SUF=_v2 python3 scripts/deriva.py > deriva_glorys_ncep_cvano_v2.log 2>&1
# HYCOM sensitivity run (default forcing): FORC=hycom python3 scripts/deriva.py
