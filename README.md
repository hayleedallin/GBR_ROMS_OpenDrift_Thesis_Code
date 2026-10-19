# Coral larval dispersal on the Great Barrier Reef: model setup and run

This notebook simulates the dispersal of coral larvae from a cluster of reefs on the
central Great Barrier Reef after a mass spawning event, and estimates where those
larvae settle.

## What it does

1. **Sets up the spawning event.** Spawning nights are defined in local time (AEST),
   and particles are released at random times within each night.
2. **Seeds larvae on source reefs.** Reefs within a chosen study area are selected
   from the Seamap Australia GBR reef layer. Particles are allocated in proportion
   to reef area (with a minimum per reef) and placed at random positions inside each
   reef polygon.
3. **Runs the dispersal model.** Larvae are tracked with OpenDrift, using a custom
   model (`CoralLarvae`) driven by ROMS ocean currents and GFS winds. Larvae are held
   at the surface for their first hours, then mix vertically with wind-driven
   turbulence and sink slowly. Particles that hit the coastline strand, and those
   leaving the model domain are removed.
4. **Checks the run.** It reports particle fates and sanity checks, and plots
   particle paths and vertical positions.
5. **Applies settlement and mortality.** Post-processing treats each particle as a
   bundle of larvae that gradually settles onto reefs once competent, with
   background mortality applied by age. The result is a reef-to-reef connectivity
   matrix, settlement and self-recruitment statistics, and per-reef metrics.

Settlement and mortality are deliberately kept out of the model run. The run only
moves particles, so biological assumptions can be changed and re-applied in seconds
without re-running the model.

## Files

| File | Purpose |
|---|---|
| `Model_Simulation_Config.ipynb` | This notebook: configure, run and check one simulation |
| `models/LarvaeDriftV4.py` | `CoralLarvae`, an OpenDrift model with coral larval vertical behaviour |
| `models/PostProcessingV2.py` | Settlement, mortality, connectivity matrices, figures and GIS export |

## Data

Reef and study-area shapefiles go in `Data/SHP/`. The ROMS ocean model output and
GFS wind forcing are **not included** because of their size; set their location in
the configuration cell. Any ROMS output readable by OpenDrift's `reader_ROMS_native`
and any CF-compliant wind file should work, with matching paths and variable names.

## How to use

1. Install OpenDrift and its dependencies (conda-forge recommended; see
   https://opendrift.github.io/install.html), plus `geopandas` and `geopy`.
2. Edit the **configuration cell** only: data paths, spawning nights, study area,
   particle numbers, physics settings and random seeds.
3. Run all cells from top to bottom. A 40,000-particle, ~22-day run takes about
   1.5 hrs at 15min timestep.

Output is written to `outputs/<RUN_NAME>/`: the trajectory file (`.nc`), a reef
lookup table, figures and post-processing CSVs.

To compare scenarios (e.g. different spawning months or sensitivity tests), run the
notebook once per scenario, then post-process the runs together. See
*Comparing runs* at the end of the notebook.

## Citation

If you use this code, please cite my thesis paper and OpenDrift :)

> Dagestad, K.-F., Röhrs, J., Breivik, Ø., & Ådlandsvik, B. (2018). OpenDrift v1.0:
> a generic framework for trajectory modelling. *Geoscientific Model Development*,
> 11, 1405–1420. https://doi.org/10.5194/gmd-11-1405-2018
