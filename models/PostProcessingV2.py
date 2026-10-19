"""
PostProcessing.py is used to apply settlement, mortality and connectivity from saved OpenDrift (.nc) runs.

Everything works on the saved output file (data = xr.open_dataset(outfile)), not the
live model object `o`, so any November or December run can be reloaded later.

Settlement methods (settle(..., method=...)):
  'bundle'        Each particle is treated as a bundle of larvae. While a bundle is competent,
                  inside the depth band and over a reef, a fraction settles every
                  output step (rate = settle_rate_per_hour); the rest keeps drifting
                  and can settle on later reefs.
  'per_visit'     As 'bundle', but a fixed fraction (visit_fraction) settles once
                  each time the bundle enters a reef. This does not account for the varying sizes of reefs
  'first_contact' The whole bundle settles on the first eligible reef contact

Mortality:
  Applied here only, as weight decay exp(-M * age). The OpenDrift runs must be made with
  biology:background_mortality_rate = 0, so every trajectory runs to the end.

Typical use (see bottom of file):
  data = xr.open_dataset(outfile)
  dest = destination_reefs(gbr)
  res  = settle(data, dest, seed_reefs, method='bundle')
  print(summary(res))
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

FATE_COLOURS = {   # muted palette for report figures
    'drifting': '#d5dde5',                           # pale blue-grey
    'settled': '#4f8a7b',                            # muted teal-green
    'mortality': '#f0f0f0',                          # light grey
    'stranded': '#d8d3c8',                           # sand
    'left_domain': '#d0d0d0',                        # slate blue
#    'left_shelf': '#666666',                         # doesnt check left shelf unless it was activated in run
    'pelagic_larval_duration_exceeded': '#d6c48f',   # muted ochre
}
FATE_LABELS = {
    'settled': 'Settled', 'mortality': 'Mortality', 'stranded': 'Stranded',
    'left_domain': 'Left domain', 'left_shelf': 'Left shelf',
    'pelagic_larval_duration_exceeded': 'PLD exceeded', 'drifting': 'Still drifting',
}

# ------------------------------------------------------------------
# Run bookkeeping
# ------------------------------------------------------------------
def status_lookup(ds):
    """OpenDrift status code -> name, read from the output file's attributes."""
    a = ds['status'].attrs
    if 'flag_meanings' not in a:
        return {}
    names = str(a['flag_meanings']).split()
    vals = np.atleast_1d(a.get('flag_values', np.arange(len(names))))
    return {int(v): n for v, n in zip(vals, names)}


def run_info(ds, label=None):
    """Start/end, release window and the longest age every particle reaches."""
    times = pd.to_datetime(ds['time'].values)
    lon = ds['lon'].values
    age_s = ds['age_seconds'].values
    seeded = np.isfinite(lon).any(axis=1)
    first = np.argmax(np.isfinite(lon), axis=1)
    rows = np.where(seeded)[0]
    release = times[first[rows]] - pd.to_timedelta(np.nan_to_num(age_s[rows, first[rows]]), unit='s')

    info = {
        'label': label or f'{times[0]:%b %Y} run',
        'start': times[0], 'end': times[-1],
        'first_release': release.min(), 'last_release': release.max(),
        'n_particles': int(seeded.sum()),
        'dt_h': (times[1] - times[0]).total_seconds() / 3600,
    }
    info['max_common_age_h'] = (info['end'] - info['last_release']).total_seconds() / 3600
    return info


def common_age_cap(*datasets):
    """Largest age (h) that every particle in every run reaches - use the same cap for
    all runs you want to compare."""
    cap = min(run_info(d)['max_common_age_h'] for d in datasets)
    return np.floor(cap * 2) / 2


def destination_reefs(gbr, area_col='Area_HA', min_area_ha=0.0):
    """Reef polygons larvae can settle on: drops the mainland (only feature spanning
    > 1 degree) and zero-area cays/islands."""
    b = gbr.bounds
    reefs = gbr[((b['maxx'] - b['minx']) <= 1) & ((b['maxy'] - b['miny']) <= 1)]
    if area_col in reefs.columns:
        reefs = reefs[reefs[area_col] > min_area_ha]
    return reefs


def _reef_labels(gdf, name_col='GBR_NAME', id_col='GBR_ID'):
    """Reef name, with the ID appended only where a name is duplicated (e.g. 'U/N Reef')."""
    names = gdf[name_col].astype(str)
    dup = names.duplicated(keep=False)
    if id_col in gdf.columns:
        names = names.where(~dup, names + ' (' + gdf[id_col].astype(str) + ')')
    else:
        names = names.where(~dup, names + ' [' + gdf.index.astype(str) + ']')
    return pd.Series(names.values, index=gdf.index)


# ------------------------------------------------------------------
# Core: settlement + mortality + connectivity in one pass
# ------------------------------------------------------------------
def settle(ds, dest_reefs, seed_reefs, method='bundle',
           competency_start_days=4, competency_end_days=28,
           depth_min=-np.inf, depth_max=0,
           settle_rate_per_day=1,                              # these are default fallbacks, actual values set
           settle_fraction_per_hour=None, visit_fraction=0.3,  # in notebook when settle() is called 
           mortality_per_day=0.05, age_max_h=None,
           name_col='GBR_NAME', id_col='GBR_ID',
           label=None, chunk_steps=48, track=None, verbose=True):
    """
    ds          : xr.open_dataset(outfile)
    dest_reefs  : reefs larvae can settle on (use destination_reefs(gbr))
    seed_reefs  : seeding reefs in the SAME order used to assign origin_marker at seeding
    age_max_h   : common age cap (h). None = the longest age every particle in this run
                  reaches. When comparing months, pass common_age_cap(data_nov, data_dec).
    track       : optional list of particle indices whose bundle weight is recorded every
                  step (for plot_bundle_tracks). Pick them with pick_example_particles().

    Returns a results dict used by summary(), connectivity_matrix() and the plot functions.
    """
    if method not in ('bundle', 'per_visit', 'first_contact'):
        raise ValueError("method must be 'bundle', 'per_visit' or 'first_contact'")

    info = run_info(ds, label)
    names = status_lookup(ds)

    # --- mortality: applied here as weight decay (runs have no model mortality)
    M = float(mortality_per_day)
    if verbose:
        print(f"[{info['label']}] Applying mortality exp(-{M} * age) in post-processing.")

    # --- age cap
    if age_max_h is None:
        age_max_h = np.floor(info['max_common_age_h'] * 2) / 2
    elif age_max_h > info['max_common_age_h'] + 1e-6:
        print(f"WARNING: age cap {age_max_h} h is longer than this run supports "
              f"({info['max_common_age_h']:.1f} h) -- late-released larvae will be cut short.")

    lon = ds['lon'].values.astype(float)
    lat = ds['lat'].values.astype(float)
    z = ds['z'].values.astype(float)
    age_h = ds['age_seconds'].values.astype(float) / 3600.0
    status = ds['status'].values
    times = pd.to_datetime(ds['time'].values)
    n, nT = lon.shape

    # --- source reef per particle (origin_marker is NaN outside the active period)
    org = ds['origin_marker'].values
    first = np.argmax(np.isfinite(org), axis=1)
    origin = org[np.arange(n), first]
    origin = np.where(np.isfinite(origin), origin, -1).astype(int)
    valid = origin >= 0
    n_src = len(seed_reefs)
    if origin.max() >= n_src:
        raise ValueError(f'origin_marker goes up to {origin.max()} but seed_reefs has {n_src} rows '
                         '-- is seed_reefs built with the same seeding_id as this run?')

    # --- apply age cap
    with np.errstate(invalid='ignore'):
        too_old = age_h > age_max_h
    lon[too_old] = np.nan
    lat[too_old] = np.nan

    # --- reef under each eligible particle-step
    reefs = dest_reefs.to_crs(epsg=4326)
    reef_geom = gpd.GeoDataFrame(geometry=reefs.geometry.values, crs=reefs.crs)
    with np.errstate(invalid='ignore'):
        eligible = (np.isfinite(lon)
                    & (age_h >= competency_start_days * 24) & (age_h <= competency_end_days * 24)
                    & (z >= depth_min) & (z <= depth_max))
    over = np.full((n, nT), -1, dtype=np.int32)
    for c0 in range(0, nT, chunk_steps):
        ii, tt = np.where(eligible[:, c0:c0 + chunk_steps])
        if len(ii) == 0:
            continue
        tt = tt + c0
        pts = gpd.GeoDataFrame({'p': ii, 't': tt},
                               geometry=gpd.points_from_xy(lon[ii, tt], lat[ii, tt]),
                               crs='EPSG:4326')
        hit = gpd.sjoin(pts, reef_geom, predicate='within', how='inner')
        hit = hit[~hit.index.duplicated(keep='first')]         # overlapping polygons -> one reef
        if len(hit) == 0:
            continue
        over[hit['p'].values.astype(int), hit['t'].values.astype(int)] = \
            hit['index_right'].values.astype(int)

    # --- walk all bundles forward together
    dt_h = info['dt_h']
    surv = np.exp(-M * dt_h / 24.0)
    if method == 'bundle':
        if settle_fraction_per_hour is not None:
            f_settle = 1.0 - (1.0 - settle_fraction_per_hour) ** dt_h   # e.g. 0.2 -> 20% per hour over a reef
        else:
            f_settle = 1.0 - np.exp(-settle_rate_per_day * dt_h / 24.0)
    elif method == 'per_visit':
        f_settle = visit_fraction
    else:
        f_settle = 1.0

    nA = int(np.floor(age_max_h)) + 2
    fate_age = {'settled': np.zeros(nA), 'mortality': np.zeros(nA)}
    C = np.zeros((n_src, len(reefs)))
    w = np.where(valid, 1.0, 0.0)
    settled_w = np.zeros(n)
    first_t = np.full(n, -1)
    final_fate = np.array(['not_released'] * n, dtype=object)
    final_fate[valid] = 'drifting'
    was_active = np.zeros(n, dtype=bool)
    drifting_at_cap = 0.0
    n_settle_visits = np.zeros(n, dtype=int)       # reef visits in which some larvae settled
    last_settle_reef = np.full(n, -1)
    first_reef = np.full(n, -1)                    # first reef the bundle settled on
    first_reef_w = np.zeros(n)                     # share of bundle settled on that first reef

    trk = None if track is None else np.asarray(track, dtype=int)
    if trk is not None:
        k = len(trk)
        T_active = np.zeros((k, nT), dtype=bool)
        T_settled = np.zeros((k, nT))                # settled this step
        T_mort = np.zeros((k, nT))                   # died this step (post-processing mortality)
        T_lost = np.zeros((k, nT))                   # removed this step by deactivation
        T_reef = np.full((k, nT), -1)                # reef the bundle is over while eligible

    def _abin(mask, t):
        return np.clip(np.floor(np.nan_to_num(age_h[mask, t])).astype(int), 0, nA - 1)

    for t in range(nT):
        active = np.isfinite(lon[:, t]) & valid
        ended = was_active & ~active
        if trk is not None:
            w_a = w[trk].copy()
        if ended.any():
            capped = ended & too_old[:, t]
            drifting_at_cap += w[capped].sum()
            deact = np.where(ended & ~capped)[0]
            if len(deact):
                codes = np.nan_to_num(status[deact, t - 1], nan=-1).astype(int)
                reasons = np.array([names.get(c, f'status_{c}') for c in codes], dtype=object)
                bins = np.clip(np.floor(np.nan_to_num(age_h[deact, t - 1])).astype(int), 0, nA - 1)
                for r in np.unique(reasons):
                    m = reasons == r
                    fate_age.setdefault(r, np.zeros(nA))
                    np.add.at(fate_age[r], bins[m], w[deact[m]])
                final_fate[deact] = reasons
            w[ended] = 0.0
        if trk is not None:
            # removal is booked on the bundle's last active step (its age is NaN from step t)
            T_lost[:, t - 1] += np.where(ended[trk] & ~too_old[trk, t], w_a - w[trk], 0.0)
            w_b = w[trk].copy()

        if M > 0 and active.any():
            dead = w[active] * (1.0 - surv)
            np.add.at(fate_age['mortality'], _abin(active, t), dead)
            w[active] -= dead

        if trk is not None:
            T_mort[:, t] = w_b - w[trk]
            w_c = w[trk].copy()
            T_active[:, t] = active[trk]
            T_reef[:, t] = np.where(active[trk], over[trk, t], -1)

        on = active & (over[:, t] >= 0) & (w > 0)
        if method == 'per_visit' and t > 0:
            on &= over[:, t] != over[:, t - 1]
        if on.any():
            new = on & (first_t < 0)
            first_t[new] = t
            first_reef[new] = over[new, t]
            s = w[on] * f_settle
            np.add.at(C, (origin[on], over[on, t]), s)
            np.add.at(fate_age['settled'], _abin(on, t), s)
            settled_w[on] += s
            idx = np.where(on)[0]
            at_first = over[idx, t] == first_reef[idx]
            first_reef_w[idx[at_first]] += s[at_first]
            w[on] -= s
            switch = on & (last_settle_reef != over[:, t])
            n_settle_visits[switch] += 1
            last_settle_reef[on] = over[on, t]
        if trk is not None:
            T_settled[:, t] = w_c - w[trk]
        was_active = active

    drifting_at_cap += w[was_active].sum()
    n0 = valid.sum()

    # --- tidy outputs
    dest_labels = _reef_labels(reefs, name_col, id_col)
    seed_own = _reef_labels(seed_reefs, name_col, id_col)
    seed_labels = [dest_labels[idx] if idx in dest_labels.index else seed_own.iloc[k]
                   for k, idx in enumerate(seed_reefs.index)]
    counts = pd.DataFrame(C, index=seed_labels, columns=dest_labels.values)
    released = pd.Series(np.bincount(origin[valid], minlength=n_src), index=seed_labels, name='released')

    # same reef split over several polygons (or listed twice) -> merge into one row/column
    counts = counts.groupby(level=0, sort=False).sum()
    counts = counts.T.groupby(level=0, sort=False).sum().T
    released = released.groupby(level=0, sort=False).sum()

    fate_by_age = pd.DataFrame(fate_age) / n0
    fate_by_age.index.name = 'age_h'
    fates = fate_by_age.sum()
    fates['drifting'] = drifting_at_cap / n0

    ft = np.where(first_t >= 0, first_t, 0)
    has = first_t >= 0
    settlement_df = pd.DataFrame({
        'particle': np.arange(n),
        'origin_marker': origin,
        'origin_reef': [seed_labels[o] if o >= 0 else None for o in origin],
        'settled': has,                                              # reached an eligible reef
        'settlement_time': pd.Series(times[ft]).where(has).values,   # first eligible reef contact
        'settlement_age_days': np.where(has, age_h[np.arange(n), ft] / 24, np.nan),
        'settlement_reef': [dest_labels.iloc[over[i, first_t[i]]] if first_t[i] >= 0 else None
                            for i in range(n)],
        'settled_fraction': settled_w,                               # of this bundle, all reefs
        'settled_first_reef': first_reef_w,                         # of this bundle, on its first reef
        'settled_elsewhere': settled_w - first_reef_w,              # of this bundle, on later reefs
        'n_settle_visits': n_settle_visits,                          # reef visits with settlement
        'final_fate': final_fate,
    })

    tracks = None
    if trk is not None:
        tracks = dict(ids=trk, age_h=age_h[trk], active=T_active, settled=T_settled,
                      mortality=T_mort, lost=T_lost, reef=T_reef,
                      reef_labels=np.asarray(dest_labels.values, dtype=object),
                      origin=[seed_labels[origin[i]] if origin[i] >= 0 else None for i in trk],
                      final_fate=final_fate[trk])

    params = dict(method=method, competency_days=(competency_start_days, competency_end_days),
                  depth_band=(depth_min, depth_max), settle_rate_per_day=settle_rate_per_day,
                  settle_fraction_per_hour=settle_fraction_per_hour,
                  visit_fraction=visit_fraction, mortality_per_day=M, age_max_h=age_max_h)
    return dict(info=info, params=params, counts=counts, released=released,
                fate_by_age=fate_by_age, fates=fates, settlement_df=settlement_df,
                tracks=tracks)


# ------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------
def connectivity_matrix(res, normalise='released'):
    """
    normalise:
      'released' (default) - proportion of each source reef's RELEASED larvae settling at
                  each sink. Rows sum to that source's total settlement success.
      'source'   - rows sum to 1: where a source's settlers ended up.
      'sink'     - columns sum to 1: where a sink's recruits came from.
      'none'     - settled larvae in particle-equivalents.
    Returns (matrix, counts); zero columns dropped.
    """
    counts = res['counts']
    counts = counts.loc[res['released'] > 0, counts.sum() > 0]   # drop unseeded sources + unreached sinks
    if normalise == 'released':
        matrix = counts.div(res['released'].replace(0, np.nan), axis=0)
    elif normalise == 'source':
        matrix = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalise == 'sink':
        matrix = counts.div(counts.sum(axis=0), axis=1)
    elif normalise == 'none':
        matrix = counts
    else:
        raise ValueError("normalise must be 'released', 'source', 'sink' or 'none'")
    return matrix, counts


def _weighted_quantile(values, weights, q):
    if weights.sum() == 0:
        return np.nan
    cw = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cw, q)])


def summary(res):
    """One-column summary of a run (percentages are of all released larvae)."""
    counts, released = res['counts'], res['released']
    n0 = released.sum()
    settled = counts.values.sum()
    seed_cols = [c for c in counts.columns if c in set(counts.index)]
    local = counts[seed_cols].values.sum()
    self_rec = sum(counts.at[r, r] for r in counts.index if r in counts.columns)
    sa = res['fate_by_age']['settled']
    ages_d = sa.index.values / 24 + 0.5 / 24
    f = res['fates']
    other = f.drop(['settled', 'mortality', 'drifting'], errors='ignore')

    out = {
        'run': res['info']['label'],
        'method': res['params']['method'],
        'age cap (days)': res['params']['age_max_h'] / 24,
        'mortality applied (/day)': res['params']['mortality_per_day'],
        'larvae released (particles)': int(n0),
        'settled (%)': 100 * f.get('settled', 0),
        'mortality (%)': 100 * f.get('mortality', 0),
        **{f'{FATE_LABELS.get(k, k)} (%)': 100 * v for k, v in other.items()},
        'still drifting at age cap (%)': 100 * f.get('drifting', 0),
        'settled on seeding reefs (%)': 100 * local / n0,
        'self-recruitment (% released)': 100 * self_rec / n0,
        'self-recruitment (% of settlers)': 100 * self_rec / settled if settled else np.nan,
        'sink reefs reached': int((counts.sum() > 0).sum()),
        'mean settlement age (days)': (ages_d * sa.values).sum() / sa.sum() if sa.sum() else np.nan,
        'median settlement age (days)': _weighted_quantile(ages_d, sa.values, 0.5),
    }
    return pd.Series(out, name=res['info']['label'])


def source_table(res):
    """Per source reef: released, % settled anywhere, % self-recruited, sinks reached."""
    counts, released = res['counts'], res['released']
    self_rec = pd.Series({r: counts.at[r, r] if r in counts.columns else 0.0 for r in counts.index})
    return pd.DataFrame({
        'released': released,
        'settled (%)': 100 * counts.sum(axis=1) / released.replace(0, np.nan),
        'self-recruitment (%)': 100 * self_rec / released.replace(0, np.nan),
        'sinks reached': (counts > 0).sum(axis=1),
    }).sort_values('released', ascending=False)


def compare_runs(*results):
    """Side-by-side summary, e.g. compare_runs(res_nov, res_dec)."""
    return pd.concat([summary(r) for r in results], axis=1)


def save_outputs(res, outdir, prefix=''):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, prefix)
    summary(res).to_csv(p + 'summary.csv')
    source_table(res).to_csv(p + 'source_table.csv')
    matrix, counts = connectivity_matrix(res, 'released')
    matrix.to_csv(p + 'connectivity_proportion_of_released.csv')
    counts.to_csv(p + 'connectivity_counts.csv')
    res['fate_by_age'].to_csv(p + 'fate_by_age.csv')
    res['settlement_df'].to_csv(p + 'settlement_by_particle.csv', index=False)
    print(f'Saved outputs to {outdir}')


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_connectivity_matrix(matrix, title='Larval connectivity matrix',
                             cbar_label='Larvae settling (% of released)',
                             top_n=None, scale='log', low_pct=10, vmin=None, vmax=None,
                             as_percent=True, cmap='viridis', figsize=None,
                             annot=None, annot_fmt='{:.0f}', annot_size=5, annot_min=0.5):
    """
    Heatmap of a connectivity matrix.

    scale     : 'log' (default) or 'linear'
    low_pct   : log scale only - the lower colour limit is this percentile of the non-zero
                links, so a long tail of near-empty links doesn't use up the colour range.
                Links below it are drawn in the lowest colour (arrow on the colour bar).
    vmin/vmax : set the colour limits explicitly (in the plotted units, i.e. % if as_percent),
                e.g. to give November and December the same scale.
    as_percent: plot values x100 so the colour bar reads in % of released larvae.
    top_n     : keep only the sinks receiving the most larvae.
    Zero / missing links are left blank (light grey).
    """
    m = matrix.copy()
    if top_n is not None and m.shape[1] > top_n:
        m = m[m.sum().sort_values(ascending=False).index[:top_n]]
    m = m.loc[(m.fillna(0) > 0).any(axis=1)]          # drop source reefs with no links shown. comment out if you want to see them
    raw = m.values.astype(float) * (100.0 if as_percent else 1.0)
    vals = np.ma.masked_where(~np.isfinite(raw) | (raw <= 0), raw)
    pos = vals.compressed()

    hi = vmax if vmax is not None else (float(pos.max()) if pos.size else 1.0)
    if scale == 'log':
        lo = vmin if vmin is not None else (max(float(np.percentile(pos, low_pct)), hi / 1e4)
                                            if pos.size else hi / 100)
        lo = min(lo, hi / 10)
        norm = LogNorm(vmin=lo, vmax=hi)
    else:
        lo = vmin if vmin is not None else 0.0
        norm = Normalize(vmin=lo, vmax=hi)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('#f0f0f0')                      # zero / no link
    cm.set_under(cm(0.0))                      # below the lower limit: lowest colour
    extend = 'min' if (pos.size and pos.min() < lo) else 'neither'

    figsize = figsize or (max(6, 0.3 * m.shape[1] + 2), max(4, 0.3 * m.shape[0] + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vals, cmap=cm, norm=norm, aspect='auto', interpolation='nearest')
    if annot is not None:                       # numbers over each cell, e.g. annot=counts
        A = annot.reindex(index=m.index, columns=m.columns).values.astype(float)
        empty = np.ma.getmaskarray(vals)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if empty[i, j] or not np.isfinite(v) or v < annot_min:
                    continue
                r, g, b, _ = cm(norm(max(raw[i, j], lo)))
                txt_col = 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'
                label = annot_fmt(v) if callable(annot_fmt) else annot_fmt.format(v)
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=annot_size, color=txt_col)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index, fontsize=7)
    ax.set_xlabel('Sink reef (settlement)')
    ax.set_ylabel('Source reef')
    ax.set_title(title, loc='left')
    cb = fig.colorbar(im, ax=ax, extend=extend, label=cbar_label)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))   # 0.1, 1, 10 not 10^-1
    fig.tight_layout()
    return fig, ax


def plot_fates_over_age(res, ax=None, title=None, figsize=(7, 3.6)):
    """
    Share of released larvae by fate against larval age. All larvae start as 'still
    drifting' (bottom layer, 100% at day 0); losses to each fate accumulate above it.
    """
    fa = res['fate_by_age']
    cum = fa.cumsum()                                         # cumulative share removed by each fate
    known = ['settled', 'left_domain', 'left_shelf', 'stranded', 'pelagic_larval_duration_exceeded']
    losses = [c for c in known if c in cum.columns] + \
             [c for c in cum.columns if c not in known + ['mortality']] + \
             (['mortality'] if 'mortality' in cum.columns else [])
    drifting = (1 - cum[losses].sum(axis=1)).clip(lower=0)

    names = ['drifting'] + losses                             # bottom -> top
    layers = 100 * np.vstack([drifting.values] + [cum[c].values for c in losses])
    days = cum.index.values / 24

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.stackplot(days, layers, colors=[FATE_COLOURS.get(c, '0.75') for c in names],
                 labels=[FATE_LABELS.get(c, c) for c in names],
                 edgecolor='white', linewidth=0.4)

    ax.set_xlim(0, days.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel('Larval age (days)')
    ax.set_ylabel('Share of released larvae (%)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc='left', fontsize=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc='center left', bbox_to_anchor=(1.01, 0.5),   # same order as the stack
              fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_settlement_age(res, ax=None, title=None):
    """Daily settlement (% of released larvae) with cumulative line."""
    sa = res['fate_by_age']['settled']
    daily = sa.groupby(sa.index // 24).sum() * 100
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    ax.bar(daily.index + 0.5, daily.values, width=0.9, color=FATE_COLOURS['settled'])
    ax.set_xlabel('Larval age (days)')
    ax.set_ylabel('Settled that day (% of released)')
    ax2 = ax.twinx()
    ax2.plot(daily.index + 1, daily.cumsum().values, 'k-', lw=1.2)
    ax2.set_ylabel('Cumulative settled (%)')
    ax.set_title(title or f"Settlement by age - {res['info']['label']}")
    fig.tight_layout()
    return fig, ax


def plot_mortality_check(res, mortality_per_day=None, ax=None):
    """
    Mortality-only survival among larvae still in the water, against exp(-M t).
    Removes the effect of settlement, stranding and the age cap, so the curve should
    sit exactly on the theoretical line - a check that mortality was applied as intended.
    """
    fa = res['fate_by_age']
    M = mortality_per_day
    if M is None:
        M = res['params']['mortality_per_day']
    losses = fa.sum(axis=1)
    in_water = 1 - losses.cumsum().shift(fill_value=0)
    hazard = (fa['mortality'] / in_water.replace(0, np.nan)).fillna(0)
    survival = np.cumprod(1 - hazard.values)
    days = (fa.index.values + 1) / 24
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ax.plot(days, 100 * survival, color='steelblue', lw=1.5, label='Model output')
    ax.plot(days, 100 * np.exp(-M * days), 'k--', lw=1.2, label=f'exp(-{M}·t)')
    ax.set_xlabel('Larval age (days)')
    ax.set_ylabel('Surviving mortality (%)')
    ax.set_title(f"Background mortality check -- {res['info']['label']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# Individual bundles: weight vs age
# ------------------------------------------------------------------
def pick_example_particles(res, n=5, seed=0, top=0.1, late_days=10, verbose=True):
    """
    Pick contrasting example bundles from a settle() result. Each category is ranked by a
    score and one bundle is drawn at random from the top `top` fraction, so examples are
    clear cases rather than borderline ones:
      self-recruiter      first settled on its own source reef and stayed there
      multi-reef settler  largest share settled on reefs AFTER its first one
      late settler        first reached an eligible reef after `late_days`, still settled well
      never reached reef  drifted to the age cap without settling
      lost                stranded or left the domain
    Categories that are empty in this run are skipped; gaps are filled at random.
    """
    df = res['settlement_df']
    df = df[df['origin_marker'] >= 0]
    if 'settled_elsewhere' not in df:
        raise ValueError('Re-run settle() with the updated PostProcessingV2 first')
    rng = np.random.default_rng(seed)
    sf = df['settled_fraction']
    zero = pd.Series(0.0, index=df.index)
    cats = [
        ('self-recruiter',
         (df['settlement_reef'] == df['origin_reef']) & (df['settled_elsewhere'] < 0.02),
         df['settled_first_reef']),
        ('multi-reef settler', df['settled_elsewhere'] > 0.05, df['settled_elsewhere']),
        (f'late settler (>{late_days} d)',
         df['settlement_age_days'] >= late_days, sf),
        ('never reached a reef', (sf == 0) & (df['final_fate'] == 'drifting'), zero),
        ('lost (stranded / left domain)',
         df['final_fate'].isin(['stranded', 'left_domain']), -sf),
    ]
    picked = []
    for name, mask, score in cats:
        if len(picked) >= n:
            break
        sub = df[mask & ~df['particle'].isin(picked)]
        if not len(sub):
            if verbose:
                print(f'{"-":>6}  {name}: none in this run')
            continue
        sc = score.loc[sub.index]
        cand = sub[sc >= sc.quantile(1 - top)]
        p = int(rng.choice(cand['particle'].values))
        picked.append(p)
        if verbose:
            r = df.loc[df['particle'] == p].iloc[0]
            print(f'{p:>6}  {name:<30} settled {100 * r.settled_fraction:4.0f}% '
                  f'(first reef {100 * r.settled_first_reef:.0f}%, later reefs '
                  f'{100 * r.settled_elsewhere:.0f}%)')
    rest = df[~df['particle'].isin(picked)]['particle'].values
    while len(picked) < n and len(rest):
        p = int(rng.choice(rest))
        picked.append(p)
        rest = rest[rest != p]
        if verbose:
            print(f'{p:>6}  random')
    return picked


def plot_bundle_tracks(res, max_reefs=4, min_pct=1.0, figsize=None, title=None):
    """
    One panel per tracked bundle (run settle(..., track=ids) first). Each bundle starts as
    100% still drifting (bottom layer); settlement (by reef), other losses and mortality
    accumulate above it as the bundle empties. Reefs taking < min_pct % of the bundle are
    lumped together (and left out of the legend if too thin to see). Dotted line: competency.
    """
    tr = res.get('tracks')
    if tr is None:
        raise ValueError('No tracks in this result -- run settle(..., track=ids) first')
    k = len(tr['ids'])
    comp0 = res['params']['competency_days'][0]
    x_end = res['params']['age_max_h'] / 24
    greens = ['#00543C', '#009E73', '#5CC6A0', '#A8E0CB', '#D3EFE4']
    fig, axes = plt.subplots(k, 1, figsize=figsize or (9, 1.9 * k + 0.6), sharex=True, squeeze=False)

    for r, ax in enumerate(axes[:, 0]):
        a = tr['age_h'][r] / 24
        keep = np.where(np.isfinite(a) & tr['active'][r])[0]
        if len(keep) == 0:
            ax.set_visible(False)
            continue
        x = a[keep]
        s = tr['settled'][r, keep]
        reef = tr['reef'][r, keep]
        fate = tr['final_fate'][r]

        # settled by reef: largest first; reefs below min_pct lumped together
        tot = {rid: s[reef == rid].sum() for rid in np.unique(reef[s > 0])}
        order = sorted(tot, key=tot.get, reverse=True)
        major = [rid for rid in order if 100 * tot[rid] >= min_pct][:max_reefs]
        minor = [rid for rid in order if rid not in major]

        loss, labels, colours = [], [], []
        for j, rid in enumerate(major):
            loss.append(np.cumsum(np.where(reef == rid, s, 0.0)))
            labels.append(f"{tr['reef_labels'][rid]} ({100 * tot[rid]:.0f}%)")
            colours.append(greens[j % len(greens)])
        if minor:
            minor_tot = 100 * sum(tot[rid] for rid in minor)
            loss.append(np.cumsum(np.where(np.isin(reef, minor), s, 0.0)))
            colours.append(greens[-1])
            labels.append(f"{len(minor)} other reef{'s' if len(minor) > 1 else ''} ({minor_tot:.0f}%)"
                          if minor_tot >= 0.5 else '_nolegend_')
        lost = tr['lost'][r, keep]
        if lost.sum() > 0:
            loss.append(np.cumsum(lost))
            labels.append(FATE_LABELS.get(fate, fate))
            colours.append(FATE_COLOURS.get(fate, '0.5'))
        loss.append(np.cumsum(tr['mortality'][r, keep]))           # always present
        labels.append('Mortality')
        colours.append(FATE_COLOURS['mortality'])

        # still drifting at the bottom, so the bundle starts full and empties upwards
        drifting = np.clip(1 - np.sum(np.vstack(loss), axis=0), 0, 1)
        L = np.vstack([drifting] + loss)
        labels = [FATE_LABELS['drifting']] + labels
        colours = [FATE_COLOURS['drifting']] + colours

        # hold the final make-up flat to the age cap, so every panel spans the same ages
        if x[-1] < x_end:
            x = np.append(x, [x[-1] + 1e-6, x_end])
            L = np.column_stack([L, L[:, -1], L[:, -1]])
        ax.stackplot(x, 100 * L, colors=colours, labels=labels, edgecolor='white', linewidth=0.3)
        ax.axvline(comp0, color='0.3', ls=':', lw=0.8)

        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel('% of bundle', fontsize=8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        n_main = len(major)
        extra = f' (+{len(minor)} minor)' if minor else ''
        ax.set_title(f"Particle {tr['ids'][r]} · Source: {tr['origin'][r]} · Settled "
                     f"{100 * s.sum():.0f}% at {n_main} reef{'s' if n_main != 1 else ''}{extra}",
                     fontsize=9, loc='left', pad=4)
        h, l = ax.get_legend_handles_labels()
        hl = [(hh, ll) for hh, ll in zip(h, l) if not ll.startswith('_')]
        ax.legend([hh for hh, _ in hl][::-1], [ll for _, ll in hl][::-1],
                  loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
        ax.tick_params(labelsize=8)

    axes[-1, 0].set_xlabel('Larval age (days)')
    axes[-1, 0].set_xlim(0, x_end)
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha='left')
    fig.text(0.01, 0.005, f'Dotted line: competency begins ({comp0:g} d)', fontsize=7, color='0.35')
    top = 1 - (0.9 if title else 0.4) / fig.get_figheight()
    fig.subplots_adjust(left=0.08, right=0.72, top=top, bottom=0.9 / fig.get_figheight(),
                        hspace=0.55)
    return fig, axes

# ------------------------------------------------------------------
# Reef-level metrics and GIS export (QGIS)
# ------------------------------------------------------------------
def reef_metrics(res):
    """
    One row per reef (as source and/or sink). Larvae are in particle-equivalents.
      released    particles released from the reef
      src_pct     % of the reef's released larvae that settled anywhere (source strength)
      self_pct    % of the reef's released larvae that settled back on itself (self-recruitment)
      export      larvae it supplied to OTHER reefs
      received    larvae that settled on it, from all sources
      import      larvae it received from OTHER reefs
      recv_share  % of all settlement in the run that landed on this reef (sink strength)
      n_sinks     number of reefs it supplied;  n_srcs  number of reefs that supplied it
      ss_index    (export - import) / (export + import): +1 pure source, -1 pure sink, 0 balanced
    """
    counts, released = res['counts'], res['released']
    reefs = list(dict.fromkeys(list(counts.index) + list(counts.columns)))
    self_rec = pd.Series({r: counts.at[r, r] if (r in counts.index and r in counts.columns) else 0.0
                          for r in reefs})
    out_tot = counts.sum(axis=1).reindex(reefs).fillna(0.0)
    in_tot = counts.sum(axis=0).reindex(reefs).fillna(0.0)
    rel = released.reindex(reefs)
    export = out_tot - self_rec
    imp = in_tot - self_rec
    df = pd.DataFrame({
        'released': rel,
        'src_pct': 100 * out_tot / rel.replace(0, np.nan),
        'self_pct': 100 * self_rec / rel.replace(0, np.nan),
        'export': export,
        'received': in_tot,
        'import': imp,
        'recv_share': 100 * in_tot / max(in_tot.sum(), 1e-12),
        'n_sinks': (counts > 0).sum(axis=1).reindex(reefs).fillna(0).astype(int),
        'n_srcs': (counts > 0).sum(axis=0).reindex(reefs).fillna(0).astype(int),
        'ss_index': ((export - imp) / (export + imp).replace(0, np.nan)),
    })
    df.index.name = 'reef'
    return df


def _reef_geometries(dest_reefs, name_col='GBR_NAME', id_col='GBR_ID'):
    """Reef polygons keyed by the same labels settle() uses (duplicate labels dissolved)."""
    g = dest_reefs.to_crs(epsg=4326).copy()
    g['reef'] = _reef_labels(g, name_col, id_col).values
    keep = ['reef', 'geometry'] + ([id_col] if id_col in g.columns else []) + \
           (['Area_HA'] if 'Area_HA' in g.columns else [])
    g = g[keep]
    agg = {c: 'first' for c in keep if c not in ('reef', 'geometry')}
    if 'Area_HA' in agg:
        agg['Area_HA'] = 'sum'
    return g.dissolve(by='reef', aggfunc=agg).reset_index()


def export_reef_metrics(res, dest_reefs, path, name_col='GBR_NAME', id_col='GBR_ID', only_active=True):
    """
    Write reef polygons with the reef_metrics() columns for QGIS. Use a .gpkg path (keeps
    full field names); .shp works too but field names are cut to 10 characters.
    only_active=True keeps reefs that released or received larvae.
    """
    geo = _reef_geometries(dest_reefs, name_col, id_col)
    met = reef_metrics(res).reset_index()
    out = geo.merge(met, on='reef', how='inner' if only_active else 'left')
    if 'Area_HA' in out.columns:
        out['recv_per_km2'] = out['received'] / (out['Area_HA'] / 100).replace(0, np.nan)
    out['run'] = res['info']['label']
    out.to_file(path)
    print(f"wrote {len(out)} reefs -> {path}")
    return out


def export_links(res, dest_reefs, path, min_pct=0.1, max_links=300, include_self=False,
                 name_col='GBR_NAME', id_col='GBR_ID'):
    """
    Write source -> sink connections as lines (source reef to sink reef) for QGIS.
      pct_src   % of the source reef's released larvae that settled on the sink
      larvae    particle-equivalents settled
    Keeps links with pct_src >= min_pct, strongest first, at most max_links.
    """
    from shapely.geometry import LineString
    geo = _reef_geometries(dest_reefs, name_col, id_col).set_index('reef')
    pts = geo.geometry.representative_point()
    counts, released = res['counts'], res['released']
    long = counts.stack().rename('larvae').reset_index()
    long.columns = ['source', 'sink', 'larvae']
    long = long[long['larvae'] > 0]
    long['pct_src'] = 100 * long['larvae'] / long['source'].map(released).replace(0, np.nan)
    if not include_self:
        long = long[long['source'] != long['sink']]
    long = long[(long['pct_src'] >= min_pct) & long['source'].isin(pts.index) & long['sink'].isin(pts.index)]
    long = long.sort_values('pct_src', ascending=False).head(max_links)
    geoms = [LineString([pts[a], pts[b]]) for a, b in zip(long['source'], long['sink'])]
    out = gpd.GeoDataFrame(long.reset_index(drop=True), geometry=geoms, crs='EPSG:4326')
    out['run'] = res['info']['label']
    out.to_file(path)
    print(f"wrote {len(out)} links (>= {min_pct}% of source) -> {path}")
    return out


def export_metrics_difference(res_a, res_b, dest_reefs, path, name_col='GBR_NAME', id_col='GBR_ID'):
    """
    Reef polygons with (b - a) differences, e.g. res_a = November, res_b = December:
      d_src_pct, d_self_pct, d_recv_share, d_ss_index   plus each run's own values.
    Positive = larger in run b.
    """
    a, b = reef_metrics(res_a), reef_metrics(res_b)
    cols = ['src_pct', 'self_pct', 'recv_share', 'ss_index']
    both = a[cols].add_suffix('_a').join(b[cols].add_suffix('_b'), how='outer')
    for c in cols:
        both[f'd_{c}'] = both[f'{c}_b'].fillna(0) - both[f'{c}_a'].fillna(0)
    geo = _reef_geometries(dest_reefs, name_col, id_col)
    out = geo.merge(both.reset_index(), on='reef', how='inner')
    out['runs'] = f"{res_b['info']['label']} minus {res_a['info']['label']}"
    out.to_file(path)
    print(f"wrote {len(out)} reefs -> {path}")
    return out
