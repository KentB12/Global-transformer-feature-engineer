"""
Intermittency-aware features for the Global Transformer (missing-aware version).

Missing (NaN) is NEVER treated as zero. Only genuine observed zeros count as zero production.
All features at step s use only y[0..s], so there is no future leakage.

  zero_run     : consecutive OBSERVED zeros ending at s (a missing quarter resets it to 0,
                 because it is not a confirmed zero). Raw count here; the model script
                 caps it at 12 and scales it to 0-1.
  zero_ratio4  : observed zeros / observed quarters in the last 4 (0 if none observed)
  trans_rate4  : zero<->nonzero changes between adjacent OBSERVED pairs / number of such pairs

Also included: the Missing-History Window Filtering rule (has_observed_history).
"""
import numpy as np
import pandas as pd

FEATURE_COLS = ["zero_run", "zero_ratio4", "trans_rate4"]


def intermittency_features(y, window=4):
    """y: 1-D array of production values (NaN = missing).
    Returns array [T, 3] = zero_run, zero_ratio4, trans_rate4."""
    y = np.asarray(y, dtype=float)
    obs = np.isfinite(y)
    z = ((y == 0) & obs).astype(float)                 # genuine zeros only
    out = np.zeros((len(y), 3))
    run = 0
    for s in range(len(y)):
        run = run + 1 if z[s] else 0
        lo = max(0, s - window + 1)
        zw, ow = z[lo: s + 1], obs[lo: s + 1]          # only past + current values
        n_obs = ow.sum()
        valid = ow[1:] & ow[:-1]                       # adjacent pairs where both are observed
        out[s, 0] = run
        out[s, 1] = zw.sum() / n_obs if n_obs > 0 else 0.0
        out[s, 2] = (np.abs(np.diff(zw)) * valid).sum() / valid.sum() if valid.sum() > 0 else 0.0
    return out


def has_observed_history(history):
    """Missing-History Window Filtering rule (deterministic, nothing is learned).
    True  -> keep the window: at least one observed value (a genuine zero counts as observed).
    False -> drop the window: all quarters are missing (NaN)."""
    return bool(np.isfinite(np.asarray(history, dtype=float)).any())


def add_features(df, group_cols=("crop", "province"), time_col="t", value_col="y"):
    """Long-format panel -> same panel with the three feature columns added.
    Each crop-province series is processed separately and in time order."""
    df = df.sort_values(list(group_cols) + [time_col]).copy()
    parts = []
    for _, g in df.groupby(list(group_cols), sort=False):
        f = intermittency_features(g[value_col].values)
        parts.append(pd.DataFrame(f, index=g.index, columns=FEATURE_COLS))
    return df.join(pd.concat(parts))


def check_no_leakage(y, t, window=4):
    """Features at step t must not change if values AFTER t are altered."""
    y = np.asarray(y, dtype=float)
    a = intermittency_features(y, window)[t]
    y2 = y.copy(); y2[t + 1:] = np.random.rand(len(y2) - t - 1) * 100
    b = intermittency_features(y2, window)[t]
    return np.allclose(a, b)


if __name__ == "__main__":
    nan = np.nan
    # --- no missing values: same behavior as before ---
    y = [0, 0, 0, 5, 0, 7, 8, 0, 0, 0]
    f = intermittency_features(y)
    print(pd.DataFrame(f, columns=FEATURE_COLS).assign(y=y).round(3))
    assert np.allclose(f[5], [0, 0.5, 1.0])
    assert np.allclose(f[9], [3, 0.75, 1 / 3])
    assert all(check_no_leakage(y, t) for t in range(len(y)))

    # --- missing-aware: 100, NaN, 0, NaN, 50 ---
    m = intermittency_features([100, nan, 0, nan, 50])
    assert m[1, 0] == 0                       # missing is not a zero -> run is 0
    assert m[2, 0] == 1                       # genuine zero starts a run
    assert m[3, 0] == 0                       # missing after a zero resets the run
    assert np.isclose(m[3, 1], 0.5)           # window [100,NaN,0,NaN]: 1 zero / 2 observed
    assert np.isclose(m[3, 2], 0.0)           # no adjacent observed pair in the window -> 0
    # missing must not be counted as zero anywhere
    assert intermittency_features([nan, nan, nan, nan])[:, 1].sum() == 0
    assert intermittency_features([0, 0, 0, 0])[3, 0] == 4

    # --- Missing-History Window Filtering rule ---
    assert has_observed_history([0] * 12)                  # 12 genuine zeros  -> keep
    assert not has_observed_history([nan] * 12)            # 12 missing        -> drop
    assert has_observed_history([100, nan, 0, nan, 50])    # mixed             -> keep
    # leakage check also holds with missing values
    yy = [3, nan, 0, 0, nan, 5, 0, nan, 0, 0]
    assert all(check_no_leakage(yy, t) for t in range(len(yy)))
    print("\nAll checks passed (including missing-aware and history-filter checks).")