"""
Global Transformer: baseline vs intermittency-aware feature-augmented.
Quarterly PSA spice-crop production, 12-quarter lookback, 1-step-ahead.

Train targets : 2013Q1-2022Q4   Val targets : 2023Q1-2024Q4   Test targets : 2025Q1-Q4
2026 Q1-Q2 are loaded but kept out of the main experiment.

Usage:  python global_transformer.py --xlsx 2E4EVCP3_2013-2026_combined_1.xlsx
"""
import argparse, copy, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

LOOKBACK = 12
FEATS = ["zero_run", "zero_ratio4", "trans_rate4"]


# ----------------------------------------------------------------------------
# 1. DATA LOADING  (wide PSA sheet -> long panel)
# ----------------------------------------------------------------------------
def load_panel(xlsx):
    raw = pd.read_excel(xlsx, sheet_name="Combined Data", header=None)
    years = raw.iloc[2, 2:].ffill()
    qlabels = raw.iloc[3, 2:]
    # keep only Quarter 1-4 columns (drop Semester / Annual)
    cols = {}
    for j, (yr, q) in enumerate(zip(years, qlabels), start=2):
        if isinstance(q, str) and q.startswith("Quarter"):
            cols[j] = (int(yr), int(q.split()[-1]))
    body = raw.iloc[4:].copy()
    body[0] = body[0].ffill()                       # crop name is only on the first row of block
    crop = body[0].astype(str)
    area = body[1].astype(str)

    is_top_crop = ~crop.str.startswith(".")         # drops nested duplicated crops (..Onion, Bermuda ...)
    is_province = area.str.startswith("....") & ~area.str.contains("City", case=False)
    body = body[is_top_crop & is_province]
    body["province"] = body[1].str.replace(".", "", regex=False).str.strip()
    body["crop"] = body[0].str.strip()

    recs = []
    for j, (yr, q) in cols.items():
        v = pd.to_numeric(body[j], errors="coerce")   # '..' (missing notation) -> NaN, real 0 stays 0
        recs.append(pd.DataFrame({"crop": body["crop"].values, "province": body["province"].values,
                                  "year": yr, "q": q, "y": v.values}))
    df = pd.concat(recs, ignore_index=True)
    df["t"] = (df["year"] - 2013) * 4 + (df["q"] - 1)      # 0 = 2013Q1
    return df


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (each feature at step s only uses y[<= s])
# ----------------------------------------------------------------------------
def intermittency_features(y):
    z = (np.nan_to_num(y, nan=0.0) == 0).astype(float)
    T = len(z)
    run, ratio, trans = np.zeros(T), np.zeros(T), np.zeros(T)
    r = 0
    for s in range(T):
        r = r + 1 if z[s] else 0
        run[s] = r
        w = z[max(0, s - 3): s + 1]
        ratio[s] = w.mean()
        trans[s] = (np.abs(np.diff(w)).sum() / (len(w) - 1)) if len(w) > 1 else 0.0
    return np.stack([run, ratio, trans], axis=1)


def build_series(df, t_max):
    """Returns arrays: Y[N,T] (raw, NaN = missing), F[N,T,3], ids, quarter-of-year[T]."""
    df = df[df["t"] <= t_max]
    piv = df.pivot_table(index=["crop", "province"], columns="t", values="y", aggfunc="first", dropna=False)
    piv = piv.reindex(columns=range(t_max + 1))
    keys = piv.index.to_frame(index=False)
    keys["crop_id"] = keys["crop"].astype("category").cat.codes
    keys["prov_id"] = keys["province"].astype("category").cat.codes
    Y = piv.values.astype(float)
    F = np.stack([intermittency_features(Y[i]) for i in range(len(Y))])
    F[..., 0] = np.minimum(F[..., 0], LOOKBACK) / LOOKBACK      # scale zero-run to [0,1]
    qoy = np.arange(t_max + 1) % 4
    return Y, F, keys, qoy


# ----------------------------------------------------------------------------
# 3. WINDOWING + SPLITS + LEAKAGE-SAFE SCALING
# ----------------------------------------------------------------------------
def make_windows(Y, F, keys, qoy, mu, sd, t_lo, t_hi, use_feats):
    """One sample per (series, target quarter t): inputs t-12..t-1, target t."""
    Yl = (np.log1p(np.nan_to_num(Y, nan=0.0)) - mu) / sd
    X, C, P, Q, y_t, raw_t, sid, tt = [], [], [], [], [], [], [], []
    for i in range(len(Y)):
        for t in range(max(t_lo, LOOKBACK), t_hi + 1):
            if np.isnan(Y[i, t]):
                continue                                   # skip missing targets
            sl = slice(t - LOOKBACK, t)
            x = Yl[i, sl][:, None]
            if use_feats:
                x = np.concatenate([x, F[i, sl]], axis=1)
            X.append(x); C.append(keys.crop_id[i]); P.append(keys.prov_id[i]); Q.append(qoy[sl])
            y_t.append(Yl[i, t]); raw_t.append(Y[i, t]); sid.append(i); tt.append(t)
    T = lambda a, d: torch.tensor(np.array(a), dtype=d)
    return dict(x=T(X, torch.float32), crop=T(C, torch.long), prov=T(P, torch.long), q=T(Q, torch.long),
                y=T(y_t, torch.float32), raw=np.array(raw_t), sid=np.array(sid), t=np.array(tt))


# ----------------------------------------------------------------------------
# 4. MODEL  (identical in both configs except n_in)
# ----------------------------------------------------------------------------
class GlobalTransformer(nn.Module):
    def __init__(self, n_in, n_crop, n_prov, d=64, heads=4, layers=2, ff=128, drop=0.1):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.crop = nn.Embedding(n_crop, d)
        self.prov = nn.Embedding(n_prov, d)
        self.qoy = nn.Embedding(4, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, crop, prov, q):
        h = self.inp(x) + self.qoy(q) + self.pos
        h = h + (self.crop(crop) + self.prov(prov)).unsqueeze(1)    # series identifiers
        h = self.enc(h)
        return self.head(h[:, -1]).squeeze(-1)


# ----------------------------------------------------------------------------
# 5. TRAIN / PREDICT / METRICS
# ----------------------------------------------------------------------------
def batches(D, bs, shuffle, dev):
    n = len(D["y"]); idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        yield [D[k][b].to(dev) for k in ("x", "crop", "prov", "q", "y")]


@torch.no_grad()
def predict_scaled(model, D, dev):
    model.eval()
    return torch.cat([model(x, c, p, q).cpu() for x, c, p, q, _ in batches(D, 1024, False, dev)]).numpy()


def to_raw(pred_scaled, mu, sd):
    return np.maximum(np.expm1(pred_scaled * sd + mu), 0.0)      # back to metric tons, no negatives


def fit(train, val, n_in, n_crop, n_prov, seed, dev, epochs=60, patience=8, lr=1e-3, bs=256):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = GlobalTransformer(n_in, n_crop, n_prov).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.HuberLoss(delta=1.0)
    best, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        model.train()
        for x, c, p, q, y in batches(train, bs, True, dev):
            opt.zero_grad(); loss_fn(model(x, c, p, q), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = np.mean((predict_scaled(model, val, dev) - val["y"].numpy()) ** 2)
        sched.step(v)
        if v < best - 1e-5: best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


def metrics(y, yh):
    e = y - yh
    den = np.abs(y).sum()
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                WMAPE=(np.abs(e).sum() / den * 100) if den > 0 else np.nan)   # undefined if all actual = 0


def evaluate(y, yh, sid, zero_share):
    out = {"overall": metrics(y, yh)}
    out["zero_actual"] = metrics(y[y == 0], yh[y == 0]) if (y == 0).any() else {}
    out["nonzero_actual"] = metrics(y[y > 0], yh[y > 0]) if (y > 0).any() else {}
    # intermittency groups by each series' zero share in the training period
    s = zero_share[sid]
    for name, m in {"zero_share<0.5": s < 0.5, "0.5-0.9": (s >= 0.5) & (s < 0.9), ">=0.9": s >= 0.9}.items():
        if m.any(): out[name] = metrics(y[m], yh[m])
    return out


def per_series_mae(y, yh, sid):
    return pd.DataFrame({"sid": sid, "ae": np.abs(y - yh)}).groupby("sid")["ae"].mean()


# ----------------------------------------------------------------------------
# 6. EXPERIMENT
# ----------------------------------------------------------------------------
def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_panel(a.xlsx)
    T_TRAIN_END, T_VAL_END, T_TEST_END = 39, 47, 51      # 2022Q4, 2024Q4, 2025Q4 (t=0 is 2013Q1)
    Y, F, keys, qoy = build_series(df, T_TEST_END)       # 2026 data excluded from main experiment
    print(f"series={len(Y)} crops={keys.crop_id.nunique()} provinces={keys.prov_id.nunique()} quarters={Y.shape[1]}")
    print(f"zero share = {(Y == 0).sum() / np.isfinite(Y).sum():.3f}")

    # scaling fitted on TRAIN period only
    tr = np.log1p(Y[:, :T_TRAIN_END + 1]); tr = tr[np.isfinite(tr)]
    mu, sd = tr.mean(), tr.std() + 1e-8
    zero_share = np.nanmean(np.where(np.isnan(Y[:, :T_TRAIN_END + 1]), np.nan, Y[:, :T_TRAIN_END + 1] == 0), axis=1)

    results, per_series = {}, {}
    for name, use_f in {"baseline": False, "feature_engineered": True}.items():
        mk = lambda lo, hi: make_windows(Y, F, keys, qoy, mu, sd, lo, hi, use_f)
        train, val, test = mk(0, T_TRAIN_END), mk(T_TRAIN_END + 1, T_VAL_END), mk(T_VAL_END + 1, T_TEST_END)
        n_in = train["x"].shape[-1]
        print(f"\n[{name}] n_in={n_in} train={len(train['y'])} val={len(val['y'])} test={len(test['y'])}")
        runs = []
        for seed in range(a.seeds):
            m = fit(train, val, n_in, keys.crop_id.max() + 1, keys.prov_id.max() + 1, seed, dev, a.epochs)
            yh = to_raw(predict_scaled(m, test, dev), mu, sd)
            r = evaluate(test["raw"], yh, test["sid"], zero_share)
            runs.append((r, per_series_mae(test["raw"], yh, test["sid"])))
            print(f"  seed {seed}: " + ", ".join(f"{k}={v:.3f}" for k, v in r["overall"].items()))
        results[name] = [r for r, _ in runs]
        per_series[name] = pd.concat([s for _, s in runs], axis=1).mean(axis=1)

    # aggregate across seeds
    print("\n=== TEST (2025) — mean ± std over seeds ===")
    rows = []
    for name, rl in results.items():
        for grp in rl[0]:
            for met in ("MAE", "RMSE", "WMAPE"):
                vals = [r[grp][met] for r in rl if grp in r and met in r[grp]]
                if vals: rows.append((name, grp, met, np.mean(vals), np.std(vals)))
    tab = pd.DataFrame(rows, columns=["config", "group", "metric", "mean", "std"])
    print(tab.pivot_table(index=["group", "metric"], columns="config", values="mean").round(3))
    tab.to_csv("results_summary.csv", index=False)

    # paired per-series comparison
    both = pd.concat([per_series["baseline"], per_series["feature_engineered"]], axis=1, keys=["base", "fe"]).dropna()
    diff = both["base"] - both["fe"]
    if (diff != 0).any():
        stat, p = wilcoxon(diff, alternative="two-sided")
        print(f"\nPer-series MAE: FE better in {(diff > 0).mean() * 100:.1f}% of {len(diff)} series; "
              f"Wilcoxon p={p:.4g}")
    both.to_csv("per_series_mae.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="2E4EVCP3_2013-2026_combined_1.xlsx")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    main(ap.parse_args())