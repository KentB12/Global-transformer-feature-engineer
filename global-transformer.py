"""
Standalone BASELINE Global Transformer (no intermittency-aware features).
Input per quarter: scaled log1p(production), observed-value mask (1 = observed, 0 = missing)
+ series/time identifiers only. Missing values are NEVER converted to zero.
Uses the same preprocessing as the feature-engineered file (missing-target skip,
Missing-History Window Filtering, observed mask) so the two are directly comparable.
12-quarter lookback -> 1-quarter-ahead forecast.

Train targets 2013Q1-2022Q4 | Val 2023Q1-2024Q4 | Test 2025Q1-Q4 (2026 excluded)

Usage: python global_transformer_baseline.py   (dataset path is set as the default)
"""
import argparse, copy, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LOOKBACK = 12
T_TRAIN_END, T_VAL_END, T_TEST_END = 39, 47, 51      # t=0 is 2013Q1


# ---------------- data ----------------
def load_panel(xlsx):
    raw = pd.read_excel(xlsx, sheet_name="Combined Data", header=None)
    years, qlabels = raw.iloc[2, 2:].ffill(), raw.iloc[3, 2:]
    cols = {j: (int(y), int(q.split()[-1])) for j, (y, q) in enumerate(zip(years, qlabels), start=2)
            if isinstance(q, str) and q.startswith("Quarter")}
    body = raw.iloc[4:].copy()
    body[0] = body[0].ffill()
    keep = ~body[0].astype(str).str.startswith(".") & body[1].astype(str).str.startswith("....") \
           & ~body[1].astype(str).str.contains("City", case=False)
    body = body[keep]
    crop = body[0].str.strip().values
    prov = body[1].str.replace(".", "", regex=False).str.strip().values
    recs = [pd.DataFrame({"crop": crop, "province": prov, "t": (yr - 2013) * 4 + q - 1,
                          "y": pd.to_numeric(body[j], errors="coerce").values})   # '..' -> NaN
            for j, (yr, q) in cols.items()]
    return pd.concat(recs, ignore_index=True)


def build_series(df):
    df = df[df["t"] <= T_TEST_END]
    piv = df.pivot_table(index=["crop", "province"], columns="t", values="y", aggfunc="first", dropna=False)
    piv = piv.reindex(columns=range(T_TEST_END + 1))
    keys = piv.index.to_frame(index=False)
    keys["crop_id"] = keys["crop"].astype("category").cat.codes
    keys["prov_id"] = keys["province"].astype("category").cat.codes
    return piv.values.astype(float), keys, np.arange(T_TEST_END + 1) % 4


def make_windows(Y, keys, qoy, mu, sd, t_lo, t_hi):
    """Builds one sample per (series, target quarter): inputs t-12..t-1, target t.
    Data preprocessing rules applied here (deterministic, nothing is learned):
      1. Missing target            -> sample skipped.
      2. Missing-History Window Filtering: if all 12 history quarters are NaN
         (no observed value), the sample is skipped. Genuine zeros count as
         observed values, so windows of 12 zeros or mixed windows are kept.
    Only individual windows are filtered; no series or raw rows are deleted."""
    dropped_no_target = dropped_no_history = 0
    Yl = (np.log1p(Y) - mu) / sd                       # NaN stays NaN (missing is not turned into zero)
    OBS = np.isfinite(Yl)                              # observed-value mask (True = observed, incl. genuine zeros)
    Yfill = np.where(OBS, Yl, 0.0)                     # tensor placeholder only (training mean); the mask tells the model it is missing
    X, C, P, Q, y_t, raw_t, sid = [], [], [], [], [], [], []
    for i in range(len(Y)):
        for t in range(max(t_lo, LOOKBACK), t_hi + 1):
            if np.isnan(Y[i, t]):
                dropped_no_target += 1
                continue
            sl = slice(t - LOOKBACK, t)
            history = Y[i, sl]
            if not np.isfinite(history).any():          # Missing-History Window Filtering
                dropped_no_history += 1
                continue
            X.append(np.concatenate([Yfill[i, sl][:, None], OBS[i, sl][:, None].astype(float)], axis=1)); C.append(keys.crop_id[i]); P.append(keys.prov_id[i])
            Q.append(qoy[sl]); y_t.append(Yl[i, t]); raw_t.append(Y[i, t]); sid.append(i)
    T = lambda a, d: torch.tensor(np.array(a), dtype=d)
    return dict(x=T(X, torch.float32), crop=T(C, torch.long), prov=T(P, torch.long), q=T(Q, torch.long),
                y=T(y_t, torch.float32), raw=np.array(raw_t), sid=np.array(sid),
                dropped_no_target=dropped_no_target, dropped_no_history=dropped_no_history)


# ---------------- model ----------------
class GlobalTransformer(nn.Module):
    def __init__(self, n_in, n_crop, n_prov, d=64, heads=4, layers=2, ff=128, drop=0.1):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.crop, self.prov, self.qoy = nn.Embedding(n_crop, d), nn.Embedding(n_prov, d), nn.Embedding(4, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, crop, prov, q):
        h = self.inp(x) + self.qoy(q) + self.pos + (self.crop(crop) + self.prov(prov)).unsqueeze(1)
        return self.head(self.enc(h)[:, -1]).squeeze(-1)


# ---------------- training ----------------
def batches(D, bs, shuffle, dev):
    n = len(D["y"]); idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        yield [D[k][b].to(dev) for k in ("x", "crop", "prov", "q", "y")]


@torch.no_grad()
def predict(model, D, dev):
    model.eval()
    return torch.cat([model(x, c, p, q).cpu() for x, c, p, q, _ in batches(D, 1024, False, dev)]).numpy()


def fit(train, val, n_crop, n_prov, dev, seed=0, epochs=60, patience=8, lr=1e-3, bs=256):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = GlobalTransformer(train["x"].shape[-1], n_crop, n_prov).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.HuberLoss(delta=1.0)
    best, state, bad = math.inf, None, 0
    for ep in range(epochs):
        model.train()
        for x, c, p, q, y in batches(train, bs, True, dev):
            opt.zero_grad(); loss_fn(model(x, c, p, q), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = np.mean((predict(model, val, dev) - val["y"].numpy()) ** 2)
        sched.step(v)
        print(f"epoch {ep + 1:02d}  val_mse(scaled)={v:.4f}")
        if v < best - 1e-5: best, state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state)
    return model


def metrics(y, yh):
    e = y - yh; den = np.abs(y).sum()
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                WMAPE=(np.abs(e).sum() / den * 100) if den > 0 else np.nan)


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Y, keys, qoy = build_series(load_panel(a.xlsx))
    print(f"series={len(Y)}  zero share={(Y == 0).sum() / np.isfinite(Y).sum():.3f}")
    tr = np.log1p(Y[:, :T_TRAIN_END + 1]); tr = tr[np.isfinite(tr)]
    mu, sd = tr.mean(), tr.std() + 1e-8                       # fitted on training period only
    mk = lambda lo, hi: make_windows(Y, keys, qoy, mu, sd, lo, hi)
    train, val, test = mk(0, T_TRAIN_END), mk(T_TRAIN_END + 1, T_VAL_END), mk(T_VAL_END + 1, T_TEST_END)
    for name, D in {"train": train, "val": val, "test": test}.items():
        print(f"{name}: kept={len(D['y'])}  dropped_no_target={D['dropped_no_target']}  "
              f"dropped_no_history={D['dropped_no_history']}")
    model = fit(train, val, keys.crop_id.max() + 1, keys.prov_id.max() + 1, dev, a.seed, a.epochs)
    yh = np.maximum(np.expm1(predict(model, test, dev) * sd + mu), 0.0)
    y = test["raw"]
    print("\nTEST 2025 (baseline)")
    for name, m in {"overall": slice(None), "zero actual": y == 0, "nonzero actual": y > 0}.items():
        print(f"{name:15s}", {k: round(float(v), 3) for k, v in metrics(y[m], yh[m]).items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=r"C:\Users\Kent\Downloads\2E4EVCP3_2013-2026_combined (1).xlsx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    main(ap.parse_args())