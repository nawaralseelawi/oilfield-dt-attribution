"""Paired episode-bootstrap of macro-F1 differences (variant minus proposed), per site and pooled."""
import json
import numpy as np

rng = np.random.default_rng(123)
VARS = ["arr_only", "dd_only", "raw_only", "no_cov", "no_topo", "no_replay", "no_inst"]


def conf_by_ep(z, key, name):
    y, eid, pred = z[key], z["eid"], z[f"pred_{name}"]
    n = eid.max() + 1
    C = np.zeros((n, 4, 4))
    m = y >= 0
    for k in range(n):
        s = m & (eid == k)
        np.add.at(C[k], (y[s], pred[s]), 1)
    return C


def mf1(C):
    tp = np.diag(C).astype(float); p = C.sum(0); r = C.sum(1)
    return np.where(p + r > 0, 2 * tp / np.maximum(p + r, 1), 0.0).mean()


out = {}
for key, tag in (("y", "manifest"), ("y_all", "all_steady")):
    out[tag] = {}
    Z = {s: np.load(f"results/preds_site{s}.npz") for s in (1, 2, 3)}
    base = {s: conf_by_ep(Z[s], key, "proposed") for s in Z}
    for v in VARS:
        cv = {s: conf_by_ep(Z[s], key, v) for s in Z}
        row = {}
        for s in Z:
            d0 = mf1(cv[s].sum(0)) - mf1(base[s].sum(0))
            n = len(base[s]); ds = []
            for _ in range(2000):
                i = rng.integers(0, n, n)
                ds.append(mf1(cv[s][i].sum(0)) - mf1(base[s][i].sum(0)))
            row[f"site{s}"] = dict(diff=float(d0), ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))])
        # pooled over sites: resample episodes within each site, sum confusions across sites
        d0 = mf1(sum(cv[s].sum(0) for s in Z)) - mf1(sum(base[s].sum(0) for s in Z)); ds = []
        for _ in range(2000):
            cb = cvv = 0
            for s in Z:
                n = len(base[s]); i = rng.integers(0, n, n)
                cb = cb + base[s][i].sum(0); cvv = cvv + cv[s][i].sum(0)
            ds.append(mf1(cvv) - mf1(cb))
        row["pooled"] = dict(diff=float(d0), ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                             excludes_zero=bool(np.percentile(ds, 2.5) > 0 or np.percentile(ds, 97.5) < 0))
        out[tag][v] = row
json.dump(out, open("results/paired_ablation.json", "w"), indent=1)
for tag in out:
    print(f"== {tag}: macro-F1 difference vs proposed (pooled over 3 sites, 95% paired bootstrap CI)")
    for v, r in out[tag].items():
        p = r["pooled"]
        print(f"  {v:10s} {p['diff']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]  {'excludes 0' if p['excludes_zero'] else 'includes 0'}   per-site: " + " ".join(f"{r[f'site{s}']['diff']:+.3f}" for s in (1, 2, 3)))
