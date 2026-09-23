"""
Post-processing of saved results (no model training):
  * episode-level bootstrap CIs for window-level macro-F1 of every feature-set variant
  * streaming / cost metrics with CIs for every method (primary site)
  * sensitivity of the policy ranking to the (illustrative) cost matrix and class prior
  * mean +/- sd across the three independent sites
Writes results/summary.json
"""
import json, pickle, glob
import numpy as np
from sklearn.metrics import f1_score
from oilfield_dt import pipeline as pl
from oilfield_dt.util import jsonable, log

SITES = [1, 2, 3]
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- window-level bootstrap
def macro_f1_from_conf(C):
    tp = np.diag(C).astype(float)
    p = C.sum(0); r = C.sum(1)
    f1 = np.where(p + r > 0, 2 * tp / np.maximum(p + r, 1), 0.0)
    return f1.mean(), f1


def window_ci(site, names, n_boot=500):
    z = np.load(f"results/preds_site{site}.npz")
    out = {}
    for lab_key in ("y", "y_all"):
        y = z[lab_key]; eid = z["eid"]; n_ep = eid.max() + 1
        for name in names:
            pred = z[f"pred_{name}"]
            m = y >= 0
            conf = np.zeros((n_ep, 4, 4))
            for k in range(n_ep):
                s = m & (eid == k)
                np.add.at(conf[k], (y[s], pred[s]), 1)
            tot = conf.sum(0)
            f_all, _ = macro_f1_from_conf(tot)
            bs = []
            for _ in range(n_boot):
                idx = rng.integers(0, n_ep, n_ep)
                bs.append(macro_f1_from_conf(conf[idx].sum(0))[0])
            out.setdefault(name, {})[lab_key] = dict(
                macro_f1=float(f_all), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                accuracy=float(np.trace(tot) / tot.sum()))
    return out


# ---------------------------------------------------------------- method definitions
METHODS = {
    "Proposed (argmax)":                 ("proposed", "argmax"),
    "Proposed (min-cost)":               ("proposed", "bayes"),
    "Proposed (min-cost + review)":      ("proposed", "bayes_review"),
    "Proposed (min-cost + review, decide 6 h later)": ("proposed", "bayes_review@settled"),
    "Physics twin only (argmax)":        ("arr_only", "argmax"),
    "Data-driven twin only (argmax)":    ("dd_only", "argmax"),
    "No twin, raw signals (argmax)":     ("raw_only", "argmax"),
    "Generic detector -> maintenance":   ("generic", "fixed:maintenance"),
    "Generic detector -> escalate":      ("generic_esc", "fixed:escalate"),
    "Attack-only detector -> escalate":  ("attack_only", "fixed:escalate"),
    "Fault-only detector -> maintenance": ("fault_only", "fixed:maintenance"),
}
KEYS = ["cost_equal", "cost_deploy", "attack_escalated", "attack_missed_or_benign",
        "false_escalation_rate", "fa_per_day", "detect_fault", "detect_environment", "detect_attack",
        "attr_fault", "attr_environment", "attr_attack", "review_rate_anom"]


def stream_table(site, n_boot=200):
    tabs = pickle.load(open(f"results/tables_site{site}.pkl", "rb"))
    out = {}
    for label, (tab_name, pol) in METHODS.items():
        tab = tabs[tab_name]
        pol, key = (pol.split("@")[0], "p_settled") if "@" in pol else (pol, "p_alarm")
        pf = pl.make_policy(pol)
        m = pl.table_metrics(tab, pf, key=key)
        ci = pl.bootstrap(tab, pf, KEYS, n=n_boot, seed=1, key=key)
        out[label] = dict(metrics=m, ci=ci)
    return out


# ---------------------------------------------------------------- cost sensitivity
def expected_cost(tab, policy_kind, C, prior):
    key = "p_settled" if "@" in policy_kind else "p_alarm"
    pf = pl.make_policy(policy_kind.split("@")[0], C)
    m = pl.table_metrics(tab, pf, C=C, prior=prior, key=key)
    return m["cost_deploy"], m["cost_equal"]


def sensitivity(site):
    tabs = pickle.load(open(f"results/tables_site{site}.pkl", "rb"))
    cands = {"Proposed (argmax)": ("proposed", "argmax"),
             "Proposed (min-cost)": ("proposed", "bayes"),
             "Proposed (min-cost + review)": ("proposed", "bayes_review"),
             "Proposed (min-cost + review, 6 h later)": ("proposed", "bayes_review@settled"),
             "Generic -> maintenance": ("generic", "fixed:maintenance"),
             "Generic -> escalate": ("generic_esc", "fixed:escalate"),
             "Attack-only -> escalate": ("attack_only", "fixed:escalate"),
             "Fault-only -> maintenance": ("fault_only", "fixed:maintenance")}
    grid = {}
    for miss in (20, 60, 200, 600):
        for fe_ in (2, 6, 20):
            for r in (0.5, 0.8, 0.95, 1.0):
                C = pl.scaled_cost(miss, fe_, r)
                for pa in (0.005, 0.02, 0.10):
                    pr = np.array([1 - 0.10 - pa, 0.05, 0.05, pa]); pr[0] = 1 - pr[1:].sum()
                    row = {}
                    for label, (tn, pol) in cands.items():
                        row[label] = expected_cost(tabs[tn], pol, C, pr)[0]
                    grid[f"miss={miss}|fe={fe_}|r={r}|pa={pa}"] = row
    # best-policy tables
    best = {}
    for k, row in grid.items():
        best[k] = min(row, key=row.get)
    return dict(grid=grid, best=best)


def main():
    res = {}
    names = list(pl.FEATURE_SETS)
    log("window CIs")
    res["window_ci_site1"] = window_ci(1, names)
    res["window_ci_by_site"] = {s: window_ci(s, ["proposed"], n_boot=200)["proposed"] for s in SITES}
    log("stream tables")
    res["stream_site1"] = stream_table(1)
    res["stream_by_site"] = {}
    for s in SITES:
        tabs = pickle.load(open(f"results/tables_site{s}.pkl", "rb"))
        res["stream_by_site"][s] = {
            label: pl.table_metrics(tabs[tn], pl.make_policy(pol.split("@")[0]),
                                    key="p_settled" if "@" in pol else "p_alarm")
            for label, (tn, pol) in METHODS.items()}
    log("sensitivity")
    res["sensitivity_site1"] = sensitivity(1)
    # main JSON per site (window metrics incl. calibration) -> pass through
    for s in SITES:
        res[f"main_site{s}"] = json.load(open(f"results/main_site{s}.json"))
    res["aux_site1"] = json.load(open("results/aux_site1.json"))
    json.dump(jsonable(res), open("results/summary.json", "w"), indent=1)
    log("summary saved")


if __name__ == "__main__":
    main()
