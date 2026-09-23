"""
Models, calibration, streaming evaluation and operator-cost analysis.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score, confusion_matrix, log_loss
from .sim import STEPS_PER_HOUR, STEPS_PER_DAY, CLASSES

# ---------------------------------------------------------------- feature sets
FEATURE_SETS = {
    "proposed":   ["arr", "dd", "inst", "cov", "topo", "replay", "ctx"],
    "arr_only":   ["arr", "inst", "cov", "topo", "replay", "ctx"],
    "dd_only":    ["dd", "inst", "cov", "replay", "ctx"],
    "raw_only":   ["raw", "inst", "replay", "ctx"],
    "no_cov":     ["arr", "dd", "inst", "topo", "replay", "ctx"],
    "no_topo":    ["arr", "dd", "inst", "cov", "replay", "ctx"],
    "no_replay":  ["arr", "dd", "inst", "cov", "topo", "ctx"],
    "no_inst":    ["arr", "dd", "cov", "topo", "replay", "ctx"],
}


def cols(ds, name):
    return np.where(np.isin(ds["groups"], FEATURE_SETS[name]))[0]


# ---------------------------------------------------------------- models
HGB_KW = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=24, l2_regularization=1.0,
              min_samples_leaf=30, early_stopping=False, random_state=0)


def fit_multiclass(ds, c, classes=(0, 1, 2, 3)):
    m = np.isin(ds["y"], classes)
    clf = HistGradientBoostingClassifier(**HGB_KW).fit(ds["F"][m][:, c], ds["y"][m])
    return clf


def calibrate(clf, ds, c, classes=(0, 1, 2, 3), method="isotonic"):
    m = np.isin(ds["y"], classes)
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method)
    cal.fit(ds["F"][m][:, c], ds["y"][m])
    return cal


def fit_binary(ds, c, pos_classes, neg_classes=(0,)):
    """Detector trained on normal vs. selected anomaly classes only."""
    m = np.isin(ds["y"], list(pos_classes) + list(neg_classes))
    y = np.isin(ds["y"][m], pos_classes).astype(int)
    return HistGradientBoostingClassifier(**HGB_KW).fit(ds["F"][m][:, c], y)


def proba_by_episode(model, ds, c, n_ep, binary=False, n_cls=4):
    """Return list of (n_windows, n_cls) probability arrays (binary -> [1-p, p] in cols 0/1)."""
    P = model.predict_proba(ds["F"][:, c])
    out = []
    for k in range(n_ep):
        out.append(P[ds["eid"] == k])
    return out


# ---------------------------------------------------------------- window-level metrics
def ece_top(P, y, bins=10):
    conf = P.max(1); pred = P.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def _wm(P, ds, ep_meta, key):
    m = ds[key] >= 0
    y = ds[key][m]; Pm = P[m]; Pm = Pm / Pm.sum(1, keepdims=True); pred = Pm.argmax(1)
    out = dict(
        n_windows=int(m.sum()), accuracy=float((pred == y).mean()),
        macro_f1=float(f1_score(y, pred, average="macro")),
        f1_per_class=[float(v) for v in f1_score(y, pred, average=None, labels=[0, 1, 2, 3])],
        ece=float(ece_top(Pm, y)),
        brier=float(np.mean(np.sum((Pm - np.eye(4)[y]) ** 2, 1))),
        logloss=float(log_loss(y, np.clip(Pm, 1e-6, 1), labels=[0, 1, 2, 3])),
        confusion=confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
    )
    modes = np.array([ep_meta[k]["mode"] for k in ds["eid"]])
    rec = {}
    for md in sorted(set(modes[m & (ds[key] > 0)])):
        mm = m & (modes == md) & (ds[key] > 0)
        rec[md] = dict(n=int(mm.sum()), recall=float((Pm_all_pred(P, mm) == ds[key][mm]).mean()))
    out["per_mode_recall"] = rec
    return out


def Pm_all_pred(P, mask):
    return P[mask].argmax(1)


def window_metrics(P, ds, ep_meta):
    """Metrics on manifest windows (primary) and on all steady windows incl. latent (secondary)."""
    return dict(manifest=_wm(P, ds, ep_meta, "y"), all_steady=_wm(P, ds, ep_meta, "y_all"))


# ---------------------------------------------------------------- streaming alarm rule
def first_alarm(score, tau, k, start=0):
    """First index i >= start+k-1 such that score[i-k+1..i] > tau (runs restart at `start`)."""
    run = 0
    for i in range(start, len(score)):
        run = run + 1 if score[i] > tau else 0
        if run >= k:
            return i
    return -1


def calibrate_tau(scores_normal, k, target_fa_per_day, days_per_ep=3.0):
    """Smallest threshold whose latched false-alarm rate on normal episodes <= target."""
    allv = np.concatenate(scores_normal)
    grid = np.unique(np.concatenate([np.quantile(allv, np.linspace(0.5, 0.99999, 400)),
                                     [allv.max() + 1e-9]]))
    n_days = days_per_ep * len(scores_normal)
    for tau in grid:
        n_fa = sum(first_alarm(s, tau, k) >= 0 for s in scores_normal)
        if n_fa / n_days <= target_fa_per_day:
            return float(tau)
    return float(grid[-1])


# ---------------------------------------------------------------- costs
ACTIONS = ["ignore", "maintenance", "compensate", "escalate", "review"]
COST = np.array([          # rows: truth [normal, fault, env, attack]; cols: actions
    [0.0,  4.0, 2.0,  6.0, 3.0],
    [10.0, 1.0, 3.0,  8.0, 3.0],
    [6.0,  4.0, 1.0,  6.0, 3.0],
    [60.0, 50.0, 50.0, 2.0, 6.0],     # review resolves an attack w.p. 0.95: 3 + 0.05*60
])
NONNORMAL_TO_ACTION = {1: 1, 2: 2, 3: 3}
PRIOR_DEPLOY = np.array([0.88, 0.05, 0.05, 0.02])     # illustrative per-episode prior


def scaled_cost(miss_attack=60.0, false_escalate=6.0, review_eff=0.95):
    """Cost matrix with attack-miss cost, false-escalation cost and analyst-review
    effectiveness (probability that review correctly resolves an attack) rescaled."""
    C = COST.copy()
    C[3, [0, 1, 2]] = miss_attack
    C[3, 3] = 2.0
    C[3, 4] = 3.0 + (1.0 - review_eff) * miss_attack
    C[0, 3] = false_escalate
    C[1, 3] = false_escalate + 2.0
    C[2, 3] = false_escalate
    return C


def act_argmax(p):
    q = np.asarray(p, float).copy(); q[0] = -1
    return NONNORMAL_TO_ACTION[int(q.argmax())]


def act_bayes(p, C=COST, allow_review=True):
    exp = np.asarray(p) @ C
    if not allow_review:
        exp = exp[:4]
    return int(np.argmin(exp))


def make_policy(kind, C=COST):
    if kind == "argmax":
        return lambda p: act_argmax(p)
    if kind == "bayes":
        return lambda p: act_bayes(p, C, allow_review=False)
    if kind == "bayes_review":
        return lambda p: act_bayes(p, C, allow_review=True)
    if kind.startswith("fixed:"):
        a = ACTIONS.index(kind.split(":")[1])
        return lambda p: a
    raise ValueError(kind)


# ---------------------------------------------------------------- per-episode outcome table
def episode_table(P_list, ends_list, ep_meta, score_fn, tau, k=2, settle=5):
    """One row per test episode.  Detection is searched only in windows ending at/after
    onset; alarms before onset are counted as false alarms (pre_alarm)."""
    n = len(P_list); nc = P_list[0].shape[1]
    tab = dict(label=np.zeros(n, int), mode=np.empty(n, object), onset=np.zeros(n, int),
               alarm=np.zeros(n, bool), delay_h=np.full(n, np.nan),
               pre_alarm=np.zeros(n, bool), pre_days=np.zeros(n),
               p_alarm=np.zeros((n, nc)), p_settled=np.zeros((n, nc)))
    for e, (P, ends, meta) in enumerate(zip(P_list, ends_list, ep_meta)):
        s = score_fn(P); lab, onset = meta["label"], meta["onset"]
        tab["label"][e] = lab; tab["mode"][e] = meta["mode"]; tab["onset"][e] = onset
        if lab == 0:
            a = first_alarm(s, tau, k, 0); tab["pre_days"][e] = len(s) and (ends[-1] + 1) / STEPS_PER_DAY
            tab["pre_alarm"][e] = a >= 0
            if a >= 0:
                tab["alarm"][e] = True
                tab["p_alarm"][e] = P[max(0, a - k + 1):a + 1].mean(0)
                tab["p_settled"][e] = P[a:min(len(P), a + settle + 1)].mean(0)
            continue
        i0 = int(np.searchsorted(ends, onset))                    # first window with end >= onset
        pre = first_alarm(s[:i0], tau, k, 0) if i0 > 0 else -1
        tab["pre_alarm"][e] = pre >= 0
        tab["pre_days"][e] = max(onset - 0, 0) / STEPS_PER_DAY
        a = first_alarm(s, tau, k, i0)
        if a >= 0:
            tab["alarm"][e] = True
            tab["delay_h"][e] = (ends[a] - onset) / STEPS_PER_HOUR
            tab["p_alarm"][e] = P[max(i0, a - k + 1):a + 1].mean(0)
            tab["p_settled"][e] = P[a:min(len(P), a + settle + 1)].mean(0)
    return tab


def table_actions(tab, policy, key="p_alarm"):
    a = np.zeros(len(tab["label"]), int)
    for e in range(len(a)):
        a[e] = policy(tab[key][e]) if tab["alarm"][e] else 0
    return a


def table_metrics(tab, policy, C=COST, prior=PRIOR_DEPLOY, idx=None, key="p_alarm"):
    """Aggregate metrics for a subset `idx` of episodes (used for bootstrap)."""
    if idx is None:
        idx = np.arange(len(tab["label"]))
    lab = tab["label"][idx]; alarm = tab["alarm"][idx]
    act = table_actions({k: (v[idx] if hasattr(v, "__len__") else v) for k, v in tab.items()}, policy, key)
    cost = C[lab, act]
    out = dict()
    normal = lab == 0
    fa = tab["pre_alarm"][idx]
    days = tab["pre_days"][idx]
    out["fa_per_day"] = float(fa.sum() / days.sum()) if days.sum() > 0 else np.nan
    pc = {}
    for c in range(4):
        m = lab == c
        pc[c] = float(cost[m].mean()) if m.any() else np.nan
        if c > 0:
            out[f"detect_{CLASSES[c]}"] = float(alarm[m].mean()) if m.any() else np.nan
            d = tab["delay_h"][idx][m & alarm]
            out[f"delay_med_{CLASSES[c]}"] = float(np.median(d)) if len(d) else np.nan
            out[f"delay_p90_{CLASSES[c]}"] = float(np.percentile(d, 90)) if len(d) else np.nan
            if len(policy_attr := tab["p_alarm"][idx][m & alarm]) and tab["p_alarm"].shape[1] == 4:
                ok = [act_argmax(p) == NONNORMAL_TO_ACTION[c] for p in policy_attr]
                out[f"attr_{CLASSES[c]}"] = float(np.mean(ok))
                oks = [act_argmax(p) == NONNORMAL_TO_ACTION[c] for p in tab["p_settled"][idx][m & alarm]]
                out[f"attr_settled_{CLASSES[c]}"] = float(np.mean(oks))
    out["cost_by_class"] = pc
    out["cost_equal"] = float(np.nanmean(list(pc.values())))
    out["cost_deploy"] = float(np.nansum([prior[c] * pc[c] for c in range(4)]))
    # attack handling
    m = lab == 3
    out["attack_escalated"] = float(np.mean(act[m] == 3)) if m.any() else np.nan
    out["attack_missed_or_benign"] = float(np.mean(np.isin(act[m], [0, 1, 2]))) if m.any() else np.nan
    m = (lab != 3)
    out["false_escalation_rate"] = float(np.mean(act[m] == 3)) if m.any() else np.nan
    out["review_rate_anom"] = float(np.mean(act[lab > 0] == 4)) if (lab > 0).any() else np.nan
    out["actions"] = {c: np.bincount(act[lab == c], minlength=5).tolist() for c in range(4)}
    return out


def bootstrap(tab, policy, keys, C=COST, prior=PRIOR_DEPLOY, n=300, seed=0, key="p_alarm"):
    rng = np.random.default_rng(seed)
    N = len(tab["label"]); vals = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        m = table_metrics(tab, policy, C, prior, idx, key)
        for k in keys:
            vals[k].append(m.get(k, np.nan))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in vals.items()}
