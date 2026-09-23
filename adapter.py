"""
Apply the attribution pipeline to real historian data.

SCOPE / LIMITS (read first)
  * Fixed topology: a 4-well pad + header, 15 channels in the order of oilfield_dt.sim.NAMES
    (P1..P4, T1..T4, Q1..Q4, Ph, Th, Qh) plus ambient temperature (and optionally a dust index).
    A different topology needs the ARR structure in oilfield_dt/twin.py to be edited.
  * The classifier is trained on SIMULATED data (train_bundle.py).  The twins are re-identified on
    YOUR normal-operation data, and the alarm threshold is re-calibrated on held-out normal days
    (variants B/C of experiment E4).  Attribution quality on real data is untested until you
    validate it on labelled incidents; treat outputs as decision support, not ground truth.
  * Normal-operation data must be free of known faults/attacks.  >= ~30 days recommended for the
    data-driven twin; the physics twin needs far less.

USAGE
  python train_bundle.py                                   # once; writes bundle.pkl
  python adapter.py --selftest                             # round-trip on a simulated CSV
  python adapter.py --csv hist.csv --map colmap.json \
        --normal 2026-01-01 2026-02-15 --gate 2026-02-15 2026-03-01 --score 2026-03-01 2026-04-01

colmap.json: {"time": "timestamp", "P1": "WHP_1", ..., "Qh": "HDR_FLOW", "Ta": "AMB_T", "Dust": "DUST"}
             (Dust optional -> zeros).  Sampling is resampled to 10 minutes (mean).
"""
import argparse, json, pickle, sys
import numpy as np
import pandas as pd
from oilfield_dt import sim, features as fe, pipeline as pl
from oilfield_dt.twin import PhysicsTwin, DataDrivenTwin
from oilfield_dt.util import log

EP = sim.T_EPISODE           # 432 steps = 72 h
K = 2


def load_csv(path, colmap):
    df = pd.read_csv(path, parse_dates=[colmap["time"]]).set_index(colmap["time"]).sort_index()
    cols = [colmap[n] for n in sim.NAMES]
    y = df[cols].resample(f"{sim.STEP_MIN}min").mean().interpolate(limit=3)
    amb = df[colmap["Ta"]].resample(f"{sim.STEP_MIN}min").mean().interpolate(limit=3)
    dust = (df[colmap["Dust"]].resample(f"{sim.STEP_MIN}min").mean().interpolate(limit=3)
            if colmap.get("Dust") else pd.Series(0.0, index=y.index))
    out = pd.concat([y, amb, dust], axis=1).dropna()
    out.columns = list(sim.NAMES) + ["Ta", "D"]
    return out


def to_episodes(frame, length=EP):
    y = frame[list(sim.NAMES)].to_numpy(float); c = frame[["Ta", "D"]].to_numpy(float)
    return [dict(y=y[i:i + length], covar=c[i:i + length], label=0, onset=-1,
                 y_nominal=y[i:i + length], sigma=np.ones(sim.NS), mode="none", x=None)
            for i in range(0, len(y) - length + 1, length)]


class SiteDetector:
    def __init__(self, bundle_path="bundle.pkl"):
        self.b = pickle.load(open(bundle_path, "rb"))

    def fit_site(self, normal_eps):
        self.pt = PhysicsTwin().fit(normal_eps)
        self.dd = DataDrivenTwin().fit(normal_eps)
        self.nm = fe.Normaliser().fit(normal_eps)
        return self

    def _proba(self, eps):
        P, ends = [], []
        for e in eps:
            r = fe.episode_features(e, self.pt, self.dd, self.nm)
            P.append(self.b["cal"].predict_proba(r["F"][:, self.b["cols"]])); ends.append(r["ends"])
        return P, ends

    def calibrate_gate(self, gate_eps, target_fa_per_day=0.05):
        P, _ = self._proba(gate_eps)
        self.tau = pl.calibrate_tau([1 - p[:, 0] for p in P], K, target_fa_per_day)
        return self.tau

    def score(self, eps, timestamps=None):
        """Returns per-episode dict with probabilities, first alarm, and recommended action."""
        P, ends = self._proba(eps)
        pol = pl.make_policy("bayes_review")
        out = []
        for p, en in zip(P, ends):
            a = pl.first_alarm(1 - p[:, 0], self.tau, K, 0)
            if a < 0:
                out.append(dict(alarm=False, P=p, ends=en)); continue
            pa = p[max(0, a - K + 1):a + 1].mean(0)
            out.append(dict(alarm=True, window=int(a), end_step=int(en[a]), P=p, ends=en,
                            posterior={c: round(float(v), 3) for c, v in zip(sim.CLASSES, pa)},
                            action=pl.ACTIONS[pol(pa)]))
        return out


def selftest(bundle="bundle.pkl"):
    """Simulate a 'historian' for an unseen site, export CSV, and run the full adapter path."""
    site = sim.make_site(77, shift=0.5); rng = np.random.default_rng(5)
    normal = [sim.make_episode(site, rng, cls=0) for _ in range(30)]
    gate = [sim.make_episode(site, rng, cls=0) for _ in range(20)]
    test = [sim.make_episode(site, rng, cls=c) for c in (0, 1, 2, 3, 3, 1)]
    allep = normal + gate + test
    t0 = pd.Timestamp("2026-01-01")
    rows = []
    for k, e in enumerate(allep):
        idx = pd.date_range(t0 + pd.Timedelta(minutes=sim.STEP_MIN * EP * k), periods=EP, freq=f"{sim.STEP_MIN}min")
        d = pd.DataFrame(e["y"], index=idx, columns=[f"raw_{n}" for n in sim.NAMES])
        d["amb"] = e["covar"][:, 0]; d["dust"] = e["covar"][:, 1]
        rows.append(d)
    df = pd.concat(rows); df.index.name = "timestamp"; df.to_csv("selftest_historian.csv")
    cmap = {"time": "timestamp", **{n: f"raw_{n}" for n in sim.NAMES}, "Ta": "amb", "Dust": "dust"}
    json.dump(cmap, open("selftest_colmap.json", "w"), indent=1)
    fr = load_csv("selftest_historian.csv", cmap)
    eps = to_episodes(fr)
    det = SiteDetector(bundle).fit_site(eps[:30])
    tau = det.calibrate_gate(eps[30:50]); log("gate threshold", round(tau, 3))
    res = det.score(eps[50:])
    print(f"{'episode':8s} {'truth':16s} {'alarm':6s} posterior / action")
    for e, r in zip(test, res):
        truth = f"{sim.CLASSES[e['label']]}/{e['mode']}"
        print(f"{'':8s} {truth:16s} {str(r['alarm']):6s}", (r["posterior"], r["action"]) if r["alarm"] else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv"); ap.add_argument("--map"); ap.add_argument("--bundle", default="bundle.pkl")
    ap.add_argument("--normal", nargs=2); ap.add_argument("--gate", nargs=2); ap.add_argument("--score", nargs=2)
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.bundle)
    cmap = json.load(open(a.map)); fr = load_csv(a.csv, cmap)
    sl = lambda r: to_episodes(fr.loc[r[0]:r[1]])
    det = SiteDetector(a.bundle).fit_site(sl(a.normal))
    log("threshold", round(det.calibrate_gate(sl(a.gate)), 3))
    for k, r in enumerate(det.score(sl(a.score))):
        print(k, "ALARM" if r["alarm"] else "ok", r.get("posterior", ""), r.get("action", ""))


if __name__ == "__main__":
    main()
