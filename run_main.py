"""
Main experiment for one simulated site.

usage:  python run_main.py <site_seed> [--smoke] [--aux]

Splits (all disjoint, all from the same site):
  fit   250 normal episodes  -> twin identification + z-score statistics
  gate  150 normal episodes  -> alarm-threshold calibration (fixed false-alarm rate)
  train 800 mixed episodes   -> classifiers
  cal   250 mixed episodes   -> probability calibration (isotonic)
  test  500 mixed episodes   -> everything reported
"""
import sys, time, json, pickle, os
import numpy as np
from oilfield_dt import sim, features as fe, pipeline as pl
from oilfield_dt.twin import PhysicsTwin, DataDrivenTwin
from oilfield_dt.util import log, ends_by_ep, meta_of, jsonable

TARGET_FA = 0.05      # false alarms per day (about one per 20 days)
K_PERSIST = 2         # consecutive windows (2 h) above threshold
os.makedirs("results", exist_ok=True)


def run(site_seed, smoke=False, aux=False):
    n_fit, n_gate, n_tr, n_cal, n_te = (60, 40, 120, 60, 100) if smoke else (250, 150, 800, 250, 500)
    site = sim.make_site(site_seed)
    base = 10000 * site_seed
    fit_eps = sim.make_dataset(site, n_fit, seed=base + 1, cls=0)
    gate_eps = sim.make_dataset(site, n_gate, seed=base + 2, cls=0)
    tr_eps = sim.make_dataset(site, n_tr, seed=base + 3)
    cal_eps = sim.make_dataset(site, n_cal, seed=base + 4)
    te_eps = sim.make_dataset(site, n_te, seed=base + 5)
    log("data generated")

    pt = PhysicsTwin().fit(fit_eps)
    dd = DataDrivenTwin().fit(fit_eps)
    nm = fe.Normaliser().fit(fit_eps)
    log("twins fitted")
    ds = {n: fe.build(e, pt, dd, nm) for n, e in
          (("gate", gate_eps), ("tr", tr_eps), ("cal", cal_eps), ("te", te_eps))}
    log("features built", ds["tr"]["F"].shape)
    te_meta = meta_of(te_eps); te_ends = ends_by_ep(ds["te"], len(te_eps))
    gate_ends = ends_by_ep(ds["gate"], len(gate_eps))

    res = dict(site_seed=site_seed, n=dict(fit=n_fit, gate=n_gate, train=n_tr, cal=n_cal, test=n_te),
               target_fa_per_day=TARGET_FA, k_persist=K_PERSIST, window={}, stream={}, ci={})
    models, cals, taus = {}, {}, {}
    tables, preds = {}, {}

    # ---------------- multi-class variants
    for name in pl.FEATURE_SETS:
        t = time.time()
        c = pl.cols(ds["tr"], name)
        clf = pl.fit_multiclass(ds["tr"], c)
        cal = pl.calibrate(clf, ds["cal"], c)
        models[name], cals[name] = clf, cal
        P = cal.predict_proba(ds["te"]["F"][:, c])
        res["window"][name] = pl.window_metrics(P, ds["te"], te_meta)
        if name == "proposed":
            Pu = clf.predict_proba(ds["te"]["F"][:, c])
            res["window"]["proposed_uncalibrated"] = pl.window_metrics(Pu, ds["te"], te_meta)
            np.savez_compressed(f"results/test_probs_site{site_seed}{'_smoke' if smoke else ''}.npz", P=P, Pu=Pu, y=ds["te"]["y"],
                                y_all=ds["te"]["y_all"], manifest=ds["te"]["manifest"],
                                eid=ds["te"]["eid"], ends=ds["te"]["ends"],
                                modes=np.array([te_meta[k]["mode"] for k in ds["te"]["eid"]]))
        # streaming
        Pg = cal.predict_proba(ds["gate"]["F"][:, c])
        Pg_list = [Pg[ds["gate"]["eid"] == k] for k in range(len(gate_eps))]
        score = lambda P: 1.0 - P[:, 0]
        tau = pl.calibrate_tau([score(p) for p in Pg_list], K_PERSIST, TARGET_FA)
        taus[name] = tau
        P_list = [P[ds["te"]["eid"] == k] for k in range(len(te_eps))]
        tab = pl.episode_table(P_list, te_ends, te_meta, score, tau, K_PERSIST)
        tables[name] = tab; preds[name] = P.argmax(1)
        for pol in ("argmax", "bayes", "bayes_review"):
            m = pl.table_metrics(tab, pl.make_policy(pol))
            res["stream"][f"{name}/{pol}"] = m
        log(f"{name}: macroF1(manifest)={res['window'][name]['manifest']['macro_f1']:.3f} "
            f"tau={tau:.3f} [{time.time()-t:.0f}s]")

    # ---------------- binary baselines (same features as the proposed model)
    c = pl.cols(ds["tr"], "proposed")
    bases = {"generic": ((1, 2, 3), "fixed:maintenance"), "generic_esc": ((1, 2, 3), "fixed:escalate"),
             "attack_only": ((3,), "fixed:escalate"), "fault_only": ((1,), "fixed:maintenance")}
    fitted = {}
    for name, (pos, pol) in bases.items():
        key = tuple(pos)
        if key not in fitted:
            fitted[key] = pl.fit_binary(ds["tr"], c, pos)
        clf = fitted[key]
        score = lambda P: P[:, 1]
        Pg = clf.predict_proba(ds["gate"]["F"][:, c])
        tau = pl.calibrate_tau([score(Pg[ds["gate"]["eid"] == k]) for k in range(len(gate_eps))],
                               K_PERSIST, TARGET_FA)
        Pt = clf.predict_proba(ds["te"]["F"][:, c])
        P_list = [Pt[ds["te"]["eid"] == k] for k in range(len(te_eps))]
        tab = pl.episode_table(P_list, te_ends, te_meta, score, tau, K_PERSIST)
        tables[name] = tab
        res["stream"][f"{name}/{pol}"] = pl.table_metrics(tab, pl.make_policy(pol))
        taus[name] = tau
        log(f"baseline {name}: tau={tau:.3f}")

    # ---------------- bootstrap CIs for the headline rows
    keys = ["cost_equal", "cost_deploy", "attack_escalated", "attack_missed_or_benign",
            "false_escalation_rate", "fa_per_day", "detect_fault", "detect_environment", "detect_attack",
            "attr_fault", "attr_environment", "attr_attack"]
    for name in ("proposed/argmax", "proposed/bayes", "proposed/bayes_review"):
        nm_, pol = name.split("/")
        tab = tables["proposed"]
        res["ci"][name] = pl.bootstrap(tab, pl.make_policy(pol), keys, n=200)
    res["taus"] = taus
    pickle.dump(tables, open(f"results/tables_site{site_seed}{'_smoke' if smoke else ''}.pkl", "wb"))
    np.savez_compressed(f"results/preds_site{site_seed}{'_smoke' if smoke else ''}.npz",
                        y=ds["te"]["y"], y_all=ds["te"]["y_all"], eid=ds["te"]["eid"],
                        manifest=ds["te"]["manifest"], **{f"pred_{k}": v for k, v in preds.items()})

    json.dump(jsonable(res), open(f"results/main_site{site_seed}{'_smoke' if smoke else ''}.json", "w"), indent=1)
    log("saved results")

    if aux:
        import aux_experiments as ax
        ax.run_aux(site, site_seed, pt, dd, nm, models, cals, taus, fit_eps, ds, smoke=smoke)
    return res


if __name__ == "__main__":
    seed = int(sys.argv[1]); smoke = "--smoke" in sys.argv; aux = "--aux" in sys.argv
    run(seed, smoke, aux)
