"""
Auxiliary experiments on the primary site:
  E3  magnitude sweep         (how do detection / attribution degrade as deviations shrink?)
  E4  site transfer           (new site: no adaptation / twin re-identified / +threshold / retrained oracle)
  E5  stress tests            (mass-balance-preserving, replay, and twin-aware 'full' attacker)
  E7  example episodes        (for the qualitative figure)
"""
import copy, pickle, json
import numpy as np
from oilfield_dt import sim, features as fe, pipeline as pl
from oilfield_dt.twin import PhysicsTwin, DataDrivenTwin
from oilfield_dt.util import log, ends_by_ep, meta_of, jsonable

K = 2
TARGET_FA = 0.05
SCORE = lambda P: 1.0 - P[:, 0]


def stream_tab(eps, twins, cal, c, tau):
    pt, dd, nm = twins
    d = fe.build(eps, pt, dd, nm)
    P = cal.predict_proba(d["F"][:, c])
    P_list = [P[d["eid"] == k] for k in range(len(eps))]
    tab = pl.episode_table(P_list, ends_by_ep(d, len(eps)), meta_of(eps), SCORE, tau, K)
    return tab, d, P


def gate_tau(eps, twins, cal, c):
    pt, dd, nm = twins
    d = fe.build(eps, pt, dd, nm)
    P = cal.predict_proba(d["F"][:, c])
    return pl.calibrate_tau([SCORE(P[d["eid"] == k]) for k in range(len(eps))], K, TARGET_FA)


def run_aux(site, site_seed, pt, dd, nm, models, cals, taus, fit_eps, ds, smoke=False):
    out = {}
    twins = (pt, dd, nm)
    c = pl.cols(ds["tr"], "proposed")
    cal, tau = cals["proposed"], taus["proposed"]
    n = 25 if smoke else 100
    base = 10000 * site_seed + 500

    # ------------------------------------------------ E3 magnitude sweep
    log("E3 magnitude sweep")
    mags = [0.25, 0.5, 1.0, 2.0, 4.0]
    e3 = {}
    for cls in (1, 2, 3):
        cn = sim.CLASSES[cls]
        e3[cn] = {}
        for mg in mags:
            eps = sim.make_dataset(site, n, seed=base + 10 * cls + int(mg * 8), cls=cls, mag=mg)
            tab, _, _ = stream_tab(eps, twins, cal, c, tau)
            m = pl.table_metrics(tab, pl.make_policy("bayes_review"))
            am = pl.table_metrics(tab, pl.make_policy("argmax"))
            e3[cn][str(mg)] = dict(detect=m[f"detect_{cn}"], attr=am[f"attr_{cn}"],
                                   joint=m[f"detect_{cn}"] * am[f"attr_{cn}"] if am[f"attr_{cn}"] == am[f"attr_{cn}"] else 0.0,
                                   review_rate=m["review_rate_anom"],
                                   delay_med=m[f"delay_med_{cn}"], cost=m["cost_by_class"][cls])
        log("  ", cn, {k: round(v["detect"], 2) for k, v in e3[cn].items()})
    out["E3_magnitude"] = e3

    # ------------------------------------------------ E5 stress tests
    log("E5 stress tests")
    e5 = {}
    for mode in ("massbal", "replay", "full"):
        eps = sim.make_dataset(site, n, seed=base + 700 + len(mode), cls=3, mode=mode)
        rem_days = np.mean([(sim.T_EPISODE - e["onset"]) / sim.STEPS_PER_DAY for e in eps])
        chance = 1 - np.exp(-TARGET_FA * rem_days)
        row = dict(chance_detect=float(chance))
        for mname in ("proposed", "no_replay"):
            cm = pl.cols(ds["tr"], mname)
            tau_m = taus[mname]
            tab, _, _ = stream_tab(eps, twins, cals[mname], cm, tau_m)
            am = pl.table_metrics(tab, pl.make_policy("argmax"))
            row[mname] = dict(detect=am["detect_attack"], attr=am["attr_attack"],
                              delay_med=am["delay_med_attack"])
        e5[mode] = row
        log("  ", mode, row)
    out["E5_stress"] = e5

    # ------------------------------------------------ E4 site transfer
    log("E4 site transfer")
    site2 = sim.make_site(site_seed + 100, shift=1.0)
    b2 = 10000 * (site_seed + 100)
    n_ad, n_g, n_t = (30, 30, 60) if smoke else (60, 80, 300)
    ad_eps = sim.make_dataset(site2, n_ad, seed=b2 + 1, cls=0)
    g_eps = sim.make_dataset(site2, n_g, seed=b2 + 2, cls=0)
    te2 = sim.make_dataset(site2, n_t, seed=b2 + 3)
    meta2 = meta_of(te2)
    ptB = PhysicsTwin().fit(ad_eps)
    ddB = copy.deepcopy(dd).renormalise(ad_eps)
    nmB = fe.Normaliser().fit(ad_eps)
    twinsB = (ptB, ddB, nmB)
    e4 = {}

    def evaluate(tag, tw, cal_, c_, tau_):
        tab, d, P = stream_tab(te2, tw, cal_, c_, tau_)
        wm = pl.window_metrics(P, d, meta2)
        pol_am, pol_br = pl.make_policy("argmax"), pl.make_policy("bayes_review")
        ma, mb = pl.table_metrics(tab, pol_am), pl.table_metrics(tab, pol_br)
        e4[tag] = dict(macro_f1=wm["manifest"]["macro_f1"], accuracy=wm["manifest"]["accuracy"],
                       ece=wm["manifest"]["ece"], fa_per_day=ma["fa_per_day"],
                       detect_attack=ma["detect_attack"], attr_attack=ma["attr_attack"],
                       detect_fault=ma["detect_fault"], attr_fault=ma["attr_fault"],
                       detect_env=ma["detect_environment"], attr_env=ma["attr_environment"],
                       cost_equal_argmax=ma["cost_equal"], cost_equal_bayes_review=mb["cost_equal"],
                       cost_deploy_bayes_review=mb["cost_deploy"], tau=float(tau_))
        log("  ", tag, {k: round(v, 3) for k, v in e4[tag].items()})

    evaluate("A_no_adaptation", twins, cal, c, tau)
    evaluate("B_twin_reidentified", twinsB, cal, c, tau)
    tauC = gate_tau(g_eps, twinsB, cal, c)
    evaluate("C_twin_and_threshold", twinsB, cal, c, tauC)
    # D: oracle - classifier retrained on the new site
    n_tr2, n_cal2 = (80, 40) if smoke else (300, 120)
    tr2 = fe.build(sim.make_dataset(site2, n_tr2, seed=b2 + 4), ptB, ddB, nmB)
    cal2 = fe.build(sim.make_dataset(site2, n_cal2, seed=b2 + 5), ptB, ddB, nmB)
    clf2 = pl.fit_multiclass(tr2, c)
    calm2 = pl.calibrate(clf2, cal2, c)
    tauD = gate_tau(g_eps, twinsB, calm2, c)
    evaluate("D_retrained_oracle", twinsB, calm2, c, tauD)
    out["E4_transfer"] = e4

    # ------------------------------------------------ E7 example episodes
    log("E7 example episodes")
    rng = np.random.default_rng(2024)
    ex = {}
    def pick(cls, mode, cond):
        for _ in range(400):
            e = sim.make_episode(site, rng, cls=cls, mode=mode)
            if cond(e):
                return e
        return e
    ex["fault"] = pick(1, "drift", lambda e: len(e["sensors"]) == 1 and sim.STYPE[e["sensors"][0]] == 1)
    ex["environment"] = pick(2, "heat", lambda e: sum(sim.STYPE[s] == 1 for s in e["sensors"]) >= 3)
    ex["attack"] = pick(3, "bias", lambda e: e["gateway_compromise"] and len(e["sensors"]) >= 3)
    pack = {}
    for k, e in ex.items():
        d = fe.build([e], pt, dd, nm)
        P = cal.predict_proba(d["F"][:, c])
        pack[k] = dict(y=e["y"], y_nominal=e["y_nominal"], sensors=e["sensors"], onset=e["onset"],
                       label=e["label"], mode=e["mode"], covar=e["covar"],
                       R=pt.residuals(e["y"], e["covar"]), P=P, ends=d["ends"], tau=tau)
    pickle.dump(pack, open(f"results/example_episodes_site{site_seed}.pkl", "wb"))

    json.dump(jsonable(out), open(f"results/aux_site{site_seed}{'_smoke' if smoke else ''}.json", "w"), indent=1)
    log("aux done")
    return out
