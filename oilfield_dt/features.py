"""
Sliding-window feature extraction (window W steps, stride S steps).

Feature groups (used for ablations):
  arr    - statistics of the 12 physics-twin (ARR) residual channels
  dd     - statistics of the 15 data-driven-twin residual channels
  raw    - statistics of z-scored raw sensors (the "no twin" baseline uses only this + inst)
  inst   - instrument-health features: noise ratio and flatness per sensor
  cov    - coupling of residuals to weather covariates (ambient T, dust, cumulative dust)
  topo   - sensor isolation scores and gateway-level aggregation (coordination signature)
  replay - noise-fingerprint replay score per sensor (max lagged correlation with history)
  ctx    - operating-context features (total-flow transient size)
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve
from .sim import NS, NAMES, GW, N_GW, STEPS_PER_DAY
from .twin import ARR_NAMES, N_ARR, arr_structure

W = 36          # 6 h
STRIDE = 6      # 1 h
S_ARR = arr_structure()

STATS = ["mean", "std", "absmax", "slope", "ac1", "kurt", "frac3"]


def _stats(win):
    """win: (nW, C, W) -> (nW, C, 7)"""
    m = win.mean(-1)
    d = win - m[..., None]
    sd = d.std(-1) + 1e-9
    tc = np.arange(win.shape[-1]) - (win.shape[-1] - 1) / 2
    slope = (d * tc).sum(-1) / (tc ** 2).sum() * win.shape[-1]
    ac1 = (d[..., :-1] * d[..., 1:]).mean(-1) / (sd ** 2)
    kurt = np.clip((d ** 4).mean(-1) / sd ** 4 - 3, -3, 50)
    return np.stack([m, sd, np.abs(win).max(-1), slope, ac1, kurt,
                     (np.abs(win) > 3).mean(-1)], -1)


def _corr(win, cwin):
    """win: (nW, C, W); cwin: (nW, W) -> (nW, C) Pearson correlation (0 if cov is flat)."""
    a = win - win.mean(-1, keepdims=True)
    c = cwin - cwin.mean(-1, keepdims=True)
    num = (a * c[:, None, :]).sum(-1)
    den = np.sqrt((a ** 2).sum(-1) * (c ** 2).sum(-1)[:, None]) + 1e-12
    out = num / den
    flat = (c ** 2).sum(-1) < 1e-4
    out[flat] = 0.0
    return out


def _windows(A, ends):
    """A: (T, C) -> (nW, C, W) for windows ending at `ends` (inclusive)."""
    sw = sliding_window_view(A, W, axis=0)          # (T-W+1, C, W)
    return sw[ends - (W - 1)]


def replay_score(y, ends):
    """Max normalised cross-correlation between the window's high-passed signal and any
    earlier non-overlapping segment of the same sensor (noise-fingerprint replay check)."""
    dy = np.diff(y, axis=0)                          # dy[k] = y[k+1]-y[k]
    out = np.zeros((len(ends), NS))
    cs = np.concatenate([np.zeros((NS, 1)), np.cumsum(dy.T ** 2, axis=1)], axis=1)
    for wi, e in enumerate(ends):
        lo, hi = e - W + 1, e                        # window dy indices lo..hi-1
        win = dy[lo:hi].T                            # (NS, W-1)
        Lw = win.shape[1]
        hist_end = lo                                # history dy indices 0..lo-1
        if hist_end < Lw + 4:
            continue
        hist = dy[:hist_end].T
        cc = fftconvolve(hist, win[:, ::-1], mode="valid", axes=1)   # (NS, hist_end-Lw+1)
        nseg = cc.shape[1]
        seg_E = cs[:, Lw:Lw + nseg] - cs[:, :nseg]
        E_w = (win ** 2).sum(1, keepdims=True)
        out[wi] = (cc / np.sqrt(E_w * seg_E + 1e-12)).max(1)
    return out



def _agg_stats(s, prefix, thr_mean=3.0, thr_std=2.0):
    """Permutation-invariant aggregates over channels of a (nW, C, 7) stats block."""
    mean_abs = np.abs(s[..., 0]); sd = s[..., 1]; amax = s[..., 2]; kurt = s[..., 5]
    cols = np.stack([mean_abs.max(1), mean_abs.sum(1), sd.max(1), amax.max(1), kurt.max(1),
                     (mean_abs > thr_mean).sum(1), (sd > thr_std).sum(1)], 1)
    nm = [f"{prefix}:agg_{n}" for n in ("maxabsmean", "sumabsmean", "maxstd", "maxabsmax",
                                        "maxkurt", "n_mean_gt3", "n_std_gt2")]
    return cols, nm


class Normaliser:
    """Per-sensor statistics of normal data used to z-score raw signals / diff-noise."""
    def fit(self, normal_eps):
        y = np.concatenate([e["y"] for e in normal_eps]); self.mu = np.median(y, 0)
        self.sd = 1.4826 * np.median(np.abs(y - self.mu), 0) + 1e-9
        dsd = np.concatenate([np.diff(e["y"], axis=0) for e in normal_eps])
        self.dsd = dsd.std(0) + 1e-9
        return self


def episode_features(ep, ptwin, ddtwin, norm: Normaliser):
    """Returns dict(F=(nW, nF) float32, names, groups, ends, frac_post)."""
    y, covar = ep["y"], ep["covar"]
    T = y.shape[0]
    ends = np.arange(W - 1, T, STRIDE)
    Ra = ptwin.residuals(y, covar)
    Rd = ddtwin.residuals(y, covar)
    Rr = (y - norm.mu) / norm.sd
    Ta, D = covar[:, 0], covar[:, 1]
    cumD = np.cumsum(D) / 72.0

    blocks, names, groups = [], [], []

    def add(mat, nm, grp):
        blocks.append(mat.reshape(len(ends), -1))
        names.extend(nm); groups.extend([grp] * len(nm))

    wa = _windows(Ra, ends); wd = _windows(Rd, ends); wr = _windows(Rr, ends)
    sa, sd_, sr = _stats(wa), _stats(wd), _stats(wr)
    add(sa, [f"arr:{c}:{s}" for c in ARR_NAMES for s in STATS], "arr")
    add(sd_, [f"dd:{c}:{s}" for c in NAMES for s in STATS], "dd")
    add(sr, [f"raw:{c}:{s}" for c in NAMES for s in STATS], "raw")
    for blk, pre, grp in ((sa, "arr", "arr"), (sd_, "dd", "dd"), (sr, "raw", "raw")):
        a_, n_ = _agg_stats(blk, pre); add(a_, n_, grp)

    # instrument health: noise ratio and flat fraction of first differences
    dy = np.diff(y, axis=0)
    dyw = sliding_window_view(dy, W - 1, axis=0)[ends - (W - 1)]      # (nW, NS, W-1)
    noise_ratio = dyw.std(-1) / norm.dsd[None, :]
    flat = (np.abs(dyw) < 0.05 * norm.dsd[None, :, None]).mean(-1)
    add(noise_ratio, [f"inst:{c}:noiseratio" for c in NAMES], "inst")
    add(flat, [f"inst:{c}:flat" for c in NAMES], "inst")
    inst_agg = np.stack([noise_ratio.max(1), noise_ratio.min(1), (noise_ratio > 2).sum(1),
                         (noise_ratio < 0.3).sum(1), flat.max(1), (flat > 0.5).sum(1)], 1)
    add(inst_agg, [f"inst:agg_{n}" for n in ("nr_max", "nr_min", "n_nr_gt2", "n_nr_lt03",
                                              "flat_max", "n_flat_gt05")], "inst")

    # covariate coupling
    cw_T = _windows(Ta[:, None], ends)[:, 0, :]
    cw_D = _windows(D[:, None], ends)[:, 0, :]
    cw_C = _windows(cumD[:, None], ends)[:, 0, :]
    act_a = (np.abs(sa[..., 0]) > 3.0) | (sa[..., 1] > 2.0)          # active ARR channels
    act_d = (np.abs(sd_[..., 0]) > 3.0) | (sd_[..., 1] > 2.0)        # active DD channels
    for cname, cw in (("Ta", cw_T), ("D", cw_D), ("cumD", cw_C)):
        ca, cd = _corr(wa, cw), _corr(wd, cw)
        add(ca, [f"cov:{c}:corr_{cname}" for c in ARR_NAMES], "cov")
        add(cd, [f"cov:dd_{c}:corr_{cname}" for c in NAMES], "cov")
        for cc, act, tag in ((ca, act_a, "arr"), (cd, act_d, "dd")):
            n_act = act.sum(1)
            mx = np.where(act, np.abs(cc), 0).max(1)
            mean_act = np.where(act, np.abs(cc), 0).sum(1) / np.maximum(n_act, 1)
            add(np.stack([mx, mean_act], 1),
                [f"cov:agg_{tag}_{cname}_max", f"cov:agg_{tag}_{cname}_mean"], "cov")
    cov_stats = np.stack([cw_T.mean(-1), cw_T.max(-1), cw_D.mean(-1), cw_D.max(-1)], 1)
    add(cov_stats, ["cov:Ta_mean", "cov:Ta_max", "cov:D_mean", "cov:D_max"], "cov")

    # topology / isolation: sensor-level suspicion from ARR mean residuals
    mz = np.abs(sa[:, :, 0])                                          # (nW, 12)
    iso = (mz @ S_ARR) / S_ARR.sum(0)[None, :]                        # (nW, NS)
    active = (iso > 3.0)
    gw_iso = np.stack([iso[:, GW == g].sum(1) for g in range(N_GW)], 1)
    gw_act = np.stack([active[:, GW == g].sum(1) for g in range(N_GW)], 1)
    topo = np.concatenate([iso, gw_iso, gw_act, active.sum(1, keepdims=True),
                           (gw_act > 0).sum(1, keepdims=True)], 1)
    add(topo, [f"topo:iso_{c}" for c in NAMES] + [f"topo:gw_iso{g}" for g in range(N_GW)]
        + [f"topo:gw_active{g}" for g in range(N_GW)] + ["topo:n_active", "topo:n_gw_active"],
        "topo")

    # replay fingerprint
    rep = replay_score(y, ends)
    add(rep, [f"replay:{c}" for c in NAMES], "replay")
    rep_agg = np.concatenate([rep.max(1, keepdims=True), (rep > 0.9).sum(1, keepdims=True),
                              (rep > 0.95).sum(1, keepdims=True),
                              np.stack([rep[:, GW == g].max(1) for g in range(N_GW)], 1)], 1)
    add(rep_agg, ["replay:agg_max", "replay:agg_n_gt09", "replay:agg_n_gt095"]
        + [f"replay:agg_gw{g}" for g in range(N_GW)], "replay")

    # context: total-flow transient
    Qtot = y[:, 8:12].sum(1)
    qw = _windows(Qtot[:, None], ends)[:, 0, :]
    ctx = np.stack([np.abs(np.diff(qw, axis=1)).max(1) / Qtot.mean(),
                    qw.std(1) / Qtot.mean()], 1)
    add(ctx, ["ctx:dQ_max", "ctx:Q_std"], "ctx")

    F = np.concatenate(blocks, 1).astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=50.0, neginf=-50.0)
    onset = ep["onset"]
    frac_post = np.zeros(len(ends)) if onset < 0 else \
        np.clip((ends - np.maximum(onset, ends - W + 1) + 1) / W, 0, 1) * (ends >= onset)
    return dict(F=F, names=names, groups=np.array(groups), ends=ends, frac_post=frac_post)


MANIFEST_SIGMA = 1.5   # window is "manifest" if some sensor's RMS injected deviation > 1.5 sigma_meas


def manifest_flags(ep, ends):
    """Oracle (simulation-only) flag: does the window contain an observable injected deviation?"""
    if ep["label"] == 0:
        return np.zeros(len(ends), bool)
    dev = (ep["y"] - ep["y_nominal"]) / ep["sigma"][None, :]
    win = sliding_window_view(dev ** 2, W, axis=0)[ends - (W - 1)]      # (nW, NS, W)
    return np.sqrt(win.mean(-1)).max(1) > MANIFEST_SIGMA


def window_labels(ep, frac_post, manifest):
    """y_all: class if >=50% of the window is post-onset (latent effects included).
       y    : class only if additionally manifest; -1 = excluded (transition / latent)."""
    y_all = np.zeros(len(frac_post), dtype=int)
    if ep["label"] == 0:
        return y_all.copy(), y_all.copy()
    y_all[frac_post >= 0.5] = ep["label"]
    y_all[(frac_post > 0) & (frac_post < 0.5)] = -1
    y = y_all.copy()
    y[(y_all == ep["label"]) & ~manifest] = -1
    return y_all, y


def build(episodes, ptwin, ddtwin, norm, verbose=False):
    Fs, ys, yas, mf, eid, ends, fpost, names, groups = [], [], [], [], [], [], [], None, None
    for k, ep in enumerate(episodes):
        r = episode_features(ep, ptwin, ddtwin, norm)
        man = manifest_flags(ep, r["ends"])
        ya, yy = window_labels(ep, r["frac_post"], man)
        Fs.append(r["F"]); ys.append(yy); yas.append(ya); mf.append(man)
        eid.append(np.full(len(r["ends"]), k)); ends.append(r["ends"]); fpost.append(r["frac_post"])
        names, groups = r["names"], r["groups"]
        if verbose and k % 100 == 0:
            print("  features", k, "/", len(episodes), flush=True)
    return dict(F=np.concatenate(Fs), y=np.concatenate(ys), y_all=np.concatenate(yas),
                manifest=np.concatenate(mf), eid=np.concatenate(eid),
                ends=np.concatenate(ends), frac_post=np.concatenate(fpost),
                names=names, groups=groups)
