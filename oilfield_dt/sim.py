"""
Synthetic oilfield wellpad simulator for cause attribution of sensor deviations.

A wellpad with M=4 wells feeds a common header.  15 instruments:
  P1..P4 (wellhead pressure, bar), T1..T4 (wellhead temperature, degC),
  Q1..Q4 (well flow, m3/h), Ph, Th, Qh (header pressure, temperature, flow).
Sensors are served by 3 gateways/RTUs: G0 (wells 1-2), G1 (wells 3-4), G2 (header).

Physics (all relations hold for the TRUE process; instruments add noise):
  * mass balance:        Qh = sum_i Qi
  * choke/flowline:      Pi = Ph + k_i * Qi^2
  * header pressure:     Ph = P_set + a * (sum Qi - Qnom)/Qnom
  * wellhead temperature T_i follows T_ss(Qi, Ta) through a first-order lag
  * header temperature   Th = flow-weighted mean(T_i) - lam * (Tbar - Ta)
Legitimate operating changes (choke steps, short shut-ins, slugging) and weather
(diurnal ambient temperature, dust events) are always present.

Four causes of measurement deviation are generated (ground-truth label):
  0 NORMAL       instruments within spec
  1 FAULT        independent hardware degradation (drift, stuck, step, gain, noise bursts)
  2 ENVIRONMENT  degradation coupled to observable weather covariates (heat, dust)
  3 ATTACK       false-data injection via compromised gateway / field device
                 (bias, ramp, replay, scale, mass-balance-preserving)
Stress-test only (never used in training):  ATTACK/full  - an attacker who knows the
twin equations and keeps every analytical-redundancy residual at zero.

IMPORTANT: this is a *generative assumption set*, not field data.  Every claim
made from results on it is conditional on these assumptions (see README).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import lfilter

M = 4
NAMES = ([f"P{i+1}" for i in range(M)] + [f"T{i+1}" for i in range(M)] +
         [f"Q{i+1}" for i in range(M)] + ["Ph", "Th", "Qh"])
IDX = {n: i for i, n in enumerate(NAMES)}
NS = len(NAMES)                                   # 15
STYPE = np.array([0] * M + [1] * M + [2] * M + [0, 1, 2])   # 0=P 1=T 2=Q
GW = np.array([0, 0, 1, 1,  0, 0, 1, 1,  0, 0, 1, 1,  2, 2, 2])
N_GW = 3

STEP_MIN = 10
STEPS_PER_HOUR = 60 // STEP_MIN                   # 6
STEPS_PER_DAY = 24 * STEPS_PER_HOUR               # 144
T_EPISODE = 3 * STEPS_PER_DAY                     # 72 h

CLASSES = ["normal", "fault", "environment", "attack"]
FAULT_MODES = ["drift", "stuck", "step", "gain", "noise"]
ENV_MODES = ["heat", "dust", "both"]
ATTACK_MODES = ["bias", "ramp", "replay", "scale", "massbal"]

# nominal generative mixes (training/test)
CLASS_PRIOR_GEN = np.array([0.30, 0.25, 0.22, 0.23])
FAULT_MIX = np.array([0.25, 0.15, 0.20, 0.15, 0.25])
ENV_MIX = np.array([0.35, 0.35, 0.30])
ATTACK_MIX = np.array([0.25, 0.20, 0.15, 0.15, 0.25])

# measurement noise std by sensor type (P bar, T degC, Q m3/h)
SIGMA_MEAS_TYPE = np.array([0.05, 0.20, 0.50])
# "unit" magnitude for injected deviations by sensor type (about 3x instrument accuracy)
UNIT_MAG_TYPE = np.array([0.15, 0.60, 1.50])


@dataclass
class Site:
    q_base: np.ndarray
    k: np.ndarray
    T_res: np.ndarray
    q_c: np.ndarray
    tau_T: np.ndarray          # temperature lag (steps)
    P_set: float
    a_hp: float
    lam: float
    T_mean: float
    T_amp: float
    noise_mult: float = 1.0    # instrument noise multiplier (site shift knob)

    @property
    def sigma_meas(self):
        return SIGMA_MEAS_TYPE[STYPE] * self.noise_mult

    @property
    def unit_mag(self):
        return UNIT_MAG_TYPE[STYPE]


def make_site(seed: int, shift: float = 0.0) -> Site:
    """shift>0 makes a site that differs more from the default (used for transfer tests)."""
    r = np.random.default_rng(seed)
    q_base = r.uniform(45, 75, M) * (1 + 0.15 * shift * r.standard_normal(M))
    dp_nom = r.uniform(6, 12, M)
    return Site(
        q_base=q_base, k=dp_nom / q_base ** 2,
        T_res=r.uniform(80, 95, M) + 3 * shift * r.standard_normal(M),
        q_c=q_base * r.uniform(0.5, 0.9, M),
        tau_T=r.uniform(3, 8, M),
        P_set=15.0 + 1.5 * shift, a_hp=r.uniform(1.0, 2.0),
        lam=r.uniform(0.03, 0.07), T_mean=32.0 + 4 * shift + r.uniform(-3, 3),
        T_amp=r.uniform(6, 10), noise_mult=1.0 + 0.3 * shift,
    )


def T_ss(Q, Ta_eff, T_res, q_c):
    return Ta_eff + (T_res - Ta_eff) * (1.0 - np.exp(-Q / q_c))


def _ar1(rng, n, phi, sigma):
    e = rng.standard_normal(n) * sigma * np.sqrt(1 - phi ** 2)
    return lfilter([1.0], [1.0, -phi], e)


def _resonant(rng, n, period, amp):
    """AR(2) resonator -> quasi-periodic slugging, normalised to std=amp."""
    r = 0.93
    th = 2 * np.pi / period
    e = rng.standard_normal(n + 200)
    x = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], e)[200:]
    return amp * x / (x.std() + 1e-12)


def simulate_base(site: Site, rng, T=T_EPISODE, force_dust_after: int | None = None):
    """Return the nominal (no anomaly) episode."""
    t = np.arange(T)
    h0 = rng.uniform(0, 24)
    hour = h0 + t / STEPS_PER_HOUR
    Ta = (site.T_mean + site.T_amp * np.sin(2 * np.pi * (hour - 9) / 24)
          + _ar1(rng, T, 0.995, 1.2))
    # dust events
    D = np.zeros(T)
    n_ev = rng.poisson(0.8)
    starts = list(rng.integers(0, T - 20, n_ev))
    if force_dust_after is not None:
        starts.append(int(rng.integers(force_dust_after, T - 30)))
    for s0 in starts:
        dur = int(rng.integers(12, 72))
        e = min(T, s0 + dur)
        D[s0:e] += rng.uniform(0.3, 1.0) * np.hanning(e - s0 + 2)[1:-1]
    D = np.clip(D, 0, 1)

    # legitimate operating changes
    Q = np.zeros((T, M))
    for i in range(M):
        c = np.ones(T)
        for _ in range(rng.poisson(1.2)):
            c[int(rng.integers(10, T - 10)):] *= rng.uniform(0.8, 1.2)
        c = np.clip(c, 0.6, 1.4)
        c = np.convolve(c, np.ones(3) / 3, mode="same")
        if rng.random() < 0.06:                                  # short shut-in
            s0 = int(rng.integers(20, T - 30)); dur = int(rng.integers(6, 18))
            m = np.ones(T); m[s0:s0 + dur] = 0.02
            c = c * np.convolve(m, np.ones(3) / 3, mode="same")
        slug = _resonant(rng, T, rng.uniform(6, 18), rng.uniform(0.015, 0.04))
        slow = _ar1(rng, T, 0.999, 0.02)
        Q[:, i] = site.q_base[i] * c * (1 + slug + slow)
    Qtot = Q.sum(1)
    Qnom = site.q_base.sum()
    Ph = site.P_set + site.a_hp * (Qtot - Qnom) / Qnom + _ar1(rng, T, 0.9, 0.10)
    P = Ph[:, None] + site.k[None, :] * Q ** 2 + np.stack(
        [_ar1(rng, T, 0.8, 0.07) for _ in range(M)], 1)
    Ta_eff = 0.5 * Ta + 0.5 * site.T_mean
    Tss = T_ss(Q, Ta_eff[:, None], site.T_res[None, :], site.q_c[None, :])
    Tw = np.zeros((T, M))
    for i in range(M):
        a = np.exp(-1.0 / site.tau_T[i])
        Tw[:, i] = lfilter([1 - a], [1, -a], Tss[:, i], zi=[a * Tss[0, i]])[0]
    Tw += np.stack([_ar1(rng, T, 0.8, 0.12) for _ in range(M)], 1)
    Tbar = (Q * Tw).sum(1) / Qtot
    Th = Tbar - site.lam * (Tbar - Ta) + _ar1(rng, T, 0.8, 0.10)
    Qh = Qtot + _ar1(rng, T, 0.8, 0.30)

    x = np.concatenate([P, Tw, Q, Ph[:, None], Th[:, None], Qh[:, None]], axis=1)
    y = x + rng.standard_normal((T, NS)) * site.sigma_meas[None, :]
    covar = np.stack([Ta + rng.standard_normal(T) * 0.3,
                      np.clip(D + rng.standard_normal(T) * 0.03, 0, None)], 1)
    return dict(x=x, y=y, covar=covar, Ta=Ta, D=D)


# ----------------------------------------------------------------------------
# injectors
# ----------------------------------------------------------------------------
def _ramp(T, t0):
    r = np.zeros(T)
    r[t0:] = np.linspace(0, 1, T - t0)
    return r


def _sign(rng):
    return rng.choice([-1.0, 1.0])


def _loguni(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def inject_fault(base, site, rng, t0, mode, mag=1.0):
    y = base["y"].copy(); T = y.shape[0]
    n_s = 1 if rng.random() < 0.85 else 2
    sensors = list(rng.choice(NS, n_s, replace=False))
    modes = [mode] + ([rng.choice(FAULT_MODES, p=FAULT_MIX)] if n_s == 2 else [])
    for s, md in zip(sensors, modes):
        u = site.unit_mag[s]; sg = site.sigma_meas[s]
        if md == "drift":
            y[:, s] += _sign(rng) * _loguni(rng, 0.5, 8) * mag * u * _ramp(T, t0)
        elif md == "step":
            y[t0:, s] += _sign(rng) * _loguni(rng, 0.5, 8) * mag * u
        elif md == "stuck":
            y[t0:, s] = y[t0, s] + rng.standard_normal(T - t0) * 0.02 * sg
        elif md == "gain":
            y[:, s] *= 1 + _sign(rng) * _loguni(rng, 0.005, 0.08) * mag * _ramp(T, t0)
        elif md == "noise":
            on = np.zeros(T, bool); st = False
            for t in range(t0, T):
                st = (rng.random() < 0.85) if st else (rng.random() < 0.06)
                on[t] = st
            amp = _loguni(rng, 2, 8) * mag * sg
            y[:, s] += on * rng.standard_normal(T) * amp
            sp = (rng.random(T) < 0.01) & (np.arange(T) >= t0)
            y[:, s] += sp * rng.choice([-1, 1], T) * rng.uniform(6, 15, T) * sg
    return y, dict(sensors=sensors, modes=modes)


def inject_env(base, site, rng, t0, mode, mag=1.0):
    y = base["y"].copy(); T = y.shape[0]
    Ta, D = base["Ta"], base["D"]
    k = int(rng.integers(2, 9))
    sensors = list(rng.choice(NS, k, replace=False))
    post = np.arange(T) >= t0
    cumD = np.cumsum(D * post) / 72.0
    for s in sensors:
        ty = STYPE[s]; sg = site.sigma_meas[s]
        if mode in ("heat", "both"):
            if ty == 1:
                y[:, s] += rng.uniform(0.1, 0.5) * mag * np.maximum(0, Ta - 34) * post
            elif ty == 0:
                y[:, s] += _sign(rng) * rng.uniform(0.004, 0.02) * mag * (Ta - 30) * post
            else:
                y[:, s] += _sign(rng) * rng.uniform(0.02, 0.15) * mag * (Ta - 30) * post
        if mode in ("dust", "both") and ty != 1:
            kap = rng.uniform(2, 8) * mag
            y[:, s] += rng.standard_normal(T) * kap * sg * D * post
            out = (rng.random(T) < 0.08 * D) & post
            y[:, s] += out * rng.choice([-1, 1], T) * rng.uniform(5, 12, T) * sg
            if ty == 2:
                y[:, s] *= 1 - rng.uniform(0.01, 0.06) * mag * cumD
    return y, dict(sensors=sensors, modes=[mode])


def inject_attack(base, site, rng, t0, mode, mag=1.0):
    y = base["y"].copy(); T = y.shape[0]
    x = base["x"]
    meta = dict(modes=[mode])
    if mode in ("massbal", "full"):
        g = int(rng.integers(0, 2))
        i, j = (0, 1) if g == 0 else (2, 3)
        if rng.random() < 0.5:
            i, j = j, i
        d = _sign(rng) * _loguni(rng, 0.05, 0.30) * mag * site.q_base[i]
        on = (np.arange(T) >= t0).astype(float)
        y[:, IDX[f"Q{i+1}"]] += d * on
        y[:, IDX[f"Q{j+1}"]] -= d * on
        sensors = [IDX[f"Q{i+1}"], IDX[f"Q{j+1}"]]
        if mode == "full":            # attacker knows the twin: keeps every ARR consistent
            Ta = base["Ta"]; Ta_eff = 0.5 * Ta + 0.5 * site.T_mean
            Tw = x[:, 4:8]
            Q_old = x[:, 8:12]
            Q_new = Q_old.copy()
            Q_new[:, i] = np.maximum(Q_old[:, i] + d, 0.5)
            Q_new[:, j] = np.maximum(Q_old[:, j] - d, 0.5)
            Tw_new = Tw.copy()
            for w in (i, j):
                y[:, IDX[f"P{w+1}"]] += site.k[w] * (Q_new[:, w] ** 2 - Q_old[:, w] ** 2) * on
                dT = (T_ss(Q_new[:, w], Ta_eff, site.T_res[w], site.q_c[w])
                      - T_ss(Q_old[:, w], Ta_eff, site.T_res[w], site.q_c[w]))
                y[:, IDX[f"T{w+1}"]] += dT * on
                Tw_new[:, w] = Tw[:, w] + dT
            Tbar0 = (Q_old * Tw).sum(1) / Q_old.sum(1)
            Tbar1 = (Q_new * Tw_new).sum(1) / Q_new.sum(1)
            y[:, IDX["Th"]] += (1 - site.lam) * (Tbar1 - Tbar0) * on
            sensors += [IDX[f"P{i+1}"], IDX[f"P{j+1}"], IDX[f"T{i+1}"],
                        IDX[f"T{j+1}"], IDX["Th"]]
        meta["sensors"] = sensors
        return y, meta

    if rng.random() < 0.7:            # gateway compromise
        g = int(rng.integers(0, N_GW))
        pool = np.where(GW == g)[0]
        k = int(rng.integers(2, len(pool) + 1))
        sensors = list(rng.choice(pool, k, replace=False))
    else:                              # single compromised field device
        sensors = [int(rng.integers(0, NS))]
    meta["gateway_compromise"] = len(sensors) > 1
    if mode == "replay":
        Ls = int(rng.integers(36, min(t0, 108) + 1))
        idx = t0 - Ls + (np.arange(T - t0) % Ls)
        for s in sensors:
            y[t0:, s] = base["y"][idx, s]
    else:
        for s in sensors:
            u = site.unit_mag[s]
            if mode == "bias":
                y[t0:, s] += _sign(rng) * _loguni(rng, 0.5, 8) * mag * u
            elif mode == "ramp":
                y[:, s] += _sign(rng) * _loguni(rng, 0.5, 8) * mag * u * _ramp(T, t0)
            elif mode == "scale":
                y[t0:, s] *= 1 + _sign(rng) * _loguni(rng, 0.01, 0.15) * mag
    meta["sensors"] = sensors
    return y, meta


def make_episode(site: Site, rng, cls: int | None = None, mode: str | None = None,
                 mag: float = 1.0, T: int = T_EPISODE, class_prior=CLASS_PRIOR_GEN):
    """Generate one labelled episode (persisting anomaly starting at `onset`)."""
    if cls is None:
        cls = int(rng.choice(4, p=class_prior))
    onset = int(rng.integers(int(0.25 * T), int(0.6 * T)))
    if cls == 2:
        mode = mode or rng.choice(ENV_MODES, p=ENV_MIX)
    base = simulate_base(site, rng, T,
                         force_dust_after=onset if (cls == 2 and mode in ("dust", "both")) else None)
    meta = dict(modes=[])
    y = base["y"]
    if cls == 1:
        mode = mode or rng.choice(FAULT_MODES, p=FAULT_MIX)
        y, meta = inject_fault(base, site, rng, onset, mode, mag)
    elif cls == 2:
        y, meta = inject_env(base, site, rng, onset, mode, mag)
    elif cls == 3:
        mode = mode or rng.choice(ATTACK_MODES, p=ATTACK_MIX)
        y, meta = inject_attack(base, site, rng, onset, mode, mag)
    return dict(y=y, y_nominal=base["y"], x=base["x"], covar=base["covar"], label=cls,
                sigma=site.sigma_meas.copy(),
                mode=(meta["modes"][0] if meta["modes"] else "none"),
                onset=onset if cls != 0 else -1, sensors=meta.get("sensors", []),
                gateway_compromise=meta.get("gateway_compromise", False))


def make_dataset(site: Site, n: int, seed: int, **kw):
    rng = np.random.default_rng(seed)
    return [make_episode(site, rng, **kw) for _ in range(n)]
