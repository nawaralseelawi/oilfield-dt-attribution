"""
Digital-twin residual generators.

PhysicsTwin  - analytical-redundancy relations (ARRs) whose *form* comes from
               process physics and whose parameters are identified from normal
               operating data only (no attack/fault labels, no simulator internals):
                 mass balance, choke/flowline dP-vs-Q^2, wellhead T vs (Q, ambient),
                 header T vs flow-weighted well T, header P vs total flow (x2).
DataDrivenTwin - leave-one-out gradient-boosting predictors (each sensor predicted
               from all other sensors + ambient temperature), trained on normal data.

Both return standardised residuals (z-scores w.r.t. normal-data statistics), so the
same code can be re-identified on a new site with a few normal days (see `refit`).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from .sim import M, NS, IDX, NAMES

ARR_NAMES = (["mass"] + [f"dp{i+1}" for i in range(M)] + [f"T{i+1}" for i in range(M)]
             + ["Th", "Ph_sumQ", "Ph_Qh"])
N_ARR = len(ARR_NAMES)          # 12


def arr_structure() -> np.ndarray:
    """Incidence matrix S[r, s] = 1 if ARR r involves sensor s."""
    S = np.zeros((N_ARR, NS))
    a = {n: i for i, n in enumerate(ARR_NAMES)}
    for n in ("Q1", "Q2", "Q3", "Q4", "Qh"):
        S[a["mass"], IDX[n]] = 1
    for i in range(M):
        for n in (f"P{i+1}", "Ph", f"Q{i+1}"):
            S[a[f"dp{i+1}"], IDX[n]] = 1
        for n in (f"T{i+1}", f"Q{i+1}"):
            S[a[f"T{i+1}"], IDX[n]] = 1
    for n in ("Th", "T1", "T2", "T3", "T4", "Q1", "Q2", "Q3", "Q4"):
        S[a["Th"], IDX[n]] = 1
    for n in ("Ph", "Q1", "Q2", "Q3", "Q4"):
        S[a["Ph_sumQ"], IDX[n]] = 1
    for n in ("Ph", "Qh"):
        S[a["Ph_Qh"], IDX[n]] = 1
    return S


def _stack(episodes, key="y"):
    return np.concatenate([e[key] for e in episodes], axis=0)


class PhysicsTwin:
    def fit(self, normal_eps, max_rows=40000, seed=0):
        y = _stack(normal_eps, "y"); c = _stack(normal_eps, "covar")
        rng = np.random.default_rng(seed)
        if len(y) > max_rows:
            sel = rng.choice(len(y), max_rows, replace=False); y, c = y[sel], c[sel]
        Ta = c[:, 0]
        Q = y[:, 8:12]; P = y[:, 0:4]; T = y[:, 4:8]
        Ph, Th, Qh = y[:, 12], y[:, 13], y[:, 14]
        # choke relation: (P_i - Ph) = k_i Q_i^2 (through origin)
        self.k = np.array([np.sum((P[:, i] - Ph) * Q[:, i] ** 2) / np.sum(Q[:, i] ** 4)
                           for i in range(M)])
        # wellhead T: a0 + a1*Ta + a2*(1-exp(-Q/qc))
        self.th = []
        for i in range(M):
            def f(p, Qi=Q[:, i], Tai=Ta, Ti=T[:, i]):
                return p[0] + p[1] * Tai + p[2] * (1 - np.exp(-Qi / abs(p[3]))) - Ti
            p0 = [30.0, 0.3, 40.0, 0.7 * Q[:, i].mean()]
            self.th.append(least_squares(f, p0, loss="soft_l1", f_scale=0.5).x)
        # header T: linear in flow-weighted T and Ta
        Tbar = (Q * T).sum(1) / np.maximum(Q.sum(1), 1e-6)
        A = np.stack([np.ones_like(Tbar), Tbar, Ta], 1)
        self.cTh = np.linalg.lstsq(A, Th, rcond=None)[0]
        # header P: linear in total flow (well sum) and in header meter
        A = np.stack([np.ones(len(y)), Q.sum(1)], 1)
        self.cPs = np.linalg.lstsq(A, Ph, rcond=None)[0]
        A = np.stack([np.ones(len(y)), Qh], 1)
        self.cPh = np.linalg.lstsq(A, Ph, rcond=None)[0]
        # normal-data residual statistics for z-scoring
        r = self._raw(y, c)
        self.mu = np.median(r, 0)
        self.sd = 1.4826 * np.median(np.abs(r - self.mu), 0) + 1e-9
        return self

    def refit(self, normal_eps):
        """Re-identify on a new site's normal data (parameters + z-score statistics)."""
        return self.fit(normal_eps)

    def _raw(self, y, c):
        Ta = c[:, 0]
        Q = y[:, 8:12]; P = y[:, 0:4]; T = y[:, 4:8]
        Ph, Th, Qh = y[:, 12], y[:, 13], y[:, 14]
        r = np.zeros((len(y), N_ARR))
        r[:, 0] = Qh - Q.sum(1)
        for i in range(M):
            r[:, 1 + i] = (P[:, i] - Ph) - self.k[i] * Q[:, i] ** 2
            p = self.th[i]
            r[:, 5 + i] = T[:, i] - (p[0] + p[1] * Ta + p[2] * (1 - np.exp(-Q[:, i] / abs(p[3]))))
        Tbar = (Q * T).sum(1) / np.maximum(Q.sum(1), 1e-6)
        r[:, 9] = Th - (self.cTh[0] + self.cTh[1] * Tbar + self.cTh[2] * Ta)
        r[:, 10] = Ph - (self.cPs[0] + self.cPs[1] * Q.sum(1))
        r[:, 11] = Ph - (self.cPh[0] + self.cPh[1] * Qh)
        return r

    def residuals(self, y, covar):
        return (self._raw(y, covar) - self.mu) / self.sd


class DataDrivenTwin:
    def __init__(self, max_iter=120, max_leaf_nodes=15, lr=0.1):
        self.kw = dict(max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, learning_rate=lr,
                       l2_regularization=1.0, random_state=0)

    def _X(self, y, c, s):
        others = np.delete(y, s, axis=1)
        return np.concatenate([others, c[:, :1]], axis=1)

    def fit(self, normal_eps, stride=2):
        y = _stack(normal_eps, "y")[::stride]; c = _stack(normal_eps, "covar")[::stride]
        self.models = []
        for s in range(NS):
            m = HistGradientBoostingRegressor(**self.kw).fit(self._X(y, c, s), y[:, s])
            self.models.append(m)
        r = self._raw(y, c)
        self.mu = np.median(r, 0)
        self.sd = 1.4826 * np.median(np.abs(r - self.mu), 0) + 1e-9
        return self

    def _raw(self, y, c):
        return np.stack([y[:, s] - self.models[s].predict(self._X(y, c, s))
                         for s in range(NS)], 1)

    def residuals(self, y, covar):
        return (self._raw(y, covar) - self.mu) / self.sd

    def renormalise(self, normal_eps):
        """Cheap site adaptation: keep the predictors, re-estimate residual statistics only."""
        y = _stack(normal_eps, "y"); c = _stack(normal_eps, "covar")
        r = self._raw(y, c)
        self.mu = np.median(r, 0)
        self.sd = 1.4826 * np.median(np.abs(r - self.mu), 0) + 1e-9
        return self
