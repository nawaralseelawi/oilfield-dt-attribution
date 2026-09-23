"""Generate all manuscript figures from results/ (run after run_all.sh and analysis.py)."""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from oilfield_dt import sim, pipeline as pl
from oilfield_dt.twin import ARR_NAMES

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.frameon": False})
COL = {"normal": "#7f7f7f", "fault": "#0072B2", "environment": "#009E73", "attack": "#D55E00"}
CN = ["normal", "fault", "environment", "attack"]
S = json.load(open("results/summary.json"))
Z = np.load("results/test_probs_site1.npz", allow_pickle=True)
OUT = "figures/"


def save(fig, name):
    fig.savefig(OUT + name + ".png"); plt.close(fig); print("saved", name)


# ------------------------------------------------------------------ F1 framework
def fig_framework():
    fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)
    W_, H_ = 27, 13
    top = [(3, "1  Field data", "15 instruments + weather\nstation (ambient T, dust)"),
           (36.5, "2  Digital twin", "physics ARR residuals +\ndata-driven residuals"),
           (70, "3  Window features", "6 h windows, 1 h stride:\nresidual, health, weather\ncoupling, topology, replay")]
    bot = [(70, "4  Calibrated attribution", "P(normal, fault,\nenvironment, attack)"),
           (36.5, "5  Alarm gate", "1 - P(normal) above threshold\nfor 2 h; fixed false-alarm rate"),
           (3, "6  Cost-based decision", "ignore / maintain / compensate /\nescalate / analyst review")]
    def box(x, y, title, body):
        ax.add_patch(FancyBboxPatch((x, y), W_, H_, boxstyle="round,pad=0.5", fc="#eef3f8", ec="#2b4a6b", lw=1))
        ax.text(x + W_ / 2, y + H_ - 3.2, title, ha="center", va="center", fontsize=7.6, weight="bold")
        ax.text(x + W_ / 2, y + 4.3, body, ha="center", va="center", fontsize=6.6, linespacing=1.25)
    for x, t, b in top: box(x, 27, t, b)
    for x, t, b in bot: box(x, 5, t, b)
    arr = dict(arrowstyle="-|>", mutation_scale=10, color="#2b4a6b", lw=1.1)
    ax.add_patch(FancyArrowPatch((3 + W_ + 1.2, 33.5), (36.5 - 1.2, 33.5), **arr))
    ax.add_patch(FancyArrowPatch((36.5 + W_ + 1.2, 33.5), (70 - 1.2, 33.5), **arr))
    ax.add_patch(FancyArrowPatch((83.5, 27 - 0.8), (83.5, 18 + 1.2), **arr))
    ax.add_patch(FancyArrowPatch((70 - 1.2, 11.5), (36.5 + W_ + 1.2, 11.5), **arr))
    ax.add_patch(FancyArrowPatch((36.5 - 1.2, 11.5), (3 + W_ + 1.2, 11.5), **arr))
    ax.text(50, 44.5, "Cause attribution of sensor deviations in an oilfield digital twin", ha="center", fontsize=9, weight="bold")
    ax.text(50, 22.6, "twin identified from normal data only; re-identified when moving to a new site", ha="center", fontsize=6.4, style="italic", color="#444")
    ax.text(50, 0.8, "alarm threshold set on held-out normal operation (not on attack data); review = human resolves ambiguity", ha="center", fontsize=6.4, style="italic", color="#444")
    save(fig, "fig1_framework")


# ------------------------------------------------------------------ F2 examples
def fig_examples():
    ex = pickle.load(open("results/example_episodes_site1.pkl", "rb"))
    fig, axs = plt.subplots(3, 3, figsize=(7.4, 6.2), sharex="col",
                            gridspec_kw=dict(height_ratios=[1, 1.2, 1]))
    for j, k in enumerate(["fault", "environment", "attack"]):
        e = ex[k]; T = e["y"].shape[0]; t = np.arange(T) / sim.STEPS_PER_HOUR
        s = e["sensors"][0]
        a = axs[0, j]
        a.plot(t, e["y"][:, s], color=COL[k], lw=0.9, label=f"measured {sim.NAMES[s]}")
        a.plot(t, e["y_nominal"][:, s], color="k", lw=0.7, ls="--", label="counterfactual (unobservable)")
        a.axvline(e["onset"] / sim.STEPS_PER_HOUR, color="#999", lw=0.8)
        a.set_title(f"{k.capitalize()} ({e['mode']}, {len(e['sensors'])} sensor{'s' if len(e['sensors'])>1 else ''})", color=COL[k])
        a.legend(fontsize=6, loc="best")
        if j == 0: a.set_ylabel("sensor value")
        b = axs[1, j]
        im = b.imshow(np.clip(np.abs(e["R"]).T, 0, 6), aspect="auto", cmap="magma_r",
                      extent=[0, T / sim.STEPS_PER_HOUR, len(ARR_NAMES) - 0.5, -0.5], vmin=0, vmax=6)
        b.set_yticks(range(len(ARR_NAMES))); b.set_yticklabels(ARR_NAMES if j == 0 else [], fontsize=5.5)
        b.axvline(e["onset"] / sim.STEPS_PER_HOUR, color="c", lw=0.8)
        if j == 0: b.set_ylabel("physics residual |z|")
        c = axs[2, j]
        we = e["ends"] / sim.STEPS_PER_HOUR
        c.stackplot(we, e["P"].T, colors=[COL[n] for n in CN], alpha=0.85)
        al = pl.first_alarm(1 - e["P"][:, 0], e["tau"], 2, 0)
        c.axvline(e["onset"] / sim.STEPS_PER_HOUR, color="k", lw=0.8, ls=":")
        if al >= 0:
            c.axvline(we[al], color="k", lw=1.2); c.text(we[al] + 1, 0.06, "alarm", fontsize=6.5, color="w")
        c.set_ylim(0, 1); c.set_xlabel("time (h)")
        if j == 0: c.set_ylabel("posterior probability")
    h = [plt.Rectangle((0, 0), 1, 1, fc=COL[n]) for n in CN]
    fig.legend(h, CN, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(); save(fig, "fig2_examples")


# ------------------------------------------------------------------ F3 confusion + reliability
def fig_confusion_reliability():
    y, P, Pu = Z["y"], Z["P"], Z["Pu"]; m = y >= 0
    P = P / P.sum(1, keepdims=True); Pu = Pu / Pu.sum(1, keepdims=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    C = np.zeros((4, 4))
    np.add.at(C, (y[m], P[m].argmax(1)), 1)
    Cn = C / C.sum(1, keepdims=True)
    a.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{Cn[i, j]:.2f}\n({int(C[i, j])})", ha="center", va="center", fontsize=7,
                   color="w" if Cn[i, j] > 0.55 else "k")
    a.set_xticks(range(4)); a.set_yticks(range(4))
    a.set_xticklabels(["normal", "fault", "env.", "attack"]); a.set_yticklabels(["normal", "fault", "env.", "attack"])
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title("Confusion (manifest windows)")
    edges = np.linspace(0, 1, 11)
    for Pm, name, ls, mk in ((Pu, "uncalibrated", "--", "s"), (P, "isotonic-calibrated", "-", "o")):
        conf = Pm[m].max(1); acc = (Pm[m].argmax(1) == y[m]).astype(float)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (conf > lo) & (conf <= hi)
            if s.sum() > 30:
                xs.append(conf[s].mean()); ys.append(acc[s].mean())
        b.plot(xs, ys, ls=ls, marker=mk, ms=3.5, label=name)
    b.plot([0, 1], [0, 1], color="#999", lw=0.8)
    b.set_xlabel("top-class confidence"); b.set_ylabel("empirical accuracy"); b.legend(fontsize=7)
    b.set_title("Reliability of the attribution posterior")
    fig.tight_layout(); save(fig, "fig3_confusion_reliability")


# ------------------------------------------------------------------ F4 ablation
def fig_ablation():
    W = S["window_ci_site1"]
    order = ["proposed", "arr_only", "dd_only", "raw_only", "no_cov", "no_topo", "no_replay", "no_inst"]
    lab = {"proposed": "Proposed (all groups)", "arr_only": "Physics twin only", "dd_only": "Data-driven twin only",
           "raw_only": "No twin (raw signals)", "no_cov": "− weather coupling", "no_topo": "− topology/isolation",
           "no_replay": "− replay fingerprint", "no_inst": "− instrument health"}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(order))[::-1]
    for yi, n in zip(y, order):
        v = W[n]["y"]
        ax.barh(yi, v["macro_f1"], color="#0072B2" if n == "proposed" else "#9ecae1", height=0.6)
        ax.errorbar(v["macro_f1"], yi, xerr=[[v["macro_f1"] - v["ci"][0]], [v["ci"][1] - v["macro_f1"]]],
                    color="k", capsize=2, lw=0.8)
        ax.plot(W[n]["y_all"]["macro_f1"], yi, "D", color="#D55E00", ms=4)
    ax.set_yticks(y); ax.set_yticklabels([lab[n] for n in order])
    ax.set_xlabel("macro-F1 on held-out windows (95% episode-bootstrap CI)")
    ax.plot([], [], "D", color="#D55E00", ms=4, label="macro-F1 including latent post-onset windows")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1)
    ax.set_xlim(0, 1); save(fig, "fig4_ablation")


# ------------------------------------------------------------------ F5 cost / trade-off
def fig_cost():
    R = S["stream_site1"]
    names = list(R)
    num = {n: i + 1 for i, n in enumerate(names)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    mk = {"Proposed": ("o", "#0072B2"), "Physics": ("s", "#56B4E9"), "Data-driven": ("s", "#56B4E9"),
          "No twin": ("s", "#999999"), "Generic": ("^", "#E69F00"), "Attack-only": ("v", "#D55E00"),
          "Fault-only": ("v", "#CC79A7")}
    for n in names:
        m, ci = R[n]["metrics"], R[n]["ci"]
        sym, col = next(v for k, v in mk.items() if n.startswith(k))
        x, y = m["false_escalation_rate"], m["attack_missed_or_benign"]
        a.errorbar(x, y, xerr=[[x - ci["false_escalation_rate"][0]], [ci["false_escalation_rate"][1] - x]],
                   yerr=[[y - ci["attack_missed_or_benign"][0]], [ci["attack_missed_or_benign"][1] - y]],
                   fmt=sym, color=col, ms=5, lw=0.6, capsize=1.5)
        off = {8: (-11, 4), 11: (5, 4)}.get(num[n], (4, 4))
        a.annotate(str(num[n]), (x, y), xytext=off, textcoords="offset points", fontsize=7)
    a.set_xlabel("false-escalation rate\n(non-attack episodes escalated)")
    a.set_ylabel("attacks missed or treated as benign\n(analyst review counts as handled)")
    a.set_title("Handling of attacks"); a.set_xlim(-0.04, 0.85); a.set_ylim(-0.05, 1.08)
    order = sorted(names, key=lambda n: R[n]["metrics"]["cost_equal"])
    yy = np.arange(len(order))[::-1]
    for yi, n in zip(yy, order):
        m, ci = R[n]["metrics"], R[n]["ci"]
        v = m["cost_equal"]
        b.barh(yi, v, color="#0072B2" if n.startswith("Proposed") else "#bdbdbd", height=0.65)
        b.errorbar(v, yi, xerr=[[v - ci["cost_equal"][0]], [ci["cost_equal"][1] - v]], color="k", capsize=2, lw=0.8)
    lab = {"Proposed (min-cost + review, decide 6 h later)": "Proposed: min-cost + review, decide 6 h later",
           "Attack-only detector -> escalate": "Attack-only detector, escalate",
           "Fault-only detector -> maintenance": "Fault-only detector, maintain",
           "Generic detector -> escalate": "Generic detector, escalate",
           "Generic detector -> maintenance": "Generic detector, maintain"}
    b.set_yticks(yy); b.set_yticklabels([f"{num[n]}  {lab.get(n, n)}" for n in order], fontsize=6.6)
    b.set_xlabel("mean operator cost per episode\n(equal class weights, 95% CI)")
    b.set_title("Operator cost (illustrative matrix)")
    fig.tight_layout(); save(fig, "fig5_cost")


# ------------------------------------------------------------------ F6 magnitude
def fig_magnitude():
    E = S["aux_site1"]["E3_magnitude"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3))
    for cn in ("fault", "environment", "attack"):
        mg = sorted(float(k) for k in E[cn]); d = [E[cn][str(k)]["detect"] for k in mg]
        at = [E[cn][str(k)]["attr"] for k in mg]
        a.plot(mg, d, "-o", color=COL[cn], ms=3.5, label=cn); b.plot(mg, at, "-o", color=COL[cn], ms=3.5, label=cn)
    for ax_, t in ((a, "Detection rate (episode)"), (b, "Correct attribution among detected")):
        ax_.set_xscale("log", base=2); ax_.set_xlabel("deviation magnitude (× nominal distribution)")
        ax_.set_ylim(0, 1.02); ax_.set_title(t)
    a.legend(fontsize=7); fig.tight_layout(); save(fig, "fig6_magnitude")


# ------------------------------------------------------------------ F7 transfer + stress
def fig_transfer_stress():
    E4, E5 = S["aux_site1"]["E4_transfer"], S["aux_site1"]["E5_stress"]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    tags = ["A_no_adaptation", "B_twin_reidentified", "C_twin_and_threshold", "D_retrained_oracle"]
    short = ["A\nnone", "B\ntwin", "C\ntwin+τ", "D\nretrain"]
    axs[0].bar(range(4), [E4[t]["macro_f1"] for t in tags], color="#0072B2"); axs[0].set_xticks(range(4)); axs[0].set_xticklabels(short, fontsize=7)
    axs[0].set_title("Macro-F1, new site"); axs[0].set_ylim(0, 1)
    axs[1].bar(range(4), [E4[t]["fa_per_day"] for t in tags], color="#D55E00"); axs[1].axhline(0.05, color="k", ls="--", lw=0.8)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(short, fontsize=7); axs[1].set_title("False alarms / day (target 0.05)")
    modes = ["massbal", "replay", "full"]; x = np.arange(3); w = 0.27
    axs[2].bar(x - w, [E5[m]["proposed"]["detect"] for m in modes], w, label="proposed", color="#0072B2")
    axs[2].bar(x, [E5[m]["no_replay"]["detect"] for m in modes], w, label="− replay feature", color="#9ecae1")
    axs[2].bar(x + w, [E5[m]["chance_detect"] for m in modes], w, label="chance", color="#bdbdbd")
    axs[2].set_xticks(x); axs[2].set_xticklabels(["mass-balance\npreserving", "replay", "twin-aware\n('full')"], fontsize=7)
    axs[2].set_title("Stress tests: detection rate"); axs[2].set_ylim(0, 1.32); axs[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[2].legend(fontsize=6.5, ncol=3, loc="upper center", columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(); save(fig, "fig7_transfer_stress")


# ------------------------------------------------------------------ F8 sensitivity
def fig_sensitivity():
    G = S["sensitivity_site1"]["grid"]
    pol = list(next(iter(G.values())))
    sty = {"Proposed (argmax)": ("-", "#56B4E9"), "Proposed (min-cost)": ("-", "#0072B2"),
           "Proposed (min-cost + review)": ("-", "#002b5c"),
           "Proposed (min-cost + review, 6 h later)": ("-", "#6a3d9a"), "Generic -> maintenance": ("--", "#E69F00"),
           "Generic -> escalate": ("--", "#D55E00"), "Attack-only -> escalate": (":", "#CC79A7"),
           "Fault-only -> maintenance": (":", "#999999")}
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3))
    miss = [20, 60, 200, 600]
    for p in pol:
        axs[0].plot(miss, [G[f"miss={m}|fe=6|r=0.95|pa=0.02"][p] for m in miss], marker="o", ms=3, ls=sty[p][0], color=sty[p][1], label=p)
    axs[0].set_xscale("log"); axs[0].set_xticks(miss); axs[0].set_xticklabels([str(m) for m in miss])
    axs[0].minorticks_off(); axs[0].set_xlabel("cost of a missed attack"); axs[0].set_ylabel("expected cost / episode")
    rr = [0.5, 0.8, 0.95, 1.0]
    for p in pol:
        axs[1].plot(rr, [G[f"miss=200|fe=6|r={r}|pa=0.02"][p] for r in rr], marker="o", ms=3, ls=sty[p][0], color=sty[p][1])
    axs[1].set_xlabel("analyst-review effectiveness")
    pa = [0.005, 0.02, 0.1]
    for p in pol:
        axs[2].plot(pa, [G[f"miss=60|fe=6|r=0.95|pa={a}"][p] for a in pa], marker="o", ms=3, ls=sty[p][0], color=sty[p][1])
    axs[2].set_xscale("log"); axs[2].set_xticks(pa); axs[2].set_xticklabels(["0.5%", "2%", "10%"])
    axs[2].minorticks_off(); axs[2].set_xlabel("attack prior (per episode)")
    axs[0].set_title("vs. attack-miss cost\n(fe=6, r=0.95, prior 2%)", fontsize=8.5)
    axs[1].set_title("vs. review effectiveness\n(miss=200, fe=6, prior 2%)", fontsize=8.5)
    axs[2].set_title("vs. attack prior\n(miss=60, fe=6, r=0.95)", fontsize=8.5)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(); save(fig, "fig8_sensitivity")


if __name__ == "__main__":
    fig_framework(); fig_examples(); fig_confusion_reliability(); fig_ablation()
    fig_cost(); fig_magnitude(); fig_transfer_stress(); fig_sensitivity()
