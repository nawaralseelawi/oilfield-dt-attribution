// Builds the manuscript .docx. Every number is read from results/*.json (no hand-typed results).
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel,
  AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber,
} = require("docx");

const S = JSON.parse(fs.readFileSync("results/summary.json").toString().replace(/\bNaN\b/g, "null"));
const BE = JSON.parse(fs.readFileSync("results/breakeven_wait.json"));
const A = S.aux_site1;
const PA = JSON.parse(fs.readFileSync("results/paired_ablation.json"));
const SITES = [1, 2, 3];
const st1 = S.stream_site1;
const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const f1 = (x) => Number(x).toFixed(1);
const pct = (x) => `${Math.round(100 * x)}%`;
const ci = (c, d = 2) => `[${c[0].toFixed(d)}, ${c[1].toFixed(d)}]`;
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const win = (s, n, k = "manifest") => S[`main_site${s}`].window[n][k];
const wmean = (n, key, k = "manifest") => mean(SITES.map((s) => win(s, n, k)[key]));
const M = (label) => st1[label].metrics;
const C = (label) => st1[label].ci;
const bySite = (label, key) => SITES.map((s) => S.stream_by_site[String(s)][label][key]);
const range = (a) => `${f2(Math.min(...a))}–${f2(Math.max(...a))}`;
const W1 = S.window_ci_site1;
const G = S.sensitivity_site1.grid;

// ---------------------------------------------------------------- docx helpers
const FONT = "Times New Roman";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: o.size || 22, ...o });
const para = (parts, o = {}) =>
  new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: o.align || AlignmentType.JUSTIFIED,
    ...o,
    children: (Array.isArray(parts) ? parts : [parts]).map((p) => (typeof p === "string" ? run(p) : p)),
  });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: t, font: FONT, size: 26, bold: true })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 }, children: [new TextRun({ text: t, font: FONT, size: 23, bold: true })] });
const bullet = (parts) =>
  new Paragraph({
    numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 270 }, alignment: AlignmentType.JUSTIFIED,
    children: (Array.isArray(parts) ? parts : [parts]).map((p) => (typeof p === "string" ? run(p) : p)),
  });
const it = (t) => run(t, { italics: true });
const bd = (t) => run(t, { bold: true });

function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, caption, maxW = 600) {
  const { w, h } = pngSize(file);
  const width = Math.min(maxW, w), height = Math.round((h * width) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height },
        altText: { title: caption.slice(0, 40), description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 180 },
      children: [run(caption, { size: 19 })] }),
  ];
}

const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const sz = opts.size || 18;
  const mkCell = (t, w, head, first, keep) =>
    new TableCell({
      borders: BORDERS, width: { size: w, type: WidthType.DXA },
      shading: head ? { fill: "DCE6F1", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 50, bottom: 50, left: 80, right: 80 },
      children: [new Paragraph({ alignment: first ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: keep,
        children: [new TextRun({ text: String(t), font: FONT, size: sz, bold: head })] })],
    });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => mkCell(h, widths[i], true, i === 0, true)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => mkCell(c, widths[i], false, i === 0, ri < rows.length - 1)) })),
    ],
  });
}
const cap = (t) => new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true, children: [run(t, { size: 19, bold: true })] });
const gap = () => new Paragraph({ spacing: { after: 100 }, children: [] });

// ---------------------------------------------------------------- numbers used in the text
const F1p = W1.proposed.y, F1a = W1.proposed.y_all;
const eceCal = SITES.map((s) => win(s, "proposed").ece), eceUn = SITES.map((s) => win(s, "proposed_uncalibrated").ece);
const cls1 = win(1, "proposed").f1_per_class;
const pm = {}; // per-mode recall by site
SITES.forEach((s) => Object.entries(win(s, "proposed").per_mode_recall).forEach(([k, v]) => { (pm[k] = pm[k] || []).push(v.recall); }));
const cP = C("Proposed (argmax)"), mA = M("Proposed (argmax)");
const mMin = M("Proposed (min-cost)"), mRev = M("Proposed (min-cost + review)"), mLate = M("Proposed (min-cost + review, decide 6 h later)");
const mGE = M("Generic detector -> escalate"), mAE = M("Attack-only detector -> escalate");
const be = SITES.map((s) => BE[String(s)].breakeven_attack_per_hour);
const gAll = Object.values(G); const polNames = Object.keys(gAll[0]);
const bestCounts = {}; Object.entries(S.sensitivity_site1.best).forEach(([k, v]) => { bestCounts[v] = (bestCounts[v] || 0) + 1; });
const allProposedBest = Object.values(S.sensitivity_site1.best).every((v) => v.startsWith("Proposed"));
const nGrid = Object.keys(G).length;
const E4 = A.E4_transfer, E5 = A.E5_stress, E3 = A.E3_magnitude;

// ---------------------------------------------------------------- content
const kids = [];
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Attributing Sensor Deviations to Degradation, Weather, or Attack in Oilfield Digital Twins: A Simulation Study of Probabilistic Attribution and Cost-Based Decisions", font: FONT, size: 32, bold: true })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run("[Author names and affiliations]", { italics: true })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run("DRAFT manuscript. All results are from simulation; see Section 9 before drawing conclusions.", { italics: true, size: 20, color: "9C2A00" })] }));

kids.push(H1("Abstract"));
kids.push(para(
  `When an oilfield digital twin disagrees with its instruments, the operator must decide whether the cause is hardware degradation, harsh-weather effects, or malicious data manipulation. Existing digital-twin work appears to treat these causes separately: sensor-validation architectures target faults, and attack-focused twins have been evaluated on water-sector testbeds. We study joint cause attribution on a simulated four-well wellpad with 15 coupled instruments, legitimate operating transients, and weather. A physics-informed twin, identified from normal data only, produces analytical-redundancy residuals; windowed features feed a gradient-boosted classifier whose posterior drives an alarm gate with a fixed false-alarm rate and a cost-based decision rule that may defer to an analyst. On three independently generated sites, the classifier reaches macro-F1 of ${f3(wmean("proposed", "macro_f1"))} on windows where the injected deviation is observable (${f3(F1a.macro_f1)} when latent post-onset windows are included at the primary site). The physics twin accounts for essentially all of this: removing the data-driven twin changes macro-F1 by less than 0.01, whereas removing all twins drops it to about ${f2(wmean("raw_only", "macro_f1"))}. Attacks are detected quickly (median ${f1(mA.delay_med_attack)} h) but attributed correctly at alarm time only ${pct(mA.attr_attack)} of the time, rising to ${pct(mA.attr_settled_attack)} six hours later. A cost-aware policy that defers ambiguous cases had the lowest expected cost among all policies in every one of ${nGrid} cost and prior settings tested, sometimes by a small margin; the result depends on illustrative costs and on analysts resolving deferred cases. A twin-aware attacker was detected in ${pct(E5.full.proposed.detect)} of episodes yet almost never attributed to attack. These results are conditional on the simulator's generative assumptions and have not been validated on field data.`));
kids.push(para([bd("Keywords: "), run("digital twin; oil and gas; sensor fault diagnosis; false data injection; cyber-physical security; calibrated classification; industrial control systems")], { align: AlignmentType.LEFT }));

// ---- 1 Introduction
kids.push(H1("1. Introduction"));
kids.push(para("A digital twin of an oil-and-gas asset is only as trustworthy as the instruments feeding it. In the field, those instruments face heat, dust, vibration, and long maintenance intervals, and, increasingly, network-connected gateways that an adversary may compromise. When the twin and an instrument disagree, three explanations compete: the instrument is degrading, the environment is distorting it, or someone is manipulating the value. The right response differs sharply. Degradation calls for a maintenance visit, weather effects for compensation or shielding, and manipulation for incident response and isolation of the affected gateway. Choosing wrongly is costly in both directions: dismissing an attack as wear leaves it running, and escalating every drifting transmitter to the security team exhausts analysts."));
kids.push(para("Prior work on digital-twin sensor validation and on attack detection has largely proceeded on separate tracks. To our reading, sensor fault detection, isolation and accommodation (SFDIA) for industrial twins [1, 2] addresses sensor failure, while attack-discriminating twins [3] address malicious manipulation and have been evaluated on water-treatment and water-distribution testbeds [4, 5]. We are not aware of work that jointly attributes a twin-instrument deviation to degradation, weather, or attack under oilfield-like conditions, although our literature search was brief and this claim should be checked before it is relied upon."));
kids.push(para("This paper makes four contributions, each limited to a simulation setting. First, we define the three-way attribution problem and an open simulator for it, including legitimate operating transients that a naive detector would flag. Second, we evaluate a physics-informed twin, identified from normal data only, as the source of residual features, against a data-driven twin and against no twin. Third, we report posterior reliability, detection delay and attribution quality as a function of time since alarm, and we show that a cost-aware decision rule which can defer to an analyst outperforms both argmax attribution and the ‘escalate everything’ baseline under a wide range of cost settings. Fourth, we report where the approach fails, including single-sensor faults versus attacks that are indistinguishable by construction and a twin-aware attacker."));
kids.push(para("We emphasize what this paper does not show. No field data were used. The simulator's assumptions determine what is identifiable, the cost matrix is illustrative, and every conclusion is conditional on both."));

// ---- 2 Related work
kids.push(H1("2. Background"));
kids.push(para("Model-based fault diagnosis builds residuals from analytical redundancy among measured variables and isolates faults by their signature across residuals [8]. Digital-twin variants combine physics with learned models; a recurrent graph-convolutional architecture for sensor fault detection, isolation and accommodation was reported for industrial twins [1], following an earlier real-time scheme [2]. On the security side, false data injection can evade residual-based detectors when the injected vector lies in the range of the measurement model [6], and replay attacks can be countered by exploiting the noise fingerprint of genuine measurements [7]. A recent self-defending twin discriminates single- and multi-stage attacks on SWaT and WADI [3, 4, 5]. Our ‘twin-aware’ stress test follows the undetectability logic of [6] but is deliberately imperfect (Section 9)."));
kids.push(para("For decision-making, we use gradient-boosted trees [11] with isotonic recalibration [9, 10] and evaluate calibration by expected calibration error (ECE) with ten bins."));

// ---- 3 Problem
kids.push(H1("3. Problem formulation and threat model"));
kids.push(para("At each time step an operator observes 15 instrument readings y_t and a weather-station record (ambient temperature and a dust index). A persistent deviation may begin at an unknown onset. The task is to (i) raise an alarm at a controlled false-alarm rate, (ii) produce a calibrated posterior over {normal, fault, environment, attack}, and (iii) choose an action from {ignore, maintenance, compensate, escalate, analyst review} minimizing expected cost."));
kids.push(para("The attacker can alter sensor values by compromising a gateway (several sensors on the same RTU) or a single field device. The weather station is assumed trustworthy. Actuator manipulation, network-layer attacks, denial of service, and attacks on the twin itself are out of scope."));

// ---- 4 Simulation
kids.push(H1("4. Simulated wellpad"));
kids.push(para("The simulator (open code accompanies this paper) represents four wells feeding a header, with 15 instruments (wellhead pressure, temperature and flow per well; header pressure, temperature and flow) served by three gateways. The true process satisfies mass balance, a quadratic choke/flowline pressure-flow relation, a header pressure–flow relation, and a first-order thermal lag at each wellhead. Legitimate operating changes (choke steps, short shut-ins, slugging) and weather (diurnal ambient temperature, random dust events) are always present, so ‘something changed’ is not evidence of an anomaly. Episodes last 72 h at 10-minute sampling; an anomaly, when present, starts between 25% and 60% of the episode and persists."));
kids.push(cap("Table 1. Generative model of the four causes."));
kids.push(table(["Cause", "Modes", "Affected sensors", "Distinguishing structure"], [
  ["Normal", "none", "none", "operating transients and weather present"],
  ["Fault", "drift, stuck, step, gain, noise bursts", "1 sensor (85%) or 2, independent", "sensor-local; independent of weather"],
  ["Environment", "heat, dust, both", "2–8 sensors", "coupled to ambient temperature / dust index"],
  ["Attack", "bias, ramp, replay, scale, mass-balance-preserving", "70%: 2+ sensors on one gateway; 30%: one device", "abrupt or coordinated; replay repeats the noise pattern"],
  ["Stress test only", "twin-aware (‘full’)", "7 sensors", "keeps all physics residuals near zero"],
], [1500, 2500, 2600, 2760], { size: 17 }));
kids.push(gap());
kids.push(para("Fault and attack magnitudes are drawn from the same log-uniform range (0.5–8 units, a unit being about three times instrument accuracy), and single-sensor drift or step faults are statistically similar to single-sensor ramp or bias attacks. These cases are therefore partly indistinguishable by construction; the classifier's confusion there reflects the simulator, not only the method."));

// ---- 5 Method
kids.push(H1("5. Method"));
kids.push(...figure("figures/fig1_framework.png", "Figure 1. Processing pipeline. The twin is identified from normal data only; the alarm threshold is set on held-out normal operation.", 560));
kids.push(H2("5.1 Twins and residuals"));
kids.push(para("The physics twin evaluates twelve analytical-redundancy relations (ARRs): header mass balance; one choke relation per well (pressure drop against squared flow); one wellhead temperature relation per well as a function of flow and ambient temperature; header temperature against the flow-weighted well temperature; and header pressure against total flow computed two ways. Functional forms come from process physics; parameters are identified from normal data only, with no simulator internals. Residuals are standardized with robust statistics from normal data. The data-driven twin predicts each sensor from all others and ambient temperature using gradient boosting trained on normal data, and its residuals are standardized the same way."));
kids.push(H2("5.2 Features"));
kids.push(para("Six-hour windows with a one-hour stride yield 494 features in eight groups (the proposed model uses seven groups, 382 features, and leaves out the raw-signal statistics that serve the no-twin baseline): residual statistics (arr, dd), z-scored raw-signal statistics (raw), instrument health (noise ratio and flatness), coupling of residuals to weather covariates (cov), sensor-isolation and gateway-level aggregates (topo), a replay fingerprint (maximum lagged correlation of a sensor's high-passed signal with earlier history), and operating context. Permutation-invariant aggregates (maxima, counts, per-gateway maxima) accompany per-sensor features."));
kids.push(H2("5.3 Attribution, alarm gate and decision"));
kids.push(para("A gradient-boosted classifier outputs class probabilities, recalibrated by isotonic regression on a separate calibration set. An alarm is raised when 1 − P(normal) exceeds a threshold for two consecutive windows; the threshold is set on 150 held-out normal episodes for 0.05 false alarms per day. Given the posterior p at alarm time, the policy chooses the action minimizing Σ_c p_c·Cost(c, action), with analyst review as an action of fixed cost. We also evaluate deciding six hours after the alarm using the mean posterior over that interval."));
kids.push(cap("Table 2. Illustrative operator cost matrix (rows: true cause; columns: action). Review resolves an attack with probability 0.95 (cost 3 + 0.05 × 60)."));
kids.push(table(["True cause", "Ignore", "Maintain", "Compensate", "Escalate", "Review"], [
  ["Normal", "0", "4", "2", "6", "3"], ["Fault", "10", "1", "3", "8", "3"],
  ["Environment", "6", "4", "1", "6", "3"], ["Attack", "60", "50", "50", "2", "6"],
], [1900, 1400, 1400, 1560, 1550, 1550]));
kids.push(gap());

// ---- 6 Protocol
kids.push(H1("6. Experimental protocol"));
kids.push(para("For each of three independently generated sites (different wells, climate and noise), we generate disjoint sets: 250 normal episodes to identify twins, 150 normal episodes to set the alarm threshold, 800 mixed episodes to train classifiers, 250 to calibrate probabilities, and 500 for testing. Class proportions in generated episodes are 30% normal, 25% fault, 22% environment and 23% attack. Baselines and ablations use the same splits and the same false-alarm target."));
kids.push(para([bd("Window labels. "), run("A post-onset window is ‘manifest’ if some sensor's RMS injected deviation within the window exceeds 1.5 instrument-noise standard deviations, a criterion fixed before results were inspected. Non-manifest post-onset windows are excluded from training and from the primary window metric, because no method could recover their label; we report results on all post-onset windows alongside. Detection delay is always measured from the true onset.")]));
kids.push(para([bd("Methods compared. "), run("Proposed (all feature groups); physics twin only; data-driven twin only; no twin (raw signals only); four leave-one-group-out ablations; and four detector baselines that share the alarm gate but cannot attribute: a generic anomaly detector followed by always-maintain or always-escalate, an attack-only detector trained on normal versus attack data, and a fault-only detector. Confidence intervals are episode-level bootstrap intervals (500 resamples for window metrics, 200 for streaming metrics).")]));
kids.push(para([bd("Deviations from a purely pre-specified protocol. "), run("Permutation-invariant aggregate features were added after an initial prototype run showed that replay attacks, which can hit any of 15 sensors, were learned poorly from per-sensor features (replay recall 0.47 rose to 0.75 on the prototype). This design decision was made on the same simulator, although evaluation used freshly generated data. The cost matrix, the manifest threshold and the false-alarm target were fixed before results.")]));

// ---- 7 Results
kids.push(H1("7. Results"));
kids.push(H2("7.1 Attribution quality and calibration"));
kids.push(para(`On manifest windows the proposed model reaches accuracy ${f3(F1p.accuracy)} and macro-F1 ${f3(F1p.macro_f1)} ${ci(F1p.ci, 3)} at the primary site, and ${SITES.map((s) => f3(win(s, "proposed").macro_f1)).join(", ")} at the three sites. Per-class F1 at the primary site is ${cls1.map(f2).join(" / ")} for normal, fault, environment and attack. Including latent post-onset windows lowers macro-F1 to ${f3(F1a.macro_f1)} ${ci(F1a.ci, 3)}. The dominant error is the fault–attack confusion (Figure 2, primary site): about 16% of fault windows are labelled attack and 14% of attack windows fault, and 19% of manifest fault windows are labelled normal.`));
kids.push(...figure("figures/fig3_confusion_reliability.png", "Figure 2. Left: row-normalized confusion matrix on manifest test windows, primary site (counts in parentheses). Right: reliability of the top-class posterior at the primary site, before and after isotonic recalibration.", 560));
kids.push(para(`Recalibration did not help consistently. ECE fell from ${f3(eceUn[0])} to ${f3(eceCal[0])} at the primary site (Figure 2, right), but rose from ${f3(eceUn[1])} to ${f3(eceCal[1])} and from ${f3(eceUn[2])} to ${f3(eceCal[2])} at the other two sites; at the primary site the recalibrated posterior is under-confident in the 0.6–0.9 band. The posterior is moderately reliable in every case (ECE ${f3(Math.min(...eceUn, ...eceCal))}–${f3(Math.max(...eceUn, ...eceCal))}), but our data do not support the claim that isotonic recalibration is necessary.`));
kids.push(H2("7.2 What contributes: ablations"));
const abl = [["proposed", "Proposed (all groups)"], ["arr_only", "Physics twin only"], ["dd_only", "Data-driven twin only"], ["raw_only", "No twin (raw signals)"], ["no_cov", "− weather coupling"], ["no_topo", "− topology / isolation"], ["no_replay", "− replay fingerprint"], ["no_inst", "− instrument health"]];
kids.push(cap("Table 3. Window-level macro-F1 on held-out windows. Primary site with 95% episode-bootstrap CI, mean over the three sites, and the paired difference from the full model pooled over the three sites (paired episode bootstrap, 2000 resamples)."));
const dlt = (k) => { const p = PA.manifest[k].pooled; return `${p.diff >= 0 ? "+" : "−"}${Math.abs(p.diff).toFixed(3)} [${p.ci[0] >= 0 ? "+" : "−"}${Math.abs(p.ci[0]).toFixed(3)}, ${p.ci[1] >= 0 ? "+" : "−"}${Math.abs(p.ci[1]).toFixed(3)}]`; };
kids.push(table(["Variant", "Manifest windows, primary site [95% CI]", "All post-onset windows", "Mean of 3 sites", "Δ vs proposed, pooled [paired 95% CI]"],
  abl.map(([k, l]) => [l, `${f3(W1[k].y.macro_f1)} ${ci(W1[k].y.ci, 3)}`, f3(W1[k].y_all.macro_f1), f3(wmean(k, "macro_f1")), k === "proposed" ? "—" : dlt(k)]),
  [2150, 2200, 1250, 1100, 2660], { size: 17 }));
kids.push(gap());
kids.push(para(`The physics twin carries almost all of the signal: the physics-only model differs from the full model by ${dlt("arr_only")} in macro-F1 (variant minus full, pooled over sites); the paired interval includes zero, so the data-driven twin's marginal contribution is indistinguishable from nothing here, to within about ±0.007. The data-driven twin alone is ${f3(-PA.manifest.dd_only.pooled.diff)} lower than the full model, and without any twin macro-F1 falls by ${f3(-PA.manifest.raw_only.pooled.diff)} to ${f3(wmean("raw_only", "macro_f1"))}. Each remaining group contributes a small but reliable amount: removing weather coupling costs ${f3(-PA.manifest.no_cov.pooled.diff)}, instrument health ${f3(-PA.manifest.no_inst.pooled.diff)}, the replay fingerprint ${f3(-PA.manifest.no_replay.pooled.diff)} and topology features ${f3(-PA.manifest.no_topo.pooled.diff)}, with paired intervals excluding zero and the same sign at all three sites. These intervals reflect sampling variability in the test episodes; they do not capture uncertainty about the simulator's assumptions, and the three sites share one generative process. Weather coupling matters most for the environment class (F1 ${f2(win(1, "proposed").f1_per_class[2])} versus ${f2(win(1, "no_cov").f1_per_class[2])} without it at the primary site).`));
kids.push(...figure("figures/fig4_ablation.png", "Figure 3. Macro-F1 by feature configuration at the primary site. Bars: manifest windows with 95% episode-bootstrap CI; diamonds: including latent post-onset windows.", 470));
kids.push(para(`Per-mode recall on manifest windows (mean over sites) is highest for stuck sensors (${f2(mean(pm.stuck))}), mass-balance-preserving attacks (${f2(mean(pm.massbal))}) and replay (${f2(mean(pm.replay))}), and lowest for drift (${f2(mean(pm.drift))}), noise bursts (${f2(mean(pm.noise))}), steps (${f2(mean(pm.step))}), gain errors (${f2(mean(pm.gain))}) and ramps (${f2(mean(pm.ramp))}); see Appendix A.`));

kids.push(H2("7.3 Streaming detection, attribution over time, and operator cost"));
kids.push(para(`At the alarm threshold set for 0.05 false alarms per day, the realized rate on test normal episodes was ${f3(mA.fa_per_day)} per day at the primary site. Median detection delay from true onset was ${f1(mA.delay_med_fault)} h for faults, ${f1(mA.delay_med_environment)} h for environment effects and ${f1(mA.delay_med_attack)} h for attacks; the 90th percentiles were ${f1(mA.delay_p90_fault)}, ${f1(mA.delay_p90_environment)} and ${f1(mA.delay_p90_attack)} h. Detection rates were ${pct(mA.detect_fault)}, ${pct(mA.detect_environment)} and ${pct(mA.detect_attack)}. Attribution improves with waiting (Table 4).`));
kids.push(cap("Table 4. Detection and attribution by cause (argmax attribution), primary site; attribution ranges over the three sites in parentheses."));
const rowsT4 = ["fault", "environment", "attack"].map((c) => {
  const key = (p) => `${p}_${c}`;
  return [c[0].toUpperCase() + c.slice(1), pct(mA[key("detect")]), f1(mA[key("delay_med")]),
    `${f2(mA[key("attr")])} (${range(bySite("Proposed (argmax)", key("attr")))})`,
    `${f2(mA[key("attr_settled")])} (${range(bySite("Proposed (argmax)", key("attr_settled")))})`];
});
kids.push(table(["Cause", "Detected", "Median delay (h)", "Correct at alarm", "Correct 6 h later"], rowsT4, [1500, 1300, 1800, 2380, 2380]));
kids.push(gap());
kids.push(para(`Attacks are detected fastest but attributed worst at the moment of alarm (${pct(mA.attr_attack)} correct), because the alarm fires after only two windows and early evidence resembles a fault. Six hours later, ${pct(mA.attr_settled_attack)} are attributed correctly. Fault and environment attribution changes little with waiting.`));
kids.push(cap("Table 5. Operator cost and attack handling, primary site (500 test episodes). Cost is the mean over episodes under Table 2, with equal class weights and with an illustrative deployment prior (88% normal, 5% fault, 5% environment, 2% attack). All methods share the alarm-gate false-alarm target. For policies with an analyst-review action, attacks not escalated or treated as benign were sent to review."));
const order = ["Proposed (argmax)", "Proposed (min-cost)", "Proposed (min-cost + review)", "Proposed (min-cost + review, decide 6 h later)", "Physics twin only (argmax)", "Data-driven twin only (argmax)", "No twin, raw signals (argmax)", "Generic detector -> maintenance", "Generic detector -> escalate", "Attack-only detector -> escalate", "Fault-only detector -> maintenance"];
kids.push(table(["Method", "Cost, equal weights [95% CI]", "Cost, deployment prior", "Attacks escalated", "Attacks missed or treated as benign", "Non-attacks escalated"],
  order.map((l) => [l.replace("->", "→"), `${f2(M(l).cost_equal)} ${ci(C(l).cost_equal)}`, f2(M(l).cost_deploy), f2(M(l).attack_escalated), f2(M(l).attack_missed_or_benign), f2(M(l).false_escalation_rate)]),
  [2900, 1900, 1150, 1050, 1300, 1060], { size: 16 }));
kids.push(gap());
kids.push(...figure("figures/fig5_cost.png", "Figure 4. Left: attack handling (numbers match the method list on the right). Right: mean operator cost with 95% bootstrap CI under the illustrative cost matrix.", 600));
kids.push(para(`Three observations follow. First, argmax attribution is not useful as an action rule under these costs: its equal-weight cost (${f2(mA.cost_equal)} ${ci(cP.cost_equal)}) is about twice that of escalating every alarm (${f2(mGE.cost_equal)}), because ${pct(mA.attack_missed_or_benign)} of attacks are labelled fault or environment at alarm time and treated as benign. Second, minimizing expected cost under the recalibrated posterior brings cost to ${f2(mMin.cost_equal)} ${ci(C("Proposed (min-cost)").cost_equal)}, comparable to the escalate-everything baselines whose intervals it overlaps at equal weights; its advantage is clearer at the deployment prior (${f2(mMin.cost_deploy)} versus ${f2(mGE.cost_deploy)} and ${f2(mAE.cost_deploy)}). Third, allowing analyst review lowers cost further to ${f2(mRev.cost_equal)} ${ci(C("Proposed (min-cost + review)").cost_equal)}, and deciding six hours after the alarm to ${f2(mLate.cost_equal)} ${ci(C("Proposed (min-cost + review, decide 6 h later)").cost_equal)}. Under the review policy ${pct(mRev.review_rate_anom)} of anomalous episodes are sent to review, so this result depends on review being effective, which we test next.`));

kids.push(H2("7.4 Sensitivity to the cost model"));
kids.push(para(`We recomputed expected cost over ${nGrid} settings: attack-miss cost 20–600, false-escalation cost 2–20, review effectiveness 0.5–1.0, and attack prior 0.5–10%. ${allProposedBest ? `A proposed variant had the lowest expected cost in every setting (the six-hour-deferral variant in ${bestCounts["Proposed (min-cost + review, 6 h later)"] || 0}, review at alarm time in ${bestCounts["Proposed (min-cost + review)"] || 0}, and plain minimum-cost in ${bestCounts["Proposed (min-cost)"] || 0})` : "The ranking changed across settings (see Figure 5)"}, including when review resolves only half of attacks. The margin is not always large. When false escalation is very cheap (cost 2 against a missed-attack cost of 60), escalating every alarm costs ${f2(G["miss=60|fe=2|r=0.95|pa=0.02"]["Generic -> escalate"])} versus ${f2(G["miss=60|fe=2|r=0.95|pa=0.02"]["Proposed (min-cost + review, 6 h later)"])} for the deferred policy, and attribution offers little. The six-hour deferral is not free: our cost model assigns no cost to waiting. Deciding at alarm time is preferable if an attack in progress inflicts more than about ${f2(Math.min(...be))}–${f2(Math.max(...be))} cost units per hour of extra dwell time (roughly ${f1((100 * Math.min(...be)) / 60)}–${f1((100 * Math.max(...be)) / 60)}% of the missed-attack cost per hour), which is a low bar.`));
kids.push(...figure("figures/fig8_sensitivity.png", "Figure 5. Expected cost per episode (deployment-style prior) as attack-miss cost, analyst-review effectiveness and attack prior vary; other parameters as labelled above each panel.", 600));

kids.push(H2("7.5 Deviation magnitude"));
const e3 = (c, m, k) => E3[c][m][k];
kids.push(para(`Scaling deviation magnitude from 0.25× to 4× the nominal distribution raises detection of faults from ${pct(e3("fault", "0.25", "detect"))} to ${pct(e3("fault", "4.0", "detect"))} and of environment effects from ${pct(e3("environment", "0.25", "detect"))} to ${pct(e3("environment", "4.0", "detect"))}. Attack detection stays at ${pct(e3("attack", "0.25", "detect"))}–${pct(e3("attack", "4.0", "detect"))} across the range. This should not be read as robustness to stealthy attacks: the multiplier does not scale replay attacks at all and mass-balance-preserving attacks remain large in absolute terms, so the sweep varies amplitude, not adversarial stealth. Attribution among detected episodes is not monotone in magnitude because small deviations are detected only when they are unusually distinctive (a selection effect); the joint rate of detecting and correctly attributing attacks rises from ${pct(e3("attack", "0.25", "joint"))} to ${pct(e3("attack", "4.0", "joint"))}.`));
kids.push(...figure("figures/fig6_magnitude.png", "Figure 6. Detection rate and correct attribution among detected episodes versus deviation magnitude, primary site, 100 episodes per cause and magnitude.", 520));

kids.push(H2("7.6 New sites and stress tests"));
kids.push(para(`Applying the primary-site pipeline unchanged to a new simulated site (different wells, a warmer climate and higher instrument noise) fails: macro-F1 ${f2(E4.A_no_adaptation.macro_f1)} and ${f2(E4.A_no_adaptation.fa_per_day)} false alarms per day. Re-identifying only the twin (physics parameters and residual statistics) on 60 normal episodes from the new site restores macro-F1 to ${f2(E4.B_twin_reidentified.macro_f1)} with ${f3(E4.B_twin_reidentified.fa_per_day)} false alarms per day; recalibrating the alarm threshold as well changes little (${f3(E4.C_twin_and_threshold.fa_per_day)}). A classifier retrained on the new site reached ${f2(E4.D_retrained_oracle.macro_f1)}, but it used only 300 training episodes against 800, so this is not an upper bound. Transfer is easy here because every simulated site follows the same functional forms; real sites will violate them.`));
kids.push(cap("Table 6. Site transfer (300 test episodes at a new simulated site)."));
kids.push(table(["Configuration", "Macro-F1", "ECE", "False alarms / day", "Attack detected", "Attack attributed at alarm"],
  [["A. No adaptation", "A_no_adaptation"], ["B. Twin re-identified", "B_twin_reidentified"], ["C. Twin + threshold re-set", "C_twin_and_threshold"], ["D. Retrained on new site", "D_retrained_oracle"]]
    .map(([l, k]) => [l, f2(E4[k].macro_f1), f3(E4[k].ece), f3(E4[k].fa_per_day), f2(E4[k].detect_attack), f2(E4[k].attr_attack)]),
  [2600, 1100, 1000, 1500, 1500, 1660]));
kids.push(para(`In configuration A every alarm was labelled attack (attribution of faults and environment effects was ${f2(E4.A_no_adaptation.attr_fault)} and ${f2(E4.A_no_adaptation.attr_env)}), so its attack-attribution figure of ${f2(E4.A_no_adaptation.attr_attack)} is an artifact of labelling everything as attack, not a success.`, { spacing: { before: 100, after: 120, line: 276 } }));
kids.push(para(`Stress tests use 100 attack episodes each (Table 7; single run, no confidence intervals). Mass-balance-preserving and replay attacks are always detected, but attribution to attack at alarm time is only ${pct(E5.massbal.proposed.attr)} and ${pct(E5.replay.proposed.attr)}. Removing the replay-fingerprint feature raised at-alarm attribution for both (to ${pct(E5.massbal.no_replay.attr)} and ${pct(E5.replay.no_replay.attr)}), so the feature's window-level gain (Section 7.2; replay recall on manifest windows is ${f2(Math.min(...pm.replay))}–${f2(Math.max(...pm.replay))} across sites) does not carry over to alarm-time attribution of these attacks. A possible reason is that the fingerprint needs a window that is mostly replayed and so is weak at the first alarm; we did not test this. The twin-aware attacker was detected in ${pct(E5.full.proposed.detect)} of episodes against roughly ${pct(E5.full.chance_detect)} expected by chance, but attributed to attack in ${pct(E5.full.proposed.attr)}. Detection here most plausibly reflects residual leakage because our attacker ignores thermal lag, not a designed defense.`));
kids.push(cap("Table 7. Attack stress tests, primary site."));
kids.push(table(["Attack", "Detected (proposed)", "Attributed to attack at alarm", "Detected (no replay feature)", "Attributed (no replay feature)", "Chance detection"],
  [["Mass-balance-preserving", "massbal"], ["Replay", "replay"], ["Twin-aware (‘full’)", "full"]]
    .map(([l, k]) => [l, f2(E5[k].proposed.detect), f2(E5[k].proposed.attr), f2(E5[k].no_replay.detect), f2(E5[k].no_replay.attr), f2(E5[k].chance_detect)]),
  [2400, 1400, 1500, 1500, 1500, 1060]));
kids.push(gap());
kids.push(...figure("figures/fig7_transfer_stress.png", "Figure 7. Left and centre: transfer to a new simulated site under four adaptation levels (dashed line: false-alarm target). Right: detection rate under attack stress tests versus chance.", 600));

kids.push(H2("7.7 Illustrative episodes"));
kids.push(...figure("figures/fig2_examples.png", "Figure 8. One episode per cause (first random draws meeting a structural condition, not selected for performance). Top: measured signal and unobservable counterfactual. Middle: physics-residual magnitude by relation (start-up transient at left is a twin initialization effect). Bottom: attribution posterior; vertical black line marks the alarm. The fault example is a subtle drift that never triggers an alarm within 72 h.", 590));

// ---- 8 Discussion
kids.push(H1("8. Discussion"));
kids.push(para("What the evidence supports, within the simulator: a physics-informed twin identified from normal data supplies most of the information needed to separate causes; reasonably reliable posteriors (ECE ${f3(Math.min(...eceUn, ...eceCal))}–${f3(Math.max(...eceUn, ...eceCal))}) are usable for cost-aware decisions even though attribution at alarm time is often uncertain; and the practical value lies in deferral (analyst review or waiting) rather than in point predictions. Argmax attribution, taken literally, is worse than escalating everything under a cost structure where a missed attack is expensive."));
kids.push(para("What it does not support: that the data-driven twin or isotonic recalibration are worthwhile (they were not, here; topology features gave only a small gain of about 0.007 macro-F1); that attribution works on real instruments; that the cost advantage survives a real cost structure; or that a capable adversary can be attributed. The consistent results across three sites reflect three draws from one generative process, not three independent fields."));
kids.push(para("An operational reading is a two-stage response: raise the alarm quickly, treat early attribution as provisional, and commit to an attribution after the evidence has accumulated, with fault-versus-attack ambiguity routed to an analyst. Whether that trade-off is acceptable depends on the dwell-time cost of attacks, which the break-even analysis in Section 7.4 quantifies but which only operators can estimate."));

// ---- 9 Limitations
kids.push(H1("9. Limitations"));
[
  "Simulation only. Every result depends on the generative assumptions, and the simulator's overlap between single-sensor faults and attacks sets an identifiability floor that a real system may or may not share.",
  "The cost matrix is illustrative and was chosen by the authors, not elicited from operators. Sensitivity analysis covers four of its parameters, not its structure.",
  "The decision-time comparison ignores the cost of waiting; break-even values are reported instead.",
  "Attackers are limited to false-data injection on sensor values via gateways or field devices. The twin-aware attacker ignores thermal dynamics and is not an optimal adversary. Adaptive attackers who observe the detector are not modelled.",
  "The weather station is assumed trustworthy; spoofing it would defeat the weather-coupling features.",
  "One process model, one topology, one sampling rate. The ARR structure is specific to a four-well pad.",
  "Aggregate features were added after seeing prototype results on the same simulator; the manifest criterion excludes windows that no method could label, which applies to all methods alike but changes absolute numbers (results on all windows are reported).",
  "Related-work characterizations rest on a brief literature search, and the claim that no prior work jointly attributes these three causes has not been established by a systematic review. References 4–11 have not been checked.",
].forEach((t) => kids.push(bullet(t)));

// ---- 10 Conclusion
kids.push(H1("10. Conclusion"));
kids.push(para("On a simulated oilfield wellpad, a physics-informed twin plus probabilistic, cost-aware decision-making can separate sensor degradation, weather effects and attacks well enough to reduce operator cost relative to escalating every alarm, chiefly by deferring ambiguous cases. The approach cannot resolve single-sensor faults from single-sensor attacks that the simulator makes indistinguishable, and its behaviour under a knowledgeable attacker is untested beyond a deliberately imperfect stress test. The natural next step is validation on real historian data with labelled incidents; an adapter for that purpose accompanies the code."));

// ---- Data and code
kids.push(H1("Data and code availability"));
kids.push(para("Simulator, twins, feature extraction, experiments, analysis and figure code accompany this manuscript, with a script that reproduces all results. No field data were used."));

// ---- References
kids.push(H1("References"));
kids.push(para([it("Entries 1–3 were checked against the source pages during drafting. Entries 4–11 are cited from memory and have not been checked; verify authors, venue, pages and DOI before submission.")], { align: AlignmentType.LEFT }));
[
  "[1] H. Darvishi, D. Ciuonzo, P. Salvo Rossi, “Deep recurrent graph convolutional architecture for sensor fault detection, isolation, and accommodation in digital twins,” IEEE Sensors Journal, vol. 23, no. 23, pp. 29877–29891, Dec. 2023.",
  "[2] H. Darvishi, D. Ciuonzo, P. Salvo Rossi, “Real-time sensor fault detection, isolation and accommodation for industrial digital twins,” in Proc. IEEE ICNSC, 2021, doi:10.1109/ICNSC52481.2021.9702175.",
  "[3] M. Homaei, I. Khazrak, R. Molano, A. Caro, M. Ávila, “Cyber-resilient digital twins: discriminating attacks for safe critical infrastructure control,” arXiv:2603.18613, Mar. 2026.",
  "[4] J. Goh, S. Adepu, K. N. Junejo, A. Mathur, “A dataset to support research in the design of secure water treatment systems,” in Proc. CRITIS, 2016.",
  "[5] C. M. Ahmed, V. R. Palleti, A. P. Mathur, “WADI: a water distribution testbed for research in the design of secure cyber physical systems,” in Proc. CySWater, 2017.",
  "[6] Y. Liu, P. Ning, M. K. Reiter, “False data injection attacks against state estimation in electric power grids,” ACM Trans. Inf. Syst. Secur., vol. 14, no. 1, 2011.",
  "[7] Y. Mo, B. Sinopoli, “Secure control against replay attacks,” in Proc. Allerton Conf., 2009.",
  "[8] R. Isermann, “Model-based fault-detection and diagnosis: status and applications,” Annual Reviews in Control, vol. 29, no. 1, 2005.",
  "[9] A. Niculescu-Mizil, R. Caruana, “Predicting good probabilities with supervised learning,” in Proc. ICML, 2005.",
  "[10] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.",
  "[11] F. Pedregosa et al., “Scikit-learn: machine learning in Python,” J. Mach. Learn. Res., vol. 12, 2011.",
].forEach((t) => kids.push(new Paragraph({ spacing: { after: 70 }, indent: { left: 360, hanging: 360 }, children: [run(t, { size: 20 })] })));

// ---- Appendix
kids.push(H1("Appendix A. Per-mode recall"));
kids.push(cap("Table A1. Recall of the proposed classifier on manifest windows by injection mode, at each site."));
const modeOrder = [["stuck", "Fault: stuck"], ["drift", "Fault: drift"], ["step", "Fault: step"], ["gain", "Fault: gain"], ["noise", "Fault: noise bursts"], ["heat", "Environment: heat"], ["dust", "Environment: dust"], ["both", "Environment: both"], ["bias", "Attack: bias"], ["ramp", "Attack: ramp"], ["replay", "Attack: replay"], ["scale", "Attack: scale"], ["massbal", "Attack: mass-balance"]];
kids.push(table(["Mode", "Site 1", "Site 2", "Site 3", "Mean"], modeOrder.map(([k, l]) => [l, ...pm[k].map(f2), f2(mean(pm[k]))]), [3400, 1490, 1490, 1490, 1490]));
kids.push(gap());
kids.push(para("Recall is the fraction of manifest windows of that mode assigned to the correct cause (not the correct mode)."));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Draft", title: "Cause attribution of sensor deviations in oilfield digital twins",
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, right: 1440, bottom: 1300, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: kids,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("oilfield_twin_attribution_manuscript.docx", b); console.log("manuscript written", b.length); });
