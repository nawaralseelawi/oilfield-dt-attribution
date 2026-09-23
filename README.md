# Cause attribution of sensor deviations in an oilfield digital twin

Simulation-based implementation of a research gap: given a deviation between a digital twin and
its instruments on an oil-and-gas wellpad, decide whether it is **hardware degradation (fault)**,
**weather-driven degradation (environment)** or **malicious manipulation (attack)**, with calibrated
confidence, and turn that into a cost-aware operator decision.

> **Everything here runs on a simulator whose assumptions are the paper's assumptions.**
> No field data were used. Results show the method works *under those generative assumptions*;
> they do not show it works on a real wellpad. `adapter.py` exists so that step can be taken.

## Layout

| path | purpose |
|---|---|
| `oilfield_dt/sim.py` | wellpad simulator (4 wells + header, 15 instruments, 3 gateways), fault / environment / attack injectors |
| `oilfield_dt/twin.py` | physics twin (analytical-redundancy residuals, parameters identified from normal data) and data-driven twin |
| `oilfield_dt/features.py` | 6 h / 1 h-stride windows; feature groups `arr dd raw inst cov topo replay ctx` |
| `oilfield_dt/pipeline.py` | classifiers, isotonic calibration, alarm gate, per-episode tables, cost model, bootstrap |
| `run_main.py` | one site: train, calibrate, evaluate all methods and ablations |
| `aux_experiments.py` | magnitude sweep, site transfer, stress tests, example episodes |
| `analysis.py` | bootstrap CIs, cost/prior sensitivity, cross-site aggregation -> `results/summary.json` |
| `make_figures.py` | all manuscript figures -> `figures/` |
| `train_bundle.py`, `adapter.py` | apply the pipeline to a real historian CSV (see limits in `adapter.py`) |
| `paired_ablation.py`, `breakeven.py` | paired-bootstrap ablation tests; break-even cost of deferring the decision |
| `build_manuscript.js` | builds the Word manuscript from `results/` (all numbers read from files) |
| `run_all.sh` | full reproduction |

## Reproduce

```bash
pip install numpy scipy scikit-learn matplotlib pandas
export PYTHONPATH=$(pwd)
python run_main.py 1 --aux      # primary site + auxiliary experiments   (~15 min, 1 CPU)
python run_main.py 2            # replication site                        (~8 min)
python run_main.py 3            # replication site                        (~8 min)
python analysis.py && python paired_ablation.py && python breakeven.py && python make_figures.py
node build_manuscript.js        # optional; needs `npm install docx`
```

Note: in the sandbox where this was developed, background jobs were killed when the session was idle. Run long jobs in the foreground or under a process manager.

## Protocol (fixed before looking at results unless stated)

* Splits per site, all disjoint: twin-fit 250 normal, alarm-gate 150 normal, train 800, calibration 250, test 500 episodes (72 h, 10-min sampling).
* Alarm rule: `1 - P(normal)` above a threshold for 2 consecutive windows; threshold set on held-out **normal** episodes for 0.05 false alarms/day. Baselines get the same target.
* "Manifest" windows: the injected deviation exceeds 1.5 sigma of instrument noise (RMS over the window). Fixed a priori. Non-manifest post-onset windows are excluded from training and from the primary window metric; results on *all* post-onset windows are always reported alongside.
* Aggregate (permutation-invariant) features were added **after** a prototype run showed replay attacks were poorly learned from per-sensor features. The prototype and final test sets are different draws, but this is a design decision made after seeing a result on the same simulator.
* Cost matrix (`pipeline.COST`) is **illustrative**. Conclusions that depend on it are stress-tested in `analysis.py` (attack-miss cost, false-escalation cost, review effectiveness, attack prior).

## Known limitations

1. Generative assumptions decide identifiability. Single-sensor drift/step faults and single-sensor bias/ramp attacks are generated from overlapping distributions, so they are partly indistinguishable by construction; the classifier's confusion there is a property of the simulator.
2. Attacks are false-data injection on sensor values via a gateway or field device. Weather-station data are assumed trustworthy. Actuator attacks, network-layer attacks and denial of service are out of scope.
3. The "twin-aware" attacker ignores thermal lag and is therefore not optimal; it is not a proof of what a perfect attacker can do.
4. Site transfer is easy here because every simulated site uses the same functional forms. Real sites will violate the physics forms.
5. One process model, one sampling rate, one topology.

## Figure files vs manuscript numbering

Files are named in generation order; the manuscript numbers figures by first appearance.

| file | manuscript |
|---|---|
| `fig1_framework` | Figure 1 |
| `fig3_confusion_reliability` | Figure 2 |
| `fig4_ablation` | Figure 3 |
| `fig5_cost` | Figure 4 |
| `fig8_sensitivity` | Figure 5 |
| `fig6_magnitude` | Figure 6 |
| `fig7_transfer_stress` | Figure 7 |
| `fig2_examples` | Figure 8 |

## What was and was not established

* Results (three simulated sites, 500 test episodes each) are in `results/summary.json` and the manuscript. Read the manuscript's limitations before quoting any number.
* Isotonic recalibration did **not** consistently improve calibration (better at site 1, worse at sites 2 and 3).
* The data-driven twin added nothing measurable beyond the physics twin (paired bootstrap includes zero).
* Cost conclusions depend on an illustrative cost matrix and on analyst review resolving deferred cases.
* `adapter.py --selftest` round-trips a simulated CSV; it is a plumbing check, not validation. No real data have been used.
* All eleven manuscript references were verified against their source pages (DOI / publisher / arXiv) prior to submission.
