"""Break-even cost of waiting: how much per hour of extra dwell would cancel the benefit of deciding
6 h after the alarm instead of at alarm time (attack episodes; proposed model, min-cost + review)."""
import pickle, json
import numpy as np
from oilfield_dt import pipeline as pl

out = {}
for s in (1, 2, 3):
    tab = pickle.load(open(f"results/tables_site{s}.pkl", "rb"))["proposed"]
    pf = pl.make_policy("bayes_review")
    a = pl.table_metrics(tab, pf, key="p_alarm"); b = pl.table_metrics(tab, pf, key="p_settled")
    row = {n: (a["cost_by_class"][c], b["cost_by_class"][c])
           for c, n in enumerate(["normal", "fault", "environment", "attack"])}
    d_att = row["attack"][0] - row["attack"][1]
    out[s] = dict(per_class=row, breakeven_attack_per_hour=d_att / 6.0)
    print(f"site {s}: break-even wait cost on attacks {d_att / 6:.2f}/h")
json.dump(out, open("results/breakeven_wait.json", "w"), indent=1)
