"""Train the simulation-based classifier bundle used by adapter.py  (python train_bundle.py)."""
import pickle
import numpy as np
from oilfield_dt import sim, features as fe, pipeline as pl
from oilfield_dt.twin import PhysicsTwin, DataDrivenTwin
from oilfield_dt.util import log

site = sim.make_site(1)
fit = sim.make_dataset(site, 250, seed=9001, cls=0)
tr = sim.make_dataset(site, 800, seed=9002)
cal = sim.make_dataset(site, 250, seed=9003)
pt, dd, nm = PhysicsTwin().fit(fit), DataDrivenTwin().fit(fit), fe.Normaliser().fit(fit)
dtr, dcal = fe.build(tr, pt, dd, nm), fe.build(cal, pt, dd, nm)
c = pl.cols(dtr, "proposed")
clf = pl.fit_multiclass(dtr, c)
cal_m = pl.calibrate(clf, dcal, c)
pickle.dump(dict(clf=clf, cal=cal_m, cols=c, names=[dtr["names"][i] for i in c]), open("bundle.pkl", "wb"))
log("bundle.pkl written;", len(c), "features")
