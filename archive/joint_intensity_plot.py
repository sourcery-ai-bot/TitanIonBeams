from scipy.io import readsav
from util import generate_mass_bins
import datetime
from cassinipy.caps.mssl import ELS_intensityplot, IBS_intensityplot, CAPS_slicenumber, ELS_backgroundremoval
import matplotlib.pyplot as plt
import numpy as np

filedates_times = {"t17": ["07-sep-2006", "20:13:00"],
                   "t20": "25-oct-2006",
                   "t21": "12-dec-2006",
                   "t25": ["22-feb-2007", "03:08:00"],
                   "t26": ["10-mar-2007", "01:43:00"],
                   "t27": "26-mar-2007",
                   "t28": ["10-apr-2007", "22:55:00"],
                   "t29": ["26-apr-2007", "21:29:00"],
                   "t30": "12-may-2007",
                   "t32": "13-jun-2007",
                   "t46": "03-nov-2008",
                   "t47": "19-nov-2008"}

flyby = "t29"
anode = 4
starttime = datetime.datetime(2007, 4, 26, 21, 29, 30)
endtime = datetime.datetime(2007, 4, 26, 21, 35, 30)

elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
generate_mass_bins(elsdata, flyby, "els")
ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
generate_mass_bins(ibsdata, flyby, "ibs")

fig, ax = plt.subplots(subplot_kw={"yscale": "log"})
ax2 = ax.twinx()
ax2.set_yscale("log")
ELS_intensityplot(elsdata, 400, anode, starttime, endtime, peaks=True, prominence=1e10, ax=ax)
IBS_intensityplot(ibsdata, 55, starttime, endtime, peaks=True, prominence=300, ax=ax2)
fig.legend()

startslice_els, endslice_els = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
dataslice_els = ELS_backgroundremoval(elsdata, startslice_els, endslice_els)[:, anode - 1, :]
startslice_ibs, endslice_ibs = CAPS_slicenumber(ibsdata, starttime), CAPS_slicenumber(ibsdata, endtime)

ax.plot(elsdata['times_utc'][startslice_els:endslice_els], np.sum(dataslice_els, axis=0), linestyle='--',
        label="ELS Flux Sum", color='C0')
ax2.plot(ibsdata['times_utc'][startslice_ibs:endslice_ibs],
        np.sum(ibsdata['ibsdata'][:, 1, startslice_ibs:endslice_ibs], axis=0), linestyle='--', label="IBS Count Sum" , color='C1')

plt.show()
