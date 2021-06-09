from scipy.io import readsav
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from util import generate_mass_bins
from sunpy.time import TimeRange
import datetime
from heliopy.data.cassini import caps_ibs
from cassinipy.caps.mssl import *
from cassinipy.caps.util import *
from util import *

matplotlib.rcParams.update({'font.size': 15})

ibscalib = readsav('calib/ibsdisplaycalib.dat')
sngcalib = readsav('calib/sngdisplaycalib.dat')

TAELSdata = readsav("data/els/elsres_26-oct-2004.dat")
TASNGdata = readsav("data/sng/sngres_26-oct-2004.dat")
TAIBSdata = readsav("data/ibs/ibsres_26-oct-2004.dat")

generate_mass_bins(TAELSdata, "ta", "els")
generate_mass_bins(TASNGdata, "ta", "sng")
generate_aligned_ibsdata(TAIBSdata, TAELSdata, "ta")

start = datetime.datetime(2004, 10, 26, 15, 23)
end = datetime.datetime(2004, 10, 26, 15, 37)
TAELS_startslice = CAPS_slicenumber(TAELSdata, start)
TAELS_endslice = CAPS_slicenumber(TAELSdata, end)
TAIBS_startslice = CAPS_slicenumber(TAIBSdata, start)
TAIBS_endslice = CAPS_slicenumber(TAIBSdata, end)
TASNG_startslice = CAPS_slicenumber(TASNGdata, start)
TASNG_endslice = CAPS_slicenumber(TASNGdata, end)

fig, (elsax, ibsax, sngax) = plt.subplots(3, sharex="all", sharey="all")
elsax.pcolormesh(TAELSdata['times_utc'][TAELS_startslice:TAELS_endslice], elscalib['earray'],
                 TAELSdata['data'][:, 3, TAELS_startslice:TAELS_endslice], norm=LogNorm(vmin=50, vmax=5e4),
                 cmap='viridis')
ibsax.pcolormesh(TAIBSdata['times_utc'][TAIBS_startslice:TAIBS_endslice], ibscalib['ibsearray'],
                 TAIBSdata['ibsdata'][:, 1, TAIBS_startslice:TAIBS_endslice], norm=LogNorm(vmin=50, vmax=5e4),
                 cmap='viridis')
sngax.pcolormesh(TASNGdata['times_utc'][TASNG_startslice:TASNG_endslice], sngcalib['sngearray'],
                 TASNGdata['sngdata'][:, 3, TASNG_startslice:TASNG_endslice], norm=LogNorm(vmin=50, vmax=5e4),
                 cmap='viridis')
sngax.set_yscale("log")
ibsax.set_ylabel("IBS Fan 2 \n ev/q")
sngax.set_ylabel("SNG A4 \n ev/q")
# plt.show()

# slicetime = datetime.datetime(2004,10,26,15,29,45)
slicetime = datetime.datetime(2004, 10, 26, 15, 29, 46)
end_slicetime = datetime.datetime(2004, 10, 26, 15, 29, 52)
TAIBS_slice = CAPS_slicenumber(TAIBSdata, slicetime)
TAIBS_slice_end = CAPS_slicenumber(TAIBSdata, end_slicetime)
TASNG_slice = CAPS_slicenumber(TASNGdata, slicetime)
TASNG_slice_end = CAPS_slicenumber(TASNGdata, end_slicetime)
TA_IBSslicetime = TAIBSdata['times_utc'][TAIBS_slice]
TA_IBSslicetime_end = TAIBSdata['times_utc'][TAIBS_slice_end]
TA_SNGslicetime = TASNGdata['times_utc'][TASNG_slice]
TA_SNGslicetime_end = TASNGdata['times_utc'][TASNG_slice_end]

TAIBS_data = np.mean(TAIBSdata['ibsdata'][:, 1, TAIBS_slice:TAIBS_slice_end],axis=1) / (ibscalib['ibsgeom'] * 1e-4)
TAIBS_series = pd.Series(TAIBS_data, index=ibscalib['ibsearray'])
TAIBS_bins = pd.cut(ibscalib['ibsearray'], sngcalib['sngpolyearray'])
print(TAIBS_bins)
TAIBS_sampledseries = TAIBS_series.groupby(TAIBS_bins).agg(['sum'])
print(TAIBS_sampledseries)
IBS_scaling = 0.06
IBS_shift = 1e13

energyfig, ax = plt.subplots()
ax.step(ibscalib['ibsearray'], TAIBS_data, where='mid',
        label="IBS Data averaged, " + str(TA_IBSslicetime) + " to " + str(TA_IBSslicetime_end))
ax.step(sngcalib['sngearray'], TAIBS_sampledseries.to_numpy() * IBS_scaling, where='mid',
        label="IBS data, binned to SNG bins," + str(IBS_scaling) + " scale")
# ax.step(sngcalib['sngearray'],TAIBS_sampledseries.to_numpy()-IBS_shift,where='mid',label="IBS data, binned to SNG bins, -" + str(IBS_shift) +" shift")
ax.step(sngcalib['sngearray'], np.mean(TASNGdata['sngdef'][:, 9, TASNG_slice:TASNG_slice_end],axis=1), where='mid',
        label="SNG Data averaged, " + str(TA_SNGslicetime) + " to " + str(TA_SNGslicetime_end))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("DEF")
ax.set_xlabel("eV/q")
ax.set_xlim(1, 100)
ax.legend()
ax.set_title(str(slicetime) + " to " + str(end_slicetime))
plt.show()
