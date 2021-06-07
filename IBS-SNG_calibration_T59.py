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

T59ELSdata = readsav("data/els/elsres_24-jul-2009.dat")
T59SNGdata = readsav("data/sng/sngres_24-jul-2009.dat")
T59IBSdata = readsav("data/ibs/ibsres_24-jul-2009.dat")

generate_mass_bins(T59ELSdata, "t59", "els")
generate_mass_bins(T59SNGdata, "t59", "sng")
generate_aligned_ibsdata(T59IBSdata, T59ELSdata, "t59")

# start = datetime.datetime(2009,7,24,15,25, 35)
# end = datetime.datetime(2009,7,24,15,27, 31)
# T59IBS_startslice = CAPS_slicenumber(T59IBSdata,start)
# T59IBS_endslice = CAPS_slicenumber(T59IBSdata,end)
# T59SNG_startslice = CAPS_slicenumber(T59SNGdata,start)
# T59SNG_endslice = CAPS_slicenumber(T59SNGdata,end)



# fig, (ibsax, sngax) = plt.subplots(2,sharex="all",sharey="all")
# ibsax.pcolormesh(T59IBSdata['times_utc'][T59IBS_startslice:T59IBS_endslice], ibscalib['ibsearray'], T59IBSdata['ibsdata'][:,1,T59IBS_startslice:T59IBS_endslice], norm=LogNorm(vmin=100, vmax=1e4), cmap='viridis')
# sngax.pcolormesh(T59SNGdata['times_utc'][T59SNG_startslice:T59SNG_endslice], sngcalib['sngearray'], T59SNGdata['sngdata'][:,9,T59SNG_startslice:T59SNG_endslice], norm=LogNorm(vmin=50, vmax=1e4), cmap='viridis')
# sngax.set_yscale("log")
# ibsax.set_ylabel("IBS Fan 2 \n ev/q")
# sngax.set_ylabel("SNG summed \n ev/q")
# plt.show()

slicetime = datetime.datetime(2009,7,24,15,25,39)
T59IBS_slice = CAPS_slicenumber(T59IBSdata,slicetime)
T59SNG_slice = CAPS_slicenumber(T59SNGdata,slicetime)

T59IBS_series = pd.Series(T59IBSdata['ibsdata'][:,1,T59IBS_slice]/ (ibscalib['ibsgeom']),index=ibscalib['ibsearray'])
T59IBS_bins = pd.cut(ibscalib['ibsearray'], sngcalib['sngpolyearray'])
print(T59IBS_bins)
T59IBS_sampledseries = T59IBS_series.groupby(T59IBS_bins).agg(['sum'])
print(T59IBS_sampledseries)
IBS_scaling = 500

energyfig, ax = plt.subplots()
ax.step(ibscalib['ibsearray'],T59IBSdata['ibsdata'][:,1,T59IBS_slice]/ (ibscalib['ibsgeom']),where='mid',label="IBS Data")
ax.step(sngcalib['sngearray'],T59IBS_sampledseries.to_numpy()*IBS_scaling,where='mid',label="IBS data, binned to SNG bins, x" + str(IBS_scaling) +" scale")
ax.step(sngcalib['sngearray'],T59SNGdata['sngdef'][:,9,T59SNG_slice],where='mid',label="SNG Data")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("DEF")
ax.set_xlabel("eV/q")
ax.set_xlim(10,100)
ax.legend()
plt.show()
