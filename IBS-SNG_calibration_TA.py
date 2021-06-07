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

start = datetime.datetime(2004,10,26,15,23)
end = datetime.datetime(2004,10,26,15,37)
TAIBS_startslice = CAPS_slicenumber(TAIBSdata,start)
TAIBS_endslice = CAPS_slicenumber(TAIBSdata,end)
TASNG_startslice = CAPS_slicenumber(TASNGdata,start)
TASNG_endslice = CAPS_slicenumber(TASNGdata,end)



fig, (ibsax, sngax) = plt.subplots(2,sharex="all",sharey="all")
ibsax.pcolormesh(TAIBSdata['times_utc'][TAIBS_startslice:TAIBS_endslice], ibscalib['ibsearray'], TAIBSdata['ibsdata'][:,1,TAIBS_startslice:TAIBS_endslice], norm=LogNorm(vmin=50, vmax=5e4), cmap='viridis')
sngax.pcolormesh(TASNGdata['times_utc'][TASNG_startslice:TASNG_endslice], sngcalib['sngearray'], TASNGdata['sngdata'][:,9,TASNG_startslice:TASNG_endslice], norm=LogNorm(vmin=50, vmax=5e4), cmap='viridis')
sngax.set_yscale("log")
ibsax.set_ylabel("IBS Fan 2 \n ev/q")
sngax.set_ylabel("SNG summed \n ev/q")
plt.show()

# slicetime = datetime.datetime(2004,10,26,15,30)
# TAIBS_slice = CAPS_slicenumber(TAIBSdata,slicetime)
# TASNG_slice = CAPS_slicenumber(TASNGdata,slicetime)
#
# TAIBS_series = pd.Series(TAIBSdata['ibsdata'][:,1,TAIBS_slice]/ (ibscalib['ibsgeom'] * 1e-4),index=ibscalib['ibsearray'])
# TAIBS_bins = pd.cut(ibscalib['ibsearray'], sngcalib['sngpolyearray'])
# print(TAIBS_bins)
# TAIBS_sampledseries = TAIBS_series.groupby(TAIBS_bins).agg(['sum'])
# print(TAIBS_sampledseries)
# IBS_scaling = 0.04
# IBS_shift = 1e13
#
# energyfig, ax = plt.subplots()
# ax.step(ibscalib['ibsearray'],TAIBSdata['ibsdata'][:,1,TAIBS_slice]/ (ibscalib['ibsgeom'] * 1e-4),where='mid',label="IBS Data")
# ax.step(sngcalib['sngearray'],TAIBS_sampledseries.to_numpy()*IBS_scaling,where='mid',label="IBS data, binned to SNG bins," + str(IBS_scaling) +" scale")
# ax.step(sngcalib['sngearray'],TAIBS_sampledseries.to_numpy()-IBS_shift,where='mid',label="IBS data, binned to SNG bins, -" + str(IBS_shift) +" shift")
# ax.step(sngcalib['sngearray'],TASNGdata['sngdef'][:,9,TASNG_slice],where='mid',label="SNG Data")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_ylabel("DEF")
# ax.set_xlabel("eV/q")
# ax.set_xlim(10,100)
# ax.legend()
# plt.show()
