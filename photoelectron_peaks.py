from util import generate_mass_bins, generate_aligned_ibsdata
from scipy.io import readsav
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.signal import find_peaks, peak_widths
from astropy.modeling import models, fitting
import datetime
import spiceypy as spice
import glob

from cassinipy.caps.mssl import CAPS_slicenumber, CAPS_energyslice, ELS_backgroundremoval, caps_ramdirection_time, CAPS_actuationtimeslice
from cassinipy.caps.spice import caps_ramdirection_azielv

elscalib = readsav('calib/geometricfactor.dat')


def base_spectrogram(data, datacalib, datanorm, Z, slicenumbers, ax=None, cax=None, fig=None):
    """
    Plots a spectrogram
    """
    CS = ax.pcolormesh(data['times_utc'][slicenumbers[0]:slicenumbers[-1] + 1], datacalib, Z, norm=datanorm,
                       cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=.05)
    cbar = fig.colorbar(cm.ScalarMappable(norm=datanorm, cmap='viridis'), ax=ax, cax=cax)
    cbar.set_label("DEF \n [$m^{-2} s^{1} str^{-1} eV^{-1}$]")
    ax.set_yscale("log")


def ELS_spectrogram(elsdata, anode, starttime, seconds, ax=None, cax=None, fig=None):
    """
    Plots single els anode
    """
    datacalib = elscalib['polyearray']
    elsnorm = LogNorm(vmin=1e8, vmax=1e12)

    for counter, i in enumerate(elsdata['times_utc_strings']):
        if i >= starttime:
            slicenumber = counter
            break
    slicenumbers = np.arange(slicenumber, slicenumber + (seconds / 2), 1, dtype=int)
    Z = elsdata['def'][:, anode - 1, slicenumbers[0]:slicenumbers[-1]]

    base_spectrogram(elsdata, datacalib, elsnorm, Z, slicenumbers, ax=ax, cax=cax, fig=fig)
    ax.set_ylabel("CAPS ELS \nAnode {0} \n eV/q ".format(anode))

def ELS_maxflux_anode(elsdata, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)
    anodesums = np.sum(np.sum(dataslice, axis=2), axis=0)
    maxflux_anode = np.argmax(anodesums)
    return maxflux_anode

filedates_times = {"t16": ["22-jul-2006", "00:22:00"],
                   "t17": ["07-sep-2006", "20:13:00"],
                   "t20": ["25-oct-2006", "15:54:00"],
                   "t21": ["12-dec-2006", "11:38:00"],
                   "t25": ["22-feb-2007", "03:08:00"],
                   "t26": ["10-mar-2007", "01:43:00"],
                   "t27": ["26-mar-2007", "00:20:00"],
                   "t28": ["10-apr-2007", "22:53:00"],
                   "t29": ["26-apr-2007", "21:29:00"],
                   "t30": ["12-may-2007", "20:06:00"],
                   "t32": "13-jun-2007",
                   "t42": ["25-mar-2008", "14:24:00"],
                   "t46": ["03-nov-2008", "17:32:00"],
                   "t47": "19-nov-2008"}

flyby_datetimes = {"t16": [datetime.datetime(2006, 7, 22, 0, 22), datetime.datetime(2006, 7, 22, 0, 28, 40)],
                   "t17": [datetime.datetime(2006, 9, 7, 20, 14, 30), datetime.datetime(2006, 9, 7, 20, 19, 40)],
                   "t20": [datetime.datetime(2006, 10, 25, 15, 55, 30), datetime.datetime(2006, 10, 25, 15, 57, 45)],
                   "t21": [datetime.datetime(2006, 12, 12, 11, 39, 45), datetime.datetime(2006, 12, 12, 11, 43, 20)],
                   "t25": [datetime.datetime(2007, 2, 22, 3, 10), datetime.datetime(2007, 2, 22, 3, 15)],
                   "t26": [datetime.datetime(2007, 3, 10, 1, 45, 30), datetime.datetime(2007, 3, 10, 1, 52, 20)],
                   "t27": [datetime.datetime(2007, 3, 26, 0, 21, 30), datetime.datetime(2007, 3, 26, 0, 26)],
                   "t28": [datetime.datetime(2007, 4, 10, 22, 55, 40), datetime.datetime(2007, 4, 10, 23)],
                   "t29": [datetime.datetime(2007, 4, 26, 21, 29, 30), datetime.datetime(2007, 4, 26, 21, 35, 30)],
                   "t30": [datetime.datetime(2007, 5, 12, 20, 8, 20), datetime.datetime(2007, 5, 12, 20, 11, 45)],
                   "t32": [datetime.datetime(2007, 6, 13, 17, 44), datetime.datetime(2007, 6, 13, 17, 48)],
                   "t42": [datetime.datetime(2008, 3, 25, 14, 25), datetime.datetime(2008, 3, 25, 14, 30, 20)],
                   "t46": [datetime.datetime(2008, 11, 3, 17, 33), datetime.datetime(2008, 11, 3, 17, 36, 30)]}

flyby = "t17"

lowerenergy = 2
upperenergy = 750

elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
generate_mass_bins(elsdata, flyby, "els")
maxfluxanode = ELS_maxflux_anode(elsdata,flyby_datetimes[flyby][0],flyby_datetimes[flyby][1])

starttime = flyby_datetimes[flyby][0]
endtime = flyby_datetimes[flyby][1]

fig, elsaxes = plt.subplots(3, figsize=(18, 6), sharex=True)
for i, ax in enumerate(elsaxes):
    ELS_spectrogram(elsdata, maxfluxanode-1+i, filedates_times[flyby][1], 420, ax=ax, fig=fig)

plt.show()




