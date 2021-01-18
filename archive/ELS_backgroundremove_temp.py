import datetime

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import readsav
from cassinipy.caps.mssl import CAPS_slicenumber
from util import generate_mass_bins
import spiceypy as spice
import glob

# Loading Kernels
if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = False
plt.rcParams['date.epoch'] = '0000-12-31'

filedates_times = {"t17": ["07-sep-2006", "20:11:00"],
                   "t20": "25-oct-2006",
                   "t21": ["12-dec-2006", "11:35:00"],
                   "t25": ["22-feb-2007", "03:07:00"],
                   "t26": ["10-mar-2007", "01:43:00"],
                   "t27": ["26-mar-2007", "00:19:00"],
                   "t28": ["10-apr-2007", "22:55:00"],
                   "t29": ["26-apr-2007", "21:29:00"],
                   "t30": "12-may-2007",
                   "t32": ["13-jun-2007", "17:41:00"],
                   "t46": "03-nov-2008",
                   "t47": "19-nov-2008"}

flyby_datetimes = {"t17": [datetime.datetime(2006, 9, 7, 20, 14, 30), datetime.datetime(2006, 9, 7, 20, 19, 30)],
                   "t21": [datetime.datetime(2006, 12, 12, 11, 39, 45), datetime.datetime(2006, 12, 12, 11, 43, 20)],
                   "t25": [datetime.datetime(2007, 2, 22, 3, 7), datetime.datetime(2007, 2, 22, 3, 17)],
                   "t27": [datetime.datetime(2007, 3, 26, 0, 19, 00), datetime.datetime(2007, 3, 26, 0, 29)],
                   "t29": [datetime.datetime(2007, 4, 26, 21, 29, 30), datetime.datetime(2007, 4, 26, 21, 35, 30)],
                   "t32": [datetime.datetime(2007, 6, 13, 17, 41), datetime.datetime(2007, 6, 13, 17, 51)]
                   }

ibscalib = readsav('calib\\ibsdisplaycalib.dat')
elscalib = readsav('calib\\geometricfactor.dat')

def ELS_backgroundremoval(data, startslice, endslice):
    def_backgroundremoved = np.zeros((63, 8, endslice - startslice))
    # Background defined as average of 5 loweest count, negative ions unlikely to appear across 3 anodes
    for backgroundcounter, timecounter in enumerate(np.arange(startslice, endslice, 1)):

        for energycounter in range(63):
            backgroundremoved_temp = np.array(data['def'][energycounter, :8, timecounter]) - np.mean(
                sorted(data['def'][energycounter, :8, timecounter])[:6])
            backgroundremoved_anodes = [0 if i < 0 else i for i in backgroundremoved_temp]
            def_backgroundremoved[energycounter, :, backgroundcounter] = backgroundremoved_anodes

    return def_backgroundremoved

def cassini_titan_altlatlon(tempdatetime):
    et = spice.datetime2et(tempdatetime)

    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    lon, lat, alt = spice.recpgr('TITAN', state[:3], spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    return alt, lat * spice.dpr(), lon * spice.dpr()

def base_spectrogram(data, datacalib, datanorm, Z, slicenumbers, ax=None, cax=None, fig=None):
    """
    Plots a spectrogram
    """
    CS = ax.pcolormesh(data['times_utc'][slicenumbers[0]:slicenumbers[-1] + 1], datacalib, Z, norm=datanorm, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=.05)
    cbar = fig.colorbar(cm.ScalarMappable(norm=datanorm, cmap='jet'), ax=ax, cax=cax)
    cbar.set_label("DEF [$m^{-2} s^{1} str^{-1} eV^{-1}$]")
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
    data = ELS_backgroundremoval(elsdata, slicenumbers[0], slicenumbers[-1])
    Z = data[:, anode - 1, :]

    base_spectrogram(elsdata, datacalib, elsnorm, Z, slicenumbers, ax=ax, cax=cax, fig=fig)
    ax.set_ylabel("CAPS ELS \nAnode {0} \n eV/q ".format(anode))



flyby = "t32"
anodes = [4,5]
lowerenergy = 2
upperenergy = 500

elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
generate_mass_bins(elsdata, flyby, "els")

starttime = flyby_datetimes[flyby][0]
endtime = flyby_datetimes[flyby][1]


alts=[]
startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
for i in np.arange(startslice,endslice,1):
    alt, lat, lon = cassini_titan_altlatlon(elsdata['times_utc'][i])
    alts.append(alt)

fig, axes = plt.subplots(len(anodes)+1, figsize=(18, 6), sharex='all')
for anode, ax in zip(anodes,axes[:-1]):
    ELS_spectrogram(elsdata, anode, filedates_times[flyby][1], 600, ax=ax, fig=fig)
axes[-1].plot(elsdata['times_utc'][startslice:endslice],alts)
axes[-1].set_ylabel("Altitude [/km]")
plt.show()