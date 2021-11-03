from scipy.io import readsav
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from util import generate_mass_bins
from sunpy.time import TimeRange
import datetime
#from heliopy.data.cassini import caps_ibs
from cassinipy.caps.mssl import *
from cassinipy.caps.util import *
from util import *

from lmfit import CompositeModel, Model
from lmfit.models import GaussianModel
from lmfit import Parameters
from lmfit import Minimizer
matplotlib.rcParams.update({'font.size': 15})

ibscalib = readsav('calib/ibsdisplaycalib.dat')
sngcalib = readsav('calib/sngdisplaycalib.dat')

filedates_times = {"t55": ["21-may-2009", "21:27:00"],
                   "t56": ["06-jun-2009", "20:00:00"],
                   "t57": ["22-jun-2009", "18:33:00"],
                   "t58": ["08-jul-2009", "17:04:00"],
                   "t59": ["24-jul-2009", "15:31:00"],
                   }

flyby_datetimes = {"t55": [datetime.datetime(2009, 5, 21, 21, 18), datetime.datetime(2009, 5, 21, 21, 34)],
                   "t56": [datetime.datetime(2009, 6, 6,  19, 53), datetime.datetime(2009, 6, 6, 20, 7)],
                   "t57": [datetime.datetime(2009, 6, 22, 18, 26), datetime.datetime(2009, 6, 22, 18, 41)],
                   "t58": [datetime.datetime(2009, 7, 8, 16, 55), datetime.datetime(2009, 7, 8, 17, 16)],
                   "t59": [datetime.datetime(2009, 7, 24, 15, 25), datetime.datetime(2009, 7, 24, 15, 43)],

                   }

if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))


def IBS_spectrogram(ibsdata, fans, starttime, seconds, backgroundremoved=False):
    """
    Plot multiple ibs fan spectrograms
    """

    countlimits = {'t27': [1e1, 5e5], 't46': [1e1, 5e5], 't55': [1e1, 5e5], 't56': [1e1, 5e5], 't57': [1e2, 5e5],
                   't58': [1e1, 5e5], 't59': [6e2, 3e5]}
    deflimits = {'t27': [1e10, 1e14], 't46': [1e10, 1e14], 't55': [1e10, 1e14], 't56': [1e10, 1e14],
                 't57': [1e10, 1e14], 't58': [1e10, 1e14], 't59': [1e10, 1e14]}

    energylimits = {'t27': [3, 100], 't40': [1, 10000], 't46': [3, 100], 't55': [35, 60], 't56': [35, 70],
                    't57': [1, 80], 't58': [35, 70], 't59': [3, 60], 't83': [1, 250]}

    if len(fans) == 1:
        fig = plt.figure(figsize=(18, 6))
        axis1 = plt.axes()

    if len(fans) > 1:
        fig, axes = plt.subplots(len(fans), sharex='all', sharey='all', figsize=(18, 10))

    for counter, i in enumerate(ibsdata['times_utc_strings']):
        if i >= starttime:
            slicenumber = counter
            break

    slicenumbers = arange(slicenumber, slicenumber + (seconds / 2), 1, dtype=int)

    X = ibsdata['times_utc'][slicenumbers[0]:slicenumbers[-1] + 1]
    Y = ibscalib['ibspolyearray']

    for figcounter, fan in enumerate(fans):

        if not backgroundremoved:
            Z = zeros((653, len(slicenumbers)))
            for k in range(653):
                for slicecounter, slicenumber in enumerate(slicenumbers):
                    Z[k, slicecounter] = ((ibsdata['ibsdata'][k, fan, slicenumber]) / (ibscalib['ibsgeom'] * 1e-4))
                    # Z[k,slicecounter] = ibsdata['ibsdata'][k,fan,slicenumber]
        if backgroundremoved:
            backgroundremoveddata = zeros(shape=(653, len(slicenumbers)))
            for slicecounter, slicenumber in enumerate(slicenumbers):
                backgroundval = max([max(ibsdata['ibsdata'][:, :3, slicenumber], axis=1),
                                     max(ibsdata['ibsdata'][:, 5:8, slicenumber], axis=1)], axis=0)
                tempdata = [a - b for a, b in zip(ibsdata['ibsdata'][:, fan, slicenumber], backgroundcount)]
                backgroundremoveddata[:, slicecounter] = tempdata
            Z = backgroundremoveddata

        print(len(X), len(Y), Z.shape)
        if len(fans) > 1:
            axis1 = axes[figcounter]

        if 'flyby' in list(deflimits.keys()):
            CS = axis1.pcolormesh(X, Y, Z, norm=LogNorm(vmin=deflimits[ibsdata['flyby']][0],
                                                        vmax=deflimits[ibsdata['flyby']][1]), cmap='viridis',
                                  shading='auto')
            axis1.set_ylim(energylimits[ibsdata['flyby']][0], energylimits[ibsdata['flyby']][1])
        else:
            CS = axis1.pcolormesh(X, Y, Z, norm=LogNorm(vmin=5e10, vmax=5e14), cmap='viridis', shading='auto')

        axis1.set_yscale("log")
        axis1.set_ylabel("IBS Fan " + str(fan + 1) + "\n [eV/q]", fontsize=16)
        axis1.minorticks_on()
        axis1.tick_params(labelbottom=True, labeltop=False, bottom=True, top=True, left=True, right=True, which='both')
        axis1.tick_params(axis='y', labelleft=True, labelright=False, left=True, right=True, which='major')
        axis1.yaxis.set_major_formatter(ScalarFormatter())
        # axis1.yaxis.set_minor_formatter(ScalarFormatter())

    if 'flyby' in list(ibsdata.keys()):
        if len(fans) == 1:
            axis1.set_title("Spectrogram of " + ibsdata['instrument'].upper() + " data from the " + str(
                ibsdata['flyby']).upper() + " flyby", y=1.01, fontsize=30)
            axis1.set_xlabel("Time", fontsize=20)
        if len(fans) > 1:
            axes[0].set_title("Spectrogram of " + ibsdata['instrument'].upper() + " data from the " + str(
                ibsdata['flyby']).upper() + " flyby", y=1.08, fontsize=30)
            axes[-1].set_xlabel("Time", fontsize=20)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.11, 0.03, 0.77])
    cbar = fig.colorbar(CS, cax=cbar_ax)
    cbar.ax.set_ylabel("DEF [$m^{-2} s^{1} str^{-1} eV^{-1}$]")
    # cbar.ax.set_ylabel("Counts [/s]")
    # fig.savefig(ibsdata['instrument'] + "-" + ibsdata['flyby'] + "-IBSSpectrogram.pdf",format='pdf',

def cassini_titan_altlatlon(tempdatetime):
    et = spice.datetime2et(tempdatetime)

    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    lon, lat, alt = spice.recpgr('TITAN', state[:3], spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    return alt, lat * spice.dpr(), lon * spice.dpr()

def find_CA_height_time(data,startslice):
    currentalt = 1e10


    for tempdatetime in data['times_utc'][startslice:]:
        alt, lat, lon = cassini_titan_altlatlon(tempdatetime)
        if alt < currentalt:
            currentalt = alt
        else:
            break
        lastdatetime = tempdatetime

    return currentalt, lastdatetime


def IBS_alt_spec(flyby):

    ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(ibsdata, flyby, "ibs")
    startslice = CAPS_slicenumber(ibsdata, flyby_datetimes[ibsdata['flyby']][0])
    endslice = CAPS_slicenumber(ibsdata, flyby_datetimes[ibsdata['flyby']][1])
    CA_height, CA_time = find_CA_height_time(ibsdata,startslice)
    print(CA_time,CA_height)
    CA_slice = CAPS_slicenumber(ibsdata, CA_time)
    print(CA_slice)


    inbound_alts = [cassini_titan_altlatlon(i)[0] for i in ibsdata['times_utc'][startslice:CA_slice]]
    outbound_alts = [cassini_titan_altlatlon(i)[0] for i in ibsdata['times_utc'][CA_slice:endslice]]

    X = ibscalib['ibsearray']
    Y1 = inbound_alts
    Y2 = outbound_alts
    Z1 = ibsdata['ibsdata'][:,1,startslice:CA_slice]
    Z2 = ibsdata['ibsdata'][:,1,CA_slice:endslice]

    fig, axes = plt.subplots(nrows=1,ncols=2,sharey="all",sharex='all')
    CS1 = axes[0].pcolormesh(X, Y1, Z1.T, norm=LogNorm(vmin=50,vmax=1e6), cmap='viridis',shading='nearest')
    CS2 = axes[1].pcolormesh(X, Y2, Z2.T, norm=LogNorm(vmin=50,vmax=1e6), cmap='viridis',shading='nearest')
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(3,200)
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Altitude [km]")
    axes[0].set_title("Inbound")
    axes[1].set_title("Outbound")
    fig.suptitle(flyby)


IBS_alt_spec("t59")
plt.show()


