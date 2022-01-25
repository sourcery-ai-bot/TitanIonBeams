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



filedates = {"t16": "22-jul-2006", "t17": "07-sep-2006", "t18": "23-sep-2006", "t19": "09-oct-2006",
             "t20": "25-oct-2006", "t21": "12-dec-2006", "t23": "13-jan-2007", "t25": "22-feb-2007",
             "t26": "10-mar-2007",
             "t27": "26-mar-2007",
             "t28": "10-apr-2007", "t29": "26-apr-2007",
             "t30": "12-may-2007", "t32": "13-jun-2007", "t36": "02-oct-2007", "t39": "20-dec-2007",
             "t40": "05-jan-2008", "t41": "22-feb-2008", "t42": "25-mar-2008", "t43": "12-may-2008",
             "t46": "03-nov-2008", "t47": "19-nov-2008","t48": "05-dec-2008","t49": "21-dec-2008",
             "t50": "07-feb-2009","t51": "27-mar-2009","t71": "07-jul-2010","t83": "22-may-2012",
             "t55":"21-may-2009", "t56": "06-jun-2009", "t57": "22-jun-2009", "t58": "08-jul-2009",
             "t59": "24-jul-2009"}

flyby_datetimes = {"t16": [datetime.datetime(2006, 7, 22, 0, 20, 43), datetime.datetime(2006, 7, 22, 0, 32, 20)],
                   "t17": [datetime.datetime(2006, 9, 7, 20, 9, 30), datetime.datetime(2006, 9, 7, 20, 24, 55)],
                  "t19": [datetime.datetime(2006, 10, 9, 17, 22), datetime.datetime(2006, 10, 9, 17, 36, 30)],
                  "t20": [datetime.datetime(2006, 10, 25, 15, 55, 30), datetime.datetime(2006, 10, 25, 15, 57, 45)],
                  "t21": [datetime.datetime(2006, 12, 12, 11, 34, 30), datetime.datetime(2006, 12, 12, 11, 50)],
                  "t23": [datetime.datetime(2007, 1, 13, 8, 31), datetime.datetime(2007, 1, 13, 8, 46, 15)],
                  "t25": [datetime.datetime(2007, 2, 22, 3, 10), datetime.datetime(2007, 2, 22, 3, 15)],
                  "t26": [datetime.datetime(2007, 3, 10, 1, 41, 30), datetime.datetime(2007, 3, 10, 1, 56, 45)],
                  "t27": [datetime.datetime(2007, 3, 26, 0, 16, 30), datetime.datetime(2007, 3, 26, 0, 30, 30)],
                  "t28": [datetime.datetime(2007, 4, 10, 22, 53), datetime.datetime(2007, 4, 10, 23, 3, 20)],
                  "t29": [datetime.datetime(2007, 4, 26, 21, 28, 40), datetime.datetime(2007, 4, 26, 21, 38)],
                  "t30": [datetime.datetime(2007, 5, 12, 20, 8, 30), datetime.datetime(2007, 5, 12, 20, 15, 20)],
                  "t32": [datetime.datetime(2007, 6, 13, 17, 43, 10), datetime.datetime(2007, 6, 13, 17, 51, 25)],
                  "t36": [datetime.datetime(2007, 10, 2, 4, 39, 30), datetime.datetime(2007, 10, 2, 4, 45)],
                  "t39": [datetime.datetime(2007, 12, 20, 22, 54, 20), datetime.datetime(2007, 12, 20, 23, 1, 20)],
                  "t40": [datetime.datetime(2008, 1, 5, 21, 24), datetime.datetime(2008, 1, 5, 21, 37, 15)],
                  "t41": [datetime.datetime(2008, 2, 22, 17, 29, 40), datetime.datetime(2008, 2, 22, 17, 34, 40)],
                  "t42": [datetime.datetime(2008, 3, 25, 14, 22, 30), datetime.datetime(2008, 3, 25, 14, 33)],
                  "t43": [datetime.datetime(2008, 5, 12, 9, 55, 30), datetime.datetime(2008, 5, 12, 10, 8, 30)],
                  "t46": [datetime.datetime(2008, 11, 3, 17, 33, 10), datetime.datetime(2008, 11, 3, 17, 36, 30)],
                  "t47": [datetime.datetime(2008, 11, 19, 15, 53), datetime.datetime(2008, 11, 19, 15, 54)],
                  "t48": [datetime.datetime(2008, 12, 5, 14, 20), datetime.datetime(2008, 12, 5, 14, 32, 40)],
                  "t49": [datetime.datetime(2008, 12, 21, 12, 55, 30), datetime.datetime(2008, 12, 21, 13, 4)],
                  "t50": [datetime.datetime(2009, 2, 7, 8, 45, 20), datetime.datetime(2009, 2, 7, 8, 54, 45)],
                  "t51": [datetime.datetime(2009, 3, 27, 4, 41), datetime.datetime(2009, 3, 27, 4, 49, 30)],
                  "t71": [datetime.datetime(2010, 7, 7, 0, 17, 10), datetime.datetime(2010, 7, 7, 0, 26, 30)],
                  "t83": [datetime.datetime(2012, 5, 22, 1, 2, 30), datetime.datetime(2012, 5, 22, 1, 16, 10)],
                  "t55": [datetime.datetime(2009, 5, 21, 21, 18), datetime.datetime(2009, 5, 21, 21, 34)],
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


def IBS_alt_spec(flyby,flybydf):
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)


    startslice = CAPS_slicenumber(ibsdata, flyby_datetimes[ibsdata['flyby']][0])
    endslice = CAPS_slicenumber(ibsdata, flyby_datetimes[ibsdata['flyby']][1])
    CA_height, CA_time = find_CA_height_time(ibsdata,startslice)
    print(CA_time,CA_height)
    CA_slice = CAPS_slicenumber(ibsdata, CA_time)
    print(ibsdata['times_utc'][CA_slice])

    inbound_alts = [cassini_titan_altlatlon(i+datetime.timedelta(seconds=1))[0] for i in ibsdata['times_utc'][startslice:CA_slice]] #Fix timeoffset!!!!!!!!!!!!!!!!!!
    outbound_alts = [cassini_titan_altlatlon(i+datetime.timedelta(seconds=1))[0] for i in ibsdata['times_utc'][CA_slice:endslice]]

    print("last inbound alt",inbound_alts[-1])
    print("first outbound alt",outbound_alts[0])

    X = ibscalib['ibsearray']
    Y1 = np.array(inbound_alts)
    Y2 = outbound_alts
    Z1 = ibsdata['ibsdata'][:,1,startslice:CA_slice]
    Z2 = ibsdata['ibsdata'][:,1,CA_slice:endslice]

    print(X.shape,Y1.shape,Z1.shape)

    fig, axes = plt.subplots(nrows=1,ncols=3,sharey="all")
    axes[0].get_shared_x_axes().join(axes[0], axes[1])
    CS1 = axes[0].pcolormesh(X, Y1, Z1.T, norm=LogNorm(vmin=50,vmax=1e6), cmap='viridis',shading='nearest')
    CS2 = axes[1].pcolormesh(X, Y2, Z2.T, norm=LogNorm(vmin=50,vmax=1e6), cmap='viridis',shading='nearest')

    inbounddf = flybydf[(pd.to_datetime(flybydf["Positive Peak Time"]) < CA_time)]
    outbounddf = flybydf[(pd.to_datetime(flybydf["Positive Peak Time"]) > CA_time)]

    axes[0].plot(inbounddf['IBS Mass 17 energy'], inbounddf['Altitude'],marker='.')
    axes[0].plot(inbounddf['IBS Mass 28 energy'],inbounddf['Altitude'],marker='.')
    axes[0].plot(inbounddf['IBS Mass 40 energy'],inbounddf['Altitude'],marker='.')
    axes[0].plot(inbounddf['IBS Mass 53 energy'],inbounddf['Altitude'],marker='.')
    axes[0].plot(inbounddf['IBS Mass 66 energy'],inbounddf['Altitude'],marker='.')
    axes[0].plot(inbounddf['IBS Mass 78 energy'],inbounddf['Altitude'],marker='.')
    axes[0].plot(inbounddf['IBS Mass 91 energy'],inbounddf['Altitude'],marker='.')

    axes[1].plot(outbounddf['IBS Mass 17 energy'], outbounddf['Altitude'],marker='.')
    axes[1].plot(outbounddf['IBS Mass 28 energy'],outbounddf['Altitude'],marker='.')
    axes[1].plot(outbounddf['IBS Mass 40 energy'],outbounddf['Altitude'],marker='.')
    axes[1].plot(outbounddf['IBS Mass 53 energy'],outbounddf['Altitude'],marker='.')
    axes[1].plot(outbounddf['IBS Mass 66 energy'],outbounddf['Altitude'],marker='.')
    axes[1].plot(outbounddf['IBS Mass 78 energy'],outbounddf['Altitude'],marker='.')
    axes[1].plot(outbounddf['IBS Mass 91 energy'],outbounddf['Altitude'],marker='.')

    axes[2].plot(inbounddf['IBS alongtrack velocity'], inbounddf['Altitude'], color='C0', marker='.',label="Inbound")
    axes[2].plot(outbounddf['IBS alongtrack velocity'], outbounddf['Altitude'], color='C1', marker='.',label="Outbound")
    axes[2].set_xlabel("IBS Alongtrack Velocity")
    axes[2].legend()


    for ax in [axes[0],axes[1]]:
        ax.set_xscale("log")
        ax.set_xlim(3,200)
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Altitude [km]")
    axes[0].set_title("Inbound")
    axes[1].set_title("Outbound")
    fig.suptitle(flyby)


flyby = "t55"
#windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)
windsdf = pd.read_csv("nonactuatingflybys_alongtrackvelocity_t55.csv", index_col=0, parse_dates=True)
flybydf = windsdf[windsdf['Flyby'] == flyby]
flybydf.reset_index(inplace=True)

IBS_alt_spec(flyby, flybydf)
plt.show()


