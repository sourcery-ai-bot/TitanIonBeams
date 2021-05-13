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

from cassinipy.caps.mssl import CAPS_slicenumber, CAPS_energyslice, ELS_backgroundremoval, caps_ramdirection_time, \
    CAPS_actuationtimeslice
from cassinipy.caps.spice import caps_ramdirection_azielv

if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = False
plt.rcParams['date.epoch'] = '0000-12-31'

ibscalib = readsav('calib/ibsdisplaycalib.dat')
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
    data = ELS_backgroundremoval(elsdata, slicenumbers[0], slicenumbers[-1])
    Z = data[:, anode - 1, :]

    base_spectrogram(elsdata, datacalib, elsnorm, Z, slicenumbers, ax=ax, cax=cax, fig=fig)
    ax.set_ylabel("CAPS ELS \nAnode {0} \n eV/q ".format(anode))


def IBS_spectrogram(ibsdata, fan, starttime, seconds, ax=None, cax=None, fig=None):
    """
    Plots single ibs fan
    """
    datacalib = ibscalib['ibspolyearray']
    ibsnorm = LogNorm(vmin=1e11, vmax=1e14)

    for counter, i in enumerate(ibsdata['times_utc_strings']):
        if i >= starttime:
            slicenumber = counter
            break
    slicenumbers = np.arange(slicenumber, slicenumber + (seconds / 2), 1, dtype=int)
    Z = np.zeros((653, len(slicenumbers)))
    for k in range(653):
        for slicecounter, slicenumber in enumerate(slicenumbers):
            Z[k, slicecounter] = ((ibsdata['ibsdata'][k, fan - 1, slicenumber]) / (ibscalib['ibsgeom'] * 1e-4))

    base_spectrogram(ibsdata, datacalib, ibsnorm, Z, slicenumbers, ax=ax, cax=cax, fig=fig)
    ax.set_ylabel("CAPS IBS \nFan {0} \n eV/q ".format(fan))


def actuator_plot(elsdata, starttime, seconds, ax=None, fig=None):
    """
    Plots actuator position
    """
    for counter, i in enumerate(elsdata['times_utc_strings']):
        if i >= starttime:
            slicenumber = counter
            break
    slicenumbers = np.arange(slicenumber, slicenumber + (seconds / 2), 1, dtype=int)
    ax.plot(elsdata['times_utc'][slicenumbers[0]:slicenumbers[-1]],
            elsdata['actuator'][slicenumbers[0]:slicenumbers[-1]], color='r')
    ax.fill_between(elsdata['times_utc'][slicenumbers[0]:slicenumbers[-1]],
                    elsdata['actuator'][slicenumbers[0]:slicenumbers[-1]] - 0.5,
                    elsdata['actuator'][slicenumbers[0]:slicenumbers[-1]] + 0.5, color='r', alpha=0.25)
    ax.set_ylabel("Azimuth Angle")
    ax.set_yscale("linear")


def heavy_ion_finder(elsdata, startenergy, anode, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    tempslice = ELS_backgroundremoval(elsdata, startslice, endslice)[:, anode, :]
    maxindices = np.unravel_index(tempslice.argmax(), tempslice.shape)
    # print("Max DEF", tempslice[maxindices])
    prominence = tempslice[maxindices] / 30
    # print("prominence", prominence)
    energybin = CAPS_energyslice("els", startenergy, startenergy)[0]

    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)[energybin, anode, :]
    peaks, properties = find_peaks(dataslice, height=1e9, prominence=prominence, width=0.5, rel_height=0.5,
                                   distance=15)
    results = peak_widths(dataslice, peaks, rel_height=0.97)

    recentpeakslice = []
    left = int(results[2][0])
    right = int(results[3][0]) + 1
    tempenergybin = energybin
    singlepeakslice = ELS_backgroundremoval(elsdata, startslice, endslice)[tempenergybin, anode, left:right]

    temppeaks, tempproperties = find_peaks(singlepeakslice, height=1e9, prominence=prominence, width=0.5,
                                           rel_height=0.5,
                                           distance=5)
    if temppeaks.size == 0:
        raise ValueError("No Peaks")
    while len(temppeaks) == 1:
        recentpeakslice = singlepeakslice
        recentenergybin = tempenergybin
        tempenergybin += 1
        singlepeakslice = ELS_backgroundremoval(elsdata, startslice, endslice)[tempenergybin, anode, left:right]
        temppeaks, tempproperties = find_peaks(singlepeakslice, height=5e9, prominence=prominence, width=0.5,
                                               rel_height=0.5,
                                               distance=5)

    times = elsdata['times_utc'][startslice:endslice][left:right]
    times_adjusted = [d + (63 - recentenergybin) * datetime.timedelta(seconds=31.25e-3) for d in
                      times]
    timestamps_adjusted = [datetime.datetime.timestamp(d) for d in times_adjusted]
    testtime = np.average(timestamps_adjusted, weights=recentpeakslice)

    # plt.plot(times_adjusted, recentpeakslice)
    # plt.vlines(times_adjusted[np.argmax(recentpeakslice)],min(recentpeakslice),max(recentpeakslice))
    # plt.vlines(datetime.datetime.fromtimestamp(testtime), min(recentpeakslice), max(recentpeakslice),color='m')

    # print("currenttime", times_adjusted[np.argmax(recentpeakslice)])
    # print("testtime", datetime.datetime.fromtimestamp(testtime))
    # return times_adjusted[np.argmax(recentpeakslice)], elscalib['earray'][tempenergybin]
    return datetime.datetime.fromtimestamp(testtime), elscalib['earray'][tempenergybin]


def heavy_ion_finder_ibs(ibsdata, startenergy, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(ibsdata, starttime), CAPS_slicenumber(ibsdata, endtime)
    energybin = CAPS_energyslice("ibs", startenergy, startenergy)[0]
    tempslice = (ibsdata['ibsdata'][:, 1, startslice:endslice] / (ibscalib['ibsgeom'] * 1e-4))
    maxindices = np.unravel_index(tempslice.argmax(), tempslice.shape)
    prominence = tempslice[maxindices] / 10
    # print("prominence", prominence)
    # dataslice = ibsdata['ibsdata'][energybin, 1, startslice:endslice]
    singlepeakslice = (ibsdata['ibsdata'][energybin, 1, startslice:endslice] / (ibscalib['ibsgeom'] * 1e-4))
    # peaks, properties = find_peaks(dataslice, height=5e2, prominence=prominence,distance=15)
    # print(peaks)
    # if peaks.size == 0:
    #     raise ValueError("No Peaks")
    # while len(peaks) == 1:
    while max(singlepeakslice) > prominence:
        recentpeakslice = singlepeakslice
        recentenergybin = energybin
        recentpeaks = np.argmax(recentpeakslice)
        energybin += 1
        singlepeakslice = (ibsdata['ibsdata'][energybin, 1, startslice:endslice] / (ibscalib['ibsgeom'] * 1e-4))
        # peaks, properties = find_peaks(singlepeakslice, height=5e2, prominence=prominence)

    zerocounter = 0
    i = 0
    while i == 0:
        i = int(ibsdata['ibsdata'][zerocounter, 1, startslice + recentpeaks])
        zerocounter += 1
    numberofsteps = (255 - (recentenergybin - zerocounter))

    # print("Counter", zerocounter)
    # print("RecentEnergyBin", recentenergybin)
    # print("Number of steps", numberofsteps)

    times = ibsdata['times_utc'][startslice:endslice]
    times_adjusted = [d + numberofsteps * datetime.timedelta(seconds=7.813e-3) for d in
                      times]
    timestamps_adjusted = [datetime.datetime.timestamp(d) for d in times_adjusted]
    weightedtime = datetime.datetime.fromtimestamp(np.average(timestamps_adjusted, weights=recentpeakslice))

    # return times_adjusted[np.argmax(recentpeakslice)], ibscalib['ibsearray'][energybin]
    return weightedtime, ibscalib['ibsearray'][energybin]


def CAPS_ramtimes(elsdata, starttime, endtime):
    ramtimes = []
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    for i in np.arange(startslice, endslice, 1):
        actuatorangle = elsdata['actuator'][i]
        ramangle = caps_ramdirection_azielv(elsdata['times_utc'][i])[0]
        if abs(actuatorangle - ramangle) < 1.5:
            ramtimes.append(caps_ramdirection_time(elsdata, elsdata['times_utc'][i]))
    for counter, i in enumerate(ramtimes):
        ramtimes[counter] = i.replace(microsecond=0)
    ramtimes = sorted(set(ramtimes))
    return ramtimes


def ELS_maxflux_anode(elsdata, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)
    anodesums = np.sum(np.sum(dataslice, axis=2), axis=0)

    maxflux_anode = np.argmax(anodesums)
    # print(anodesums, maxflux_anode)
    return maxflux_anode


filedates_times = {"t16": ["22-jul-2006", "00:22:00"],
                   "t17": ["07-sep-2006", "20:13:00"],
                   "t18": ["23-sep-2006", "18:55:00"],
                   "t19": ["09-oct-2006", "17:27:00"],
                   "t20": ["25-oct-2006", "15:54:00"],
                   "t21": ["12-dec-2006", "11:38:00"],
                   "t23": ["13-jan-2007", "08:35:00"],
                   "t25": ["22-feb-2007", "03:08:00"],
                   "t26": ["10-mar-2007", "01:43:00"],
                   "t27": ["26-mar-2007", "00:20:00"],
                   "t28": ["10-apr-2007", "22:53:00"],
                   "t29": ["26-apr-2007", "21:29:00"],
                   "t30": ["12-may-2007", "20:06:00"],
                   "t32": "13-jun-2007",
                   "t36": ["02-oct-2007", "04:39:00"],
                   "t39": ["20-dec-2007", "22:54:00"],
                   "t40": ["05-jan-2008", "21:26:00"],
                   "t41": ["22-feb-2008", "17:28:00"],
                   "t42": ["25-mar-2008", "14:24:00"],
                   "t43": ["12-may-2008", "09:58:00"],
                   "t46": ["03-nov-2008", "17:32:00"],
                   "t47": "19-nov-2008",
                   "t48": ["05-dec-2008", "14:22:00"],
                   "t49": ["21-dec-2008", "12:57:00"],
                   "t50": ["07-feb-2009", "08:47:00"],
                   "t51": ["27-mar-2009", "04:40:00"],
                   "t71": ["07-jul-2010", "00:19:00"],
                   "t83": ["22-may-2012", "01:07:00"]}

flyby_datetimes = {"t16": [datetime.datetime(2006, 7, 22, 0, 22), datetime.datetime(2006, 7, 22, 0, 28, 40)],
                   "t17": [datetime.datetime(2006, 9, 7, 20, 14, 30), datetime.datetime(2006, 9, 7, 20, 19, 40)],
                   "t18": [datetime.datetime(2006, 9, 23, 18, 56), datetime.datetime(2006, 9, 23, 19, 2)],
                   # "t19": [datetime.datetime(2006, 10, 9, 17, 28), datetime.datetime(2006, 10, 9, 17, 30, 14)],
                   "t19": [datetime.datetime(2006, 10, 9, 17, 31, 15), datetime.datetime(2006, 10, 9, 17, 33, 10)],
                   "t20": [datetime.datetime(2006, 10, 25, 15, 55, 30), datetime.datetime(2006, 10, 25, 15, 57, 45)],
                   "t21": [datetime.datetime(2006, 12, 12, 11, 39, 45), datetime.datetime(2006, 12, 12, 11, 43, 20)],
                   "t23": [datetime.datetime(2007, 1, 13, 8, 35), datetime.datetime(2007, 1, 13, 8, 42)],
                   "t25": [datetime.datetime(2007, 2, 22, 3, 10), datetime.datetime(2007, 2, 22, 3, 15)],
                   "t26": [datetime.datetime(2007, 3, 10, 1, 45, 30), datetime.datetime(2007, 3, 10, 1, 52, 20)],
                   "t27": [datetime.datetime(2007, 3, 26, 0, 21, 30), datetime.datetime(2007, 3, 26, 0, 26)],
                   "t28": [datetime.datetime(2007, 4, 10, 22, 55, 40), datetime.datetime(2007, 4, 10, 23)],
                   "t29": [datetime.datetime(2007, 4, 26, 21, 29, 30), datetime.datetime(2007, 4, 26, 21, 35, 30)],
                   "t30": [datetime.datetime(2007, 5, 12, 20, 8, 20), datetime.datetime(2007, 5, 12, 20, 11, 45)],
                   "t32": [datetime.datetime(2007, 6, 13, 17, 44), datetime.datetime(2007, 6, 13, 17, 48)],
                   "t36": [datetime.datetime(2007, 10, 2, 4, 39, 30), datetime.datetime(2007, 10, 2, 4, 45)],
                   "t39": [datetime.datetime(2007, 12, 20, 22, 54, 20), datetime.datetime(2007, 12, 20, 23, 1, 20)],
                   "t40": [datetime.datetime(2008, 1, 5, 21, 27, 20), datetime.datetime(2008, 1, 5, 21, 33, 30)],
                   "t41": [datetime.datetime(2008, 2, 22, 17, 29, 40), datetime.datetime(2008, 2, 22, 17, 34, 40)],
                   "t42": [datetime.datetime(2008, 3, 25, 14, 25), datetime.datetime(2008, 3, 25, 14, 30, 20)],
                   "t43": [datetime.datetime(2008, 5, 12, 9, 59), datetime.datetime(2008, 5, 12, 10, 5)],
                   "t46": [datetime.datetime(2008, 11, 3, 17, 33), datetime.datetime(2008, 11, 3, 17, 36, 30)],
                   "t48": [datetime.datetime(2008, 12, 5, 14, 23, 30), datetime.datetime(2008, 12, 5, 14, 28)],
                   "t49": [datetime.datetime(2008, 12, 21, 12, 58), datetime.datetime(2008, 12, 21, 13, 2, 15)],
                   "t50": [datetime.datetime(2009, 2, 7, 8, 48, 45), datetime.datetime(2009, 2, 7, 8, 53)],
                   "t51": [datetime.datetime(2009, 3, 27, 4, 42), datetime.datetime(2009, 3, 27, 4, 46)],
                   "t71": [datetime.datetime(2010, 7, 7, 0, 20, 30), datetime.datetime(2010, 7, 7, 0, 25)],
                   "t83": [datetime.datetime(2012, 5, 22, 1, 7, 45), datetime.datetime(2012, 5, 22, 1, 13)],
                   }
flyby_ramanodes = {"t16": [4, 5],
                   "t17": [4, 5],
                   "t18": [4, 5],
                   "t19": [4, 5],
                   "t20": [1, 2],
                   "t21": [4, 5],
                   "t23": [4, 5],
                   "t25": [4, 5],
                   "t26": [4, 5],
                   "t27": [6, 7],
                   "t28": [4, 5],
                   "t29": [4, 5],
                   "t30": [4, 5],
                   "t32": [4, 5],
                   "t36": [4, 5],
                   "t39": [4, 5],
                   "t40": [4, 5],
                   "t41": [4, 5],
                   "t42": [4, 5],
                   "t43": [4, 5],
                   "t46": [4, 5],
                   "t48": [4, 5],
                   "t49": [4, 5],
                   "t50": [4, 5],
                   "t51": [4, 5],
                   "t71": [4, 5],
                   "t83": [4, 5],
                   }


def main():
    flyby = "t83"
    anode1 = flyby_ramanodes[flyby][0]
    anode2 = flyby_ramanodes[flyby][1]
    lowerenergy = 2
    upperenergy = 750

    CAPS_df = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
    CAPS_df = CAPS_df[CAPS_df['Flyby'] == flyby.lower()]
    CAPS_df['Bulk Time'] = pd.to_datetime(CAPS_df['Bulk Time'])
    CAPS_df['Azimuthal Ram Time'] = pd.to_datetime(CAPS_df['Azimuthal Ram Time'])
    CAPS_df['Positive Peak Time'] = pd.to_datetime(CAPS_df['Positive Peak Time'])
    CAPS_df['Negative Peak Time'] = pd.to_datetime(CAPS_df['Negative Peak Time'])

    elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)

    starttime = flyby_datetimes[flyby][0]
    endtime = flyby_datetimes[flyby][1]

    ramtimes = CAPS_ramtimes(elsdata, starttime, endtime)
    maxflux_anodes = []
    for i in ramtimes:
        maxflux_anodes.append(
            ELS_maxflux_anode(elsdata, i - datetime.timedelta(seconds=10), i + datetime.timedelta(seconds=10)))

    heavypeaktimes, heavypeakenergies = [], []
    for ramtime, maxflux_anode in zip(ramtimes, maxflux_anodes):
        # print(ramtime, maxflux_anode)
        print(maxflux_anode, ramtime - datetime.timedelta(seconds=15), ramtime + datetime.timedelta(seconds=15))
        heavypeaktime, heavypeakenergy = heavy_ion_finder(elsdata, 20, maxflux_anode,
                                                          ramtime - datetime.timedelta(seconds=15),
                                                          ramtime + datetime.timedelta(seconds=15))
        print(heavypeaktime, heavypeakenergy)
        print("------Next------")
        heavypeaktimes.append(heavypeaktime)
        heavypeakenergies.append(heavypeakenergy)

    heavypeaktimes_ibs, heavypeakenergies_ibs = [], []
    for ramtime in ramtimes:
        print("IBS", ramtime)
        # print(50, maxflux_anode, ramtime - datetime.timedelta(seconds=10), ramtime + datetime.timedelta(seconds=10))
        heavypeaktime_ibs, heavypeakenergy_ibs = heavy_ion_finder_ibs(ibsdata, 15,
                                                                      ramtime - datetime.timedelta(seconds=20),
                                                                      ramtime + datetime.timedelta(seconds=20))
        # print(heavypeaktime_ibs, heavypeakenergy_ibs)
        print("------Next------")
        heavypeaktimes_ibs.append(heavypeaktime_ibs)
        heavypeakenergies_ibs.append(heavypeakenergy_ibs)

    fig, (elsax, elsax2, ibsax, actax) = plt.subplots(4, figsize=(18, 6), sharex=True)
    ELS_spectrogram(elsdata, anode1, filedates_times[flyby][1], 420, ax=elsax, fig=fig)
    ELS_spectrogram(elsdata, anode2, filedates_times[flyby][1], 420, ax=elsax2, fig=fig)
    IBS_spectrogram(ibsdata, 2, filedates_times[flyby][1], 420, ax=ibsax, fig=fig)

    for peaktime, peakenergy, maxflux_anode in zip(heavypeaktimes, heavypeakenergies, maxflux_anodes):
        if maxflux_anode == (anode1 - 1):
            elsax.scatter(peaktime, peakenergy, color='m', marker="X", s=100)
            elsax.vlines(peaktime, lowerenergy - 0.5, upperenergy + 50, color='m', linestyle="dotted")
            actax.vlines(peaktime, 1, 110, color='m', linestyle="dotted")
        if maxflux_anode == (anode2 - 1):
            elsax2.scatter(peaktime, peakenergy, color='m', marker="X", s=100)
            elsax2.vlines(peaktime, lowerenergy - 0.5, upperenergy + 50, color='m', linestyle="dotted")
            actax.vlines(peaktime, 1, 110, color='m', linestyle="dotted")

    for peaktime, peakenergy in zip(heavypeaktimes_ibs, heavypeakenergies_ibs):
        ibsax.scatter(peaktime, peakenergy, color='m', marker="X", s=100)
        ibsax.vlines(peaktime, lowerenergy - 0.5, upperenergy + 50, color='m', linestyle="dashed")
        actax.vlines(peaktime, 1, 110, color='m', linestyle="dashed")

    actuator_plot(elsdata, filedates_times[flyby][1], 600, ax=actax)
    actax.plot(CAPS_df['Azimuthal Ram Time'], CAPS_df['Azimuthal Ram Angle'], color='k')
    actax.plot(CAPS_df['Negative Peak Time'], CAPS_df['Negative Azimuth Angle'], color='k', linestyle='dotted')
    actax.plot(CAPS_df['Positive Peak Time'], CAPS_df['Positive Azimuth Angle'], color='k', linestyle='dashed')
    actax.plot(CAPS_df['Bulk Time'], CAPS_df['Bulk Azimuth'], color='k', linestyle='dashdot')

    elsax.vlines(CAPS_df['Negative Peak Time'], lowerenergy - 0.5, upperenergy + 50, color='k', linestyle="dotted")
    elsax2.vlines(CAPS_df['Negative Peak Time'], lowerenergy - 0.5, upperenergy + 50, color='k', linestyle="dotted")
    ibsax.vlines(CAPS_df['Positive Peak Time'], lowerenergy - 0.5, upperenergy + 50, color='k', linestyle="dashed")

    for ax in (elsax, elsax2):
        ax.set_ylim(lowerenergy, upperenergy)

    ibsax.set_ylim(3, 200)

    for ax in (elsax, elsax2, ibsax, actax):
        ax.vlines(CAPS_df['Azimuthal Ram Time'], lowerenergy - 0.5, upperenergy + 50, color='k')

    actax.set_ylim(70, 105)
    divider2 = make_axes_locatable(actax)
    cax2 = divider2.append_axes("right", size="1.5%", pad=.05)
    cax2.remove()

    plt.show()


main()
