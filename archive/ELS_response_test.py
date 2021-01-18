from __future__ import division

import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from cassinipy.caps.mssl import CAPS_slicenumber, CAPS_energyslice, ELS_backgroundremoval, caps_ramdirection_time, \
    CAPS_actuation
from cassinipy.caps.spice import caps_ramdirection_azielv
from scipy.io import readsav
from scipy.signal import find_peaks, peak_widths
import spiceypy as spice
import glob

from util import generate_mass_bins

matplotlib.rcParams.update({'font.size': 15})

if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

filedates_times = {"t17": ["07-sep-2006", "20:13:00"],
                   "t20": "25-oct-2006",
                   "t21": ["12-dec-2006", "11:38:00"],
                   "t25": ["22-feb-2007", "03:08:00"],
                   "t26": ["10-mar-2007", "01:43:00"],
                   "t27": ["26-mar-2007", "00:20:00"],
                   "t28": ["10-apr-2007", "22:55:00"],
                   "t29": ["26-apr-2007", "21:29:00"],
                   "t30": "12-may-2007",
                   "t32": "13-jun-2007",
                   "t46": "03-nov-2008",
                   "t47": "19-nov-2008"}

flyby_datetimes = {"t17": [datetime.datetime(2006, 9, 7, 20, 14, 30), datetime.datetime(2006, 9, 7, 20, 19, 30)],
                   "t21": [datetime.datetime(2006, 12, 12, 11, 39, 45), datetime.datetime(2006, 12, 12, 11, 43, 20)],
                   "t25": [datetime.datetime(2007, 2, 22, 3, 10), datetime.datetime(2007, 2, 22, 3, 15)],
                   # "t27": [datetime.datetime(2007, 3, 26, 0, 21, 30), datetime.datetime(2007, 3, 26, 0, 23,30)],
                   "t27": [datetime.datetime(2007, 3, 26, 0, 21, 30), datetime.datetime(2007, 3, 26, 0, 26)],
                   "t29": [datetime.datetime(2007, 4, 26, 21, 29, 30), datetime.datetime(2007, 4, 26, 21, 35, 30)]
                   }
elscalib = readsav('calib\\geometricfactor.dat')


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def heavy_ion_finder_old(elsdata, startenergy, anode, starttime, endtime, prominence=1e10):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    energybin = CAPS_energyslice("els", startenergy, startenergy)[0]

    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)[energybin, anode, :]
    peaks, properties = find_peaks(dataslice, height=3e9, prominence=prominence, width=0.5, rel_height=0.4,
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
    # if temppeaks == []:
    #     raise ValueError("No Peaks")
    while len(temppeaks) == 1:
        recentpeaks = temppeaks
        recentproperties = tempproperties
        recentpeakslice = singlepeakslice
        recentenergybin = tempenergybin
        tempenergybin += 1
        singlepeakslice = ELS_backgroundremoval(elsdata, startslice, endslice)[tempenergybin, anode, left:right]
        temppeaks, tempproperties = find_peaks(singlepeakslice, height=5e9, prominence=prominence, width=0.5,
                                               rel_height=0.5,
                                               distance=5)

    print("Energy Bin time",
          elsdata['times_utc'][startslice + recentpeaks[0]] + (63 - recentenergybin) * datetime.timedelta(
              seconds=31.25e-3))
    # print(elsdata['times_utc'][startslice + left], elsdata['times_utc'][startslice + right])
    # print(anode, startslice, left, right)
    # print(recentpeakslice)
    times = elsdata['times_utc'][startslice:endslice][left:right]
    timestamps = [datetime.datetime.timestamp(d) for d in elsdata['times_utc'][startslice:endslice][left:right]]

    times_adjusted = [d + (63 - recentenergybin) * datetime.timedelta(seconds=31.25e-3) for d in
                      times]
    timestamps_adjusted = [
        datetime.datetime.timestamp(d + (63 - recentenergybin) * datetime.timedelta(seconds=31.25e-3)) for d in
        times]

    peak_timestamps_lower = timestamps_adjusted[int(np.floor(recentproperties['left_ips'][0]))]
    peak_timestamps_upper = timestamps_adjusted[int(np.ceil(recentproperties['right_ips'][0]))]

    stddev = (peak_timestamps_upper - peak_timestamps_lower) / 2
    time_init = models.Gaussian1D(amplitude=max(recentpeakslice), mean=timestamps_adjusted[recentpeaks[0]],
                                  stddev=stddev,
                                  bounds={
                                      "amplitude": (max(recentpeakslice), max(recentpeakslice) * 1.2),
                                      "mean": (peak_timestamps_lower, peak_timestamps_upper),
                                      "stddev": (stddev / 1.5, stddev * 4)
                                  })

    timefit = fitting.LevMarLSQFitter()(time_init, timestamps_adjusted, recentpeakslice)

    timestamp_linspace = np.linspace(timestamps_adjusted[0], timestamps_adjusted[-1], 100)
    actuatorangle = elsdata['actuator'][startslice:endslice][left:right]
    actuatorangle_adjusted = np.interp(timestamps_adjusted, timestamps, actuatorangle)
    print(timestamps, actuatorangle)
    print(timestamps_adjusted, actuatorangle_adjusted)
    act_init = models.Gaussian1D(amplitude=max(recentpeakslice), mean=actuatorangle_adjusted[recentpeaks[0]],
                                 stddev=1,
                                 bounds={
                                     "amplitude": (max(recentpeakslice), max(recentpeakslice) * 1.2),
                                     "mean": (actuatorangle_adjusted[recentpeaks[0]] - 1, actuatorangle_adjusted[recentpeaks[0]] + 1),
                                     "stddev": (1, 3)
                                 })
    actfit = fitting.LevMarLSQFitter()(act_init, actuatorangle, recentpeakslice)
    ramtime = caps_ramdirection_time(elsdata, datetime.datetime.fromtimestamp(timefit.mean.value))
    ramangle =  caps_ramdirection_azielv(
              caps_ramdirection_time(elsdata, datetime.datetime.fromtimestamp(timefit.mean.value)))[0]
    print("Ram Time", ramtime, ramangle)
    print("Time Mean", datetime.datetime.fromtimestamp(timefit.mean.value),
          CAPS_actuation(elsdata, datetime.datetime.fromtimestamp(timefit.mean.value)),
          CAPS_actuation(elsdata, datetime.datetime.fromtimestamp(timefit.mean.value))-ramangle)

    # fig, ax = plt.subplots()
    # ax.plot(elsdata['actuator'][startslice:endslice],timestamps)
    if elsdata['actuator'][startslice:endslice][0] > elsdata['actuator'][startslice:endslice][-1]:
        testtimes = np.interp(actfit.mean.value, np.flip(elsdata['actuator'][startslice:endslice][left:right]), np.flip(timestamps))
    else:
        testtimes = np.interp(actfit.mean.value, elsdata['actuator'][startslice:endslice][left:right], timestamps)
    # ax.vlines(actfit.mean.value,min(timestamps),max(timestamps),color='k')
    # ax.hlines(testtimes, min(elsdata['actuator'][startslice:endslice]), max(elsdata['actuator'][startslice:endslice]), color='k')
    print("Act Mean", datetime.datetime.fromtimestamp(testtimes), actfit.mean.value, actfit.mean.value-ramangle)
    print("No Fit Mean", times_adjusted[np.argmax(recentpeakslice)],
          CAPS_actuation(elsdata, times_adjusted[np.argmax(recentpeakslice)]),
          CAPS_actuation(elsdata, times_adjusted[np.argmax(recentpeakslice)])-ramangle)

    if actuatorangle_adjusted[0] > actuatorangle_adjusted[-1]:
        print("Negative Actuation")
        act_linspace = np.linspace(actuatorangle_adjusted[-1], actuatorangle_adjusted[0], 100)
    else:
        act_linspace = np.linspace(actuatorangle_adjusted[0], actuatorangle_adjusted[-1], 100)

    fig, axes = plt.subplots(2)

    axes[0].plot(times_adjusted, recentpeakslice, label="data")
    axes[0].plot([datetime.datetime.fromtimestamp(d) for d in timestamp_linspace], timefit(timestamp_linspace),
                 linestyle='--', label="Gaussian fit")
    axes[0].vlines(caps_ramdirection_time(elsdata, datetime.datetime.fromtimestamp(timefit.mean.value))
                   , min(recentpeakslice), max(recentpeakslice), color='k')

    axes[1].plot(actuatorangle_adjusted, recentpeakslice, label="data")
    axes[1].plot(act_linspace, actfit(act_linspace), label="interp")
    axes[1].vlines(
        caps_ramdirection_azielv(caps_ramdirection_time(elsdata, datetime.datetime.fromtimestamp(timefit.mean.value)))[
            0]
        , min(recentpeakslice), max(recentpeakslice), color='k')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("DEF")
    axes[1].set_xlabel("Actuator Angle")
    axes[1].set_ylabel("DEF")
    axes[0].legend()
    axes[1].legend()

    return datetime.datetime.fromtimestamp(timefit.mean.value), elscalib['earray'][tempenergybin]


x = np.linspace(-15, 15, 1000)
x_convolve = np.linspace(-15, 15, 1999)
mu = 0
sig = 1
y_ionbeam = gaussian(x, mu, sig)


def calc_convolve(ELS_act, x=x, y_ionbeam=y_ionbeam):
    ELS_mu = 0.41  # 0.41 at 125eV
    ELS_sig = 2.74
    y_ELS = gaussian(x, ELS_mu + ELS_act, ELS_sig)
    convolved_y = np.convolve(y_ionbeam, y_ELS, 'full')
    convolved_y = (convolved_y / max(convolved_y))

    return y_ELS, convolved_y


y_ELS, convolved_y = calc_convolve(0, x=x, y_ionbeam=y_ionbeam)


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
    # print(anodesums,maxflux_anode)
    return maxflux_anode


flyby = "t17"
anode1 = 4
anode2 = 5
lowerenergy = 2
upperenergy = 500

elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
generate_mass_bins(elsdata, flyby, "els")

starttime = flyby_datetimes[flyby][0]
endtime = flyby_datetimes[flyby][1]

ramtimes = CAPS_ramtimes(elsdata, starttime, endtime)
print(ramtimes)
maxflux_anodes = []
for i in ramtimes:
    maxflux_anodes.append(
        ELS_maxflux_anode(elsdata, i - datetime.timedelta(seconds=10), i + datetime.timedelta(seconds=10)))
# print(ramtimes,maxflux_anodes)

heavypeaktimes, heavypeakenergies = [], []
for ramtime, maxflux_anode in zip(ramtimes, maxflux_anodes):
    print(ramtime, maxflux_anode)
    print(maxflux_anode, ramtime - datetime.timedelta(seconds=15), ramtime + datetime.timedelta(seconds=15))
    heavypeaktime, heavypeakenergy = heavy_ion_finder(elsdata, 30, maxflux_anode,
                                                      ramtime - datetime.timedelta(seconds=15),
                                                      ramtime + datetime.timedelta(seconds=15), prominence=1e10)
    print(heavypeaktime, heavypeakenergy)
    print("------Next------")
    heavypeaktimes.append(heavypeaktime)
    heavypeakenergies.append(heavypeakenergy)

# fig, ax = plt.subplots()
# ax.plot(x, y_ELS, label="ELS azimuth response")
# ax.plot(x, y_ionbeam , label="Ion Beam")
# ax.plot(x_convolve, convolved_y, label="Convolution", color='C2')
# ax.set_xlabel("Azimuth Angle")
# ax.set_ylabel("Normalised Intensity")
# fig.legend()

# plt.legend()
plt.show()
