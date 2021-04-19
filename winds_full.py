from __future__ import unicode_literals

import csv
import time

import matplotlib
import scipy.signal
from astropy.modeling import models, fitting
from cassinipy.caps.mssl import *
from cassinipy.caps.spice import *
from cassinipy.caps.util import *
from cassinipy.misc import *
from cassinipy.spice import *
from scipy.signal import peak_widths
import pandas as pd
import spiceypy as spice

import datetime
from util import *

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

# Loading Kernels
if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

ibscalib = readsav('calib\\ibsdisplaycalib.dat')
elscalib = readsav('calib\\geometricfactor.dat')
sngcalib = readsav('calib\\sngdisplaycalib.dat')

AMU = scipy.constants.physical_constants['atomic mass constant'][0]
AMU_eV = scipy.constants.physical_constants['atomic mass unit-electron volt relationship'][0]
e = scipy.constants.physical_constants['atomic unit of charge'][0]
e_mass = scipy.constants.physical_constants['electron mass'][0]
e_mass_eV = scipy.constants.physical_constants['electron mass energy equivalent in MeV'][0] * 1e6
c = scipy.constants.physical_constants['speed of light in vacuum'][0]
k = scipy.constants.physical_constants['Boltzmann constant'][0]

MCPefficiency = 0.05
Af = 0.33e-4

ELS_FWHM = 0.167
SNG_FWHM = 0.167
IBS_FWHM = 0.017

titan_flybyvelocities = {'t16': 6e3, 't17': 6e3, 't18': 6e3, 't19': 6e3,
                         't20': 6e3, 't21': 5.9e3, 't23': 6e3, 't25': 6.2e3, 't26': 6.2e3, 't27': 6.2e3, 't28': 6.2e3,
                         't29': 6.2e3,
                         't30': 6.2e3, 't32': 6.2e3, 't39': 6.3e3,
                         't40': 6.3e3, 't41': 6.3e3, 't42': 6.3e3, 't43': 6.3e3, 't46': 6.3e3, 't47': 6.3e3,
                         't83': 5.9e3}
titan_CAheight = {'t16': 950, 't17': 1000, 't18': 960, 't19': 980,
                  't20': 1029, 't21': 1000, 't23': 1000, 't25': 1000, 't26': 981, 't27': 1010, 't28': 991, 't29': 981,
                  't30': 959, 't32': 965, 't39': 970,
                  't40': 1010, 't41': 1000, 't42': 1000, 't43': 1000, 't46': 1100, 't47': 1023,
                  't83': 990}
titan_flybydates = {'t16': [2006, 7, 22], 't17': [2006, 9, 7], 't18': [2006, 9, 23], 't19': [2006, 10, 9],
                    't20': [2006, 10, 25], 't21': [2006, 12, 12], 't23': [2007, 1, 13], 't25': [2007, 2, 22],
                    't26': [2007, 3, 10],
                    't27': [2007, 3, 26], 't28': [2007, 4, 10], 't29': [2007, 4, 26],
                    't30': [2007, 5, 12], 't32': [2007, 6, 13], 't39': [2007, 12, 20],
                    't40': [2008, 1, 5], 't41': [2008, 2, 22], 't42': [2008, 3, 25], 't43': [2008, 5, 12],
                    't46': [2008, 11, 3], 't47': [2008, 11, 19],
                    't83': [2012, 5, 22]}


def heavy_ion_finder(elsdata, startenergy, anode, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    tempslice = ELS_backgroundremoval(elsdata, startslice, endslice)[:, anode, :]
    maxindices = np.unravel_index(tempslice.argmax(), tempslice.shape)
    prominence = tempslice[maxindices] / 30
    energybin = CAPS_energyslice("els", startenergy, startenergy)[0]

    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)[energybin, anode, :]
    peaks, properties = find_peaks(dataslice, height=1e9, prominence=prominence, width=0.5, rel_height=0.5,
                                   distance=15)
    results = peak_widths(dataslice, peaks, rel_height=0.95)

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
        recentpeaks = temppeaks
        recentproperties = tempproperties
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

    return datetime.datetime.fromtimestamp(testtime), elscalib['earray'][tempenergybin]


def heavy_ion_finder_ibs(ibsdata, startenergy, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(ibsdata, starttime), CAPS_slicenumber(ibsdata, endtime)
    energybin = CAPS_energyslice("ibs", startenergy, startenergy)[0]
    tempslice = (ibsdata['ibsdata'][:, 1, startslice:endslice] / (ibscalib['ibsgeom'] * 1e-4))
    maxindices = np.unravel_index(tempslice.argmax(), tempslice.shape)
    prominence = tempslice[maxindices] / 10

    singlepeakslice = (ibsdata['ibsdata'][energybin, 1, startslice:endslice] / (ibscalib['ibsgeom'] * 1e-4))
    while max(singlepeakslice) > prominence:
        recentpeakslice = singlepeakslice
        recentenergybin = energybin
        recentpeaks = np.argmax(recentpeakslice)
        energybin += 1
        singlepeakslice = (ibsdata['ibsdata'][energybin, 1, startslice:endslice] / (ibscalib['ibsgeom'] * 1e-4))

    zerocounter = 0
    i = 0
    while i == 0:
        i = int(ibsdata['ibsdata'][zerocounter, 1, startslice + recentpeaks])
        zerocounter += 1
    numberofsteps = (255 - (recentenergybin - zerocounter))

    times = ibsdata['times_utc'][startslice:endslice]
    times_adjusted = [d + numberofsteps * datetime.timedelta(seconds=7.813e-3) for d in
                      times]
    timestamps_adjusted = [datetime.datetime.timestamp(d) for d in times_adjusted]
    testtime = np.average(timestamps_adjusted, weights=recentpeakslice)

    return datetime.datetime.fromtimestamp(testtime), ibscalib['ibsearray'][energybin]


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
    return maxflux_anode


def cassini_titan_altlatlon(tempdatetime):
    et = spice.datetime2et(tempdatetime)

    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    lon, lat, alt = spice.recpgr('TITAN', state[:3], spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    return alt, lat * spice.dpr(), lon * spice.dpr()


def els_alongtrack_velocity(elsdata, tempdatetime):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[4]) ** 2 + (state[5]) ** 2 + (state[6]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(elsdata, tempdatetime)

    plt.plot(elscalib['earray'], elsdata['def'][:, 4, slicenumber])
    print("cassini_speed", cassini_speed)

    # return alongtrackvelocity


# def ibs_alongtrack_velocity(ibsdata,tempdatetime):


filedates = {"t16": "22-jul-2006", "t17": "07-sep-2006", "t18": "23-sep-2006", "t19": "09-oct-2006",
             "t20": "25-oct-2006", "t21": "12-dec-2006", "t23": "13-jan-2007", "t25": "22-feb-2007",
             "t26": "10-mar-2007",
             "t27": "26-mar-2007",
             "t28": "10-apr-2007", "t29": "26-apr-2007", "t39": "20-dec-2007",
             "t30": "12-may-2007", "t32": "13-jun-2007",
             "t40": "05-jan-2008", "t41": "22-feb-2008", "t42": "25-mar-2008", "t43": "12-may-2008",
             "t46": "03-nov-2008", "t47": "19-nov-2008"}

flyby_maxbeamtimegap = {"t16": 13, "t17": 13, "t18": 13, "t19": 13,
                        "t20": 13, "t21": 13, "t23": 13, "t25": 13, "t26": 13, "t27": 13, "t28": 13, "t29": 13,
                        "t30": 13, "t32": 13, "t39": 13,
                        "t40": 13, "t41": 13, "t42": 13, "t43": 13, "t46": 40, "t47": 30}

data_times_pairs = [
    ["t16", [datetime.datetime(2006, 7, 22, 0, 22), datetime.datetime(2006, 7, 22, 0, 28, 45)], 20, 15, 30],
    ["t17", [datetime.datetime(2006, 9, 7, 20, 13, 30), datetime.datetime(2006, 9, 7, 20, 19, 40)], 20, 15, 30],
    # ["t18", [datetime.datetime(2006, 9, 23, 20, 13, 30), datetime.datetime(2006, 9, 23, 20, 19, 40)], 20, 15, 30],
    ["t19", [datetime.datetime(2006, 10, 9, 17, 28), datetime.datetime(2006, 10, 9, 17, 30, 14)], 20, 15, 30],
    ["t19", [datetime.datetime(2006, 10, 9, 17, 31, 15), datetime.datetime(2006, 10, 9, 17, 33, 10)], 20, 15, 30],
    ["t20", [datetime.datetime(2006, 10, 25, 15, 55, 30), datetime.datetime(2006, 10, 25, 15, 57, 45)], 20, 15, 40],
    ["t21", [datetime.datetime(2006, 12, 12, 11, 40, 30), datetime.datetime(2006, 12, 12, 11, 43, 20)], 20, 15, 30],
    ["t23", [datetime.datetime(2007, 1, 13, 8, 35), datetime.datetime(2007, 1, 13, 8, 42)], 20, 15, 30],
    ["t25", [datetime.datetime(2007, 2, 22, 3, 10), datetime.datetime(2007, 2, 22, 3, 15)], 20, 15, 30],
    ["t26", [datetime.datetime(2007, 3, 10, 1, 45, 30), datetime.datetime(2007, 3, 10, 1, 52, 20)], 20, 15, 30],
    ["t27", [datetime.datetime(2007, 3, 26, 0, 21, 30), datetime.datetime(2007, 3, 26, 0, 26)], 20, 15, 30],
    ["t28", [datetime.datetime(2007, 4, 10, 22, 55, 40), datetime.datetime(2007, 4, 10, 23)], 20, 15, 30],
    ["t29", [datetime.datetime(2007, 4, 26, 21, 29, 30), datetime.datetime(2007, 4, 26, 21, 35, 30)], 20, 15, 30],
    ["t30", [datetime.datetime(2007, 5, 12, 20, 8, 30), datetime.datetime(2007, 5, 12, 20, 11, 45)], 40, 15, 30],
    ["t32", [datetime.datetime(2007, 6, 13, 17, 44), datetime.datetime(2007, 6, 13, 17, 48)], 20, 15, 30],
    ["t39", [datetime.datetime(2007, 12, 20, 22, 54, 20), datetime.datetime(2007, 12, 20, 23, 1, 20)], 20, 15, 30],
    ["t40", [datetime.datetime(2008, 1, 5, 21, 27, 20), datetime.datetime(2008, 1, 5, 21, 33, 30)], 20, 15, 30],
    ["t41", [datetime.datetime(2008, 2, 22, 17, 29, 40), datetime.datetime(2008, 2, 22, 17, 34, 40)], 20, 15, 30],
    ["t42", [datetime.datetime(2008, 3, 25, 14, 25), datetime.datetime(2008, 3, 25, 14, 30, 20)], 20, 15, 30],
    ["t43", [datetime.datetime(2008, 5, 12, 9, 59), datetime.datetime(2008, 5, 12, 10, 5)], 20, 15, 30],
    ["t46", [datetime.datetime(2008, 11, 3, 17, 33, 10), datetime.datetime(2008, 11, 3, 17, 36, 30)], 20, 15, 30],
    ["t47", [datetime.datetime(2008, 11, 19, 15, 53), datetime.datetime(2008, 11, 19, 15, 54)], 14, 15, 50],
]

usedflybys = ['t16', 't17', 't19', 't20', 't21', 't23', 't25', 't26', 't27', 't28', 't29', 't30', 't32', 't39', 't40',
              't41', 't42', 't43', 't46']
#oldflybys = ['t16', 't17', 't20', 't21', 't25', 't26', 't27', 't28', 't29', 't30', 't32', 't42', 't46']
#newflybys = ['t19', 't23', 't39', 't40', 't41', 't43']


# usedflybys = ['t42', 't46']
# usedflybys = ['t16', 't17', 't29']


def CAPS_winds(data_times_pairs):
    elspeakslist = []
    for flyby, times, negativemass, positivemass, timewindow in data_times_pairs:
        if flyby in usedflybys:
            elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
            generate_mass_bins(elsdata, flyby, "els")
            ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
            generate_aligned_ibsdata(ibsdata, elsdata, flyby)

            ramtimes = CAPS_ramtimes(elsdata, times[0], times[1])
            maxflux_anodes = []
            for i in ramtimes:
                maxflux_anodes.append(ELS_maxflux_anode(elsdata, i - datetime.timedelta(seconds=10),
                                                        i + datetime.timedelta(seconds=10)))
            # print(ramtimes,maxflux_anodes)

            for ramtime, maxflux_anode in zip(ramtimes, maxflux_anodes):
                heavypeaktime_neg, heavypeakenergy_neg = heavy_ion_finder(elsdata, negativemass, maxflux_anode,
                                                                          ramtime - datetime.timedelta(
                                                                              seconds=timewindow / 2),
                                                                          ramtime + datetime.timedelta(
                                                                              seconds=timewindow / 2))
                heavypeaktime_pos, heavypeakenergy_pos = heavy_ion_finder_ibs(ibsdata, positivemass,
                                                                              ramtime - datetime.timedelta(
                                                                                  seconds=timewindow / 2),
                                                                              ramtime + datetime.timedelta(
                                                                                  seconds=timewindow / 2))
                heavypeakangle_neg = CAPS_ELS_FOVcentre_azi_elv(heavypeaktime_neg, elsdata)[0]
                heavypeakangle_pos = CAPS_IBS_FOVcentre_azi_elv(heavypeaktime_pos, elsdata)[0]

                peaks = [heavypeaktime_neg, heavypeakenergy_neg, heavypeakangle_neg,
                         heavypeaktime_pos, heavypeakenergy_pos, heavypeakangle_pos]

                # print("--------Next----------")
                peaks.append(caps_ramdirection_time(elsdata, heavypeaktime_neg))
                peaks.append(caps_ramdirection_azielv(heavypeaktime_neg)[0])
                peaks.append(flyby)
                peaks.append(str(titan_flybydates[flyby][2]) + '/' + str(titan_flybydates[flyby][1]) + '/' + str(
                    titan_flybydates[flyby][0]))
                peaks.append(titan_flybyvelocities[flyby])
                peaks.append(CAPS_actuationtimeslice(ramtime, elsdata)[2])
                elspeakslist.append(list(peaks))
            del elsdata
            del ibsdata
    capsdf = pd.DataFrame(elspeakslist, columns=["Negative Peak Time", "Negative Peak Energy", "Negative Azimuth Angle",
                                                 "Positive Peak Time", "Positive Peak Energy", "Positive Azimuth Angle",
                                                 "Azimuthal Ram Time", "Azimuthal Ram Angle",
                                                 "Flyby", "FlybyDate", "Flyby velocity", "Actuation Direction"])

    capsdf['Bulk Azimuth'] = capsdf[["Negative Azimuth Angle", "Positive Azimuth Angle"]].mean(axis=1)
    capsdf['Bulk Time'] = capsdf["Negative Peak Time"] + (
            (capsdf["Positive Peak Time"] - capsdf["Negative Peak Time"]) / 2)
    capsdf["Bulk Deflection from Ram Angle"] = capsdf["Bulk Azimuth"] - capsdf["Azimuthal Ram Angle"]
    capsdf["Negative Deflection from Ram Angle"] = capsdf["Negative Azimuth Angle"] - capsdf["Azimuthal Ram Angle"]
    capsdf["Positive Deflection from Ram Angle"] = capsdf["Positive Azimuth Angle"] - capsdf["Azimuthal Ram Angle"]
    capsdf.drop_duplicates(subset=['Positive Peak Time'], inplace=True)
    capsdf.drop_duplicates(subset=['Negative Peak Time'], inplace=True)

    return capsdf


data = CAPS_winds(data_times_pairs)
data["ELS crosstrack velocity"] = np.sin(data["Negative Deflection from Ram Angle"] * spice.rpd()) * data[
    'Flyby velocity']
data["IBS crosstrack velocity"] = np.sin(data["Positive Deflection from Ram Angle"] * spice.rpd()) * data[
    'Flyby velocity']

data["Crosstrack velocity"] = np.sin(data["Bulk Deflection from Ram Angle"] * spice.rpd()) * data['Flyby velocity']
data["Absolute Crosstrack velocity"] = data["Crosstrack velocity"].abs()

alts, lats, lons = [], [], []
for tempdatetime in data['Bulk Time']:
    alt, lat, lon = cassini_titan_altlatlon(tempdatetime)
    alts.append(alt)
    lats.append(lat)
    lons.append(lon)
data['Altitude'] = alts
data['Longitude'] = lons
data['Latitude'] = lats

data.to_csv("crosswinds_full.csv")
