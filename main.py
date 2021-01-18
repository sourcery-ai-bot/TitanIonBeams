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

titan_flybyvelocities = {'t16': 6e3 ,'t17': 6e3,
                         't20': 6e3, 't21': 5.9e3, 't25': 6.2e3, 't26': 6.2e3, 't27': 6.2e3, 't28': 6.2e3, 't29': 6.2e3,
                         't30': 6.2e3, 't32': 6.2e3,
                         't40': 6.3e3, 't42': 6.3e3, 't46': 6.3e3, 't47': 6.3e3,
                         't83': 5.9e3}
titan_CAheight = {'t16': 950,'t17': 1000,
                  't20': 1029, 't21': 1000, 't25': 1000, 't26': 981, 't27': 1010, 't28': 991, 't29': 981,
                  't30': 959, 't32': 965,
                  't42': 1000, 't46': 1100, 't47': 1023,
                  't83': 990}
titan_flybydates = {'t16': [2006, 7, 22], 't17': [2006, 9, 7],
                    't20': [2006, 10, 25], 't21': [2006, 12, 12], 't25': [2007, 2, 22], 't26': [2007, 3, 10],
                    't27': [2007, 3, 26], 't28': [2007, 4, 10], 't29': [2007, 4, 26],
                    't30': [2007, 5, 12], 't32': [2007, 6, 13],
                    't40': [2008, 1, 5],  't42': [2008, 3, 25], 't46': [2008, 11, 3], 't47': [2008, 11, 19],
                    't83': [2012, 5, 22]}


def ELS_beamanodes(elevation):
    beamanodes = [0, 0]
    anoderanges = np.arange(-70, 90, 20)
    if elevation < -70:
        return [0]
    if elevation > 70:
        return [7]

    for counter, elv in enumerate(anoderanges):
        # print(counter,elv, beamanodes)
        if elevation >= elv and elevation <= anoderanges[counter + 1]:
            beamanodes[1] += 1
            return beamanodes
        else:
            beamanodes[0] += 1
            beamanodes[1] += 1


def ELS_ramanodes(tempdatetime):
    ramelv = caps_ramdirection_azielv(tempdatetime)[1]
    ramanodes = ELS_beamanodes(ramelv)
    return ramanodes


def ELS_nonramanodes(tempdatetime):
    ramelv = caps_ramdirection_azielv(tempdatetime)[1]
    ramanodes = ELS_beamanodes(ramelv)
    nonramanodes = [i for i in np.arange(0, 8, 1) if i not in ramanodes]
    return nonramanodes


def ELS_beamsfinder(data, lowerenergylim, upperenergylim, starttime, endtime, prominence=1e3):
    '''
    Finds peaks in a given time and energy range

    Returns:

    peakenergy - energy in eV of the peak energy in time/energy/anode
    peakelv -  elevation in degrees of the peak energy in time/energy/anode
    peaktime - datetime when peak occurs
    peakanode -  anode which peak occurs in
    peakbin - bin which peak occurs in
    peakazi - azimuth which peak occurs in
    peakazidef - DEF of the peak azimuth
    peakenergybins - energy bin that the peak occurs for each time

    '''

    startslicenumber = CAPS_slicenumber(data, starttime)
    endslicenumber = CAPS_slicenumber(data, endtime)
    ramanodes = list(set(ELS_ramanodes(starttime) + ELS_ramanodes(endtime)))
    # print(ramanodes)
    startenergyslice, endenergyslice = CAPS_energyslice("els", lowerenergylim, upperenergylim)
    # print("startenergyslice",startenergyslice)
    # dataslice = data['data'][startenergyslice:endenergyslice,:8,startslicenumber:endslicenumber]
    dataslice = ELS_backgroundremoval(data, startslicenumber, endslicenumber)[startenergyslice:endenergyslice, :, :]

    peakindex = np.unravel_index(dataslice.argmax(), dataslice.shape)
    # print(peakindex[0])
    peakbin = startenergyslice + peakindex[0]
    peakslice = startslicenumber + peakindex[2]
    peakenergy = elscalib['earray'][peakbin]
    peakanode = peakindex[1]
    peaktime = data['times_utc'][startslicenumber + peakindex[2]]

    # Creates a array that encompasses the highest energy peak across every slice
    peakenergybins = np.zeros(shape=(endslicenumber - startslicenumber))
    peakanodes = np.zeros(shape=(endslicenumber - startslicenumber))
    fixeddataslice = np.zeros(shape=(endslicenumber - startslicenumber))
    for slicecounter, slicenumber in enumerate(np.arange(startslicenumber, endslicenumber, 1)):

        tempindex = np.unravel_index(data['def'][startenergyslice:endenergyslice, :8, slicenumber].argmax(),
                                     data['def'][startenergyslice:endenergyslice, :8, slicenumber].shape)
        if tempindex[1] not in ramanodes:
            # TODO write better method for checking which is actual ram anode
            # print(ELS_ramanodes(data['times_utc'][slicenumber]))
            peakanodes[slicecounter] = peakanodes[slicecounter - 1]
        else:
            peakanodes[slicecounter] = int(tempindex[1])
        #         if tempindex[1] > 1:
        #             print(tempindex,data['def'][startenergyslice:endenergyslice,:8,slicenumber])
        peakenergybin = tempindex[0]

        peakenergybins[slicecounter] = peakenergybin + startenergyslice
        # print(peakenergybin,slicecounter)
        fixeddataslice[slicecounter] = dataslice[peakenergybin, tempindex[1], slicecounter]
    # print(fixeddataslice)

    # plots across time and fits a gaussian
    times = data['times_utc'][startslicenumber:endslicenumber]
    timestamps = [datetime.datetime.timestamp(d) for d in times]
    peaks, properties = find_peaks(fixeddataslice, prominence=prominence, width=1, rel_height=0.5, distance=5)
    results = peak_widths(fixeddataslice, peaks, rel_height=1)
    # print(results[2],results[3])
    peakenergies = []
    peaktimes = []
    peakelvs = []
    # print(peakanodes[peaks])
    for peakcounter, peak in enumerate(peaks):
        left = int(results[2][peakcounter])
        right = int(results[3][peakcounter]) + 1

        peak_timestamps_lower = timestamps[int(np.floor(properties['left_ips'][peakcounter]))]
        peak_timestamps_upper = timestamps[int(np.ceil(properties['right_ips'][peakcounter]))]
        stddev = (peak_timestamps_upper - peak_timestamps_lower) / 2
        time_init = models.Gaussian1D(amplitude=max(fixeddataslice), mean=timestamps[peak], stddev=stddev,
                                      bounds={"mean": (peak_timestamps_lower, peak_timestamps_upper),
                                              "stddev": (stddev / 2, stddev * 2)
                                              })
        timefit = fitting.LevMarLSQFitter()(time_init, timestamps[left:right], fixeddataslice[left:right])
        peaktime = datetime.datetime.fromtimestamp(timefit.mean.value)

        # Find the peak location with a gaussian
        # Background defined as average of 6 lowest count
        energies = elscalib['earray'][startenergyslice:endenergyslice]

        # print(peak,dataslice[:,peak])
        # print(peak,peakanodes[peak])
        # print(peaktime, peakenergy, elscalib['earray'][startenergyslice + dataslice[:, int(peakanodes[peak]), peak].argmax()])
        energy_init = models.Gaussian1D(amplitude=max(dataslice[:, int(peakanodes[peak]), peak]), mean=peakenergy,
                                        stddev=peakenergy * 0.167, bounds={"mean": (lowerenergylim, upperenergylim)})
        energyfit_fitter = fitting.LevMarLSQFitter()
        energyfit = energyfit_fitter(energy_init, energies, dataslice[:, int(peakanodes[peak]), peak])
        # cov_diag = np.diag(energyfit_fitter.fit_info['param_cov'])

        # print('Mean: {} +\- {}'.format(energyfit.mean.value, np.sqrt(cov_diag[1])))

        fittedpeakenergy = energyfit.mean.value
        # print(max(dataslice[:, int(peakanodes[peak]), peak]), fittedpeakenergy)

        if fittedpeakenergy == lowerenergylim:
            print(lowerenergylim, upperenergylim, peaktime, peakenergy, "Lower Bound energy Fitting Warning")
        if fittedpeakenergy == upperenergylim:
            print(lowerenergylim, upperenergylim, peaktime, peakenergy, "Upper bound energy Fitting Warning")

        # fig, ax = plt.subplots()
        # ax.plot(energies, dataslice[:, int(peakanodes[peak]), peak], label="first energies")
        # ax.plot(energies, energyfit(energies), label="fitted energies")
        # fig.legend()

        # Background defined as average of 5 loweest count, negative ions unlikely to appear across 3 anodes
        # print(peakenergybins[peak]-startenergyslice)
        # print(dataslice[int(peakenergybins[peak]-startenergyslice),:,peak])
        datacut = dataslice[int(peakenergybins[peak] - startenergyslice), :, peak]
        backgroundremoved_temp = np.array(datacut) - np.mean(sorted(datacut)[:5])
        backgroundremoved_anodes = [0 if i < 0 else i for i in backgroundremoved_temp]
        # Fit a gaussian across the anodes to find elevation peak flux
        anodeangles = np.arange(-70, 90, 20)
        elv_init = models.Gaussian1D(amplitude=max(backgroundremoved_anodes), mean=anodeangles[int(peakanodes[peak])],
                                     stddev=5)
        elvfit = fitting.LevMarLSQFitter()(elv_init, anodeangles, backgroundremoved_anodes)
        peakelv = elvfit.mean.value
        if peakelv < -80:
            peakelv = -80
        if peakelv > 80:
            peakelv = 80
            # print("peakelv",peakelv)
        # plt.plot(anodeangles,backgroundremoved_anodes)
        # plt.plot(anodeangles,elvfit(anodeangles))

        # plt.plot(times[left:right],fixeddataslice[left:right])
        # plt.plot(times[left:right],timefit(timestamps[left:right]))

        peakenergies.append(fittedpeakenergy)
        peaktimes.append(peaktime)
        peakelvs.append(peakelv)

    # plt.show()

    return peakenergies, peakelvs, peaktimes, peakanodes, peakenergybins, peaks


def IBS_beamsfinder(data, lowerenergylim, upperenergylim, starttime, endtime, fan=2, prominence=1e3):
    '''
    Fan 2 default
    '''
    # print(starttime,endtime)
    startslicenumber = CAPS_slicenumber(data, starttime)
    endslicenumber = CAPS_slicenumber(data, endtime)
    startenergyslice, endenergyslice = CAPS_energyslice("ibs", lowerenergylim, upperenergylim)

    dataslice = data['ibsdata'][startenergyslice:endenergyslice, fan - 1,
                startslicenumber:endslicenumber]  # /(ibscalib['ibsgeom']*1e-4)

    peakindex = np.unravel_index(dataslice.argmax(), dataslice.shape)

    # print(dataslice.shape)
    peakenergy = ibscalib['ibsearray'][startenergyslice + peakindex[0]]
    # peaktime = data['times_utc'][startslicenumber+peakindex[1]]

    # Creates a array that encompasses the highest energy peak across every slice
    peakenergybins = np.zeros(shape=(endslicenumber - startslicenumber))
    fixeddataslice = np.zeros(shape=(endslicenumber - startslicenumber))
    for slicecounter, slicenumber in enumerate(np.arange(startslicenumber, endslicenumber, 1)):
        temp = list(data['ibsdata'][startenergyslice:endenergyslice, 1, slicenumber])
        peakenergybin = int(temp.index(max(temp)))
        peakenergybins[slicecounter] = peakenergybin + startenergyslice
        fixeddataslice[slicecounter] = dataslice[peakenergybin, slicecounter]

    # plots across time and fits a gaussian
    times = data['times_utc'][startslicenumber:endslicenumber]
    timestamps = [datetime.datetime.timestamp(d) for d in times]
    peaks, properties = find_peaks(fixeddataslice, prominence=prominence, width=1, rel_height=0.5)
    results = peak_widths(fixeddataslice, peaks, rel_height=1)
    # print(results[2],results[3])
    peakenergies = []
    peaktimes = []
    for peakcounter, peak in enumerate(peaks):
        left = int(results[2][peakcounter])
        right = int(results[3][peakcounter]) + 1

        peak_timestamps_lower = timestamps[int(np.floor(properties['left_ips'][peakcounter]))]
        peak_timestamps_upper = timestamps[int(np.ceil(properties['right_ips'][peakcounter]))]
        stddev = (peak_timestamps_upper - peak_timestamps_lower) / 2
        time_init = models.Gaussian1D(amplitude=max(fixeddataslice), mean=timestamps[peak], stddev=stddev,
                                      bounds={"mean": (peak_timestamps_lower, peak_timestamps_upper),
                                              "stddev": (stddev / 2, stddev * 2)})
        timefit = fitting.LevMarLSQFitter()(time_init, timestamps[left:right], fixeddataslice[left:right])
        peaktime = datetime.datetime.fromtimestamp(timefit.mean.value)

        # Find the peak location with a gaussian
        # Background defined as average of 6 lowest count
        energies = ibscalib['ibsearray'][startenergyslice:endenergyslice]

        # plt.plot(energies,dataslice[:,peak])
        # print(peak,dataslice[:,peak])
        energy_init = models.Gaussian1D(amplitude=max(dataslice[:, peak]), mean=peakenergy, stddev=1,
                                        bounds={"mean": (lowerenergylim, upperenergylim)})
        energyfit = fitting.LevMarLSQFitter()(energy_init, energies, dataslice[:, peak])
        peakenergy = energyfit.mean.value

        # plt.plot(energies,energyfit(energies))
        if peakenergy == lowerenergylim or peakenergy == upperenergylim:
            continue
        # plt.plot(times[left:right],fixeddataslice[left:right])
        # plt.plot(times[left:right],timefit(timestamps[left:right]))

        peakenergies.append(peakenergy)
        peaktimes.append(peaktime)

    return peakenergies, peaktimes, peakenergybins


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
    tempslice = (ibsdata['ibsdata'][:, 1, startslice:endslice]/ (ibscalib['ibsgeom'] * 1e-4))
    maxindices = np.unravel_index(tempslice.argmax(), tempslice.shape)
    prominence = tempslice[maxindices]/10

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


filedates = {"t16": "22-jul-2006","t17": "07-sep-2006",
             "t20": "25-oct-2006","t21": "12-dec-2006", "t25": "22-feb-2007","t26": "10-mar-2007","t27": "26-mar-2007",
             "t28": "10-apr-2007","t29": "26-apr-2007",
             "t30": "12-may-2007", "t32": "13-jun-2007",
             "t42": "25-mar-2008", "t46": "03-nov-2008", "t47": "19-nov-2008"}

flyby_maxbeamtimegap = {"t16": 13, "t17": 13,
                        "t20": 13, "t21": 13, "t25": 13, "t26": 13, "t27": 13, "t28": 13, "t29": 13,
                        "t30": 13, "t32": 13,
                        "t42": 13, "t46": 40, "t47": 30}

elsdata_times_pairs = [
    ["t16", [datetime.datetime(2006, 7, 22, 0, 22), datetime.datetime(2006, 7, 22, 0, 23, 45)],
     [[2, 5], [5, 10], [10, 70]],
     20, 3e10, 2e11, 30],
    ["t16", [datetime.datetime(2006, 7, 22, 0, 23, 45), datetime.datetime(2006, 7, 22, 0, 27, 10)],
     [[2, 5], [5, 10], [10, 700]],
     20, 3e10, 2e11, 30],
    ["t16", [datetime.datetime(2006, 7, 22, 0, 27, 10), datetime.datetime(2006, 7, 22, 0, 28, 40)],
     [[2, 5], [5, 10], [10, 70]],
     20, 3e10, 2e11, 30],
    ["t17", [datetime.datetime(2006, 9, 7, 20, 14, 30), datetime.datetime(2006, 9, 7, 20, 19, 40)],
     [[2, 5], [5, 11], [9, 19], [35, 120]],
     20, 3e10, 2e11, 30],
    ["t20", [datetime.datetime(2006, 10, 25, 15, 55, 30), datetime.datetime(2006, 10, 25, 15, 57, 45)],
     [[2, 5], [6, 10], [9, 18], [35, 100]],
     20, 3e10, 2e11, 40],
    ["t21", [datetime.datetime(2006, 12, 12, 11, 39, 45), datetime.datetime(2006, 12, 12, 11, 43, 20)],
     [[2, 5], [6, 10], [10, 35], [35, 100]],
     20, 2e10, 1e10, 30],
    ["t25", [datetime.datetime(2007, 2, 22, 3, 10), datetime.datetime(2007, 2, 22, 3, 15)],
     [[2, 6], [6, 13], [19, 190]],
     20, 2e10, 5e10, 30],
    ["t26", [datetime.datetime(2007, 3, 10, 1, 45, 30), datetime.datetime(2007, 3, 10, 1, 52, 20)],
     [[2, 6], [6, 13], [17, 230]],
     20, 1e10, 5e10, 30],
    ["t27", [datetime.datetime(2007, 3, 26, 0, 21, 30), datetime.datetime(2007, 3, 26, 0, 26)],
     [[2, 6], [6, 12], [14, 100]],
     20, 6e10, 5e10, 30],
    ["t28", [datetime.datetime(2007, 4, 10, 22, 55, 40), datetime.datetime(2007, 4, 10, 23)],
     [[2, 5.5], [6, 12], [16, 100]],
     20, 1e10, 1e10, 30],
    ["t29", [datetime.datetime(2007, 4, 26, 21, 29, 30), datetime.datetime(2007, 4, 26, 21, 35, 30)],
     [[2, 5.4], [6, 12], [16, 190]],
     20, 2e10, 5e10, 30],
    ["t30", [datetime.datetime(2007, 5, 12, 20, 8, 30), datetime.datetime(2007, 5, 12, 20, 11, 45)],
     [[2, 6], [6, 12], [16, 1000]],
     40, 3e10, 5e10, 30],
    ["t32", [datetime.datetime(2007, 6, 13, 17, 44), datetime.datetime(2007, 6, 13, 17, 48)],
     [[2, 6], [6, 12], [13, 1000]],
     20, 1e10, 5e10, 30],
    ["t42", [datetime.datetime(2008, 3, 25, 14, 25), datetime.datetime(2008, 3, 25, 14, 30, 20)],
     [[2, 6], [6, 12], [13, 200]],
     20, 1e10, 5e10, 30],
    ["t46", [datetime.datetime(2008, 11, 3, 17, 33, 10), datetime.datetime(2008, 11, 3, 17, 36, 30)],
     [[2, 5], [5, 14], [20, 100]],
     20, 2e9, 5e10, 30],
    #                         [T46ELSdata,
    #                             [datetime.datetime(2008,11,3,17,37,20),datetime.datetime(2008,11,3,17,38,40)],
    # #                         [datetime.datetime(2008,11,3,17,37,45),
    # #                             datetime.datetime(2008,11,3,17,38,4)
    # #                                     ],[[2.5,6],[6,12],[11,28]],20,8e2, 2e3],
    ["t47",
     [datetime.datetime(2008, 11, 19, 15, 53), datetime.datetime(2008, 11, 19, 15, 54)],
     [[2.5, 7], [7, 12], [12, 25]],
     14, 1e8, 1e11, 50],
]

ibsdata_times_pairs = [
    ["t16", [datetime.datetime(2006, 7, 22, 0, 22), datetime.datetime(2006, 7, 22, 0, 28, 40)],
     [[2.2, 3.9], [4, 6.2], [6.2, 8.1], [8.1, 10.4], [10.4, 12.8], [12.8, 14.5], [14.5, 16.8]],
     15, 2e3, 1e4],
    ["t17", [datetime.datetime(2006, 9, 7, 20, 14, 30), datetime.datetime(2006, 9, 7, 20, 19, 30)],
     [[2.2, 3.9], [4, 6.2], [6.2, 8.2], [8.2, 10.7], [10.7, 12.9], [12.9, 15], [15, 17.2], [17.2, 19.5]],
     15, 2e3, 1e4],
    ["t20", [datetime.datetime(2006, 10, 25, 15, 54, 20), datetime.datetime(2006, 10, 25, 15, 58, 49)],
     [[4.1, 6.3], [6.3, 8.3], [8.3, 10.7], [10.7, 12.9], [12.9, 14.9], [14.9, 17.2]],
     15, 3e3, 2e4],
    ["t21", [datetime.datetime(2006, 12, 12, 11, 39, 15), datetime.datetime(2006, 12, 12, 11, 43, 20)],
     [[4.1, 6.3], [6.3, 8.3], [8.3, 10.7], [10.7, 12.9], [12.9, 14.9], [14.9, 17.2]],
     15, 3e3, 2e4],
    ["t25", [datetime.datetime(2007, 2, 22, 3, 10), datetime.datetime(2007, 2, 22, 3, 15)],
     [[4.6, 6.8], [6.8, 9], [9, 11.6], [11.6, 14.3], [14.3, 16.4], [16.4, 19.2], [19, 21.3], [21, 75]],
     15, 2e3, 2e4],
    ["t26", [datetime.datetime(2007, 3, 10, 1, 45, 30), datetime.datetime(2007, 3, 10, 1, 52, 20)],
     [[4.5, 6.8], [6.8, 9], [9, 11.6], [11.6, 14.3], [14.3, 16.4], [16.4, 19.2], [19, 21.3], [21, 75]],
     15, 2e3, 2e4],
    ["t27", [datetime.datetime(2007, 3, 26, 0, 20, 45), datetime.datetime(2007, 3, 26, 0, 26)],
     [[4.6, 6.8], [6.8, 9], [9, 11.6], [11.6, 14.3], [14.3, 16.4], [16.4, 19.2], [19, 21.3], [21, 75]],
     15, 3e3, 2e4],
    ["t28", [datetime.datetime(2007, 4, 10, 22, 55, 40), datetime.datetime(2007, 4, 10, 23)],
     [[4.6, 6.8], [6.8, 9], [9, 11.6], [11.6, 14], [14, 16.2], [16.2, 19], [19, 21.3], [21, 100]],
     15, 2e3, 2e4],
    ["t29", [datetime.datetime(2007, 4, 26, 21, 29, 30), datetime.datetime(2007, 4, 26, 21, 35, 30)],
     [[4.6, 6.8], [6.8, 9], [9, 11.6], [11.6, 14], [14, 16.2], [16.2, 19], [19, 21.3], [21, 100]],
     15, 2e3, 2e4],
    ["t30", [datetime.datetime(2007, 5, 12, 20, 8, 20), datetime.datetime(2007, 5, 12, 20, 11, 50)],
     [[4.6, 6.8], [6.8, 9], [9, 11.6], [11.6, 14], [14, 16.2], [16.2, 19], [19, 21.3], [34, 41], [41, 100]],
     15, 2e3, 2e4],
    ["t32", [datetime.datetime(2007, 6, 13, 17, 44), datetime.datetime(2007, 6, 13, 17, 48)],
     [[4.6, 6.8], [6.8, 9], [9, 11.6], [11.6, 14.3], [14.3, 16.4], [16.4, 19.2], [19, 21.3], [21, 75]],
     15, 2e3, 1e3],
    ["t42", [datetime.datetime(2008, 3, 25, 14, 25), datetime.datetime(2008, 3, 25, 14, 30, 20)],
     [[3, 4.7], [4.6, 7.4], [7.4, 10], [10, 12.6], [12.6, 15.2], [15.2, 17.7], [17.7, 20.5], [21, 75]],
     15, 2e3, 2e3],
    ["t46", [datetime.datetime(2008, 11, 3, 17, 33), datetime.datetime(2008, 11, 3, 17, 36, 30)],
     [[3, 4.1], [4.2, 6.6], [6.6, 8.9], [8.9, 11.4], [11.4, 13.8], [13.8, 15.9], [15.9, 18.6]],
     15, 3e3, 1e3],
    #                        [T46IBSdata,T46ELSdata,
    #                         [datetime.datetime(2008,11,3,17,37,20),datetime.datetime(2008,11,3,17,38,40)],
    #                         [[3,4.3],[4.5,7],[6.9,9.4],[9.4,11.9],[11.9,14.3],[14.3,16.6],[16.6,20]],
    #                         9.5,
    #                         1e3,
    #                         2e3],
    ["t47", [datetime.datetime(2008, 11, 19, 15, 53), datetime.datetime(2008, 11, 19, 15, 54)],
     [[3, 4.6], [4.6, 7.4], [7.4, 9.9], [9.9, 12.5], [12.5, 15]],
     11, 4e3, 5e3]
]

usedflybys = ['t16', 't17','t20','t21','t25','t26','t27','t28','t29','t30','t32']
#usedflybys = ['t16']


def CAPS_beamsdata(times_elsdata_pairs, times_ibsdata_pairs):
    elspeakslist = []
    for elsflyby, times, energypairs, heavymass, heavyprominence, lightprominence, timewindow in times_elsdata_pairs:
        if elsflyby in usedflybys:
            elsdata = readsav("data/els/elsres_" + filedates[elsflyby] + ".dat")
            generate_mass_bins(elsdata, elsflyby, "els")
            LPdata = read_LP_V1(elsflyby)
            starttime_els = times[0]
            endtime_els = times[1]
            #print(elsdata, starttime_els, endtime_els)
            ramtimes = CAPS_ramtimes(elsdata, starttime_els, endtime_els)
            maxflux_anodes = []
            for i in ramtimes:
                maxflux_anodes.append(ELS_maxflux_anode(elsdata, i - datetime.timedelta(seconds=10),
                                                        i + datetime.timedelta(seconds=10)))
            #print(ramtimes,maxflux_anodes)
            heavypeaktimes, heavypeakenergies = [], []
            for ramtime, maxflux_anode in zip(ramtimes, maxflux_anodes):
                #print(ramtime, maxflux_anode)
                # print(ramtime, maxflux_anode, ramtime - datetime.timedelta(seconds=timewindow / 2),
                #       ramtime + datetime.timedelta(seconds=timewindow / 2), heavyprominence)
                heavypeaktime, heavypeakenergy = heavy_ion_finder(elsdata, heavymass, maxflux_anode,
                                                                  ramtime - datetime.timedelta(seconds=timewindow / 2),
                                                                  ramtime + datetime.timedelta(seconds=timewindow / 2))
                #print(heavypeaktime, heavypeakenergy)
                heavypeaktimes.append(heavypeaktime)
                heavypeakenergies.append(heavypeakenergy)
            # print(heavypeaktimes)
            for peaknumber, energypair in enumerate(energypairs):
                minenergy = energypair[0]
                maxenergy = energypair[1]
                # print(elsdata['flyby'],minenergy,maxenergy,starttime_els,endtime_els,lightprominence)
                peakenergies, peakelvs, peaktimes, peakanodes, peakenergybins, peakindices = ELS_beamsfinder(elsdata,
                                                                                                             minenergy,
                                                                                                             maxenergy,
                                                                                                             starttime_els,
                                                                                                             endtime_els,
                                                                                                             prominence=lightprominence)

                for (peakenergy, peaktime, peakelv, peakindex) in zip(peakenergies, peaktimes, peakelvs, peakindices):
                    peaks = [peakenergy, peaktime, peakelv]

                    heavypeak_timediffs = []
                    for i in heavypeaktimes:
                        #print(peaktime,i)
                        heavypeak_timediffs.append(abs(peaktime - i))

                    #print(heavypeak_timediffs)
                    nearestheavypeaktime = heavypeaktimes[np.argmin(heavypeak_timediffs)]
                    # print(peaktime,nearestheavypeaktime)

                    # print("--------Next----------")
                    negbulkazi = CAPS_ELS_FOVcentre_azi_elv(nearestheavypeaktime, elsdata)[0]
                    peakaziangle = CAPS_ELS_FOVcentre_azi_elv(peaktime, elsdata)[0]
                    # print(caps_ramdirection_azielv(peaks[1])[0],negbulkazi,peakaziangle)
                    peaks.append(peakaziangle)
                    peaks.append(caps_ramdirection_azielv(peaks[1])[0])
                    peaks.append(caps_ramdirection_time(elsdata, peaks[1]))
                    peaks.append(nearestheavypeaktime)
                    peaks.append(negbulkazi)
                    peaks.append(elsdata["flyby"])
                    peaks.append(inst_RPWS_LP(LPdata, peaks[1]))
                    peaks.append("els")
                    peaks.append(CAPS_actuationtimeslice(peaktime, elsdata)[2])
                    elspeakslist.append(list(peaks))
            del elsdata
    elsdf = pd.DataFrame(elspeakslist, columns=["Peak Energy", "Peak Time", "Peak Elevation", "Peak Azimuth",
                                                "Azimuthal Ram Angle", "Azimuthal Ram Time", "Heavy Anion Peak Time",
                                                "Heavy Anion Azimuth Angle",
                                                "Flyby", "Spacecraft Potential (LP)", "Instrument",
                                                "Actuation Direction"])

    ibspeakslist = []
    for flyby, times, energypairs, heavymass, heavyprominence, lightprominence in times_ibsdata_pairs:
        if flyby in usedflybys:
            # print(ibsdata["flyby"],times)
            elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
            generate_mass_bins(elsdata, flyby, "els")
            ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
            generate_aligned_ibsdata(ibsdata, elsdata, flyby)

            LPdata = read_LP_V1(flyby)
            starttime_ibs = times[0]
            endtime_ibs = times[1]
            for peaknumber, energypair in enumerate(energypairs):
                minenergy = energypair[0]
                maxenergy = energypair[1]
                peakenergies, peaktimes, peakenergybins = IBS_beamsfinder(ibsdata, minenergy, maxenergy,
                                                                          starttime_ibs, endtime_ibs,
                                                                          prominence=lightprominence)
                # print(peakenergies)
                for (peakenergy, peaktime) in zip(peakenergies, peaktimes):
                    peaks = [peakenergy, peaktime]
                    #print(peaks)
                    # print(heavymass,peaktime-datetime.timedelta(seconds=30),peaktime+datetime.timedelta(seconds=30),heavyprominence)
                    # heavypeaks, amplitudes = IBS_intensitypeaks(ibsdata, heavymass,
                    #                                             peaktime - datetime.timedelta(seconds=30),
                    #                                             peaktime + datetime.timedelta(seconds=30),
                    #                                             prominence=heavyprominence)
                    heavypeaktime_ibs, heavypeakenergy_ibs = heavy_ion_finder_ibs(ibsdata, heavymass,
                                                                                  peaktime - datetime.timedelta(
                                                                                      seconds=timewindow / 2),
                                                                                  peaktime + datetime.timedelta(
                                                                                      seconds=timewindow / 2))
                    # print(heavypeaks)
                    # if heavypeaks == []:
                    #     print(heavymass, peaktime - datetime.timedelta(seconds=30),
                    #           peaktime + datetime.timedelta(seconds=30), heavyprominence)
                    #     raise ValueError("Empty List")
                    # if len(heavypeaks) > 1:
                    #     print("List too big", peaktime, heavypeaks)
                    # print(heavypeaks,properties)


                    # Attempt to account for assysmetric IBS response
                    actuationdirection = CAPS_actuationtimeslice(peaktime, elsdata)[2]
                    # if actuationdirection == "negative":
                    #     heavypeaktime_ibs = heavypeaktime_ibs - datetime.timedelta(seconds=0.5)
                    heavypeakangle_ibs = CAPS_IBS_FOVcentre_azi_elv(heavypeaktime_ibs, elsdata)[0]

                    peakaziangle = CAPS_IBS_FOVcentre_azi_elv(peaktime, elsdata)[0]
                    # print(caps_ramdirection_azielv(peaks[1]c)[0],posbulkazi,peakaziangle)
                    peaks.append(peakaziangle)
                    peaks.append(caps_ramdirection_azielv(peaks[1])[0])
                    peaks.append(caps_ramdirection_time(elsdata, peaks[1]))
                    peaks.append(heavypeaktime_ibs)
                    peaks.append(heavypeakangle_ibs)
                    peaks.append(ibsdata["flyby"])
                    peaks.append(inst_RPWS_LP(LPdata, peaks[1]))
                    peaks.append("ibs")
                    peaks.append(actuationdirection)
                    ibspeakslist.append(list(peaks))
            del elsdata
            del ibsdata
    ibsdf = pd.DataFrame(ibspeakslist, columns=["Peak Energy", "Peak Time", "Peak Azimuth",
                                                "Azimuthal Ram Angle", "Azimuthal Ram Time", "Heavy Cation Peak Time",
                                                "Heavy Cation Azimuth Angle", "Flyby",
                                                "Spacecraft Potential (LP)", "Instrument", "Actuation Direction"])

    capsdf = pd.concat([elsdf, ibsdf], ignore_index=True)

    heavyanionangles = capsdf["Peak Azimuth"] - capsdf["Heavy Anion Azimuth Angle"]
    heavycationangles = capsdf["Peak Azimuth"] - capsdf["Heavy Cation Azimuth Angle"]
    # print(heavyanionangles,heavycationangles)
    heavyionangles = heavyanionangles.combine_first(heavycationangles)
    # print(heavyionangles)
    capsdf["Azimuthal Angle to Heavy Ion"] = heavyionangles

    capsdf["Azimuthal Angle to Ram"] = capsdf["Peak Azimuth"] - capsdf["Azimuthal Ram Angle"]

    capsdf.set_index("Peak Time", inplace=True)
    capsdf.sort_index(inplace=True)

    # print(capsdf)
    bulkdeflection_series = pd.Series(name="Deflection from Bulk", dtype=float)
    bulkseries = pd.Series(name="Bulk Azimuth", dtype=float)
    bulktimeseries = pd.Series(name="Bulk Time", dtype='datetime64[ns]')
    for flyby in capsdf.Flyby.unique():
        print(flyby)
        testgroup = capsdf[(capsdf["Flyby"] == flyby)]
        beamsgrouplist = []
        while not testgroup.empty:
            startdatetime = testgroup.index[0]
            enddatetime = testgroup.index[1]
            # print(startdatetime,enddatetime)
            # print(type(enddatetime), enddatetime-datetime.timedelta(seconds=1))
            endcounter = 0
            while testgroup.index[endcounter + 1] - enddatetime < datetime.timedelta(
                    seconds=flyby_maxbeamtimegap[flyby]):
                endcounter += 1
                enddatetime = testgroup.index[endcounter]
                if endcounter + 1 == len(testgroup.index):
                    break
            tempgroup = testgroup.truncate(before=startdatetime, after=enddatetime)
            # print("testgroup2", tempgroup)
            testgroup.drop(tempgroup.index, inplace=True)
            beamsgrouplist.append(tempgroup)

            groupeddf = tempgroup
            # print(elskey, groupeddf[["Instrument", "Peak Energy"]])
            elsgroupdf = groupeddf[groupeddf["Instrument"] == "els"]
            ibsgroupdf = groupeddf[groupeddf["Instrument"] == "ibs"]
            if len(elsgroupdf) < 2 or len(ibsgroupdf) < 2:
                print("empty")
                continue
            # print(ibsgroupdf)
            # print(elsgroupdf)
            # print(ibsgroupdf["Peak Energy"].idxmax())
            # print(elsgroupdf[elsgroupdf["Peak Energy"].idxmax(),"Heavy Anion Azimuthal Deflection"])
            # print(elsgroupdf.at[elsgroupdf["Peak Energy"].idxmax(),"Heavy Anion Azimuthal Deflection"])
            # print(ibsgroupdf.at[ibsgroupdf["Peak Energy"].idxmax(),"Heavy Cation Azimuthal Deflection"])

            elsbulkazimuth = elsgroupdf.at[
                elsgroupdf["Peak Energy"].idxmax(), "Heavy Anion Azimuth Angle"]
            ibsbulkazimuth = ibsgroupdf.at[
                ibsgroupdf["Peak Energy"].idxmax(), "Heavy Cation Azimuth Angle"]
            bulkazimuth = np.mean([elsbulkazimuth, ibsbulkazimuth])
            elsbulktime = elsgroupdf.at[
                elsgroupdf["Peak Energy"].idxmax(), "Heavy Anion Peak Time"]
            ibsbulktime = ibsgroupdf.at[
                ibsgroupdf["Peak Energy"].idxmax(), "Heavy Cation Peak Time"]
            bulktime = elsbulktime + (ibsbulktime - elsbulktime) / 2
            # print(elsbulkazimuth,ibsbulkazimuth,bulkazimuth)
            #             print(flyby,"Bulk Deflection",bulkazimuth,elstempdf.at[elstempdf["Peak Energy"].idxmax(),"Actuation Direction"])
            tempelsseries_deflection = pd.Series(elsgroupdf["Peak Azimuth"] - bulkazimuth,
                                                 name="Deflection from Bulk")
            tempibsseries_deflection = pd.Series(ibsgroupdf["Peak Azimuth"] - bulkazimuth,
                                                 name="Deflection from Bulk")
            bulkdeflection_series = bulkdeflection_series.add(tempelsseries_deflection, fill_value=0)
            bulkdeflection_series = bulkdeflection_series.add(tempibsseries_deflection, fill_value=0)

            bulkazimuthlist_els = [bulkazimuth] * len(tempelsseries_deflection)
            bulkazimuthlist_ibs = [bulkazimuth] * len(tempibsseries_deflection)
            tempelsseries = pd.Series(bulkazimuthlist_els, index=tempelsseries_deflection.index,
                                      name="Bulk Azimuth")
            tempibsseries = pd.Series(bulkazimuthlist_ibs, index=tempibsseries_deflection.index,
                                      name="Bulk Azimuth")
            bulkseries = bulkseries.add(tempelsseries, fill_value=0)
            bulkseries = bulkseries.add(tempibsseries, fill_value=0)

            bulktimelist_els = [bulktime] * len(tempelsseries_deflection)
            bulkatimelist_ibs = [bulktime] * len(tempibsseries_deflection)
            tempelsseries = pd.Series(bulktimelist_els, index=tempelsseries_deflection.index,
                                      name="Bulk Time")
            tempibsseries = pd.Series(bulkatimelist_ibs, index=tempibsseries_deflection.index,
                                      name="Bulk Time")
            # print(tempelsseries,tempibsseries)
            bulktimeseries = bulktimeseries.append(tempelsseries)
            bulktimeseries = bulktimeseries.append(tempibsseries)
            # print(bulktimeseries)
            # bulktimeseries = bulktimeseries.add(tempelsseries, fill_value=0)
            # bulktimeseries = bulktimeseries.add(tempibsseries, fill_value=0)

            # print(bulkseries)
    bulktimeseries.reindex(capsdf.index)
    capsdf = pd.concat([capsdf, bulkdeflection_series, bulkseries, bulktimeseries], axis=1)
    capsdf["Bulk Deflection from Ram Angle"] = capsdf["Bulk Azimuth"] - capsdf["Azimuthal Ram Angle"]
    return capsdf


CAPS_df = CAPS_beamsdata(elsdata_times_pairs, ibsdata_times_pairs)
CAPS_df.to_csv("Beams_database.csv")
