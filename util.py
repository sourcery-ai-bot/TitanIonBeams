from __future__ import unicode_literals

import csv
import datetime
import time

from cassinipy.caps.mssl import *
from cassinipy.caps.util import *
from cassinipy.misc import *
from cassinipy.spice import *

titan_flybyvelocities = {'t16': 6e3 ,'t17': 6e3, 't18': 6e3, 't19': 6e3,
                         't20': 6e3, 't21': 5.9e3, 't23': 6e3,  't25': 6.2e3, 't26': 6.2e3, 't27': 6.2e3, 't28': 6.2e3, 't29': 6.2e3,
                         't30': 6.2e3, 't32': 6.2e3,
                         't40': 6.3e3, 't42': 6.3e3, 't46': 6.3e3, 't47': 6.3e3,
                         't83': 5.9e3}
titan_CAheight = {'t16': 950, 't17': 1000, 't18': 960, 't19': 980,
                  't20': 1029, 't21': 1000, 't23': 1000, 't25': 1000, 't26': 981, 't27': 1010, 't28': 991, 't29': 981,
                  't30': 959, 't32': 965,
                  't46': 1100, 't47': 1023,
                  't83': 990}
titan_flybydates = {'t16': [2006, 7, 22], 't17': [2006, 9, 7], 't18': [2006, 9, 23], 't19': [2006, 10, 9],
                    't20': [2006, 10, 25], 't21': [2006, 12, 12], 't23': [2007, 1, 13], 't25': [2007, 2, 22], 't26': [2007, 3, 10],
                    't27': [2007, 3, 26], 't28': [2007, 4, 10], 't29': [2007, 4, 26],
                    't30': [2007, 5, 12], 't32': [2007, 6, 13],
                    't40': [2008, 1, 5], 't42': [2008, 3, 25], 't46': [2008, 11, 3], 't47': [2008, 11, 19],
                    't83': [2012, 5, 22]}

def read_LP_V1(flyby):
    '''
    Reads RPWS-LP data in V1.TAB format
    '''
    LPdata = {'datetime': [], 'RADIAL_DISTANCE': [], 'ELECTRON_NUMBER_DENSITY': [], 'ELECTRON_TEMPERATURE': [],
              'SPACECRAFT_POTENTIAL': []}

    if flyby[0] == 't':
        moon = 'titan'
    if flyby[0] == 'e':
        moon = 'enceladus'

    print('data/lp/RPWS_LP_T_' + str(titan_flybydates[flyby][0]) + "*" + flyby.upper() + "_V1.TAB")
    with open(glob.glob(
            'data/lp/RPWS_LP_T_' + str(titan_flybydates[flyby][0]) + "*" + flyby.upper() + "_V1.TAB")[0],
              'r') as csvfile:
        tempreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in tempreader:
            if abs(float(row[4])) < 1e32:
                LPdata['datetime'].append(datetime.datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%fZ"))
                LPdata['RADIAL_DISTANCE'].append(float(row[1]))
                LPdata['ELECTRON_NUMBER_DENSITY'].append(float(row[2]))
                LPdata['ELECTRON_TEMPERATURE'].append(float(row[3]))
                LPdata['SPACECRAFT_POTENTIAL'].append(float(row[4]))

    return LPdata


def secofday2utc(datelist, secofdaylist):
    datetimeutclist = []

    for i in secofdaylist:
        sec = datetime.timedelta(seconds=int(i))
        d = datetime.datetime(datelist[0], datelist[1], datelist[2]) + sec
        datetimeutclist.append(d)

    return datetimeutclist


def read_SC_Values(data, i, moon):
    #    print(data['flyby'] + "_IBS_SpacecraftPotentialValues.csv")
    with open(moon + "/" + data['flyby'] + "/" + data['instrument'] + "/" + data['flyby'] + "_" + data[
        'instrument'] + "_SpacecraftPotentialValues_new.csv", 'r') as csvfile:
        tempreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in tempreader:
            if data['instrument'] == 'els':
                if data['secofday'][i] == int(row[0]):
                    spacecraftpotential = float(row[1])
                    spacecraftpotential_LP = float(row[2])
                    data['spacecraftpotentials'].append(spacecraftpotential)
                    data['spacecraftpotentials_LP'].append(spacecraftpotential_LP)
                    return spacecraftpotential, spacecraftpotential_LP
            elif data['instrument'] == 'ibs':
                if data['ibssecofday'][i] == int(row[0]):
                    spacecraftpotential = float(row[1])
                    spacecraftpotential_LP = float(row[2])
                    data['spacecraftpotentials'].append(spacecraftpotential)
                    data['spacecraftpotentials_LP'].append(spacecraftpotential_LP)
                    return spacecraftpotential, spacecraftpotential_LP
            elif data['instrument'] == 'sng':
                if data[str('sngsecofday')][i] == int(row[0]):
                    spacecraftpotential = float(row[1])
                    spacecraftpotential_LP = float(row[2])
                    data['spacecraftpotentials'].append(spacecraftpotential)
                    data['spacecraftpotentials_LP'].append(spacecraftpotential_LP)
                    return spacecraftpotential, spacecraftpotential_LP

    # print("Aligned Spacecraft Potential", data['spacecraftpotentials'])
    return None, None


def inst_RPWS_LP(LPdata, tempdatetime):
    '''
    Returns Langmuir probe derived spacecraft potential at a single datetime value
    '''
    counter = 0
    while LPdata['datetime'][counter] < tempdatetime:
        counter += 1

    lptimestamps = [toTimestamp(d) for d in LPdata['datetime'][counter - 1:counter + 1]]
    spacecraftpotential = np.interp(toTimestamp(tempdatetime), lptimestamps,
                                    LPdata['SPACECRAFT_POTENTIAL'][counter - 1:counter + 1])
    return spacecraftpotential


def read_ramangle(flyby):
    ramdata = readsav("data/titan/ramangles/" + flyby.upper() + "-ram.dat")
    # print(ramdata.keys())
    ramdata['utc'] = secofday2utc(titan_flybydates[flyby.lower()], ramdata['panel1_pa_time'])

    return ramdata


def generate_mass_bins(data, flyby, instrument):
    t0 = time.time()
    data['moon'] = 'titan'
    data['flyby'] = flyby
    data['instrument'] = instrument

    if data['instrument'] == 'els':
        timearray = 'secofday'
    elif data['instrument'] == 'sng':
        timearray = 'sngsecofday'
    elif data['instrument'] == 'ibs':
        timearray = 'ibssecofday'

    v = titan_flybyvelocities[flyby]
    data['flybyvelocity'] = titan_flybyvelocities[flyby]

    # Setting up time stamps
    data['times_utc'] = secofday2utc(titan_flybydates[flyby], data[timearray])
    data['times_utc_strings'] = []
    for i in data['times_utc']:
        data['times_utc_strings'].append(datetime.datetime.strftime(i, "%H:%M:%S"))

    data['conversionfactor'] = (2 * e) / (AMU * (v ** 2))

    t1 = time.time()
    print(data['flyby'], data['instrument'], "Bin Generation Time", t1 - t0)

def generate_aligned_ibsdata(ibsdata, elsdata, flyby):
    t0 = time.time()
    ibsdata['flyby'] = flyby
    ibsdata['instrument'] = "ibs"

    ibsdata['instrument'] == 'ibs'
    elstimearray = 'secofday'
    ibstimearray = 'ibssecofday'


    acycle_times = [[], []]
    acycle_times_ibs = [[], []]
    for counter, i in enumerate(elsdata['acycle'][0, :]):
        if counter > 0:
            if i != elsdata['acycle'][0, counter - 1]:
                acycle_times[0].append(i)
                acycle_times[1].append(elsdata['secofday'][counter])
        else:
            acycle_times[0].append(i)
            acycle_times[1].append(elsdata['secofday'][counter])
    for counter, i in enumerate(ibsdata['ibsacycle']):
        if counter > 0:
            if i != ibsdata['ibsacycle'][counter - 1]:
                acycle_times_ibs[0].append(i)
                acycle_times_ibs[1].append(ibsdata['ibssecofday'][counter])
        else:
            acycle_times_ibs[0].append(i)
            acycle_times_ibs[1].append(ibsdata['ibssecofday'][counter])
    differences = []
    for bigcounter, (i, j) in enumerate(zip(acycle_times_ibs[0], acycle_times_ibs[1])):
        if i not in acycle_times[0]:
            differences.append(differences[-1])
        else:
            counter = acycle_times[0].index(i,int(bigcounter/2))
            differences.append(j - acycle_times[1][counter])
    differences_interp = np.interp(ibsdata['ibssecofday'], acycle_times_ibs[1], differences)
    adjusted_times = ibsdata['ibssecofday']-differences_interp

    v = titan_flybyvelocities[flyby]
    ibsdata['flybyvelocity'] = titan_flybyvelocities[flyby]

    ibsdata['times_utc'] = secofday2utc(titan_flybydates[flyby], adjusted_times)
    ibsdata['times_utc_strings'] = []
    for i in ibsdata['times_utc']:
        ibsdata['times_utc_strings'].append(datetime.datetime.strftime(i, "%H:%M:%S"))
    ibsdata['numofanodes'] = 3

    ibsdata['conversionfactor'] = (2 * e) / (AMU * (v ** 2))

    t1 = time.time()
    print(ibsdata['flyby'], ibsdata['instrument'], "Bin Generation Time", t1 - t0)

def conversion_factor(flyby, ionvelocity, utc=''):
    if utc == '':
        v = titan_flybyvelocities[flyby]
    else:
        tempphase = cassini_phase(utc)
        v = np.sqrt((tempphase[3]) ** 2 + (tempphase[4]) ** 2 + (tempphase[5]) ** 2) * 1e3
    return (2 * e) / (AMU * ((v + ionvelocity) ** 2))