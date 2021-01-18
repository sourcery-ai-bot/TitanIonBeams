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

filedates = {"t16": "22-jul-2006", "t17": "07-sep-2006",
             "t20": "25-oct-2006", "t21": "12-dec-2006", "t25": "22-feb-2007", "t26": "10-mar-2007",
             "t27": "26-mar-2007",
             "t28": "10-apr-2007", "t29": "26-apr-2007",
             "t30": "12-may-2007", "t32": "13-jun-2007",
             "t42": "25-mar-2008", "t46": "03-nov-2008", "t47": "19-nov-2008"}

def els_alongtrack_velocity(elsdata,tempdatetime):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(elsdata, tempdatetime)

    dataslice = elsdata['data'][:,4,slicenumber]

    # Need to add in LP data

    # mass_ebins = np.array(
    #     [(elscalib['earray'][i] + LPdata['ibstimes_scp_interp_offset'][slicecounter]) for
    #      i in range(64)])
    # mass_polyebins = np.array(
    #     [(elscalib['polyearray'][i] + LPdata['ibstimes_scp_interp_offset'][slicecounter]) for i in
    #      range(653)])

    peaks, properties = scipy.signal.find_peaks(dataslice, prominence=np.sqrt(dataslice))
    print("peaks",elscalib['earray'][peaks])
    print("conversion factor",(2*e)/(AMU*((cassini_speed)**2)))
    # peaks, properties = scipy.signal.find_peaks(counts)
    plt.plot(elscalib['earray'], elsdata['def'][:, 4, slicenumber])
    plt.scatter(elscalib['earray'][peaks], elsdata['def'][:, 4, slicenumber][peaks])
    plt.xscale("log")
    plt.yscale("log")

    #return alongtrackvelocity



def ibs_alongtrack_velocity(ibsdata,tempdatetime):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(ibsdata, tempdatetime)
    lowerenergyslice = CAPS_energyslice("ibs",4,4)[0]

    dataslice = ibsdata['ibsdata'][lowerenergyslice:, 1, slicenumber]
    negated_dataslice = 1 - dataslice
    tempconversionfactor = (2 * e) / (AMU * ((cassini_speed) ** 2))
    peaks, properties = scipy.signal.find_peaks(dataslice, prominence=3*np.sqrt(dataslice),width=2)
    peaks_minima, properties_minima = scipy.signal.find_peaks(negated_dataslice, prominence=3*np.sqrt(dataslice))
    minima = scipy.signal.argrelmin(dataslice,order=2)[0]
    masspeaks = ibscalib['ibsearray'][lowerenergyslice:][peaks]*tempconversionfactor
    print("mass peaks", masspeaks)
    massdifflist = []
    masses = [28,39,52,78,91]
    for i in masses:
        diff = masspeaks-i
        mindiff_index = np.argmin(abs(diff))
        massdifflist.append(diff[mindiff_index])
        print(i,"min diff",diff[mindiff_index])
    SCoffset = massdifflist/tempconversionfactor
    #print("SC offset", SCoffset)
    #print(ibsdata['ibsdata'].shape)

    z = np.polyfit(x=np.array(masses), y=SCoffset, deg=1)
    ionwindspeed = (z[0] * (e / AMU)) / (cassini_speed)
    #print(ibsdata['flyby']," Ion wind velocity %.1f" % ionwindspeed, "m/s")
    print(ibsdata['flyby']," Residuals %.3f" % z[1])
    p = np.poly1d(z)



    ax.scatter(masses, SCoffset)
    ax.plot(masses, p(masses), linestyle='dashed',label=tempdatetime)
    #print("conversion factor", tempconversionfactor)
    # peaks, properties = scipy.signal.find_peaks(counts)


    ax2.plot(ibscalib['ibsearray'][lowerenergyslice:]*tempconversionfactor, dataslice ,label=tempdatetime)

    ax2.scatter(ibscalib['ibsearray'][lowerenergyslice:][peaks]*tempconversionfactor, dataslice[peaks])
    ax2.scatter(ibscalib['ibsearray'][lowerenergyslice:][peaks_minima] * tempconversionfactor, dataslice[peaks_minima])
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    postcorrection_conversionfactor = (2 * e) / (AMU * ((cassini_speed+ionwindspeed) ** 2))
    postcorrection_masspeaks = ibscalib['ibsearray'][lowerenergyslice:][peaks]*postcorrection_conversionfactor
    ax3.plot(ibscalib['ibsearray'][lowerenergyslice:] * postcorrection_conversionfactor, dataslice, label=tempdatetime)
    ax3.scatter(ibscalib['ibsearray'][lowerenergyslice:][peaks] * postcorrection_conversionfactor, dataslice[peaks])
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    return ionwindspeed

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

# windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
# windsdf['Bulk Time'] = pd.to_datetime(windsdf['Bulk Time'])

# usedflybys = ['t17']
# for flyby in usedflybys:
#     els_ionwindspeeds, ibs_ionwindspeeds = [], []
#     tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
#     elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
#     generate_mass_bins(elsdata, flyby, "els")
#     ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
#     generate_aligned_ibsdata(ibsdata, elsdata, flyby)
#     for i in tempdf['Bulk Time']:
#         ibs_ionwindspeed = ibs_alongtrack_velocity(ibsdata,i)
#         print(i,ibs_ionwindspeed)
#         ibs_ionwindspeeds.append(ibs_ionwindspeed)

flyby = 't17'
elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
generate_mass_bins(elsdata, flyby, "els")
ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
generate_aligned_ibsdata(ibsdata, elsdata, flyby)
windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Bulk Time'] = pd.to_datetime(windsdf['Bulk Time'])
tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
print(tempdf['Bulk Time'])
ibs_ionwindspeed = ibs_alongtrack_velocity(ibsdata,tempdf['Bulk Time'].iloc[2])


ax.legend()
ax2.legend()
plt.show()