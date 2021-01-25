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


def els_alongtrack_velocity(elsdata, tempdatetime):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(elsdata, tempdatetime)

    dataslice = elsdata['data'][:, 4, slicenumber]

    # Need to add in LP data

    # mass_ebins = np.array(
    #     [(elscalib['earray'][i] + LPdata['ibstimes_scp_interp_offset'][slicecounter]) for
    #      i in range(64)])
    # mass_polyebins = np.array(
    #     [(elscalib['polyearray'][i] + LPdata['ibstimes_scp_interp_offset'][slicecounter]) for i in
    #      range(653)])

    peaks, properties = scipy.signal.find_peaks(dataslice, prominence=np.sqrt(dataslice))
    print("peaks", elscalib['earray'][peaks])
    print("conversion factor", (2 * e) / (AMU * ((cassini_speed) ** 2)))
    # peaks, properties = scipy.signal.find_peaks(counts)
    plt.plot(elscalib['earray'], elsdata['def'][:, 4, slicenumber])
    plt.scatter(elscalib['earray'][peaks], elsdata['def'][:, 4, slicenumber][peaks])
    plt.xscale("log")
    plt.yscale("log")

    # return alongtrackvelocity


def energy2mass(energyarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    massarray = (2 * (energyarray * e + (spacecraftpotential * charge * e) - 8 * k * iontemperature)) / (
            ((spacecraftvelocity + ionvelocity) ** 2) * AMU)
    return massarray


def mass2energy(massarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    energyarray = (0.5 * massarray * ((spacecraftvelocity + ionvelocity) ** 2) * AMU - (
            spacecraftpotential * charge * e) + 8 * k * iontemperature) / e
    return energyarray


def ibs_alongtrack_velocity(ibsdata, tempdatetime):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(ibsdata, tempdatetime)
    lowerenergyslice = CAPS_energyslice("ibs", 4, 4)[0]
    upperenergyslice = CAPS_energyslice("ibs", 20, 20)[0]
    spacecraftpotential = 0

    massarray = energy2mass(ibscalib['ibsearray'], cassini_speed, 0, spacecraftpotential)

    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, 1, slicenumber]
    print(len(dataslice))
    negated_dataslice = 1 - dataslice
    tempconversionfactor = (2 * e) / (AMU * ((cassini_speed) ** 2))
    datasliceaverage = 0.5*np.mean(dataslice)
    print("datasliceaverage",datasliceaverage)
    # TO DO tidy this

    masses = [28, 39, 52, 66, 78, 91]
    residualslist=[]
    for i in np.arange(1,4,1):
        peaks, properties = scipy.signal.find_peaks(dataslice, prominence=5 * np.sqrt(dataslice), distance=5, width=int(i),height=datasliceaverage)
        startwidth = 8
        stopwidth = 0
        variablewidth = np.arange(startwidth,stopwidth,-(startwidth-stopwidth)/len(dataslice))
        print(len(negated_dataslice),len(variablewidth))
        peaks_minima, properties_minima = scipy.signal.find_peaks(negated_dataslice, prominence=5 * np.sqrt(dataslice),width=variablewidth,distance=5)
        print("test minima widths",variablewidth[peaks_minima])
        masspeaks = massarray[lowerenergyslice:upperenergyslice][peaks]
        if len(masspeaks) < 6:
            continue
        massdifflist = masspeaks[:6] - masses
        SCoffset = mass2energy(np.array(massdifflist), cassini_speed, 0, spacecraftpotential)
        z = np.polyfit(x=np.array(masses), y=SCoffset, deg=1)
        residualslist.append(z[1])

    print("Residuals list", residualslist)
    correctpeakwidth = np.argmin(abs(np.array(residualslist)))+1
    print("Peak width",correctpeakwidth)

    peaks, properties = scipy.signal.find_peaks(dataslice, prominence=5 * np.sqrt(dataslice), distance=5, width=correctpeakwidth,height=datasliceaverage)
    masspeaks = massarray[lowerenergyslice:upperenergyslice][peaks]
    massdifflist = masspeaks[:6] - masses
    uncorrectedmasspeaks = masspeaks[:6]
    SCoffset = mass2energy(np.array(massdifflist), cassini_speed, 0, spacecraftpotential)
    z = np.polyfit(x=np.array(masses), y=SCoffset, deg=1)
    print(ibsdata['flyby'], " Residuals %.3f" % z[1])

    ionwindspeed = (z[0] * (e / AMU)) / (cassini_speed)
    print(ibsdata['flyby'], " Ion wind velocity %.1f" % ionwindspeed, "m/s")
    p = np.poly1d(z)

    ax.scatter(masses, SCoffset)
    ax.plot(masses, p(masses), linestyle='dashed', label=tempdatetime)
    ax.set_xlabel("Mass (amu/q)")
    ax.set_ylabel("Energy offset (E/q)")
    ax.hlines(0,0,100,color='k')
    ax.legend()
    # print("conversion factor", tempconversionfactor)
    # peaks, properties = scipy.signal.find_peaks(counts)

    corrected_massarray = energy2mass(ibscalib['ibsearray'], cassini_speed, ionwindspeed, spacecraftpotential)

    # Correct for SCP
    wind_correctedmasspeaks = corrected_massarray[lowerenergyslice:upperenergyslice][peaks][:6]
    scp_massdifflist = corrected_massarray[lowerenergyslice:upperenergyslice][peaks][:6] - masses
    average_scp_massdiff = np.mean(scp_massdifflist)
    scp_test = -average_scp_massdiff / tempconversionfactor
    #print("average scp mass diff", average_scp_massdiff)
    #print("Test SCP", -average_scp_massdiff / tempconversionfactor)

    scp_corrected_massarray = energy2mass(ibscalib['ibsearray'], cassini_speed, ionwindspeed,
                                          -average_scp_massdiff / tempconversionfactor)
    correctedmasspeaks = scp_corrected_massarray[lowerenergyslice:][peaks][:6]

    # Plotting
    ax3.plot(massarray[lowerenergyslice:upperenergyslice], dataslice,
             label="Pre-correction " + str(tempdatetime), color='C0')
    ax3.scatter(massarray[lowerenergyslice:upperenergyslice][peaks], dataslice[peaks], marker='x', color='C0')
    ax3.scatter(massarray[lowerenergyslice:upperenergyslice][peaks_minima], dataslice[peaks_minima], marker='o', color='C0')

    ax3.plot(scp_corrected_massarray[lowerenergyslice:upperenergyslice], dataslice, label="Post-correction " + str(tempdatetime),
             color='C1')
    ax3.scatter(scp_corrected_massarray[lowerenergyslice:upperenergyslice][peaks], dataslice[peaks], color='C1', marker='x')
    ax3.hlines(datasliceaverage,1,100,color='k')
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(10, 170)
    ax3.set_ylim(3e3, 1e6)
    ax3.set_xlabel("Mass (amu/q)")
    ax3.set_ylabel("Counts")
    ax3.legend()

    for counter, (i, j) in enumerate(zip([uncorrectedmasspeaks, wind_correctedmasspeaks, correctedmasspeaks],
                    ["no correction", "wind corrected", "wind + scp corrected"])):
        z = np.polyfit(x=np.array(masses), y=i, deg=1)
        p = np.poly1d(z)
        ax4.scatter(masses,i,label=j + ", Grad = %.2f, " % p.c[0] + "Residuals = %.2f" % z[1],color='C'+str(counter))
        ax4.plot(range(100), p(range(100)), color='C' + str(counter))
    ax4.plot(range(100), range(100), color='k')
    ax4.text(50,99,"IBS derived s/c potential = %.1f" % scp_test)
    ax4.text(50,95, "IBS derived alongtrack wind = %.1f" % ionwindspeed)
    ax4.set_xlabel("Expected Masses (amu/q)")
    ax4.set_ylabel("Found Masses (amu/q)")
    ax4.legend()

    return ionwindspeed, z[1], scp_test


fig, ax = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Bulk Time'] = pd.to_datetime(windsdf['Bulk Time'])
#
# usedflybys = ['t16']
# for flyby in usedflybys:
#     els_ionwindspeeds, ibs_ionwindspeeds, ibs_residuals, ibs_scps = [], [], [], []
#     tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
#     elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
#     generate_mass_bins(elsdata, flyby, "els")
#     ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
#     generate_aligned_ibsdata(ibsdata, elsdata, flyby)
#     for i in tempdf['Bulk Time']:
#         ibs_ionwindspeed, ibs_residual, ibs_scp  = ibs_alongtrack_velocity(ibsdata,i)
#         print(i,ibs_ionwindspeed)
#         ibs_ionwindspeeds.append(ibs_ionwindspeed)
#         ibs_residuals.append(ibs_residual)
#         ibs_scps.append(ibs_scp)
#
# testoutputdf = pd.DataFrame()
# testoutputdf['Bulk Time'] = tempdf['Bulk Time']
# testoutputdf['IBS Alongtrack velocity'] = ibs_ionwindspeeds
# testoutputdf['IBS residuals'] = ibs_residuals
# testoutputdf['IBS spacecraft potentials'] = ibs_scps
# testoutputdf.to_csv("testalongtrackvelocity.csv")

flyby = 't16'
elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
generate_mass_bins(elsdata, flyby, "els")
ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
generate_aligned_ibsdata(ibsdata, elsdata, flyby)
tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]

slicenumber = 4
print(tempdf['Bulk Time'].iloc[slicenumber])
ibs_ionwindspeed = ibs_alongtrack_velocity(ibsdata, tempdf['Bulk Time'].iloc[slicenumber])



plt.show()
