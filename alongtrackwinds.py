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
from astropy.modeling import models, fitting

from lmfit import CompositeModel, Model
from lmfit.models import GaussianModel
from lmfit import Parameters
from lmfit import Minimizer


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
    lowerenergyslice = CAPS_energyslice("ibs", 4.5, 4.5)[0]
    upperenergyslice = CAPS_energyslice("ibs", 21, 21)[0]
    spacecraftpotential = 0

    massarray = energy2mass(ibscalib['ibsearray'], cassini_speed, 0, spacecraftpotential)
    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, 1, slicenumber]
    negated_dataslice = 1 - dataslice
    ax3.plot(massarray[lowerenergyslice:upperenergyslice], dataslice,
             label="Pre-correction " + str(tempdatetime), color='C0')
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(10, 170)
    ax3.set_ylim(3e3, 1e6)
    ax3.set_xlabel("Mass (amu/q)")
    ax3.set_ylabel("Counts")
    ax3.legend()



    tempconversionfactor = (2 * e) / (AMU * ((cassini_speed) ** 2))
    # TO DO tidy this

    masses = [28, 39, 52, 66, 78, 91]
    residualslist=[]
    for i in np.arange(1,4,1):
        peaks, properties = scipy.signal.find_peaks(dataslice, prominence=3 * np.sqrt(dataslice), distance=5, width=int(i))
        masspeaks = massarray[lowerenergyslice:upperenergyslice][peaks]
        if len(masspeaks) < 6:
            print("skipped",masspeaks)
            continue
        massdifflist = masspeaks[:6] - masses
        SCoffset = mass2energy(np.array(massdifflist), cassini_speed, 0, spacecraftpotential)
        z = np.polyfit(x=np.array(masses), y=SCoffset, deg=1)
        residualslist.append(z[1])

    #Finding the minma
    startwidth = 8
    stopwidth = -5
    variablewidth = np.arange(startwidth, stopwidth, -(startwidth - stopwidth) / len(dataslice))
    peaks_minima, properties_minima = scipy.signal.find_peaks(negated_dataslice, prominence=10 * np.sqrt(dataslice),
                                                              width=variablewidth, distance=5)
    print("test minima widths", variablewidth[peaks_minima])

    print("Residuals list", residualslist)
    correctpeakwidth = np.argmin(abs(np.array(residualslist)))+1
    print("Peak width",correctpeakwidth)
    peaks, properties = scipy.signal.find_peaks(dataslice, prominence=5 * np.sqrt(dataslice), distance=5, width=correctpeakwidth)
    peaks_minima = [0] + list(peaks_minima)
    print(peaks,peaks_minima)


    # Plotting
    ax3.scatter(massarray[lowerenergyslice:upperenergyslice][peaks], dataslice[peaks], marker='x', color='C0')
    ax3.scatter(massarray[lowerenergyslice:upperenergyslice][peaks_minima], dataslice[peaks_minima], marker='o', color='C0')
    #plt.show()

    #Attempting fitting of the peaks

    fittedmasspeaks =[]
    for counter,peak in enumerate(peaks[:6]):
        #print(peaks_minima[counter],peak,peaks_minima[counter+1])
        peakslice = dataslice[peaks_minima[counter]:peaks_minima[counter+1]]
        massbins = massarray[lowerenergyslice:upperenergyslice][peaks_minima[counter]:peaks_minima[counter+1]]
        peak_init = models.Gaussian1D(amplitude=max(peakslice), mean=massarray[lowerenergyslice:upperenergyslice][peak],
                                     # stddev=1,
                                     # bounds={
                                     #     "amplitude": (max(recentpeakslice), max(recentpeakslice) * 1.2),
                                     #     "mean": (actuatorangle_adjusted[recentpeaks[0]] - 1, actuatorangle_adjusted[recentpeaks[0]] + 1),
                                     #     "stddev": (1, 3)}
                                     )
        peakfit = fitting.LevMarLSQFitter()(peak_init, massbins, peakslice)
        fittedmasspeaks.append(peakfit.mean.value)
        ax3.plot(massbins, peakfit(massbins),color='r',linestyle='--')

    print(fittedmasspeaks)
    masspeaks = massarray[lowerenergyslice:upperenergyslice][peaks]
    #massdifflist = masspeaks[:6] - masses
    massdifflist = np.array(fittedmasspeaks[:6]) - masses
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


    # ax3.plot(scp_corrected_massarray[lowerenergyslice:upperenergyslice], dataslice, label="Post-correction " + str(tempdatetime),
    #          color='C1')
    # ax3.scatter(scp_corrected_massarray[lowerenergyslice:upperenergyslice][peaks], dataslice[peaks], color='C1', marker='x')
    #ax3.hlines(datasliceaverage,1,100,color='k')


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
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])


#TO DO add LP potentials
usedflybys = ['t26']
for flyby in usedflybys:
    els_ionwindspeeds, ibs_ionwindspeeds, ibs_residuals, ibs_scps = [], [], [], []
    tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)
    for i in tempdf['Positive Peak Time']:
        print(i)
        ibs_ionwindspeed, ibs_residual, ibs_scp = ibs_alongtrack_velocity(ibsdata,i)
        ibs_ionwindspeeds.append(ibs_ionwindspeed)
        ibs_residuals.append(ibs_residual)
        ibs_scps.append(ibs_scp)

testoutputdf = pd.DataFrame()
testoutputdf['Bulk Time'] = tempdf['Bulk Time']
testoutputdf['IBS Alongtrack velocity'] = ibs_ionwindspeeds
testoutputdf['IBS residuals'] = ibs_residuals
testoutputdf['IBS spacecraft potentials'] = ibs_scps
testoutputdf.to_csv("testalongtrackvelocity.csv")
#
fig5, ax5 = plt.subplots()
ax5.plot(tempdf['Positive Peak Time'],ibs_ionwindspeeds,color='C0',label="Ion Wind Speeds")
ax5_1 = ax5.twinx()
ax5_1.plot(tempdf['Positive Peak Time'],ibs_scps,color='C1',label="S/C potential, IBS derived")
fig5.legend()

#Single slice test

# flyby = 't25'
# elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
# generate_mass_bins(elsdata, flyby, "els")
# ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
# generate_aligned_ibsdata(ibsdata, elsdata, flyby)
# tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
#
# slicenumber = 4
# print(tempdf['Positive Peak Time'].iloc[slicenumber])
# ibs_ionwindspeed = ibs_alongtrack_velocity(ibsdata, tempdf['Positive Peak Time'].iloc[slicenumber])

plt.show()

Af = 0.33e-4
MCPefficiency = 0.05

def total_fluxgaussian(xvalues, yvalues, masses, tempcassini_speed, windspeed, LPvalue, temperature):
    gaussmodels = []
    pars = Parameters()
    pars.add('windspeed', value=windspeed, min=-400, max=250)
    pars.add('scp', value=LPvalue, min=-2, max=0)
    pars.add('temp', value=temperature, min=70, max=170)
    pars.add('spacecraftvelocity', value=tempcassini_speed)
    pars['spacecraftvelocity'].vary = False

    pars.add('e', value=e)
    pars.add('AMU', value=AMU)
    pars.add('k', value=k)
    pars['e'].vary = False
    pars['AMU'].vary = False

    # fitter = Minimizer()
    # fitter.asteval.symtable['peakflux'] = peakflux

    for masscounter, mass in enumerate(masses):
        tempprefix = "mass" + str(mass) + '_'
        gaussmodels.append(GaussianModel(prefix=tempprefix))
        # if masscounter == 0:
        # pars = gaussmodels[-1].guess(yvalues, x=xvalues)
        pars.add(tempprefix, value=mass)
        pars[tempprefix].vary = False

        pars.update(gaussmodels[-1].make_params())
        # pars.add('mass',value=mass, min=mass-0.1, max=mass+0.1)

        peakenergy = (0.5 * (pars[tempprefix] * pars['AMU']) * ((pars['spacecraftvelocity'] + pars['windspeed']) ** 2) +
                      pars['scp'] * pars['e'] + 8 * pars['k'] * pars['temp']) / pars['e']
        temppeakflux = peakflux(mass, tempcassini_speed, pars['windspeed'], pars['scp'], pars['temp'], charge=-1)
        # print(peakenergy,temppeakflux)

        # pars.add(tempprefix + 'testcenter',expr='0.5*((mass*AMU)/e)*((spacecraftvelocity + flowspeed)**2) + scp*e + 8*k*temp')
        # peakenergy = 0.5*mass_kg*((spacecraftvelocity + flowspeed + charge*deflectionvelocity)**2) - charge*spacecraftpotential*e + 8*k*temperature
        # peakenergy_eV = peakenergy/e

        #         thermalvelocity = np.sqrt((2*k*temperature)/(mass*AMU))
        #         print(thermalvelocity)
        #         tempwidth = ((mass*AMU)*(thermalvelocity**2))/e
        #         print(tempwidth)
        tempwidth = temppeakflux * 0.167 / 2.355

        # COnvolve with response function to increase width?
        # Need to set the parameters to be temp,

        # pars[tempprefix + 'center'].set(value=peakflux(mass,tempcassini_speed,pars['flowspeed'],pars['scp'],pars['temp'],charge=-1), min=temppeakflux-1, max=temppeakflux+1)
        tempstring = '(0.5*(' + tempprefix + '*AMU)*((spacecraftvelocity + windspeed)**2) + scp*e + 8*k*temp)/e'

        pars[tempprefix + 'center'].set(
            value=peakflux(mass, tempcassini_speed, pars['windspeed'], pars['scp'], pars['temp'], charge=-1),
            expr=tempstring, min=temppeakflux - 1, max=temppeakflux + 1)
        # print(pars[tempprefix + 'center'])
        pars[tempprefix + 'sigma'].set(value=tempwidth, min=tempwidth / 2, max=tempwidth * 2)
        pars[tempprefix + 'amplitude'].set(value=5e10, min=1e9, max=5e11)

    for counter, model in enumerate(gaussmodels):
        if counter == 0:
            mod = model
        else:
            mod = mod + model

    init = mod.eval(pars, x=xvalues)
    out = mod.fit(yvalues, pars, x=xvalues)

    print(out.fit_report(min_correl=0.7))

    return out


def ELS_fluxfitting(elsdata, time, seconds, anode, lpvalue=-1.3):
    for counter, i in enumerate(elsdata['times_utc_strings']):
        if i >= time:
            slicenumber = counter
            break

    temputc = str(titan_flybydates[elsdata['flyby']][0]) + '-' + str(titan_flybydates[elsdata['flyby']][1]) + '-' + str(
        titan_flybydates[elsdata['flyby']][2]) + 'T' + elsdata['times_utc_strings'][slicenumber]
    tempphase = cassini_phase(temputc)
    tempcassini_speed = np.sqrt((tempphase[3]) ** 2 + (tempphase[4]) ** 2 + (tempphase[5]) ** 2) * 1e3

    flowspeed = -150
    temperature = 150

    DEF = elsdata['def'][:, anode - 1, slicenumber]

    stepplotfig, stepplotax = plt.subplots()
    stepplotax.step(elscalib['polyearray'][:-1], DEF, where='post', label=elsdata['flyby'], color='k')
    stepplotax.errorbar(elscalib['earray'], DEF, yerr=[np.sqrt(i) for i in DEF], color='k', fmt='none')
    stepplotax.set_xlim(0, 300)
    stepplotax.set_ylim(bottom=1e6)
    stepplotax.set_yscale("log")
    stepplotax.set_ylabel("DEF [$m^{-2} s^{1} str^{-1} eV^{-1}$]", fontsize=20)
    stepplotax.set_xlabel("Energy (Pre-correction) [eV/q]", fontsize=20)
    stepplotax.tick_params(axis='both', which='major', labelsize=15)
    stepplotax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    stepplotax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
    stepplotax.minorticks_on()
    stepplotax.set_title(
        "Histogram of " + elsdata['flyby'].upper() + " els data from " + elsdata['times_utc_strings'][slicenumber],
        fontsize=32)
    # stepplotax.plot(elscalib['earray'],smoothedcounts_full,color='k')

    masses = [26, 50, 74, 117]
    x = elscalib['earray']
    out = total_fluxgaussian(x, DEF, masses, tempcassini_speed, flowspeed, lpvalue, temperature)

    stepplotax.plot(x, out.best_fit, 'r-', label='best fit')
    stepplotax.text(0, 0, "Ion wind = %2.2f" % out.params['flowspeed'], transform=stepplotax.transAxes)
    stepplotax.text(0, .05, "SC Potential = %2.2f" % out.params['scp'], transform=stepplotax.transAxes)
    stepplotax.text(0, .10, "Temp = %2.2f" % out.params['temp'], transform=stepplotax.transAxes)

    comps = out.eval_components(x=x)
    for mass in masses:
        stepplotax.plot(x, comps["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")

    stepplotax.legend(loc='best')
