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

Af = 0.33e-4
MCPefficiency = 0.05
ELS_FWHM = 0.167
IBS_FWHM = 0.014

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

IBS_fluxfitting_dict = {"mass28_": {"sigma": 0.4, "amplitude": []},
                        "mass40_": {"sigma": 0.5, "amplitude": []},
                        "mass53_": {"sigma": 0.5, "amplitude": []},
                        "mass66_": {"sigma": 0.6, "amplitude": []},
                        "mass78_": {"sigma": 0.7, "amplitude": []},
                        "mass91_": {"sigma": 0.8, "amplitude": []}}

IBS_energybound_dict = {"t16": [4, 17], "t17": [3.5, 16.25],
                        "t20": [3.5, 16.5], "t21": [4.25, 16.75], "t25": [4.25, 18.25], "t26": [4.35, 18.25],
                        "t27": [4.5, 18.25],
                        "t28": [4.5, 18.25], "t29": [4.5, 18.25],
                        "t30": [4.5, 18.25], "t32": [4.5, 18.25],
                        "t42": [4.5, 19.5], "t46": [3.75, 17.5], "t47": [4.5, 18.25]}
IBS_LPoffset_dict = {"t16": 0.25, "t17": 0.25,
                "t20": 0.25, "t21": 0.25, "t25": 0.25, "t26": 0.25,
                "t27": 0.25,
                "t28": 0.25, "t29": 0.25,
                "t30": 0.25, "t32": 0.25,
                "t42": 0.25, "t46": 0.5, "t47": 0.25}


def energy2mass(energyarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    massarray = (2 * (energyarray * e + (spacecraftpotential * charge * e) - 8 * k * iontemperature)) / (
            ((spacecraftvelocity + ionvelocity) ** 2) * AMU)
    return massarray


def mass2energy(massarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    energyarray = (0.5 * massarray * ((spacecraftvelocity + ionvelocity) ** 2) * AMU - (
            spacecraftpotential * charge * e) + 8 * k * iontemperature) / e
    return energyarray


def total_fluxgaussian(xvalues, yvalues, masses, cassini_speed, windspeed, LPvalue, temperature, charge, FWHM):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    # pars.add('scp', value=LPvalue, min=LPvalue-0.25, max=LPvalue + 0.5)
    pars.add('temp', value=temperature)  # , min=130, max=170)
    pars.add('spacecraftvelocity', value=cassini_speed)
    # pars.add('windspeed', value=0, min=-400, max=400)
    pars['spacecraftvelocity'].vary = False
    pars['temp'].vary = False

    pars.add('e', value=e)
    pars.add('AMU', value=AMU)
    pars.add('k', value=k)
    pars.add('charge', value=charge)
    pars['e'].vary = False
    pars['AMU'].vary = False
    pars['k'].vary = False
    pars['charge'].vary = False

    peakfluxvalues_nowind = []
    for masscounter, mass in enumerate(masses):
        tempprefix = "mass" + str(mass) + '_'
        gaussmodels.append(GaussianModel(prefix=tempprefix))
        pars.add(tempprefix, value=mass, vary=False)
        # pars.add(tempprefix+'windspeed', value=0, min=-400, max=400)
        # effectivescpexpr = 'scp + ((' + tempprefix + '*AMU*spacecraftvelocity)/e)*' + tempprefix + 'windspeed' #Windspeed defined positive if going in same direction as Cassini
        pars.add(tempprefix + "effectivescp", value=LPvalue, min=LPvalue - 2, max=LPvalue + 2)
        pars.update(gaussmodels[-1].make_params())

        temppeakflux = peakflux(mass, pars['spacecraftvelocity'], 0, LPvalue, temperature, charge=charge)
        peakfluxvalues_nowind.append(temppeakflux)
        # print("mass", mass, "Init Flux", temppeakflux)

        peakfluxexpr = '(0.5*(' + tempprefix + '*AMU)*((spacecraftvelocity)**2) - ' + tempprefix + 'effectivescp*e*charge + 8*k*temp)/e'
        pars[tempprefix + 'center'].set(
            value=peakflux(mass, cassini_speed, 0, LPvalue, temperature, charge=charge), expr=peakfluxexpr)
        # min=temppeakflux - 2, max=temppeakflux + 2)

        sigmaval = IBS_fluxfitting_dict[tempprefix]['sigma']
        pars[tempprefix + 'sigma'].set(value=sigmaval, min=0.5 * sigmaval, max=1.5 * sigmaval)
        pars[tempprefix + 'amplitude'].set(value=np.mean(yvalues) * (1 + (0.1 * masscounter)), min=min(yvalues))

    for counter, model in enumerate(gaussmodels):
        if counter == 0:
            mod = model
        else:
            mod = mod + model

    init = mod.eval(pars, x=xvalues)
    out = mod.fit(yvalues, pars, x=xvalues)

    # SCP offset plot
    effectivescplist, effectivescplist_errors = [], []
    for masscounter, mass in enumerate(masses):
        tempprefix = "mass" + str(mass) + '_'
        effectivescplist.append(out.params[tempprefix + "effectivescp"].value)
        effectivescplist_errors.append(out.params[tempprefix + "effectivescp"].stderr)
    z, cov = np.polyfit(x=np.array(masses), y=np.array(effectivescplist), deg=1, cov=True)
    ionwindspeed = (z[0] * (e / AMU)) / (cassini_speed)
    ionwindspeed_err = (np.sqrt(np.diag(cov)[0]) * (e / AMU)) / (cassini_speed)
    print(ibsdata['flyby'], " Ion wind velocity = %2.2f ± %2.2f m/s" % (ionwindspeed, ionwindspeed_err))

    # fig, ax = plt.subplots()
    # p = np.poly1d(z)
    # ax.errorbar(masses, np.array(effectivescplist),yerr=effectivescplist_errors,fmt='.')
    # ax.plot(masses, p(masses))

    # SCP calculation
    scpvalues = []
    for masscounter, mass in enumerate(masses):
        scpvalues.append(effectivescplist[masscounter] - (mass * AMU * cassini_speed * ionwindspeed) / e)
    scp_mean = np.mean(scpvalues)
    scp_err = np.std(scpvalues)

    print(ibsdata['flyby'], " IBS-derived SCP = %2.2f ± %2.2f V" % (scp_mean, scp_err))

    print(out.fit_report(min_correl=0.7))

    return out, ionwindspeed, ionwindspeed_err, scp_mean, scp_err


def titan_linearfit_temperature(altitude):
    if altitude > 1150:
        temperature = 110 + 0.26 * (altitude - 1200)
    else:
        temperature = 133 - 0.12 * (altitude - 1100)
    return temperature


# [28, 29, 39, 41, 52, 54, 65, 66, 76, 79, 91]
def IBS_fluxfitting(ibsdata, tempdatetime, titanaltitude, ibs_masses=[28, 40, 53, 66, 78, 91]):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(ibsdata, tempdatetime)
    lpdata = read_LP_V1(ibsdata['flyby'])
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    #print("interp lpvalue", lpvalue)
    lpvaluewithoffset = lpvalue+IBS_LPoffset_dict[ibsdata['flyby']]

    lowerenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue)[0]
    upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue)[0]

    windspeed = 0
    temperature = titan_linearfit_temperature(titanaltitude)

    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, 1, slicenumber]

    x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice]
    out, ionwindspeed, ionwindspeed_err, scp_mean, scp_err = total_fluxgaussian(x, dataslice, ibs_masses, cassini_speed,
                                                                                windspeed, lpvaluewithoffset, temperature,
                                                                                charge=1,
                                                                                FWHM=IBS_FWHM)

    # print(out.fit_report(min_correl=0.7))jiop
    comps = out.eval_components(x=x)

    stepplotfig, stepplotax = plt.subplots()
    stepplotax.step(ibscalib['ibspolyearray'][lowerenergyslice:upperenergyslice], dataslice, where='post',
                    label=elsdata['flyby'], color='k')
    stepplotax.errorbar(x, dataslice, yerr=[np.sqrt(i) for i in dataslice], color='k', fmt='none')
    stepplotax.set_xlim(3, 20)
    stepplotax.set_ylim(min(dataslice), max(dataslice))
    stepplotax.set_yscale("log")
    stepplotax.set_ylabel("DEF [$m^{-2} s^{1} str^{-1} eV^{-1}$]", fontsize=20)
    stepplotax.set_xlabel("Energy (Pre-correction) [eV/q]", fontsize=20)
    stepplotax.tick_params(axis='both', which='major', labelsize=15)
    stepplotax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    stepplotax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
    stepplotax.minorticks_on()
    stepplotax.set_title(
        "Histogram of " + ibsdata['flyby'].upper() + " IBS data from " + ibsdata['times_utc_strings'][slicenumber],
        fontsize=32)
    stepplotax.plot(x, out.init_fit, 'b-', label='init fit')
    stepplotax.plot(x, out.best_fit, 'r-', label='best fit')
    stepplotax.text(0.8, 0.02, "Ion wind = %2.2f ± %2.2f m/s" % (ionwindspeed, ionwindspeed_err),
                    transform=stepplotax.transAxes)
    stepplotax.text(0.8, .05,
                    "IBS-derived SC Potential = %2.2f ± %2.2f V" % (scp_mean, scp_err),
                    transform=stepplotax.transAxes)
    stepplotax.text(0.8, .08, "LP-derived SC Potential = %2.2f" % lpvalue, transform=stepplotax.transAxes)
    stepplotax.text(0.8, .11, "Temp = %2.2f" % out.params['temp'], transform=stepplotax.transAxes)
    stepplotax.text(0.8, .14, "Reduced $\chi^{2}$ = %.2E" % out.redchi, transform=stepplotax.transAxes)
    for mass in ibs_masses:
        stepplotax.plot(x, comps["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")
    stepplotax.legend(loc='best')

    return out, lpvalue, ionwindspeed, ionwindspeed_err, scp_mean, scp_err


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
    stepplotax.set_xlim(0, 20)
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


windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])

def multiple_flybys(usedflybys)
    for flyby in usedflybys:
        els_fits, ibs_fits, lpvalues, ibs_ionwindspeeds, ibs_ionwindspeeds_err, ibs_scps, ibs_scps_err = [], [], [], [], [], [], []
        tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
        elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")
        ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
        generate_aligned_ibsdata(ibsdata, elsdata, flyby)
        for (i, j) in zip(tempdf['Positive Peak Time'], tempdf['Altitude']):
            ibs_fit, lpvalue, ibs_ionwindspeed, ibs_ionwindspeed_err, ibs_scp_mean, ibs_scp_err = IBS_fluxfitting(ibsdata,
                                                                                                                  i, j)
            ibs_fits.append(ibs_fit)
            lpvalues.append(lpvalue)
            ibs_ionwindspeeds.append(ibs_ionwindspeed)
            ibs_ionwindspeeds_err.append(ibs_ionwindspeed_err)
            ibs_scps.append(ibs_scp_mean)
            ibs_scps_err.append(ibs_scp_err)

    outputdf = pd.DataFrame()
    outputdf['Bulk Time'] = tempdf['Bulk Time']
    outputdf['IBS Alongtrack velocity'] = ibs_ionwindspeeds
    # outputdf['IBS residuals'] = ibs_residuals
    outputdf['IBS spacecraft potentials'] = ibs_scps
    #outputdf.to_csv("testalongtrackvelocity.csv")

    #
    # fig5, axes = plt.subplots(2)
    # axes[0].errorbar(tempdf['Positive Peak Time'], ibs_ionwindspeeds, yerr=ibs_ionwindspeeds_err, color='C0',
    #              label="Ion Wind Speeds", linestyle='--')
    # axes[0].set_ylabel("Ion Wind Speed (m/s)")
    # # for counter, x in enumerate(ibs_fits):
    # #     axes[0].text(tempdf['Positive Peak Time'].iloc[counter], ibs_ionwindspeeds[counter], "Chi-Sqr =  %.1E" % x.chisqr)
    # axes[0].legend()
    #
    # axes[1].errorbar(tempdf['Positive Peak Time'], ibs_scps, yerr=ibs_scps_err, color='C1', label="S/C potential, IBS derived")
    # axes[1].plot(tempdf['Positive Peak Time'], lpvalues, color='C2', label="S/C potential, LP derived")
    # axes[1].set_ylabel("S/C Potential (V)")
    # axes[1].set_xlabel("Time")
    # axes[1].legend()

    return outputdf

def single_slice_test(flyby="t16")
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)
    tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]

    slicenumber = 2
    print(tempdf['Positive Peak Time'].iloc[slicenumber])
    ibs_ionwindspeed = IBS_fluxfitting(ibsdata, tempdf['Positive Peak Time'].iloc[slicenumber],
                                       tempdf['Altitude'].iloc[slicenumber])



usedflybys = ['t16', 't17', 't20', 't21', 't25', 't26', 't27', 't28', 't29', 't30', 't32', 't42', 't46']
outputdf = multiple_flybys(usedflybys)

return outputdf