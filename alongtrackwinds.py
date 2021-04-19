from __future__ import unicode_literals

import csv
import time
import math

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
from itertools import chain

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

filedates = {"t16": "22-jul-2006", "t17": "07-sep-2006", "t18": "23-sep-2006", "t19": "09-oct-2006",
             "t20": "25-oct-2006", "t21": "12-dec-2006", "t23": "13-jan-2007", "t25": "22-feb-2007",
             "t26": "10-mar-2007",
             "t27": "26-mar-2007",
             "t28": "10-apr-2007", "t29": "26-apr-2007", "t39": "20-dec-2007",
             "t30": "12-may-2007", "t32": "13-jun-2007",
             "t40": "05-jan-2008", "t41": "22-feb-2008", "t42": "25-mar-2008", "t43": "12-may-2008",
             "t46": "03-nov-2008", "t47": "19-nov-2008"}

IBS_fluxfitting_dict = {"mass28_": {"sigma": [0.3, 0.4, 0.6], "amplitude": []},
                        "mass40_": {"sigma": [0.3, 0.6, 0.7], "amplitude": []},
                        "mass53_": {"sigma": [0.3, 0.5, 0.6], "amplitude": []},
                        "mass66_": {"sigma": [0.4, 0.6, 0.7], "amplitude": []}, \
                        "mass78_": {"sigma": [0.5, 0.7, 0.8], "amplitude": []}, \
                        "mass91_": {"sigma": [0.6, 0.8, 0.9], "amplitude": []}}

ELS_fluxfitting_dict = {"mass26_": {"sigma": [0.2, 0.4, 0.7], "amplitude": [5]},
                        "mass50_": {"sigma": [0.5, 0.6, 0.9], "amplitude": [4]},
                        "mass74_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass98_": {"sigma": [0.6, 0.8, 1.6], "amplitude": [3]},
                        "mass117_": {"sigma": [0.8, 0.9, 1.7], "amplitude": [3]}}

IBS_energybound_dict = {"t16": [4, 17], "t17": [3.5, 16.25], "t19": [3.5, 16.25],
                        "t20": [3.5, 16.5], "t21": [4.25, 16.75], "t25": [4.25, 18.25], "t26": [4.35, 18.25],
                        "t27": [4.5, 18.25],
                        "t28": [4.5, 18.25], "t29": [4.5, 18.25],
                        "t30": [4.5, 18.25], "t32": [4.5, 18.25],
                        "t42": [4.5, 19.5], "t46": [4, 17], "t47": [4.5, 18.25]}

ELS_energybound_dict = {"t16": [1, 30], "t17": [1.5, 30], "t19": [1.5, 30],
                        "t20": [1, 30], "t21": [1, 30], "t25": [1, 30], "t26": [1.5, 30], "t27": [1, 30],
                        "t28": [1, 30], "t29": [1, 30],
                        "t30": [1, 30], "t32": [1, 30],
                        "t42": [2.3, 30], "t46": [2.3, 28], "t47": [1, 35]}

IBS_init_ionvelocity_dict = {"t16": 0, "t17": 0, "t19": 0,
                             "t20": 0, "t21": 0, "t25": 0, "t26": 0,
                             "t27": 0,
                             "t28": 0, "t29": 0,
                             "t30": 0, "t32": 0,
                             "t42": 0, "t46": -350, "t47": 0}


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

    if charge == 1:
        pars.add('scp', value=LPvalue, min=LPvalue - 2, max=0)
    elif charge == -1:
        pars.add('scp', value=LPvalue, min=LPvalue - 2, max=0)
    pars.add('temp_eV', value=8 * k * temperature)  # , min=130, max=170)
    pars.add('spacecraftvelocity', value=cassini_speed)
    pars.add('ionvelocity', value=windspeed, min=-500, max=500)
    # pars['scp'].vary = False
    pars['spacecraftvelocity'].vary = False
    pars['temp_eV'].vary = False

    pars.add('e', value=e)
    pars.add('AMU', value=AMU)
    pars.add('charge', value=charge)
    pars['e'].vary = False
    pars['AMU'].vary = False
    pars['charge'].vary = False

    for masscounter, mass in enumerate(masses):
        tempprefix = "mass" + str(mass) + '_'
        gaussmodels.append(GaussianModel(prefix=tempprefix))
        pars.add(tempprefix, value=mass, vary=False)
        if charge == 1:
            sigmavals = IBS_fluxfitting_dict[tempprefix]['sigma']
            # ampval = IBS_fluxfitting_dict[tempprefix]['amplitude'][0]
        elif charge == -1:
            # sigmaval = ELS_fluxfitting_dict[tempprefix]['sigma']
            sigmavals = ELS_fluxfitting_dict[tempprefix]['sigma']
            ampval = ELS_fluxfitting_dict[tempprefix]['amplitude'][0]

        # effectivescpexpr = 'scp + ((' + tempprefix + '*AMU*spacecraftvelocity)/e)*windspeed' #Windspeed defined positive if going in same direction as Cassini
        # pars.add(tempprefix + "effectivescp", expr=effectivescpexpr)
        pars.update(gaussmodels[-1].make_params())

        temppeakflux = peakflux(mass, pars['spacecraftvelocity'], 0, LPvalue, temperature, charge=charge)
        # print("mass", mass, "Init Flux", temppeakflux)

        peakfluxexpr = '(0.5*(' + tempprefix + '*AMU)*((spacecraftvelocity+ionvelocity)**2) - scp*e*charge + temp_eV)/e'
        pars[tempprefix + 'center'].set(expr=peakfluxexpr)
        # min=temppeakflux - 2, max=temppeakflux + 2)

        pars[tempprefix + 'sigma'].set(value=sigmavals[1], min=sigmavals[0], max=sigmavals[2])
        pars[tempprefix + 'amplitude'].set(value=np.mean(yvalues) * (1 + sigmavals[1]), min=min(yvalues))

    for counter, model in enumerate(gaussmodels):
        if counter == 0:
            mod = model
        else:
            mod = mod + model

    init = mod.eval(pars, x=xvalues)
    out = mod.fit(yvalues, pars, x=xvalues)

    # if poor fit essentially
    # if out.params['windspeed'].stderr is None or out.params['scp'].stderr is None:
    #     maxscpincrease = 0.1
    #     while out.params['windspeed'].stderr is None or out.params['scp'].stderr is None:
    #         print("Trying better fit")
    #         maxscpincrease += 0.1
    #         pars["scp"].set(value=LPvalue + 0.1, min=LPvalue - 0.1, max=LPvalue + 0.15 + maxscpincrease)
    #         out = mod.fit(yvalues, pars, x=xvalues)

    # print(out.fit_report(min_correl=0.7))

    # Calculating CI's
    # print(out.ci_report(p_names=["scp","windspeed"],sigmas=[1],verbose=True,with_offset=False,ndigits=2))

    return out


def titan_linearfit_temperature(altitude):
    if altitude > 1150:
        temperature = 110 + 0.26 * (altitude - 1200)
    else:
        temperature = 133 - 0.12 * (altitude - 1100)
    return temperature


# [28, 29, 39, 41, 52, 54, 65, 66, 76, 79, 91]
def IBS_fluxfitting(ibsdata, tempdatetime, titanaltitude, lpdata, ibs_masses=[28, 40, 53, 66, 78, 91], numofflybys=1):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(ibsdata, tempdatetime)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    print("interp lpvalue", lpvalue)

    lowerenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][0],
                                        IBS_energybound_dict[ibsdata['flyby']][0])[0]
    upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1],
                                        IBS_energybound_dict[ibsdata['flyby']][1])[0]

    windspeed = IBS_init_ionvelocity_dict[ibsdata['flyby']]
    temperature = titan_linearfit_temperature(titanaltitude)

    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, 1, slicenumber]

    lowerenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue)[0]
    upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue)[0]

    #print("old x", ibscalib['ibsearray'][lowerenergyslice:upperenergyslice])
    #print("new x", ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.035 )
    #x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice]
    #x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.035
    x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.073


    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, 1, slicenumber]

    if numofflybys == 1:
        stepplotfig_ibs, ax = plt.subplots()
        stepplotfig_ibs.suptitle("Histogram of " + ibsdata['flyby'].upper() + " IBS data", fontsize=32)
        ax.step(x, dataslice, where='mid')
        ax.errorbar(x, dataslice, yerr=[np.sqrt(i) for i in dataslice], color='k', fmt='none')
        ax.set_yscale("log")
        ax.set_xlim(1, 25)
        ax.set_ylim(0.9 * min(dataslice), 1.1 * max(dataslice))
        ax.set_ylabel("Counts [/s]", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
        ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
        ax.minorticks_on()
        ax.set_title(ibsdata['times_utc_strings'][slicenumber])
        ax.set_xlabel("Energy [eV/q]", fontsize=14)

    out = total_fluxgaussian(x, dataslice, ibs_masses, cassini_speed, windspeed, lpvalue, temperature,
                             charge=1,
                             FWHM=ELS_FWHM)
    if out.params['ionvelocity'].stderr is None:
        out.params['ionvelocity'].stderr = np.nan
    if out.params['scp'].stderr is None:
        out.params['scp'].stderr = np.nan
    GOF = np.mean((abs(out.best_fit - dataslice) / dataslice) * 100)

    if numofflybys == 1:
        ax.plot(x, out.best_fit, 'r-', label='best fit')
        ax.text(0.8, 0.01,
                "Ion wind = %2.0f ± %2.0f m/s" % (out.params['ionvelocity'], out.params['ionvelocity'].stderr),
                transform=ax.transAxes)
        ax.text(0.8, .05,
                "IBS-derived S/C Potential = %2.2f ± %2.2f V" % (out.params['scp'], out.params['scp'].stderr),
                transform=ax.transAxes)
        ax.text(0.8, .09, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=ax.transAxes)
        # ax.text(0.8, .20, "Temp = %2.2f" % temperature, transform=ax.transAxes)
        # ax.text(0.8, .32, "Chi-square = %.2E" % out.chisqr, transform=ax.transAxes)
        ax.text(0.8, .13, "My GOF = %2.0f %%" % GOF, transform=ax.transAxes)
    # for mass in ibs_masses:
    #     stepplotax.plot(x, comps["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")
    # stepplotax.legend(loc='best')

    return out, GOF, lpvalue


def ELS_maxflux_anode(elsdata, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)
    anodesums = np.sum(np.sum(dataslice, axis=2), axis=0)
    maxflux_anode = np.argmax(anodesums)
    return maxflux_anode


def ELS_fluxfitting(elsdata, tempdatetime, titanaltitude, lpdata, els_masses=[26, 50, 74, 117], numofflybys=1):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    els_slicenumber = CAPS_slicenumber(elsdata, tempdatetime)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])

    windspeed = 0
    temperature = titan_linearfit_temperature(titanaltitude)

    els_lowerenergyslice = CAPS_energyslice("els", ELS_energybound_dict[elsdata['flyby']][0],
                                            ELS_energybound_dict[elsdata['flyby']][0])[0]
    els_upperenergyslice = CAPS_energyslice("els", ELS_energybound_dict[elsdata['flyby']][1],
                                            ELS_energybound_dict[elsdata['flyby']][1])[0]
    # x = elscalib['earray'][els_lowerenergyslice:els_upperenergyslice]
    anode = ELS_maxflux_anode(elsdata, tempdatetime - datetime.timedelta(seconds=10),
                              tempdatetime + datetime.timedelta(seconds=10))
    print("anode", anode)

    tempdataslice = list(
        np.float32(ELS_backgroundremoval(elsdata, els_slicenumber, els_slicenumber + 1, datatype="data")[
                   els_lowerenergyslice:els_upperenergyslice, anode, 0]))
    tempx = list(elscalib['earray'][els_lowerenergyslice:els_upperenergyslice])

    while tempdataslice[0] < 10:
        tempdataslice.pop(0)
        tempx.pop(0)

    tempdataslice = np.array(tempdataslice)
    tempdataslice[tempdataslice <= 0] = 1
    dataslice = np.array(tempdataslice)

    if numofflybys == 1:
        stepplotfig_els, ax = plt.subplots()
        stepplotfig_els.suptitle("Histogram of " + elsdata['flyby'].upper() + " ELS data", fontsize=32)
        ax.step(tempx, dataslice, where='mid')
        ax.errorbar(tempx, dataslice, yerr=[np.sqrt(i) for i in dataslice], color='k', fmt='none')
        ax.set_yscale("log")
        # ax.set_xlim(1, 25)
        ax.set_ylim(0.9 * min(dataslice), 1.1 * max(dataslice))
        ax.set_ylabel("Counts [/s]", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
        ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
        ax.minorticks_on()
        ax.set_title(elsdata['times_utc_strings'][els_slicenumber])
        ax.set_xlabel("Energy [eV/q]", fontsize=14)

    out = total_fluxgaussian(np.array(tempx), dataslice, els_masses, cassini_speed, windspeed, lpvalue, temperature,
                             charge=-1,
                             FWHM=ELS_FWHM)
    if out.params['ionvelocity'].stderr is None:
        out.params['ionvelocity'].stderr = np.nan
    if out.params['scp'].stderr is None:
        out.params['scp'].stderr = np.nan
    GOF = np.mean((abs(out.best_fit - dataslice) / dataslice) * 100)

    # for out in outputs:
    #     print(out.params['ionvelocity'], out.params['scp'])
    if numofflybys == 1:
        ax.plot(tempx, out.best_fit, 'r-', label='best fit')
        ax.text(0.8, 0.01,
                "Ion wind = %2.0f ± %2.0f m/s" % (out.params['ionvelocity'], out.params['ionvelocity'].stderr),
                transform=ax.transAxes)
        ax.text(0.8, .05,
                "ELS-derived S/C Potential = %2.2f ± %2.2f V" % (out.params['scp'], out.params['scp'].stderr),
                transform=ax.transAxes)
        ax.text(0.8, .09, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=ax.transAxes)
        # ax.text(0.8, .20, "Temp = %2.2f" % temperature, transform=ax.transAxes)
        ax.text(0.8, .13, "Chi-square = %.2E" % out.chisqr, transform=ax.transAxes)
        ax.text(0.8, .17, "My GOF = %2.0f %%" % GOF, transform=ax.transAxes)
    # comps = out.eval_components(x=x)
    # for mass in els_masses:
    #     stepplotax_els.plot(x, comps["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")

    return out, GOF, lpvalue


windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])
windsdf['Negative Peak Time'] = pd.to_datetime(windsdf['Negative Peak Time'])


def multiple_alongtrackwinds_flybys(usedflybys):
    outputdf = pd.DataFrame()
    for flyby in usedflybys:
        tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
        elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")
        ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
        generate_aligned_ibsdata(ibsdata, elsdata, flyby)
        ibs_outputs, ibs_datetimes, ibs_GOFvalues, lpvalues = [], [], [], []
        els_outputs, els_datetimes, els_GOFvalues = [], [], []
        lptimes = list(tempdf['Positive Peak Time'])
        lpdata = read_LP_V1(flyby)
        for (i, j, k) in zip(tempdf['Positive Peak Time'], tempdf['Negative Peak Time'], tempdf['Altitude']):
            ibs_output, ibs_GOFvalue, lpvalue = IBS_fluxfitting(ibsdata, i, k, lpdata=lpdata,
                                                                              numofflybys=len(
                                                                                  usedflybys))
            els_output, els_GOFvalue, temp = ELS_fluxfitting(elsdata, j, k, lpdata=lpdata,
                                                                           numofflybys=len(
                                                                               usedflybys))
            ibs_outputs.append(ibs_output)
            ibs_GOFvalues.append(ibs_GOFvalue)
            lpvalues.append(lpvalue)
            els_outputs.append(els_output)
            els_GOFvalues.append(els_GOFvalue)

        # testoutputdf = pd.DataFrame()
        # testoutputdf['Bulk Time'] = tempdf['Bulk Time']
        # testoutputdf['IBS Alongtrack velocity'] = [i.params['windspeed'] for i in ibs_fits]
        # # testoutputdf['IBS residuals'] = ibs_residuals
        # testoutputdf['IBS spacecraft potentials'] = [i.params['scp'] for i in ibs_fits]
        # testoutputdf.to_csv("testalongtrackvelocity.csv")

        if len(usedflybys) == 1:
            fig5, (windaxes, potaxes, GOFaxes) = plt.subplots(3, sharex='all')
            fig5.suptitle(usedflybys[0])
            # windaxes.errorbar(tidied_ibs_datetimes_flat, [i.params['ionvelocity'].value for i in tidied_ibs_outputs_flat],
            #                   yerr=[i.params['ionvelocity'].stderr for i in tidied_ibs_outputs_flat],
            #                   label="IBS - Ion Wind Speeds", marker='.', ls='none', color='C0')
            # windaxes.errorbar(tidied_els_datetimes_flat, [i.params['ionvelocity'].value for i in tidied_els_outputs_flat],
            #                   yerr=[i.params['ionvelocity'].stderr for i in tidied_els_outputs_flat],
            #                  label="ELS - Ion Wind Speeds",
            #                  marker='x', ls='none',  color='C1')
            windaxes.plot(tempdf['Positive Peak Time'], [i.params['ionvelocity'].value for i in ibs_outputs],
                          label="IBS - Ion Wind Speeds", marker='.', ls='none', color='C0')
            windaxes.plot(tempdf['Negative Peak Time'], [i.params['ionvelocity'].value for i in els_outputs],
                          label="ELS - Ion Wind Speeds",
                          marker='x', ls='none', color='C1')

            windaxes.legend()
            windaxes.set_ylabel("Derived Ion Velocity")
            windaxes.hlines([-500, 500], min(tempdf['Positive Peak Time']), max(tempdf['Positive Peak Time']), color='k')
            # windaxes.set_ylim(-525, 525)

            # potaxes.errorbar(tidied_ibs_datetimes_flat, [i.params['scp'].value for i in tidied_ibs_outputs_flat],
            #                  yerr=[i.params['scp'].stderr for i in tidied_ibs_outputs_flat],
            #                  label="IBS - Derived S/C Potential", marker='.', ls='none', color='C0')
            # potaxes.errorbar(tidied_els_datetimes_flat, [i.params['scp'].value for i in tidied_els_outputs_flat],
            #                 yerr=[i.params['scp'].stderr for i in tidied_els_outputs_flat],
            #                 label="ELS - Derived S/C Potential",
            #                 marker='x', ls='none',  color='C1')
            potaxes.plot(tempdf['Positive Peak Time'], [i.params['scp'].value for i in ibs_outputs],
                         label="IBS - Derived S/C Potential", marker='.', ls='none', color='C0')
            potaxes.plot(tempdf['Negative Peak Time'], [i.params['scp'].value for i in els_outputs],
                         label="ELS - Derived S/C Potential",
                         marker='x', ls='none', color='C1')
            potaxes.plot(lptimes, lpvalues, label="LP derived S/C potential", color='C8')

            potaxes.legend()
            potaxes.set_ylabel("Derived S/C Potential")
            # potaxes.plot(lptimes, np.array(lpvalues) - 0.5, color='k')
            # potaxes.plot(lptimes, np.array(lpvalues) + 0.5, color='k')
            potaxes.hlines([np.mean(lpvalues) - 2, 0], min(tempdf['Positive Peak Time']),
                           max(tempdf['Positive Peak Time']), color='k')
            potaxes.set_ylim(np.mean(lpvalues) - 2.1, 0.1)

            GOFaxes.scatter(tempdf['Positive Peak Time'], ibs_GOFvalues, label="IBS - GOF", color='C0')
            GOFaxes.set_ylabel("Goodness of Fit - IBS")
            GOFaxes.set_xlabel("Time")

            GOFaxes_els = GOFaxes.twinx()
            GOFaxes_els.scatter(tempdf['Negative Peak Time'], els_GOFvalues, label="ELS - GOF", color='C1',
                                marker='x')
            GOFaxes_els.set_ylabel("Goodness of Fit - ELS")

        # print(tempdf['Positive Peak Time'])
        # elsstartslice = CAPS_slicenumber(elsdata,tempdf['Positive Peak Time'].iloc[0])
        # elsendslice = CAPS_slicenumber(elsdata,tempdf['Positive Peak Time'].iloc[-1])
        # actaxes.plot(elsdata['times_utc'][elsstartslice:elsendslice],elsdata['actuator'][elsstartslice:elsendslice])
        # actaxes.set_ylabel("Actuator position")
        else:
            tempoutputdf = pd.DataFrame()
            tempoutputdf['Flyby'] = tempdf['Flyby']
            tempoutputdf['Positive Peak Time'] = tempdf['Positive Peak Time']
            tempoutputdf['Negative Peak Time'] = tempdf['Negative Peak Time']
            tempoutputdf['IBS alongtrack velocity'] = [i.params['ionvelocity'].value for i in ibs_outputs]
            tempoutputdf['IBS spacecraft potentials'] = [i.params['scp'].value for i in ibs_outputs]
            tempoutputdf['ELS alongtrack velocity'] = [i.params['ionvelocity'].value for i in els_outputs]
            tempoutputdf['ELS spacecraft potentials'] = [i.params['scp'].value for i in els_outputs]
            tempoutputdf['LP Potentials'] = lpvalues
            print(tempoutputdf)
            outputdf = pd.concat([outputdf, tempoutputdf])
    if len(usedflybys) != 1:
        outputdf.to_csv("alongtrackvelocity.csv")


def single_slice_test(flyby, slicenumber):
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)
    tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]

    print("Altitude", tempdf['Altitude'].iloc[slicenumber])
    print(tempdf['Positive Peak Time'].iloc[slicenumber])
    lpdata = read_LP_V1(flyby)
    ibs_out, ibs_GOF, lpvalue = IBS_fluxfitting(ibsdata, tempdf['Positive Peak Time'].iloc[slicenumber],
                                                tempdf['Altitude'].iloc[slicenumber],
                                                lpdata=lpdata)
    els_out, els_GOF, lpvalue = ELS_fluxfitting(elsdata, tempdf['Negative Peak Time'].iloc[slicenumber],
                                                tempdf['Altitude'].iloc[slicenumber],
                                                lpdata=lpdata)


#single_slice_test("t16", slicenumber=4)
#multiple_alongtrackwinds_flybys(["t16"])
multiple_alongtrackwinds_flybys(
    ['t16', 't17', 't20', 't21', 't25', 't26', 't27', 't28', 't29', 't30', 't32', 't42', 't46'])

plt.show()
