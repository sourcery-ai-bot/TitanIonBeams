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

from symfit import parameters, variables, Fit, Model, Parameter, exp
from symfit.distributions import Gaussian

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

IBS_fluxfitting_dict = {"mass28_": {"sigma": 0.4, "amplitude": [1]},
                        "mass41_": {"sigma": 0.5, "amplitude": [1]},
                        "mass53_": {"sigma": 0.5, "amplitude": [1]},
                        "mass66_": {"sigma": 0.6, "amplitude": [1]},
                        "mass78_": {"sigma": 0.7, "amplitude": [1]},
                        "mass91_": {"sigma": 0.8, "amplitude": [1]}}

ELS_fluxfitting_dict = {"mass25_": {"sigma": 0.5, "amplitude": [5]},
                        "mass50_neg_": {"sigma": 0.8, "amplitude": [4]},
                        "mass74_neg_": {"sigma": 0.9, "amplitude": [3]},
                        "mass117_neg_": {"sigma": 1.2, "amplitude": [3]},
                        "mass122_": {"sigma": 1.5, "amplitude": [3]}}

IBS_energybound_dict = {"t16": [4, 17], "t17": [3.5, 16.25],
                        "t20": [3.5, 16.5], "t21": [4.25, 16.75], "t25": [4.25, 18.25], "t26": [4.35, 18.25],
                        "t27": [4.5, 18.25],
                        "t28": [4.5, 18.25], "t29": [4.5, 18.25],
                        "t30": [4.5, 18.25], "t32": [4.5, 18.25],
                        "t42": [4.5, 19.5], "t46": [3.75, 17.5], "t47": [4.5, 18.25]}

ELS_energybound_dict = {"t16": [1, 30], "t17": [1, 35],
                        "t20": [1, 35], "t21": [1, 35], "t25": [1, 35], "t26": [1, 35],
                        "t27": [1, 30],
                        "t28": [1, 35], "t29": [1, 35],
                        "t30": [1, 35], "t32": [1, 35],
                        "t42": [1, 35], "t46": [1, 35], "t47": [1, 35]}

IBS_initvalues_dict = {"t16": [0.25, 200], "t17": [0.6, 300],
                       "t20": [0.25, 200], "t21": [0.25, 0], "t25": [0.25, 250], "t26": [0.25, 0],
                       "t27": [0.25, 200],
                       "t28": [0.25, 0], "t29": [0.25, 200],
                       "t30": [0.25, 250], "t32": [0.25, 200],
                       "t42": [0.25, 0], "t46": [0.5, 300], "t47": [0.25, 200]}

ELS_initvalues_dict = {"t16": [0.25, 100], "t17": [-0.5, 400],
                       "t20": [0.25, 200], "t21": [0.25, 0], "t25": [0.25, 250], "t26": [0.25, 0],
                       "t27": [0.25, 200],
                       "t28": [0.25, 0], "t29": [0.25, 200],
                       "t30": [0.25, 250], "t32": [0.25, 200],
                       "t42": [0.25, 0], "t46": [0.5, 300], "t47": [0.25, 200]}


def energy2mass(energyarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    massarray = (2 * (energyarray * e + (spacecraftpotential * charge * e) - 8 * k * iontemperature)) / (
            ((spacecraftvelocity + ionvelocity) ** 2) * AMU)
    return massarray


def mass2energy(massarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    energyarray = (0.5 * massarray * ((spacecraftvelocity + ionvelocity) ** 2) * AMU - (
            spacecraftpotential * charge * e) + 8 * k * iontemperature) / e
    return energyarray


def ELS_maxflux_anode(elsdata, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)
    anodesums = np.sum(np.sum(dataslice, axis=2), axis=0)
    maxflux_anode = np.argmax(anodesums)
    return maxflux_anode


def IBS_ELS_gaussian(ibs_x, ibs_dataslice, els_x, els_dataslice, cassini_speed, lpvalue, temperature):
    x_1, x_2, y_1, y_2 = variables('x_1, x_2, y_1, y_2')

    # Constants
    # k = Parameter("k", value=scipy.constants.physical_constants['Boltzmann constant'][0])
    # k.fixed = True
    # AMU = Parameter("AMU", value=scipy.constants.physical_constants['atomic mass constant'][0])
    # AMU.fixed = True
    # e = Parameter("e", scipy.constants.physical_constants['atomic unit of charge'][0])
    # e.fixed = True
    k_e = Parameter("k_e", value=scipy.constants.physical_constants['Boltzmann constant'][0] /
                                 scipy.constants.physical_constants['atomic unit of charge'][0])
    k_e.fixed = True
    AMU_e = Parameter("AMU_e", value=scipy.constants.physical_constants['atomic mass constant'][0] /
                                     scipy.constants.physical_constants['atomic unit of charge'][0])
    AMU_e.fixed = True

    # Physical parameters
    sc_velocity = Parameter("sc_velocity", value=cassini_speed)
    sc_velocity.fixed = True
    temp_eV = Parameter("temp", value=(8 * scipy.constants.physical_constants['Boltzmann constant'][0]*temperature)/scipy.constants.physical_constants['atomic unit of charge'][0])
    temp_eV.fixed = True
    lp_pot = Parameter("lp_pot", value=lpvalue)
    ionvelocity = Parameter("ionvelocity", value=0)

    # Negative Ion Parameters
    mass26_neg_cen = Parameter("mass26_neg_cen", value=3)
    mass50_neg_cen = Parameter("mass50_neg_cen", value=7)
    mass74_neg_cen = Parameter("mass74_neg_cen", value=15)
    mass117_neg_cen = Parameter("mass117_neg_cen", value=19)

    mass26_neg_amp = Parameter("mass26_neg_amp", value=1e5)
    mass50_neg_amp = Parameter("mass50_neg_amp", value=1e5)
    mass74_neg_amp = Parameter("mass74_neg_amp", value=3e4)
    mass117_neg_amp = Parameter("mass117_neg_amp", value=2e4)

    mass26_neg_sig = Parameter("mass26_neg_sig", value=0.5, min=0.1, max=2)
    mass50_neg_sig = Parameter("mass50_neg_sig", value=0.8, min=0.1, max=2)
    mass74_neg_sig = Parameter("mass74_neg_sig", value=0.9, min=0.1, max=2)
    mass117_neg_sig = Parameter("mass117_neg_sig", value=1.2, min=0.1, max=2)

    # Positive Ion Parameters
    mass28_cen = Parameter("mass28_cen", value=5)
    mass41_cen = Parameter("mass41_cen", value=7)
    mass53_cen = Parameter("mass53_cen", value=9.5)
    mass66_cen = Parameter("mass66_cen", value=11.5)
    mass78_cen = Parameter("mass78_cen", value=13)
    mass91_cen = Parameter("mass91_cen", value=16)

    mass28_amp = Parameter("mass28_amp", value=5e5)
    mass41_amp = Parameter("mass41_amp", value=4e5)
    mass53_amp = Parameter("mass53_amp", value=3e5)
    mass66_amp = Parameter("mass66_amp", value=3e5)
    mass78_amp = Parameter("mass78_amp", value=3e5)
    mass91_amp = Parameter("mass91_amp", value=3e5)

    mass28_sig = Parameter("mass28_sig", value=0.4, min=0.1, max=1.5)
    mass41_sig = Parameter("mass41_sig", value=0.5, min=0.1, max=1.5)
    mass53_sig = Parameter("mass53_sig", value=0.5, min=0.1, max=1.5)
    mass66_sig = Parameter("mass66_sig", value=0.6, min=0.1, max=1.5)
    mass78_sig = Parameter("mass78_sig", value=0.7, min=0.1, max=1.5)
    mass91_sig = Parameter("mass91_sig", value=0.8, min=0.1, max=1.5)

    # cen = ((0.5 * (mass * AMU) * ((sc_velocity+ionvelocity) ** 2) - (lpvalue * e * charge) + (8 * k * temp)) / e)
    # amp * np.exp(-(x - cen) ** 2 / (2. * sigma ** 2))

    model = Model({
        y_1: (mass26_neg_amp * exp(
            -(x_1 - (0.5 * (26 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                        2. * mass26_neg_sig ** 2))) +
             (mass50_neg_amp * exp(
                 -(x_1 - (0.5 * (50 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                             2. * mass50_neg_sig ** 2))) +
             (mass74_neg_amp * exp(
                 -(x_1 - (0.5 * (74 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                             2. * mass74_neg_sig ** 2))) +
             (mass117_neg_amp * exp(-(x_1 - (
                         0.5 * (117 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                                                2. * mass117_neg_sig ** 2))),
        y_2: (mass28_amp * exp(
            -(x_2 - (0.5 * (28 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                        2. * mass28_sig ** 2))) +
             (mass41_amp * exp(
                 -(x_2 - (0.5 * (41 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                             2. * mass41_sig ** 2))) +
             (mass53_amp * exp(
                 -(x_2 - (0.5 * (53 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                             2. * mass53_sig ** 2))) +
             (mass66_amp * exp(
                 -(x_2 - (0.5 * (66 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                             2. * mass66_sig ** 2))) +
             (mass78_amp * exp(
                 -(x_2 - (0.5 * (78 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                             2. * mass78_sig ** 2))) +
             (mass91_amp * exp(
                 -(x_2 - (0.5 * (91 * AMU_e) * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                             2. * mass91_sig ** 2)))
    })

    # init_y  = model(x_1=els_x, x_2=ibs_x, **fit_result.params)

    fit = Fit(model, x_1=els_x, x_2=ibs_x, y_1=els_dataslice, y_2=ibs_dataslice)
    fit_result = fit.execute()
    print(fit_result)

    y = model(x_1=els_x, x_2=ibs_x, **fit_result.params)
    print(y)

    plt.plot(els_x, y[0], color='C0')
    plt.step(els_x, els_dataslice, where='mid', color='k')

    plt.plot(ibs_x, y[1], color='C1')
    plt.step(ibs_x, ibs_dataslice, where='mid', color='r')
    plt.yscale("log")
    plt.show()

    # # SCP offset plot
    # effectivescplist, effectivescplist_errors = [], []
    # for masscounter, mass in enumerate(masses):
    #     tempprefix = "mass" + str(mass) + '_'
    #     effectivescplist.append(out.params[tempprefix + "effectivescp"].value)
    #     # effectivescplist_errors.append(out.params[tempprefix + "effectivescp"].stderr)
    # print("effectivescp", effectivescplist)
    # z, cov = np.polyfit(x=np.array(masses), y=np.array(effectivescplist), deg=1, cov=True)
    # ionwindspeed = (z[0] * (e / AMU)) / (cassini_speed)
    # ionwindspeed_err = (np.sqrt(np.diag(cov)[0]) * (e / AMU)) / (cassini_speed)
    # # print(ibsdata['flyby'], " Ion wind velocity = %2.2f ± %2.2f m/s" % (ionwindspeed, ionwindspeed_err))
    #
    # fig, ax = plt.subplots()
    # p = np.poly1d(z)
    # ax.errorbar(masses, np.array(effectivescplist), fmt='.')  # ,yerr=effectivescplist_errors)
    # ax.plot(masses, p(masses))
    #
    # # SCP calculation
    # scpvalues = []
    # for masscounter, mass in enumerate(masses):
    #     scpvalues.append((effectivescplist[masscounter] - ((mass * AMU * cassini_speed * ionwindspeed) / e)) / charge)
    # print("scplist", scpvalues)
    # scp_mean = np.mean(scpvalues)
    # scp_err = np.std(scpvalues)
    #
    # # print(ibsdata['flyby'], " IBS-derived SCP = %2.2f ± %2.2f V" % (scp_mean, scp_err))
    #
    # print(out.fit_report(min_correl=0.7))
    #
    # return out, ionwindspeed, ionwindspeed_err, scp_mean, scp_err


def titan_linearfit_temperature(altitude):
    if altitude > 1150:
        temperature = 110 + 0.26 * (altitude - 1200)
    else:
        temperature = 133 - 0.12 * (altitude - 1100)
    return temperature


def ELS_IBS_fluxfitting(elsdata, ibsdata, tempdatetime, titanaltitude, ibs_masses=[28, 41, 53, 66, 78, 91],
                        els_masses=[25, 50, 74, 117]):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    els_slicenumber = CAPS_slicenumber(elsdata, tempdatetime)
    ibs_slicenumber = CAPS_slicenumber(ibsdata, tempdatetime)
    lpdata = read_LP_V1(elsdata['flyby'])
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    # print("interp lpvalue", lpvalue)
    els_initwindspeed = ELS_initvalues_dict[elsdata['flyby']][1]
    ibs_initwindspeed = IBS_initvalues_dict[ibsdata['flyby']][1]

    els_lowerenergyslice = CAPS_energyslice("els", ELS_energybound_dict[elsdata['flyby']][0] - lpvalue,
                                            ELS_energybound_dict[elsdata['flyby']][0] - lpvalue)[0]
    els_upperenergyslice = CAPS_energyslice("els", ELS_energybound_dict[elsdata['flyby']][1] - lpvalue,
                                            ELS_energybound_dict[elsdata['flyby']][1] - lpvalue)[0]

    ibs_lowerenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue,
                                            IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue)[0]
    ibs_upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue,
                                            IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue)[0]

    windspeed = 0
    temperature = titan_linearfit_temperature(titanaltitude)

    ibs_dataslice = ibsdata['ibsdata'][ibs_lowerenergyslice:ibs_upperenergyslice, 1, ibs_slicenumber]
    ibs_x = ibscalib['ibsearray'][ibs_lowerenergyslice:ibs_upperenergyslice]

    anode = ELS_maxflux_anode(elsdata, tempdatetime - datetime.timedelta(seconds=10),
                              tempdatetime + datetime.timedelta(seconds=10))
    print("anode", anode)
    els_dataslice = np.float32(ELS_backgroundremoval(elsdata, els_slicenumber, els_slicenumber + 1, datatype="data")[
                               els_lowerenergyslice:els_upperenergyslice, anode, 0])
    # print("removed_dataslice", removed_dataslice,type(removed_dataslice),type(removed_dataslice[0]))

    # dataslice = elsdata['data'][lowerenergyslice:upperenergyslice, anode, slicenumber]
    # print("dataslice", dataslice,type(dataslice),type(dataslice[0]))
    print(elsdata['flyby'], "Cassini velocity", cassini_speed, "Altitude", titanaltitude)
    els_x = elscalib['earray'][els_lowerenergyslice:els_upperenergyslice]
    out, ionwindspeed, ionwindspeed_err, scp_mean, scp_err = IBS_ELS_gaussian(ibs_x, ibs_dataslice,
                                                                              els_x, els_dataslice,
                                                                              cassini_speed,
                                                                              lpvalue, temperature)

    # print(out.fit_report(min_correl=0.7))
    # comps = out.eval_components(x=x)

    stepplotfig, stepplotax = plt.subplots()
    stepplotax.step(elscalib['polyearray'][els_lowerenergyslice:els_upperenergyslice], els_dataslice, where='post',
                    label="ELS " + elsdata['times_utc_strings'][els_slicenumber], color='k')
    stepplotax.step(ibscalib['ibspolyearray'][ibs_lowerenergyslice:ibs_upperenergyslice], ibs_dataslice, where='post',
                    label="IBSS " + ibsdata['times_utc_strings'][ibs_slicenumber], color='r')

    stepplotax.errorbar(els_x, els_dataslice, yerr=[np.sqrt(i) for i in els_dataslice], color='k', fmt='none')
    stepplotax.errorbar(ibs_x, ibs_dataslice, yerr=[np.sqrt(i) for i in ibs_dataslice], color='r', fmt='none')
    stepplotax.set_xlim(1, 30)
    # stepplotax.set_ylim(min(dataslice), max(dataslice))
    stepplotax.set_yscale("log")
    stepplotax.set_ylabel("Counts [/s]", fontsize=20)
    stepplotax.set_xlabel("Energy (Pre-correction) [eV/q]", fontsize=20)
    stepplotax.tick_params(axis='both', which='major', labelsize=15)
    stepplotax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    stepplotax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
    stepplotax.minorticks_on()
    stepplotax.set_title(
        "Histogram of " + elsdata['flyby'].upper() + " CAPS data from ~" + elsdata['times_utc_strings'][
            els_slicenumber],
        fontsize=32)
    # stepplotax.plot(x, out.init_fit, 'b-', label='init fit')
    # stepplotax.plot(x, out.best_fit, 'r-', label='best fit')
    # stepplotax.text(0.8, 0.02, "Ion wind = %2.2f ± %2.2f m/s" % (ionwindspeed, ionwindspeed_err),
    #                     transform=stepplotax.transAxes)
    # stepplotax.text(0.8, .05,
    #                     "ELS-derived SC Potential = %2.2f ± %2.2f V" % (scp_mean, scp_err),
    #                     transform=stepplotax.transAxes)
    # stepplotax.text(0.8, .08, "LP-derived SC Potential = %2.2f" % lpvalue, transform=els_stepplotax.transAxes)
    # stepplotax.text(0.8, .11, "Temp = %2.2f" % out.params['temp'], transform=els_stepplotax.transAxes)
    # stepplotax.text(0.8, .14, "Reduced $\chi^{2}$ = %.2E" % out.redchi, transform=els_stepplotax.transAxes)
    # for mass in els_masses:
    #     stepplotax.plot(x, comps["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")
    stepplotax.legend(loc='best')

    # return out, lpvalue, ionwindspeed, ionwindspeed_err, scp_mean, scp_err, cassini_speed


windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])
windsdf['Negative Peak Time'] = pd.to_datetime(windsdf['Negative Peak Time'])


def multiple_alongtrackwinds_flybys(usedflybys):
    times = []
    els_fits, ibs_fits, lpvalues, ibs_ionwindspeeds, ibs_ionwindspeeds_err, ibs_scps, ibs_scps_err, cassini_speeds = [], [], [], [], [], [], [], []
    els_fits, els_ionwindspeeds, els_ionwindspeeds_err, els_scps, els_scps_err = [], [], [], [], []
    for flyby in usedflybys:
        tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
        times = times + list(tempdf['Positive Peak Time'])
        elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")
        ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
        generate_aligned_ibsdata(ibsdata, elsdata, flyby)
        for (i, j) in zip(tempdf['Positive Peak Time'], tempdf['Altitude']):
            ibs_fit, lpvalue, ibs_ionwindspeed, ibs_ionwindspeed_err, ibs_scp_mean, ibs_scp_err, cassini_speed = IBS_fluxfitting(
                ibsdata,
                i, j)
            els_fit, lpvalue, els_ionwindspeed, els_ionwindspeed_err, els_scp_mean, els_scp_err, cassini_speed = ELS_fluxfitting(
                elsdata,
                i, j)
            lpvalues.append(lpvalue)
            cassini_speeds.append(cassini_speed)

            ibs_fits.append(ibs_fit)
            ibs_ionwindspeeds.append(ibs_ionwindspeed)
            ibs_ionwindspeeds_err.append(ibs_ionwindspeed_err)
            ibs_scps.append(ibs_scp_mean)
            ibs_scps_err.append(ibs_scp_err)

            els_fits.append(els_fit)
            els_ionwindspeeds.append(els_ionwindspeed)
            els_ionwindspeeds_err.append(els_ionwindspeed_err)
            els_scps.append(els_scp_mean)
            els_scps_err.append(els_scp_err)

    outputdf = pd.DataFrame()
    outputdf['Positive Peak Time'] = times
    outputdf['IBS Alongtrack velocity'] = ibs_ionwindspeeds
    # outputdf['IBS residuals'] = ibs_residuals
    outputdf['IBS spacecraft potentials'] = ibs_scps
    outputdf['Actual spacecraft velocity'] = cassini_speeds
    outputdf.to_csv("testalongtrackvelocity.csv")
    print(outputdf['Positive Peak Time'])

    fig5, axes = plt.subplots(2)
    axes[0].errorbar(tempdf['Positive Peak Time'], ibs_ionwindspeeds, yerr=np.array(ibs_ionwindspeeds_err),
                     label="IBS - Ion Wind Speeds", linestyle='--', capsize=5)
    axes[0].errorbar(tempdf['Positive Peak Time'], els_ionwindspeeds, yerr=np.array(els_ionwindspeeds_err),
                     label="ELS - Ion Wind Speeds", linestyle='--', capsize=5)
    axes[0].set_ylabel("Ion Wind Speed (m/s)")
    # for counter, x in enumerate(ibs_fits):
    #     axes[0].text(tempdf['Positive Peak Time'].iloc[counter], ibs_ionwindspeeds[counter], "Chi-Sqr =  %.1E" % x.chisqr)
    axes[0].legend()

    axes[1].errorbar(tempdf['Positive Peak Time'], ibs_scps, yerr=np.array(ibs_scps_err), color='C1',
                     label="S/C potential, IBS derived", capsize=5)
    axes[1].errorbar(tempdf['Positive Peak Time'], els_scps, yerr=np.array(els_scps_err), color='C2',
                     label="S/C potential, ELS derived", capsize=5)
    axes[1].plot(tempdf['Positive Peak Time'], lpvalues, color='k', label="S/C potential, LP derived")
    axes[1].set_ylabel("S/C Potential (V)")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    return outputdf


def single_slice_test(flyby, slicenumber):
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)
    tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]

    print(tempdf['Positive Peak Time'].iloc[slicenumber])
    result = ELS_IBS_fluxfitting(elsdata, ibsdata, tempdf['Positive Peak Time'].iloc[slicenumber],
                                 tempdf['Altitude'].iloc[slicenumber])


# multiple_alongtrackwinds_flybys(['t17'])
single_slice_test(flyby="t17", slicenumber=4)

plt.show()
