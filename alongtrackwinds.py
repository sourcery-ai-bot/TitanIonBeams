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

filedates = {"t16": "22-jul-2006", "t17": "07-sep-2006",
             "t20": "25-oct-2006", "t21": "12-dec-2006", "t25": "22-feb-2007", "t26": "10-mar-2007",
             "t27": "26-mar-2007",
             "t28": "10-apr-2007", "t29": "26-apr-2007",
             "t30": "12-may-2007", "t32": "13-jun-2007",
             "t42": "25-mar-2008", "t46": "03-nov-2008", "t47": "19-nov-2008"}

IBS_fluxfitting_dict = {"mass28_": {"sigma": [0.3,0.4,0.6], "amplitude": []},
                        "mass40_": {"sigma": [0.3,0.6,0.7], "amplitude": []},
                        "mass53_": {"sigma": [0.3,0.5,0.6],"amplitude": []},
                        "mass66_": {"sigma": [0.4,0.6,0.7], "amplitude": []}, \
                        "mass78_": {"sigma": [0.5,0.7,0.8], "amplitude": []}, \
                        "mass91_": {"sigma": [0.6,0.8,0.9], "amplitude": []}}

ELS_fluxfitting_dict = {"mass26_": {"sigma": [0.2,0.4,0.6], "amplitude": [5]},
                        "mass50_": {"sigma": [0.3,0.5,0.8], "amplitude": [4]},
                        "mass74_": {"sigma": [0.5,0.7,1.3], "amplitude": [3]},
                        "mass98_": {"sigma": [0.6, 0.8, 1.6], "amplitude": [3]},
                        "mass117_": {"sigma": [0.6,0.9,1.7], "amplitude": [3]}}

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

    pars.add('scp', value=LPvalue, min=LPvalue-1.5, max=LPvalue + 0.5)
    pars.add('temp_eV', value=8*k*temperature)  # , min=130, max=170)
    pars.add('spacecraftvelocity', value=cassini_speed)
    pars.add('ionvelocity', value=0, min=-400, max=400)
    #pars['scp'].vary = False
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
            #ampval = IBS_fluxfitting_dict[tempprefix]['amplitude'][0]
        elif charge == -1:
            # sigmaval = ELS_fluxfitting_dict[tempprefix]['sigma']
            sigmavals = ELS_fluxfitting_dict[tempprefix]['sigma']
            ampval = ELS_fluxfitting_dict[tempprefix]['amplitude'][0]

        # effectivescpexpr = 'scp + ((' + tempprefix + '*AMU*spacecraftvelocity)/e)*windspeed' #Windspeed defined positive if going in same direction as Cassini
        # pars.add(tempprefix + "effectivescp", expr=effectivescpexpr)
        pars.update(gaussmodels[-1].make_params())

        temppeakflux = peakflux(mass, pars['spacecraftvelocity'], 0, LPvalue, temperature, charge=charge)
        print("mass", mass, "Init Flux", temppeakflux)

        peakfluxexpr = '(0.5*(' + tempprefix + '*AMU)*((spacecraftvelocity+ionvelocity)**2) - scp*e*charge + temp_eV)/e'
        pars[tempprefix + 'center'].set(
            value=peakflux(mass, cassini_speed, 0, LPvalue + 0.25, temperature, charge=charge), expr=peakfluxexpr)
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

    print(out.fit_report(min_correl=0.7))

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
def IBS_fluxfitting(ibsdata, tempdatetime, titanaltitude, ibs_masses=[28, 40, 53, 66, 78, 91]):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(ibsdata, tempdatetime)
    lpdata = read_LP_V1(ibsdata['flyby'])
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    print("interp lpvalue", lpvalue)

    lowerenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue)[0]
    upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue)[0]

    windspeed = -150
    temperature = titan_linearfit_temperature(titanaltitude)

    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, 1, slicenumber]

    x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice]
    out = total_fluxgaussian(x, dataslice, ibs_masses, cassini_speed, windspeed, lpvalue, temperature,
                             charge=1,
                             FWHM=IBS_FWHM)

    # print(out.fit_report(min_correl=0.7))jiop
    comps = out.eval_components(x=x)

    stepplotfig, stepplotax = plt.subplots()
    stepplotax.step(ibscalib['ibspolyearray'][lowerenergyslice:upperenergyslice], dataslice, where='post',
                    label=ibsdata['flyby'], color='k')
    stepplotax.errorbar(x, dataslice, yerr=[np.sqrt(i) for i in dataslice], color='k', fmt='none')
    stepplotax.set_xlim(3, 19)
    stepplotax.set_ylim(0.9*min(dataslice),1.1*max(dataslice))
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
    if out.params['ionvelocity'].stderr is None:
        out.params['ionvelocity'].stderr = out.params['ionvelocity']
    if out.params['scp'].stderr is None:
        out.params['scp'].stderr = out.params['scp']
    stepplotax.text(0.8, 0.02, "Ion wind = %2.2f ± %2.2f m/s" % (out.params['ionvelocity'], out.params['ionvelocity'].stderr),
                    transform=stepplotax.transAxes)
    stepplotax.text(0.8, .05,
                    "IBS-derived S/C Potential = %2.2f ± %2.2f V" % (out.params['scp'], out.params['scp'].stderr),
                    transform=stepplotax.transAxes)
    stepplotax.text(0.8, .08, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=stepplotax.transAxes)
    stepplotax.text(0.8, .11, "Temp = %2.2f" % temperature, transform=stepplotax.transAxes)
    stepplotax.text(0.8, .14, "Chi-square = %.2E" % out.chisqr, transform=stepplotax.transAxes)
    stepplotax.text(0.8, .17, "My GOF = %.2f %%" % np.mean((abs(out.best_fit - dataslice) / dataslice) * 100),
                        transform=stepplotax.transAxes)
    for mass in ibs_masses:
        stepplotax.plot(x, comps["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")
    stepplotax.legend(loc='best')

    return out, lpvalue

def ELS_maxflux_anode(elsdata, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)
    anodesums = np.sum(np.sum(dataslice, axis=2), axis=0)
    maxflux_anode = np.argmax(anodesums)
    return maxflux_anode

def ELS_fluxfitting(elsdata, tempdatetime, titanaltitude, els_masses = [26, 50, 74, 117]):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    els_slicenumber = CAPS_slicenumber(elsdata, tempdatetime)
    lpdata = read_LP_V1(elsdata['flyby'])
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])

    windspeed = 0
    temperature = titan_linearfit_temperature(titanaltitude)

    els_lowerenergyslice = CAPS_energyslice("els", ELS_energybound_dict[elsdata['flyby']][0] - lpvalue,
                                            ELS_energybound_dict[elsdata['flyby']][0] - lpvalue)[0]
    els_upperenergyslice = CAPS_energyslice("els", ELS_energybound_dict[elsdata['flyby']][1] - lpvalue,
                                            ELS_energybound_dict[elsdata['flyby']][1] - lpvalue)[0]
    x = elscalib['earray'][els_lowerenergyslice:els_upperenergyslice]
    anode = ELS_maxflux_anode(elsdata, tempdatetime - datetime.timedelta(seconds=10),
                              tempdatetime + datetime.timedelta(seconds=10))
    print("anode", anode)

    dataslices= []
    for i in range(3):
        dataslices.append(np.float32(ELS_backgroundremoval(elsdata, els_slicenumber-1+i, els_slicenumber+i, datatype="data")[
                               els_lowerenergyslice:els_upperenergyslice, anode, 0]))

    stepplotfig_els, stepplotax_els = plt.subplots()
    for dataslice in dataslices:
        stepplotax_els.scatter(x, dataslice)

    #stepplotax_els.errorbar(x, dataslice2, yerr=[np.sqrt(i) for i in dataslice2], color='k', fmt='none')
    stepplotax_els.set_yscale("log")
    stepplotax_els.set_xlim(1, 25)
    #stepplotax_els.set_ylim(0.9*min(dataslice2),1.1*max(dataslice2))
    stepplotax_els.set_ylabel("Counts (/s)", fontsize=20)
    stepplotax_els.set_xlabel("Energy (Pre-correction) [eV/q]", fontsize=20)
    stepplotax_els.tick_params(axis='both', which='major', labelsize=15)
    stepplotax_els.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    stepplotax_els.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
    stepplotax_els.minorticks_on()
    stepplotax_els.set_title(
        "Histogram of " + elsdata['flyby'].upper() + " els data from " + elsdata['times_utc_strings'][els_slicenumber],
        fontsize=32)
    # stepplotax.plot(elscalib['earray'],smoothedcounts_full,color='k')

    outputs = []
    for dataslice in dataslices:
        outputs.append(total_fluxgaussian(x, dataslice, els_masses, cassini_speed, windspeed, lpvalue, temperature,
                                 charge=-1,
                                 FWHM=ELS_FWHM))

    print(out1.params['ionvelocity'],out2.params['ionvelocity'],out3.params['ionvelocity'])
    print(out1.params['scp'], out2.params['scp'], out3.params['scp'])
    stepplotax_els.plot(x, out2.init_fit, 'b-', label='init fit')
    stepplotax_els.plot(x, out2.best_fit, 'r-', label='best fit')
    if out1.params['ionvelocity'].stderr is None:
        out1.params['ionvelocity'].stderr = abs(out1.params['ionvelocity'])
    if out2.params['scp'].stderr is None:
        out2.params['scp'].stderr = abs(out.params['scp'])
    stepplotax_els.text(0.8, 0.02, "Ion wind = %2.2f ± %2.2f m/s" % (out.params['ionvelocity'], out.params['ionvelocity'].stderr), transform=stepplotax_els.transAxes)
    stepplotax_els.text(0.8, .05, "ELS-derived S/C Potential = %2.2f ± %2.2f V" % (out.params['scp'], out.params['scp'].stderr), transform=stepplotax_els.transAxes)
    stepplotax_els.text(0.8, .08, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=stepplotax_els.transAxes)
    stepplotax_els.text(0.8, .11, "Temp = %2.2f" % temperature, transform=stepplotax_els.transAxes)
    stepplotax_els.text(0.8, .14, "Chi-square = %.2E" % out.chisqr, transform=stepplotax_els.transAxes)
    #stepplotax_els.text(0.8, .17, "My GOF = %.2f %%" % np.mean((abs(out.best_fit-meandataslice)/meandataslice)*100), transform=stepplotax_els.transAxes)

    comps = out.eval_components(x=x)
    for mass in els_masses:
        stepplotax_els.plot(x, comps["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")

    stepplotax_els.legend(loc='best')

windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])
windsdf['Negative Peak Time'] = pd.to_datetime(windsdf['Negative Peak Time'])

def multiple_alongtrackwinds_flybys(usedflybys):
    times = []
    for flyby in usedflybys:
        els_fits, ibs_fits, lpvalues = [], [], []
        tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
        elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")
        ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
        generate_aligned_ibsdata(ibsdata, elsdata, flyby)
        for (i, j) in zip(tempdf['Positive Peak Time'], tempdf['Altitude']):
            ibs_fit, lpvalue = IBS_fluxfitting(ibsdata, i, j)
            ibs_fits.append(ibs_fit)
            lpvalues.append(lpvalue)

    testoutputdf = pd.DataFrame()
    testoutputdf['Bulk Time'] = tempdf['Bulk Time']
    testoutputdf['IBS Alongtrack velocity'] = [i.params['windspeed'] for i in ibs_fits]
    # testoutputdf['IBS residuals'] = ibs_residuals
    testoutputdf['IBS spacecraft potentials'] = [i.params['scp'] for i in ibs_fits]
    testoutputdf.to_csv("testalongtrackvelocity.csv")
    #
    fig5, ax5 = plt.subplots()
    ax5.set_title(str(usedflybys))
    ax5.errorbar(tempdf['Positive Peak Time'], [i.params['windspeed'] for i in ibs_fits], yerr=[i.params['windspeed'].stderr for i in ibs_fits], color='C0',
                 label="Ion Wind Speeds",linestyle='--')
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Ion Wind Speed (m/s)")
    ax5_1 = ax5.twinx()
    ax5_1.errorbar(tempdf['Positive Peak Time'], [i.params['scp'] for i in ibs_fits], yerr=[i.params['scp'].stderr for i in ibs_fits], color='C1', label="S/C potential, IBS derived")
    ax5_1.plot(tempdf['Positive Peak Time'],lpvalues, color='C2', label="S/C potential, LP derived")
    ax5_1.set_ylabel("S/C Potential (V)")
    for counter,x in enumerate(ibs_fits):
        ax5.text(tempdf['Positive Peak Time'].iloc[counter], x.params['windspeed'], "Chi-Sqr =  %.1E\nCorr = %.2f " % (x.chisqr,x.params['windspeed'].correl['scp']))
    fig5.legend()

    return testoutputdf

def single_slice_test(flyby, slicenumber):
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)
    tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]

    print(tempdf['Positive Peak Time'].iloc[slicenumber])
    ibs_ionwindspeed = IBS_fluxfitting(ibsdata, tempdf['Positive Peak Time'].iloc[slicenumber],
                                       tempdf['Altitude'].iloc[slicenumber])
    els_ionwindspeed = ELS_fluxfitting(elsdata, tempdf['Positive Peak Time'].iloc[slicenumber],
                                       tempdf['Altitude'].iloc[slicenumber])

single_slice_test("t16",slicenumber=3)


plt.show()
