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
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

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

filedates = {'ta': "26-oct-2004", "t16": "22-jul-2006", "t17": "07-sep-2006", "t18": "23-sep-2006",
             "t19": "09-oct-2006",
             "t20": "25-oct-2006", "t21": "12-dec-2006", "t23": "13-jan-2007", "t25": "22-feb-2007",
             "t26": "10-mar-2007",
             "t27": "26-mar-2007",
             "t28": "10-apr-2007", "t29": "26-apr-2007",
             "t30": "12-may-2007", "t32": "13-jun-2007", "t36": "02-oct-2007", "t39": "20-dec-2007",
             "t40": "05-jan-2008", "t41": "22-feb-2008", "t42": "25-mar-2008", "t43": "12-may-2008",
             "t46": "03-nov-2008", "t47": "19-nov-2008", "t48": "05-dec-2008", "t49": "21-dec-2008",
             "t50": "07-feb-2009", "t51": "27-mar-2009", "t71": "07-jul-2010", "t83": "22-may-2012"}

IBS_fluxfitting_dict = {"mass16_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass17_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass28_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass39_": {"sigma": [0.2, 0.3, 0.6], "amplitude": []},
                        "mass40_": {"sigma": [0.2, 0.3, 0.7], "amplitude": []},
                        "mass51_": {"sigma": [0.3, 0.5, 0.65], "amplitude": []},
                        "mass52_": {"sigma": [0.3, 0.5, 0.65], "amplitude": []},
                        "mass53_": {"sigma": [0.3, 0.5, 0.65], "amplitude": []},
                        "mass65_": {"sigma": [0.4, 0.6, 0.7], "amplitude": []},
                        "mass66_": {"sigma": [0.4, 0.6, 0.7], "amplitude": []},
                        "mass76_": {"sigma": [0.5, 0.7, 0.8], "amplitude": []},
                        "mass77_": {"sigma": [0.5, 0.7, 0.8], "amplitude": []},
                        "mass78_": {"sigma": [0.5, 0.7, 0.8], "amplitude": []},
                        "mass91_": {"sigma": [0.6, 0.8, 1], "amplitude": []}}

IBS_fluxfitting_dict_log = {"mass28_": {"sigma": [2, 1.5, 3.5], "amplitude": [30]},
                        "mass40_": {"sigma": [0.2, 1.3, 0.6], "amplitude": [30]},
                        "mass53_": {"sigma": [0.3, 1.5, 2], "amplitude": [30]},
                        "mass66_": {"sigma": [0.4, 1.6, 2], "amplitude": [30]}, \
                        "mass78_": {"sigma": [0.5, 1.5, 2], "amplitude": [30]}, \
                        "mass91_": {"sigma": [0.6, 2, 2], "amplitude": [30]}}

ELS_fluxfitting_dict = {"mass13_": {"sigma": [0.1, 0.2, 0.7], "amplitude": [5]},
                        "mass26_": {"sigma": [0.1, 0.3, 0.8], "amplitude": [5]},
                        "mass38_": {"sigma": [0.1, 0.2, 0.7], "amplitude": [5]},
                        "mass50_": {"sigma": [0.5, 0.6, 1.4], "amplitude": [4]},
                        "mass74_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass77_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass79_": {"sigma": [0.5, 1, 1.8], "amplitude": [3]},
                        "mass91_": {"sigma": [0.6, 0.8, 1.6], "amplitude": [3]},
                        "mass117_": {"sigma": [0.8, 1.8, 3.3], "amplitude": [3]}}

IBS_energybound_dict = {"ta": [3.8, 18], "t16": [4, 17], "t17": [3.5, 16.25], "t18": [3.5, 16.25], "t19": [3.5, 16.25],
                        "t20": [3.5, 16.5], "t21": [4.25, 16.75], "t23": [4.25, 16.85], "t25": [4.25, 18.25],
                        "t26": [4.35, 18.25],
                        "t27": [4.5, 18.5],
                        "t28": [4.5, 18.25], "t29": [4.5, 18.25],
                        "t30": [4.5, 18.5], "t32": [4.5, 18.25], "t36": [4.5, 19], "t39": [4.5, 19],
                        "t40": [4.5, 19], "t41": [4.5, 19],
                        "t42": [4.5, 20], "t43": [4.5, 20.25], "t46": [4, 17], "t47": [4.5, 18.25], "t48": [4, 19.25],
                        "t49": [4.5, 19.5], "t50": [4, 19], "t51": [4.5, 19],  "t55": [4, 19], "t56": [4, 19],  "t57": [4, 19],
                        "t58": [4, 19],  "t59": [4, 19],
                        "t71": [3.5, 16.5], "t83": [3.5, 16.5]}


ELS_energybound_dict = {"ta": [3, 30], "t16": [1, 25], "t17": [1.5, 30], "t18": [1.5, 30], "t19": [1.5, 28],
                        "t20": [1, 30], "t21": [1, 30], "t23": [1, 30], "t25": [1.5, 25], "t26": [1.5, 30],
                        "t27": [1.5, 30],
                        "t28": [1.5, 30], "t29": [1, 30],
                        "t30": [1, 30], "t32": [1.5, 30], "t36": [1.5, 30], "t39": [1.5, 30], "t40": [2.3, 30],
                        "t41": [2.3, 30],
                        "t42": [2.3, 30], "t43": [2.3, 30], "t46": [2.3, 28], "t47": [1, 30], "t48": [1, 30],
                        "t49": [1, 30], "t50": [1, 35],
                        "t51": [1, 30], "t55": [1, 35], "t56": [1, 35], "t57": [1, 35], "t58": [1, 35], "t59": [1, 35],
                        "t71": [1, 35], "t83": [1, 25]}

ELS_midenergybound_dict = {"ta": [3, 18], "t16": [1, 16], "t17": [1.5, 20], "t18": [1.5, 20], "t19": [1.5, 20],
                        "t20": [1, 20], "t21": [1, 16], "t23": [1, 20], "t25": [1.5, 20], "t26": [1.5, 20],
                        "t27": [1.5, 20],
                        "t28": [1.5, 20], "t29": [1, 20],
                        "t30": [1, 20], "t32": [1.5, 20], "t36": [1.5, 20], "t39": [1.5, 20], "t40": [2.3, 20],
                        "t41": [2.3, 20],
                        "t42": [2.3, 20], "t43": [2.3, 20], "t46": [2.3, 20], "t47": [1, 20], "t48": [1, 20],
                        "t49": [1, 20], "t50": [1, 20],
                        "t51": [1, 22], "t71": [1, 20], "t83": [1, 20]}

ELS_smallenergybound_dict = {"ta": [1.5, 11], "t16": [1.5, 11], "t17": [1.5, 11], "t18": [1.5, 11], "t19": [1.5, 12],
                             "t20": [1, 11], "t21": [1, 10], "t23": [1, 10], "t25": [1.5, 11], "t26": [1.5, 11],
                             "t27": [1.5, 11],
                             "t28": [1.5, 11], "t29": [1, 11],
                             "t30": [1, 11], "t32": [1.5, 11], "t36": [1.5, 13], "t39": [2.3, 13.5], "t40": [2.3, 13.5],
                             "t41": [2.3, 13.5],
                             "t42": [2.3, 13.5], "t43": [2.3, 13.5], "t46": [2.3, 13.5], "t47": [1, 11],
                             "t48": [2.3, 13.5], "t49": [2.3, 13.5],
                             "t50": [2.3, 13], "t51": [2.3, 13.5], "t71": [2, 10.5], "t83": [2, 10.5]}

IBS_smallenergybound_dict = {"ta": [1.5, 11], "t16": [1.5, 10], "t17": [3.5, 7.5], "t18": [1.5, 10], "t19": [1.5, 10],
                             "t20": [1, 11], "t21": [1, 10], "t23": [1, 10], "t25": [1.5, 11], "t26": [1.5, 11],
                             "t27": [1.5, 11],
                             "t28": [1.5, 11], "t29": [1, 11],
                             "t30": [1, 11], "t32": [1.5, 11], "t36": [1.5, 13], "t39": [2.3, 13.5], "t40": [2.3, 13.5],
                             "t41": [2.3, 13.5],
                             "t42": [2.3, 13.5], "t43": [2.3, 13.5], "t46": [2.3, 13.5], "t47": [1, 11],
                             "t48": [2.3, 13.5], "t49": [2.3, 13.5],
                             "t50": [2.3, 13], "t51": [2.3, 13.5], "t71": [2, 10.5], "t83": [2, 10.5]}

IBS_smallmasses_dict = {"ta": [28, 40], "t16": [28, 40], "t17": [28, 40], "t18": [28, 40], "t19": [28, 40],
                        "t20": [28, 40], "t21": [28, 40], "t23": [28, 40], "t25": [28, 40], "t26": [28, 40],
                        "t27": [28, 40],
                        "t28": [28, 40], "t29": [28, 40],
                        "t30": [28, 40], "t32": [28, 40], "t36": [28, 40], "t39": [28, 40], "t40": [28, 40],
                        "t41": [28, 40],
                        "t42": [28, 40], "t43": [28, 40], "t46": [28, 40], "t47": [28, 40], "t48": [28, 40],
                        "t49": [28, 40],
                        "t50": [28, 40], "t51": [28, 40], "t71": [28, 40], "t83": [28, 40]}

ELS_smallmasses_dict = {"ta": [26, 50], "t16": [26, 50], "t17": [26, 50], "t18": [26, 50], "t19": [26, 50],
                        "t20": [26, 50], "t21": [26, 50], "t23": [26, 50], "t25": [26, 50], "t26": [26, 50],
                        "t27": [26, 50],
                        "t28": [26, 50], "t29": [26, 50],
                        "t30": [26, 50], "t32": [26, 50], "t36": [26, 50], "t39": [26, 50], "t40": [26, 50],
                        "t41": [26, 50],
                        "t42": [26, 50], "t43": [26, 50], "t46": [26, 50], "t47": [26, 50], "t48": [26, 50],
                        "t49": [26, 50],
                        "t50": [26, 50], "t51": [26, 50], "t71": [26, 50], "t83": [26, 50]}

IBS_init_ionvelocity_dict = {"ta": 0, "t16": 0, "t17": -100, "t19": -100,
                             "t20": -100, "t21": 200, "t23": -100, "t25": -100, "t26": -100,
                             "t27": 0,
                             "t28": 200, "t29": -100,
                             "t30": -300, "t32": -300, "t36": -300, "t39": -300,
                             "t40": -100, "t41": -300, "t42": -100, "t43": 0, "t46": -350, "t47": -300,
                             "t48": -300, "t49": -300, "t50": -300, "t51": -300, "t71": -300, "t83": -300}

IBS_init_ionvelocity_constrained_dict = {"ta": 0, "t16": -100, "t17": -400, "t19": -100,
                             "t20": -100, "t21": -100, "t23": -100, "t25": -100, "t26": -100,
                             "t27": -130,
                             "t28": -100, "t29": -100,
                             "t30": -300, "t32": -300, "t36": -300, "t39": -300,
                             "t40": -100, "t41": -300, "t42": -100, "t43": 0, "t46": -350, "t47": -300,
                             "t48": -300, "t49": -300, "t50": -300, "t51": -300, "t71": -300, "t83": -300}

ELS_init_ionvelocity_dict = {"ta": 0, "t16": 0, "t17": -50, "t19": 0,
                             "t20": 0, "t21": 0, "t23": 0, "t25": -100, "t26": -150,
                             "t27": -200, "t28": -200, "t29": -150,
                             "t30": 0, "t32": 0, "t36": 100, "t39": 100,
                             "t40": 0, "t41": 0, "t42": 200, "t43": 0, "t46": 0, "t47": 0,
                             "t48": 0, "t49": 0, "t50": 0, "t51": 0, "t71": -200, "t83": 20}


LP_offset_dict = {"ta": -0.4, "t16": -0.3, "t17": -0.2, "t19": -0.35,
                  "t20": -0.35, "t21": -0.35, "t23": -0.7, "t25": -0.35, "t26": -0.4,
                  "t27": -0.6,
                  "t28": -0.4, "t29": -0.4,
                  "t30": -0.4, "t32": -0.35, "t36": -0.35, "t39": -0.25,
                  "t40": -0.25, "t41": -0.25, "t42": -0.2, "t43": -0.1, "t46": -0.35, "t47": -0.35,
                  "t48": -0.35, "t49": -0.35, "t50": -0.35, "t51": -0.35, "t71": -0.35, "t83": -0.35}


def CAPS_slicenumber(data, tempdatetime):
    for counter, i in enumerate(data['times_utc']):
        if i >= tempdatetime:
            slicenumber = counter - 1
            break

    return slicenumber


def energy2mass(energyarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    massarray = (2 * (energyarray * e + (spacecraftpotential * charge * e) - 8 * k * iontemperature)) / (
            ((spacecraftvelocity + ionvelocity) ** 2) * AMU)
    return massarray


def mass2energy(massarray, spacecraftvelocity, ionvelocity, spacecraftpotential, iontemperature=150, charge=1):
    energyarray = (0.5 * massarray * ((spacecraftvelocity + ionvelocity) ** 2) * AMU - (
            spacecraftpotential * charge * e) + 8 * k * iontemperature) / e
    return energyarray


def total_fluxgaussian(xvalues, yvalues, masses, cassini_speed, altitude, windspeed, LPvalue, lpoffset, temperature, charge,
                       FWHM):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    if charge == 1:
        if LPvalue > 0:
            startscp = -1.2
            minscp = -1.8
        elif LPvalue+0.5 > 0:
            startscp = LPvalue/2
            minscp = LPvalue-0.5
        else:
            if altitude > 1500:
                startscp = LPvalue+0.4
                minscp = LPvalue
            else:
                startscp = LPvalue+0.3
                minscp = LPvalue-0.5
        pars.add('scp', value=startscp, min=minscp, max=0)
        # pars.add('scp', value=LPvalue+lpoffset)
        # pars['scp'].vary = False
    elif charge == -1:
        pars.add('scp', value=LPvalue-0.6, min=LPvalue - 1.5, max=LPvalue + 0.25)

        # pars.add('scp', value=LPvalue+lpoffset)
        # pars['scp'].vary = False

    pars.add('temp_eV', value=8 * k * temperature)  # , min=130, max=170)
    pars.add('spacecraftvelocity', value=cassini_speed)
    pars.add('ionvelocity', value=windspeed, min=windspeed-400, max=windspeed+400)
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
            #ampval = IBS_fluxfitting_dict_log[tempprefix]['amplitude'][0]
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
        pars[tempprefix + 'amplitude'].set(value=np.mean(yvalues) * (sigmavals[1]/0.4), min=min(yvalues))


    for counter, model in enumerate(gaussmodels):
        if counter == 0:
            mod = model
        else:
            mod = mod + model

    init = mod.eval(pars, x=xvalues)
    out = mod.fit(yvalues, pars, x=xvalues, weights=(1/yvalues))

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

    return out, init

def total_fluxgaussian_objective(xvalues, yvalues, masses, cassini_speed, windspeed, LPvalue, lpoffset, temperature, charge,
                       FWHM):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    if charge == 1:
        pars.add('scp', value=LPvalue+0.25, min=LPvalue - 2, max=-0.2)
        # pars.add('scp', value=LPvalue+lpoffset)
        # pars['scp'].vary = False
    elif charge == -1:
        pars.add('scp', value=LPvalue, min=LPvalue - 2, max=0)
        # pars.add('scp', value=LPvalue+lpoffset)
        # pars['scp'].vary = False

    pars.add('temp_eV', value=8 * k * temperature)  # , min=130, max=170)
    pars.add('spacecraftvelocity', value=cassini_speed)
    pars.add('ionvelocity', value=windspeed, min=windspeed-500, max=windspeed+500)
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
            #ampval = IBS_fluxfitting_dict_log[tempprefix]['amplitude'][0]
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

    #print(out.fit_report(min_correl=0.7))

    # Calculating CI's
    # print(out.ci_report(p_names=["scp","windspeed"],sigmas=[1],verbose=True,with_offset=False,ndigits=2))

    return out, init


def titan_linearfit_temperature(altitude):
    if altitude > 1150:
        temperature = 110 + 0.26 * (altitude - 1200)
    else:
        temperature = 133 - 0.12 * (altitude - 1100)
    return temperature


def ELS_maxflux_anode(elsdata, starttime, endtime):
    startslice, endslice = CAPS_slicenumber(elsdata, starttime), CAPS_slicenumber(elsdata, endtime)
    dataslice = ELS_backgroundremoval(elsdata, startslice, endslice)
    anodesums = np.sum(np.sum(dataslice, axis=2), axis=0)
    maxflux_anode = np.argmax(anodesums)
    return maxflux_anode



#windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])
windsdf['Negative Peak Time'] = pd.to_datetime(windsdf['Negative Peak Time'])


def ELS_fluxfitting(elsdata, tempdatetime, titanaltitude, lpdata, els_masses, numofflybys=1):  # [26, 50, 79, 117]
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    els_slicenumber = CAPS_slicenumber(elsdata, tempdatetime)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])

    windspeed = -300
    temperature = titan_linearfit_temperature(titanaltitude)

    if len(els_masses) == 2:
        energydict = ELS_smallenergybound_dict[elsdata['flyby']]
    if len(els_masses) > 2:
        energydict = ELS_energybound_dict[elsdata['flyby']]

    els_lowerenergyslice = CAPS_energyslice("els", energydict[0], energydict[0])[0]
    els_upperenergyslice = CAPS_energyslice("els", energydict[1], energydict[1])[0]
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
        ax.set_xlim(1, 25)

    out = total_fluxgaussian(np.array(tempx), dataslice, els_masses, cassini_speed, windspeed, lpvalue,
                             LP_offset_dict[elsdata['flyby']], temperature,
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
        ax.plot(tempx, out.best_fit, 'k-', label='best fit')
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



def ELS_fluxfitting_2dfluxtest(elsdata, tempdatetime, titanaltitude, windspeed, lpdata, els_masses=[26, 50, 74, 117],
                               numofflybys=1):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3

    print("Titan Altitude",titanaltitude)
    if titanaltitude < 1000:
        els_masses = [26, 50, 79, 117]
    if titanaltitude > 1000 and titanaltitude < 1100:
        els_masses = [26, 50, 79]
    elif titanaltitude > 1100:
        els_masses = [26, 50]

    # if titanaltitude < 1000:
    #     els_masses = [26, 39, 50, 79]
    # if titanaltitude > 1000 and titanaltitude < 1100:
    #     els_masses = [26, 39, 50]
    # elif titanaltitude > 1100:
    #     els_masses = [26, 39]

    slice_halfwidth_dict = {'t16' : 10, 't17' : 10, 't19' : 8, 't21' : 10, 't23' : 10, 't25' : 10, 't26' : 10,
                         't27' : 10, 't28' : 10, 't29' : 10, 't30' : 10, 't32' : 10, 't36' : 10, 't39' : 10,
                         't40' : 10, 't41' : 10, 't42' : 10, 't43' : 10, 't48' : 10, 't49' : 10, 't50' : 10,
                         't51' : 5, 't71' : 10, 't83' : 10}

    print(tempdatetime)
    els_slicenumber_start = CAPS_slicenumber(elsdata, tempdatetime) - slice_halfwidth_dict[elsdata['flyby']]
    els_slicenumber_end = CAPS_slicenumber(elsdata, tempdatetime) + slice_halfwidth_dict[elsdata['flyby']]
    #print(elsdata['times_utc'][els_slicenumber_start], elsdata['times_utc'][els_slicenumber_end ])

    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    temperature = titan_linearfit_temperature(titanaltitude)
    #windspeed = ELS_init_ionvelocity_dict[elsdata['flyby']]

    if len(els_masses) == 2:
        energydict = ELS_smallenergybound_dict[elsdata['flyby']]
    if len(els_masses) == 3:
        energydict = ELS_midenergybound_dict[elsdata['flyby']]
    if len(els_masses) > 3:
        energydict = ELS_energybound_dict[elsdata['flyby']]

    els_lowerenergyslice = CAPS_energyslice("els", energydict[0], energydict[0])[0]
    els_upperenergyslice = CAPS_energyslice("els", energydict[1], energydict[1])[0]

    # x = elscalib['earray'][els_lowerenergyslice:els_upperenergyslice]
    anode = ELS_maxflux_anode(elsdata, tempdatetime - datetime.timedelta(seconds=10),
                              tempdatetime + datetime.timedelta(seconds=10))
    #print("anode", anode)

    tempdataslice = np.array(list(
        np.float32(ELS_backgroundremoval(elsdata, els_slicenumber_start, els_slicenumber_end, datatype="data")[
                   els_lowerenergyslice:els_upperenergyslice, anode, :])))
    tempx = list(elscalib['earray'][els_lowerenergyslice:els_upperenergyslice])

    # fig, ax = plt.subplots()
    #
    # CS = ax.pcolormesh(elsdata['times_utc'][els_slicenumber_start:els_slicenumber_end], tempx, tempdataslice,norm=LogNorm(vmin=1e8, vmax=1e12),
    #                    cmap='viridis')
    # ax.set_yscale("log")



    significance_dict = {'t16' : 1e4, 't17' : 2e4, 't19' : 2e4, 't21' : 1e3, 't23' : 2e4, 't25' : 1e4, 't26' : 2e4,
                         't27' : 2e4, 't28' : 1e4, 't29' : 1e4, 't30' : 2e4, 't32' : 5e3, 't36' : 2e4, 't39' : 2e4,
                         't40' : 2e4, 't41' : 2e4, 't42' : 2e4, 't43' : 2e4, 't48' : 2e4, 't49' : 2e4, 't50' : 2e4,
                         't51' : 2e4, 't71' : 2e4, 't83' : 2e4}

    detected_peaks = detect_peaks(tempdataslice)
    # print(detected_peaks)
    peakvalue_2darray = np.where(detected_peaks, tempdataslice, 0)
    peakvalue_2darray[peakvalue_2darray < significance_dict[elsdata['flyby']]] = 0
    #print("2d array shape", peakvalue_2darray.shape)
    peakvalue_indices = np.array(np.argwhere(peakvalue_2darray > significance_dict[elsdata['flyby']]), dtype=int)
    # peakvalue_list = peakvalue_2darray[peakvalue_indices]
    #print("old", peakvalue_indices)
    # print("tempdataslice", tempdataslice.shape)

    # if numofflybys == 1:
    #     fig1, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.imshow(np.flip(tempdataslice, axis=0))
    #     ax2.imshow(np.flip(peakvalue_2darray, axis=0))
    #     fig1.subplots_adjust(wspace=0.05)
    #plt.show()

    newpeakvalue_indices = []
    for counter, peakspair in enumerate(peakvalue_indices):
        #print("Value", peakvalue_2darray[peakspair[0],peakspair[1]])
        if peakspair[0] == 0 or peakspair[0] == (peakvalue_2darray.shape[0]-1):
            continue
        elif peakspair[1] == 0 or peakspair[1] == (peakvalue_2darray.shape[1]-1):
            continue
        else:
            newpeakvalue_indices.append(list(peakspair))

    peakvalue_indices = np.array(newpeakvalue_indices)
    #print("Removed edges", peakvalue_indices)
    print(peakvalue_indices, len(peakvalue_indices))

    if len(peakvalue_indices) < len(els_masses):
        energydict = ELS_smallenergybound_dict[elsdata['flyby']]
        els_masses = els_masses[:2]

    els_lowerenergyslice = CAPS_energyslice("els", energydict[0], energydict[0])[0]
    els_upperenergyslice = CAPS_energyslice("els", energydict[1], energydict[1])[0]
    tempdataslice = np.array(list(
        np.float32(ELS_backgroundremoval(elsdata, els_slicenumber_start, els_slicenumber_end, datatype="data")[
                   els_lowerenergyslice:els_upperenergyslice, anode, :])))
    tempx = list(elscalib['earray'][els_lowerenergyslice:els_upperenergyslice])


    #Finished tidying

    midpoints = [int(i) for i in np.convolve(peakvalue_indices[:, 0], np.ones(2) / 2, mode='valid')]
    # print(midpoints)

    merged_dataslice = []
    for counter, peakspair in enumerate(peakvalue_indices):
        #print(counter)
        if counter == 0:
            merged_dataslice += list(tempdataslice[0:midpoints[counter], peakspair[1]])
            #print(tempx[0:midpoints[counter]])
        elif counter == len(peakvalue_indices) - 1:
            merged_dataslice += list(tempdataslice[midpoints[counter - 1]:, peakspair[1]])
            #print(tempx[midpoints[counter - 1]:])
        else:
            merged_dataslice += list(tempdataslice[midpoints[counter - 1]:midpoints[counter], peakspair[1]])
            #print(tempx[midpoints[counter - 1]:midpoints[counter]])
        # print("x",np.array(tempx)[midpoints[counter-1]:midpoints[counter]])
        # print("dataslice",tempdataslice[midpoints[counter-1]:midpoints[counter], peakspair[1]])
        # energyax.step(np.array(tempx)[peakspair[0]-3:peakspair[0]+4],tempdataslice[peakspair[0]-3:peakspair[0]+4,peakspair[1]], where='mid')
    # print(merged_dataslice)
    # while merged_dataslice[0] < 10:
    #     merged_dataslice.pop(0)
    #     tempx.pop(0)
    merged_dataslice = np.array(merged_dataslice)
    merged_dataslice[merged_dataslice <= 0] = 1



    newlowerenergyslice, newupperenergyslice = 0, -1
    noiselevel = 200
    while merged_dataslice[newlowerenergyslice] < noiselevel or np.mean(merged_dataslice[newlowerenergyslice:newlowerenergyslice+2]) < noiselevel:
        newlowerenergyslice += 1

    tempx = tempx[newlowerenergyslice:]
    merged_dataslice = merged_dataslice[newlowerenergyslice:]

    og_x = tempx
    og_dataslice = merged_dataslice

    # print(tempx,merged_dataslice)
    # print(len(tempx),len(merged_dataslice))
    out, init = total_fluxgaussian(np.array(tempx), merged_dataslice, els_masses, cassini_speed, titanaltitude, windspeed, lpvalue,
                             LP_offset_dict[elsdata['flyby']], temperature,
                             charge=-1,
                             FWHM=ELS_FWHM)
    if out.params['ionvelocity'].stderr is None:
        out.params['ionvelocity'].stderr = np.nan
    if out.params['scp'].stderr is None:
        out.params['scp'].stderr = np.nan
    GOF = np.mean((abs(out.best_fit - merged_dataslice) / merged_dataslice) * 100)

    if numofflybys == 1:
        energyfig, energyax = plt.subplots()
        energyax.step(og_x,og_dataslice, where='mid',color='b')
        for counter, peakspair in enumerate(peakvalue_indices):
            if counter == 0:
                energyax.step(np.array(tempx)[0:midpoints[counter] + 1],
                              merged_dataslice[0:midpoints[counter] + 1], where='mid',
                              color="C" + str(peakspair[1]),
                              label=elsdata['times_utc_strings'][els_slicenumber_start + peakspair[1]])
            elif counter == len(peakvalue_indices) - 1:
                energyax.step(np.array(tempx)[midpoints[counter - 1]:],
                              merged_dataslice[midpoints[counter - 1]:], where='mid',
                              color="C" + str(peakspair[1]),
                              label=elsdata['times_utc_strings'][els_slicenumber_start + peakspair[1]])
            else:
                energyax.step(np.array(tempx)[midpoints[counter - 1]:midpoints[counter] + 1],
                              merged_dataslice[midpoints[counter - 1]:midpoints[counter] + 1], where='mid',
                              color="C" + str(peakspair[1]),
                              label=elsdata['times_utc_strings'][els_slicenumber_start + peakspair[1]])
            # print("x",np.array(tempx)[midpoints[counter-1]:midpoints[counter]])
            # print("dataslice",tempdataslice[midpoints[counter-1]:midpoints[counter], peakspair[1]])
            # energyax.step(np.array(tempx)[peakspair[0]-3:peakspair[0]+4],tempdataslice[peakspair[0]-3:peakspair[0]+4,peakspair[1]], where='mid')
        energyax.set_yscale("log")
        energyax.set_xlim(1, 25)
        energyax.set_ylim(bottom=2e3)
        energyax.set_ylim(0.9 * min(merged_dataslice), 1.1 * max(merged_dataslice))
        energyax.set_ylabel("Counts [/s]", fontsize=16)
        energyax.tick_params(axis='both', which='major', labelsize=15)
        energyax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
        energyax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
        energyax.minorticks_on()
        energyax.set_xlabel("Energy [eV/q]", fontsize=14)
        energyax.set_xlim(1, 25)
        energyax.legend()

        energyax.set_title("Altitude = " + str(titanaltitude) + "km")

        energyax.plot(tempx, out.init_fit, 'b-', label='init fit')
        energyax.plot(tempx, out.best_fit, 'k-', label='best fit')
        for i in els_masses:
            energyax.plot(tempx,gaussian(tempx,out.params["mass"+str(i)+"_center"],out.params["mass"+str(i)+"_sigma"],out.params["mass"+str(i)+"_height"]),linestyle='--')

        energyax.text(0.6, 0.01,
                      "Ion wind = %2.0f ± %2.0f m/s" % (out.params['ionvelocity'], out.params['ionvelocity'].stderr),
                      transform=energyax.transAxes, fontsize=18)
        energyax.text(0.6, .05,
                      "ELS-derived S/C Potential = %2.2f ± %2.2f V" % (out.params['scp'], out.params['scp'].stderr),
                      transform=energyax.transAxes, fontsize=18)
        energyax.text(0.6, .09, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=energyax.transAxes, fontsize=18)
        # ax.text(0.8, .20, "Temp = %2.2f" % temperature, transform=ax.transAxes)
        energyax.text(0.6, .13, "Chi-square = %.2E" % out.chisqr, transform=energyax.transAxes, fontsize=18)
        energyax.text(0.6, .17, "My GOF = %2.0f %%" % GOF, transform=energyax.transAxes, fontsize=18)

    # energyvalues = np.array(tempx)[peakvalue_indices[:, 0]]
    # expectedenergies, expectedenergies_lp = [], []
    # for mass in els_masses[:len(energyvalues)]:
    #     expectedenergies.append((0.5 * (mass * AMU) * ((cassini_speed) ** 2)) / e)
    #     expectedenergies_lp.append(
    #         (0.5 * (mass * AMU) * ((cassini_speed) ** 2) - ((lpvalue) * e * -1) + (8 * k * temperature)) / e)
    # #print(energyvalues[:len(els_masses)], expectedenergies, expectedenergies_lp)
    # energyoffset = energyvalues[:len(els_masses)] - np.array(expectedenergies_lp)
    #print(energyoffset)


    # if len(els_masses[:len(energyvalues)]) > 2:
    #     z, cov = np.polyfit(x=np.array(els_masses[:len(energyvalues)]), y=energyoffset, deg=1, cov=True)
    #     #print(z)
    #     ionwindspeed = (z[0] * (e / AMU)) / (cassini_speed)
    #     ionwindspeed_err = (np.sqrt(np.diag(cov)[0]) * (e / AMU)) / (cassini_speed)
    #     print(elsdata['flyby'], " Ion wind velocity = %2.2f ± %2.2f m/s" % (ionwindspeed, ionwindspeed_err))
    #     if numofflybys == 1:
    #         fig, ax = plt.subplots()
    #         p = np.poly1d(z)
    #         ax.errorbar(els_masses[:len(energyvalues)], energyoffset, fmt='.')  # ,yerr=effectivescplist_errors)
    #         ax.plot(els_masses, p(els_masses))
    # else:
    #     print("Could not estimate covariance")



    # SCP calculation
    # scpvalues = []
    # for masscounter, mass in enumerate(masses):
    #     scpvalues.append((effectivescplist[masscounter] - ((mass * AMU * cassini_speed * ionwindspeed) / e))/charge)
    # print("scplist", scpvalues)
    # scp_mean = np.mean(scpvalues)
    # scp_err = np.std(scpvalues)

    # print(tempdataslice)
    # print(tempx)
    return out, GOF, lpvalue

def IBS_fluxfitting(ibsdata, tempdatetime, titanaltitude, lpdata, ibs_masses=[28, 40, 53, 66, 78, 91], numofflybys=1):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    slicenumber = CAPS_slicenumber(ibsdata, tempdatetime)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    print("interp lpvalue", lpvalue)

    lowerenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue)[0]
    upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue,
                                        IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue)[0]

    windspeed = IBS_init_ionvelocity_dict[ibsdata['flyby']]
    temperature = titan_linearfit_temperature(titanaltitude)

    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, 1, slicenumber]

    # print("old x", ibscalib['ibsearray'][lowerenergyslice:upperenergyslice])
    # print("new x", ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.035 )
    x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice]
    #x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.035
    # x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.073

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

    out = total_fluxgaussian(x, dataslice, ibs_masses, cassini_speed, windspeed, lpvalue,
                             LP_offset_dict[ibsdata['flyby']], temperature,
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
        #     ax.plot(x, out["mass" + str(mass) + '_'], '--', label=str(mass) + " amu/q")
        # ax.legend(loc='best')

    return out, GOF, lpvalue


def IBS_fluxfitting_2dfluxtest(ibsdata, tempdatetime, titanaltitude, windspeed, lpdata,  ibs_masses=[28, 40, 53, 66, 78, 91],
                               numofflybys=1):
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3

    slicewidth_dict = {'t16' : 5, 't17' : 5, 't19' : 5, 't21' : 5, 't23' : 5, 't25' : 5, 't26' : 5,
                         't27' : 5, 't28' : 5, 't29' : 5, 't30' : 5, 't32' : 5, 't36' : 5, 't39' : 5,
                         't40' : 5, 't41' : 5, 't42' : 5, 't43' : 5, 't48' : 5, 't49' : 5, 't50' : 5,
                         't51' : 5, 't71' : 5, 't83' : 5}

    ibs_slicenumber_start = CAPS_slicenumber(ibsdata, tempdatetime) - slicewidth_dict[ibsdata['flyby']]
    ibs_slicenumber_end = CAPS_slicenumber(ibsdata, tempdatetime) + slicewidth_dict[ibsdata['flyby']]
    #print(ibsdata['times_utc'][ibs_slicenumber_start], ibsdata['times_utc'][ibs_slicenumber_end])

    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])

    if titanaltitude < 1100:
        ibs_masses = [28, 40, 53, 66, 78, 91]
        lowerenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue,
                                            IBS_energybound_dict[ibsdata['flyby']][0] - lpvalue)[0]
        upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue,
                                            IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue)[0]
        significance_dict = {'t16': 2e4, 't17': 2e4, 't19': 2e4, 't21': 2e4, 't23': 2e4, 't25': 2e4, 't26': 2e4,
                             't27': 8e3, 't28': 2e4, 't29': 2e4, 't30': 2e4, 't32': 2e4, 't36': 2e4, 't39': 2e4,
                             't40': 2e4, 't41': 2e4, 't42': 2e4, 't43': 2e4, 't48': 2e4, 't49': 2e4, 't50': 2e4,
                             't51': 2e4, 't71': 2e4, 't83': 2e4}
    elif titanaltitude > 1100 and titanaltitude < 1300:
        ibs_masses = [17, 28, 40, 53, 66, 78, 91]
        lowerenergyslice = CAPS_energyslice("ibs", 2.8, 2.8)[0]
        upperenergyslice = CAPS_energyslice("ibs", IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue + 0.5,
                                            IBS_energybound_dict[ibsdata['flyby']][1] - lpvalue + 0.5)[0]
        significance_dict = {'t16': 2e4, 't17': 2e4, 't19': 2e4, 't21': 2e4, 't23': 2e4, 't25': 2e4, 't26': 2e4,
                             't27': 8e3, 't28': 2e4, 't29': 2e4, 't30': 2e4, 't32': 2e4, 't36': 2e4, 't39': 2e4,
                             't40': 2e4, 't41': 2e4, 't42': 2e4, 't43': 5e3, 't48': 2e4, 't49': 2e4, 't50': 2e4,
                             't51': 2e4, 't71': 2e4, 't83': 2e4}
    elif titanaltitude > 1300 and titanaltitude < 1450:
        ibs_masses = [17, 28, 40, 53, 66]
        lowerenergyslice = CAPS_energyslice("ibs", 2.8, 2.8)[0]
        upperenergyslice = CAPS_energyslice("ibs", 16, 16)[0]
        significance_dict = {'t16': 2e4, 't17': 2e4, 't19': 2e4, 't21': 2e4, 't23': 2e4, 't25': 2e4, 't26': 2e4,
                             't27': 8e3, 't28': 1e4, 't29': 2e4, 't30': 2e4, 't32': 2e4, 't36': 2e4, 't39': 2e4,
                             't40': 2e4, 't41': 2e4, 't42': 2e4, 't43': 5e3, 't48': 2e4, 't49': 2e4, 't50': 2e4,
                             't51': 2e4, 't71': 2e4, 't83': 2e4}
    elif titanaltitude > 1450 and titanaltitude < 1600:
        ibs_masses = [17, 28, 40]
        lowerenergyslice = CAPS_energyslice("ibs", 2.8, 2.8)[0]
        upperenergyslice = CAPS_energyslice("ibs", 12, 12)[0]
        significance_dict = {'t16': 2e4, 't17': 2e4, 't19': 1e4, 't21': 1e4, 't23': 2e4, 't25': 2e4, 't26': 2e4,
                             't27': 8e3, 't28': 2e4, 't29': 2e4, 't30': 2e4, 't32': 2e4, 't36': 2e4, 't39': 2e4,
                             't40': 2e4, 't41': 2e4, 't42': 2e4, 't43': 5e3, 't48': 2e4, 't49': 2e4, 't50': 2e4,
                             't51': 2e4, 't71': 2e4, 't83': 2e4}
    elif titanaltitude > 1600:
        ibs_masses = [17, 28]
        lowerenergyslice = CAPS_energyslice("ibs", 2.8, 2.8)[0]
        upperenergyslice = CAPS_energyslice("ibs", 10, 10)[0]
        significance_dict = {'t16': 2e4, 't17': 2e4, 't19': 1e4, 't21': 1e4, 't23': 2e4, 't25': 2e4, 't26': 2e4,
                             't27': 8e3, 't28': 2e4, 't29': 2e4, 't30': 2e4, 't32': 2e4, 't36': 2e4, 't39': 2e4,
                             't40': 2e4, 't41': 2e4, 't42': 2e4, 't43': 5e3, 't48': 2e4, 't49': 2e4, 't50': 2e4,
                             't51': 2e4, 't71': 2e4, 't83': 2e4}

    # print("old x", ibscalib['ibsearray'][lowerenergyslice:upperenergyslice])
    # print("new x", ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.035 )

    #x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.035
    #x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice] * 1.073


    temperature = titan_linearfit_temperature(titanaltitude)
    fan = 1

    x = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice]
    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, fan, ibs_slicenumber_start:ibs_slicenumber_end]



    # fig, ax = plt.subplots()
    #
    # CS = ax.pcolormesh(elsdata['times_utc'][els_slicenumber_start:els_slicenumber_end], tempx, tempdataslice,norm=LogNorm(vmin=1e8, vmax=1e12),
    #                    cmap='viridis')
    # ax.set_yscale("log")

    detected_peaks = detect_peaks(dataslice)
    # print(detected_peaks)
    peakvalue_2darray = np.where(detected_peaks, dataslice, 0)
    peakvalue_2darray[peakvalue_2darray < significance_dict[ibsdata['flyby']]] = 0
    peakvalue_indices = np.array(np.argwhere(peakvalue_2darray > significance_dict[ibsdata['flyby']]), dtype=int)
    # peakvalue_list = peakvalue_2darray[peakvalue_indices]
    #print(peakvalue_indices, len(peakvalue_indices))
    # print("tempdataslice", tempdataslice.shape)
    midpoints = [int(i) for i in np.convolve(peakvalue_indices[:, 0], np.ones(2) / 2, mode='valid')]
    #print(midpoints)


    dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice, fan, ibs_slicenumber_start:ibs_slicenumber_end]

    merged_dataslice = []
    for counter, peakspair in enumerate(peakvalue_indices):
        #print(counter,peakspair)
        if counter == 0:
            merged_dataslice += list(dataslice[0:midpoints[counter], peakspair[1]])
            #print(x[0:midpoints[counter]])
        elif counter == len(peakvalue_indices) - 1:
            merged_dataslice += list(dataslice[midpoints[counter - 1]:, peakspair[1]])
            #print(x[midpoints[counter - 1]:])
        else:
            merged_dataslice += list(dataslice[midpoints[counter - 1]:midpoints[counter], peakspair[1]])
            #print(x[midpoints[counter - 1]:midpoints[counter]])
        # print("x",np.array(tempx)[midpoints[counter-1]:midpoints[counter]])
        # print("dataslice",tempdataslice[midpoints[counter-1]:midpoints[counter], peakspair[1]])
        # energyax.step(np.array(tempx)[peakspair[0]-3:peakspair[0]+4],tempdataslice[peakspair[0]-3:peakspair[0]+4,peakspair[1]], where='mid')
    # print(merged_dataslice)
    merged_dataslice = np.array(merged_dataslice)
    merged_dataslice[merged_dataslice <= 0] = 1

    newlowerenergyslice, newupperenergyslice = 0, -1
    noiselevel = 1000
    while merged_dataslice[newlowerenergyslice] < noiselevel:
        newlowerenergyslice+=1
    while merged_dataslice[newupperenergyslice] < noiselevel:
        newupperenergyslice-=1
    #print("newlowerslice",newlowerenergyslice)
    #print("newupperslice", newupperenergyslice)

    #print(len(x),len(merged_dataslice))
    x = x[newlowerenergyslice:newupperenergyslice]
    merged_dataslice = merged_dataslice[newlowerenergyslice:newupperenergyslice]
    #print(len(x), len(merged_dataslice))

    # print(tempx,merged_dataslice)
    # print(len(tempx),len(merged_dataslice))
    out, init = total_fluxgaussian(np.array(x), merged_dataslice, ibs_masses, cassini_speed, titanaltitude, windspeed, lpvalue,
                             LP_offset_dict[ibsdata['flyby']], temperature,
                             charge=1,
                             FWHM=IBS_FWHM)
    if out.params['ionvelocity'].stderr is None:
        out.params['ionvelocity'].stderr = np.nan
    if out.params['scp'].stderr is None:
        out.params['scp'].stderr = np.nan
    GOF = np.mean((abs(out.best_fit - merged_dataslice) / merged_dataslice) * 100)

    if numofflybys == 1:
        energyfig, energyax = plt.subplots()
        energyax.step(np.array(x),merged_dataslice, where='mid',
                              color="g", label="Merged dataslice -" +str(tempdatetime))
        #for counter, peakspair in enumerate(peakvalue_indices):
            # if counter == 0:
            #     energyax.step(np.array(x)[0:midpoints[counter] + 1],
            #                   dataslice[0:midpoints[counter] + 1, peakspair[1]], where='mid',
            #                   color="C" + str(peakspair[1]),
            #                   label=ibsdata['times_utc_strings'][ibs_slicenumber_start + peakspair[1]])
            # elif counter == len(peakvalue_indices) - 1:
            #     energyax.step(np.array(x)[midpoints[counter - 1]:],
            #                   dataslice[midpoints[counter - 1]:, peakspair[1]], where='mid',
            #                   color="C" + str(peakspair[1]),
            #                   label=ibsdata['times_utc_strings'][ibs_slicenumber_start + peakspair[1]])
            # else:
            #     energyax.step(np.array(x)[midpoints[counter - 1]:midpoints[counter] + 1],
            #                   dataslice[midpoints[counter - 1]:midpoints[counter] + 1, peakspair[1]], where='mid',
            #                   color="C" + str(peakspair[1]),
            #                   label=ibsdata['times_utc_strings'][ibs_slicenumber_start + peakspair[1]])
            # print("x",np.array(tempx)[midpoints[counter-1]:midpoints[counter]])
            # print("dataslice",tempdataslice[midpoints[counter-1]:midpoints[counter], peakspair[1]])
            # energyax.step(np.array(tempx)[peakspair[0]-3:peakspair[0]+4],tempdataslice[peakspair[0]-3:peakspair[0]+4,peakspair[1]], where='mid')
        energyax.set_yscale("log")
        energyax.set_xlim(1, 25)
        energyax.set_ylim(bottom=2e3)
        energyax.set_ylim(0.9 * min(merged_dataslice), 1.1 * max(merged_dataslice))
        energyax.set_ylabel("Counts [/s]", fontsize=16)
        energyax.tick_params(axis='both', which='major', labelsize=15)
        energyax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
        energyax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
        energyax.minorticks_on()
        energyax.set_xlabel("Energy [eV/q]", fontsize=14)
        energyax.set_xlim(1, 25)
        energyax.legend()



        energyax.plot(x, out.init_fit, 'b-', label='init fit')
        energyax.plot(x, out.best_fit, 'k-', label='best fit')

        for i in ibs_masses:
            energyax.plot(x,gaussian(x,out.params["mass"+str(i)+"_center"],out.params["mass"+str(i)+"_sigma"],out.params["mass"+str(i)+"_height"]))

        energyax.text(0.6, 0.01,
                      "Ion wind = %2.0f ± %2.0f m/s" % (out.params['ionvelocity'], out.params['ionvelocity'].stderr),
                      transform=energyax.transAxes, fontsize=18)
        energyax.text(0.6, .05,
                      "IBS-derived S/C Potential = %2.2f ± %2.2f V" % (out.params['scp'], out.params['scp'].stderr),
                      transform=energyax.transAxes, fontsize=18)
        energyax.text(0.6, .09, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=energyax.transAxes, fontsize=18)
        # ax.text(0.8, .20, "Temp = %2.2f" % temperature, transform=ax.transAxes)
        energyax.text(0.6, .13, "Chi-square = %.2E" % out.chisqr, transform=energyax.transAxes, fontsize=18)
        energyax.text(0.6, .17, "My GOF = %2.0f %%" % GOF, transform=energyax.transAxes, fontsize=18)



    #print("ibsdata",ibsdata['ibsdata'].shape)
    single_dataslice = ibsdata['ibsdata'][lowerenergyslice:upperenergyslice+5, fan, ibs_slicenumber_start+peakvalue_indices[-1][1]]
    x2 = ibscalib['ibsearray'][lowerenergyslice:upperenergyslice+5]
    # if numofflybys == 1:
    #     energyax.step(np.array(x2),single_dataslice, where='mid',color="c", label="Single dataslice")
    #print(single_dataslice)
    peaks, properties = find_peaks(single_dataslice, height=1e4, prominence=1e4, width=0.2, rel_height=0.5,
                                   distance=5)
    #print("Peaks",peaks)
    energyvalues = np.array(x2)[peaks]
    expectedenergies, expectedenergies_lp = [], []
    for mass in ibs_masses[:len(energyvalues)]:
        expectedenergies.append((0.5 * (mass * AMU) * ((cassini_speed) ** 2)) / e)
        expectedenergies_lp.append(
            (0.5 * (mass * AMU) * ((cassini_speed) ** 2) - ((lpvalue) * e) + (8 * k * temperature)) / e)
    #print(energyvalues[:len(ibs_masses)], expectedenergies, expectedenergies_lp)
    energyoffset = energyvalues[:len(ibs_masses)] - np.array(expectedenergies_lp)
    #print(energyoffset)

    #if numofflybys == 1:
        # fig1, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(np.flip(dataslice, axis=0))
        # ax2.imshow(np.flip(peakvalue_2darray, axis=0))
        # fig1.subplots_adjust(wspace=0.05)

        # if len(ibs_masses) > 3:
        #     z, cov = np.polyfit(x=np.array(ibs_masses[:len(energyvalues)]), y=energyoffset, deg=1, cov=True)
        #     #print(z)
        #     ionwindspeed = (z[0] * (e / AMU)) / (cassini_speed)
        #     ionwindspeed_err = (np.sqrt(np.diag(cov)[0]) * (e / AMU)) / (cassini_speed)
        #     #print(ibsdata['flyby'], " Ion wind velocity = %2.2f ± %2.2f m/s" % (ionwindspeed, ionwindspeed_err))
        #
        #     if numofflybys == 1:
        #         fig, ax = plt.subplots()
        #         p = np.poly1d(z)
        #         ax.errorbar(ibs_masses[:len(energyvalues)], energyoffset, fmt='.')  # ,yerr=effectivescplist_errors)
        #         ax.plot(ibs_masses, p(ibs_masses))

    # SCP calculation
    # scpvalues = []
    # for masscounter, mass in enumerate(masses):
    #     scpvalues.append((effectivescplist[masscounter] - ((mass * AMU * cassini_speed * ionwindspeed) / e))/charge)
    # print("scplist", scpvalues)
    # scp_mean = np.mean(scpvalues)
    # scp_err = np.std(scpvalues)

    # print(tempdataslice)
    # print(tempx)
    return out, GOF, lpvalue, cassini_speed


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def multiple_alongtrackwinds_flybys(usedflybys):
    outputdf = pd.DataFrame()
    for flyby in usedflybys:
        print("Flyby", flyby)
        tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
        elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")
        ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
        generate_aligned_ibsdata(ibsdata, elsdata, flyby)
        ibs_outputs, ibs_datetimes, ibs_GOFvalues, lpvalues, cassini_speeds = [], [], [], [],[]
        els_outputs, els_datetimes, els_GOFvalues, els_velocities, els_scp, els_velocities_stderr, els_scp_stderr  = [], [], [], [], [], [], []

        all_possible_masses = [16, 17, 28, 40, 53, 66, 78, 91]
        mass_energies_dict = {}
        for mass in all_possible_masses:
            mass_energies_dict["ibsmass"+str(mass)] = []

        zonalangles, zonalwinds_2016, zonalwinds_2017 = [], [], []
        x_titan, y_titan, z_titan, dx_titan, dy_titan, dz_titan = [], [], [], [], [], []
        xx_attitude, xy_attitude, xz_attitude, yx_attitude, yy_attitude, yz_attitude, zx_attitude, zy_attitude, zz_attitude = [],[],[],[],[],[],[],[],[]
        lptimes = list(tempdf['Positive Peak Time'])
        lpdata = read_LP_V1(flyby)
        positive_windspeed = IBS_init_ionvelocity_dict[ibsdata['flyby']]
        negative_windspeed = ELS_init_ionvelocity_dict[elsdata['flyby']]

        for (i, j, k) in zip(tempdf['Positive Peak Time'], tempdf['Negative Peak Time'], tempdf['Altitude']):
            # ibs_output, ibs_GOFvalue, lpvalue = IBS_fluxfitting(ibsdata, i, k, lpdata=lpdata,
            #                                                     # ibs_masses=IBS_smallmasses_dict[flyby],
            #                                                     numofflybys=len(
            #                                                         usedflybys))
            ibs_output, ibs_GOFvalue, lpvalue, cassini_speed = IBS_fluxfitting_2dfluxtest(ibsdata, i, k, positive_windspeed, lpdata=lpdata,
                                                                # ibs_masses=IBS_smallmasses_dict[flyby],
                                                                numofflybys=len(
                                                                    usedflybys))
            positive_windspeed = ibs_output.params['ionvelocity'].value

            #print(ibs_output.params.keys())
            for mass in all_possible_masses:
                if 'mass' + str(mass) + "_center" in ibs_output.params.keys():
                    mass_energies_dict["ibsmass"+str(mass)].append(ibs_output.params['mass' + str(mass) + "_center"].value)
                else:
                    mass_energies_dict["ibsmass" + str(mass)].append(np.NaN)

            # els_output, els_GOFvalue, temp = ELS_fluxfitting(elsdata, j, k, lpdata=lpdata,
            #                                                  els_masses=ELS_smallmasses_dict[flyby],
            #                                                  numofflybys=len(
            #                                                      usedflybys))
            #print(j,type(j))
            if not pd.isnull(j):
                els_output, els_GOFvalue, temp = ELS_fluxfitting_2dfluxtest(elsdata, j, k, windspeed=positive_windspeed,
                                           lpdata=lpdata, numofflybys=len(
                                                                     usedflybys))
                els_velocities.append(els_output.params['ionvelocity'].value)
                els_velocities_stderr.append(els_output.params['ionvelocity'].stderr)
                els_scp.append(els_output.params['scp'].value)
                els_scp_stderr.append(els_output.params['scp'].stderr)
                els_GOFvalues.append(els_GOFvalue)
            else:
                els_velocities.append(np.NaN)
                els_velocities_stderr.append(np.NaN)
                els_scp.append(np.NaN)
                els_scp_stderr.append(np.NaN)
                els_GOFvalues.append(np.NaN)

            ibs_outputs.append(ibs_output)
            ibs_GOFvalues.append(ibs_GOFvalue)
            lpvalues.append(lpvalue)
            cassini_speeds.append(cassini_speed)

            et = spice.datetime2et(i)
            alt, lat, lon = cassini_titan_altlatlon(i)
            state, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "NONE", "titan")
            x_titan.append(state[0])
            y_titan.append(state[1])
            z_titan.append(state[2])
            dx_titan.append(state[3])
            dy_titan.append(state[4])
            dz_titan.append(state[5])
            sclk = spice.sce2c(-82,et)
            output_attitude = spice.ckgp(-82000,sclk,0.0,"J2000")
            for attitude, attitudelist in zip(output_attitude[0].flatten(),[xx_attitude, xy_attitude, xz_attitude, yx_attitude, yy_attitude, yz_attitude, zx_attitude, zy_attitude, zz_attitude]):
                attitudelist.append(attitude)

            ramdir = spice.vhat(state[3:6])
            titandir, state = titan_dir(i)
            titandir_unorm = spice.vhat(titandir)
            parallel_to_titan = spice.rotvec(titandir_unorm, 90 * spice.rpd(), 3)
            parallel_to_titan_noZ = [parallel_to_titan[0], parallel_to_titan[1], 0]

            angle_2_zonal = spice.dpr() * spice.vsep(ramdir, parallel_to_titan_noZ)
            zonalangles.append(angle_2_zonal)
            zonalwinds_2016.append(-gaussian(lat, 0, 70 / 2.355, 373)*np.cos(angle_2_zonal*spice.rpd()))
            zonalwinds_2017.append(-gaussian(lat, 3, 101 / 2.355, 196)*np.cos(angle_2_zonal*spice.rpd()))




        # testoutputdf = pd.DataFrame()
        # testoutputdf['Bulk Time'] = tempdf['Bulk Time']
        # testoutputdf['IBS Alongtrack velocity'] = [i.params['windspeed'] for i in ibs_fits]
        # # testoutputdf['IBS residuals'] = ibs_residuals
        # testoutputdf['IBS spacecraft potentials'] = [i.params['scp'] for i in ibs_fits]
        # testoutputdf.to_csv("testalongtrackvelocity.csv")

        if len(usedflybys) == 1:
            fig5, (windaxes, potaxes) = plt.subplots(2, sharex='all')
            #fig5.suptitle(usedflybys[0])
            # windaxes.errorbar(tidied_ibs_datetimes_flat, [i.params['ionvelocity'].value for i in tidied_ibs_outputs_flat],
            #                   yerr=[i.params['ionvelocity'].stderr for i in tidied_ibs_outputs_flat],
            #                   label="IBS - Ion Wind Speeds", marker='.', ls='none', color='C0')
            # windaxes.errorbar(tidied_els_datetimes_flat, [i.params['ionvelocity'].value for i in tidied_els_outputs_flat],
            #                   yerr=[i.params['ionvelocity'].stderr for i in tidied_els_outputs_flat],
            #                  label="ELS - Ion Wind Speeds",
            #                  marker='x', ls='none',  color='C1')
            windaxes.errorbar(tempdf['Positive Peak Time'], [i.params['ionvelocity'].value for i in ibs_outputs], yerr=[i.params['ionvelocity'].stderr for i in ibs_outputs],
                          label="IBS", marker='.', color='C0')
            print([i.params['ionvelocity'].value for i in els_outputs], [i.params['ionvelocity'].stderr for i in els_outputs])
            windaxes.errorbar(tempdf['Negative Peak Time'], els_velocities, yerr=els_velocities_stderr,
                          label="ELS",
                          marker='x', color='C1')

            windaxes.legend()
            windaxes.set_ylabel("Derived Ion Velocity [m/s]",fontsize=18)
            windaxes.hlines([0], min(tempdf['Positive Peak Time']), max(tempdf['Positive Peak Time']),
                            color='k',linestyle='--',alpha=0.5)
            # windaxes.set_ylim(-525, 525)

            # potaxes.errorbar(tidied_ibs_datetimes_flat, [i.params['scp'].value for i in tidied_ibs_outputs_flat],
            #                  yerr=[i.params['scp'].stderr for i in tidied_ibs_outputs_flat],
            #                  label="IBS - Derived S/C Potential", marker='.', ls='none', color='C0')
            # potaxes.errorbar(tidied_els_datetimes_flat, [i.params['scp'].value for i in tidied_els_outputs_flat],
            #                 yerr=[i.params['scp'].stderr for i in tidied_els_outputs_flat],
            #                 label="ELS - Derived S/C Potential",
            #                 marker='x', ls='none',  color='C1')
            potaxes.errorbar(tempdf['Positive Peak Time'], [i.params['scp'].value for i in ibs_outputs],yerr=[i.params['scp'].stderr for i in ibs_outputs],
                         label="IBS", marker='.', color='C0')
            potaxes.errorbar(tempdf['Negative Peak Time'], els_scp,yerr=els_scp_stderr,
                         label="ELS",
                         marker='x', color='C1')
            potaxes.plot(lptimes, lpvalues, label="LP", color='C8')

            potaxes.legend()
            potaxes.set_ylabel("Derived S/C Potential [V]",fontsize=18)
            # potaxes.plot(lptimes, np.array(lpvalues) - 0.5, color='k')
            # potaxes.plot(lptimes, np.array(lpvalues) + 0.5, color='k')
            # potaxes.hlines([np.mean(lpvalues) - 2, 0], min(tempdf['Positive Peak Time']),
            #                max(tempdf['Positive Peak Time']), color='k')
            potaxes.set_ylim(np.mean(lpvalues) - 1.5, 0.1)

            # GOFaxes.scatter(tempdf['Positive Peak Time'], ibs_GOFvalues, label="IBS - GOF", color='C0')
            # GOFaxes.set_ylabel("Goodness of Fit - IBS")
            # GOFaxes.set_xlabel("Time")
            # GOFaxes_els = GOFaxes.twinx()
            # GOFaxes_els.scatter(tempdf['Negative Peak Time'], els_GOFvalues, label="ELS - GOF", color='C1',
            #                     marker='x')
            # GOFaxes_els.set_ylabel("Goodness of Fit - ELS")

            potaxes.tick_params(top=True, right=True, labeltop=True, labelright=True,labelsize=12)
            windaxes.tick_params(top=True, right=True,labeltop=True, labelright=True,labelsize=12)
            windaxes.set_xlim(min(tempdf['Positive Peak Time']), max(tempdf['Positive Peak Time']))

        # print(tempdf['Positive Peak Time'])
        # elsstartslice = CAPS_slicenumber(elsdata,tempdf['Positive Peak Time'].iloc[0])
        # elsendslice = CAPS_slicenumber(elsdata,tempdf['Positive Peak Time'].iloc[-1])
        # actaxes.plot(elsdata['times_utc'][elsstartslice:elsendslice],elsdata['actuator'][elsstartslice:elsendslice])
        # actaxes.set_ylabel("Actuator position")

        tempoutputdf = pd.DataFrame()
        tempoutputdf['Flyby'] = tempdf['Flyby']
        tempoutputdf['Flyby velocity'] = cassini_speeds
        tempoutputdf['Positive Peak Time'] = tempdf['Positive Peak Time']
        tempoutputdf['Negative Peak Time'] = tempdf['Negative Peak Time']
        tempoutputdf['X Titan'] = x_titan
        tempoutputdf['Y Titan'] = y_titan
        tempoutputdf['Z Titan'] = z_titan
        tempoutputdf['DX Titan'] = dx_titan
        tempoutputdf['DY Titan'] = dy_titan
        tempoutputdf['DZ Titan'] = dz_titan
        tempoutputdf["xx_attitude"] = xx_attitude
        tempoutputdf["xy_attitude"] = xy_attitude
        tempoutputdf["xz_attitude"] = xz_attitude
        tempoutputdf["yx_attitude"] = yx_attitude
        tempoutputdf["yy_attitude"] = yy_attitude
        tempoutputdf["yz_attitude"] = yz_attitude
        tempoutputdf["zx_attitude"] = zx_attitude
        tempoutputdf["zy_attitude"] = zy_attitude
        tempoutputdf["zz_attitude"] = zz_attitude
        tempoutputdf['IBS alongtrack velocity'] = [i.params['ionvelocity'].value for i in ibs_outputs]
        tempoutputdf['IBS alongtrack velocity stderr'] = [i.params['ionvelocity'].stderr for i in ibs_outputs]
        tempoutputdf['IBS spacecraft potentials'] = [i.params['scp'].value for i in ibs_outputs]
        tempoutputdf['IBS spacecraft potentials stderr'] = [i.params['scp'].stderr for i in ibs_outputs]
        tempoutputdf['ELS alongtrack velocity'] = els_velocities
        tempoutputdf['ELS alongtrack velocity stderr'] = els_velocities_stderr
        tempoutputdf['ELS spacecraft potentials'] = els_scp
        tempoutputdf['ELS spacecraft potentials stderr'] = els_scp_stderr
        tempoutputdf['LP Potentials'] = lpvalues
        tempoutputdf['Angles to Zonal Wind'] = zonalangles
        tempoutputdf['2016 Zonal Winds'] = zonalwinds_2016
        tempoutputdf['2017 Zonal Winds'] = zonalwinds_2017
        for i in all_possible_masses:
            tempoutputdf['IBS Mass ' + str(i) +' energy'] = mass_energies_dict["ibsmass"+str(i)]
        #print(tempoutputdf)
        outputdf = pd.concat([outputdf, tempoutputdf])
    if len(usedflybys) != 1:
        outputdf.to_csv("alongtrackvelocity.csv")
    if len(usedflybys) == 1:
        outputdf.to_csv("singleflyby_alongtracktest.csv")


def single_slice_test(flyby, slicenumber):
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)
    tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]

    #print("Altitude", tempdf['Altitude'].iloc[slicenumber])
    #print(tempdf['Positive Peak Time'].iloc[slicenumber])
    lpdata = read_LP_V1(flyby)

    # ibs_out, ibs_GOF, lpvalue = IBS_fluxfitting(ibsdata, tempdf['Positive Peak Time'].iloc[slicenumber],
    #                                             tempdf['Altitude'].iloc[slicenumber],# ibs_masses=IBS_smallmasses_dict[flyby],
    #                                            lpdata=lpdata)
    IBS_fluxfitting_2dfluxtest(ibsdata, tempdf['Positive Peak Time'].iloc[slicenumber],
                                                tempdf['Altitude'].iloc[slicenumber],# ibs_masses=IBS_smallmasses_dict[flyby],
                                                lpdata=lpdata)

    # els_out, els_GOF, lpvalue = ELS_fluxfitting(elsdata, tempdf['Negative Peak Time'].iloc[slicenumber],
    #                                             tempdf['Altitude'].iloc[slicenumber], els_masses=ELS_masses_dict[flyby],
    #                                             lpdata=lpdata)
    # #plt.show()
    if not pd.isnull(tempdf['Negative Peak Time'].iloc[slicenumber]):
        ELS_fluxfitting_2dfluxtest(elsdata, tempdf['Negative Peak Time'].iloc[slicenumber],
                                   tempdf['Altitude'].iloc[slicenumber],
                                   lpdata=lpdata)


def cassini_titan_altlatlon(tempdatetime):
    et = spice.datetime2et(tempdatetime)

    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    lon, lat, alt = spice.recpgr('TITAN', state[:3], spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    return alt, lat * spice.dpr(), lon * spice.dpr()


def titan_dir(i):  # Only use for one flyby
    et = spice.datetime2et(i)
    titandir, ltime = spice.spkpos('TITAN', et, 'IAU_TITAN', "LT+S", 'CASSINI')
    state = cassini_phase(i.strftime('%Y-%m-%dT%H:%M:%S'))

    return titandir, state#, parallel_to_surface

def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def non_actuating(flyby, tempdatetime):
    elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
    generate_aligned_ibsdata(ibsdata, elsdata, flyby)

    alt, lat, lon = cassini_titan_altlatlon(tempdatetime)
    lpdata = read_LP_V1(flyby)

    ibs_out, ibs_GOF, lpvalue = IBS_fluxfitting(ibsdata, tempdatetime, alt,
                                                lpdata=lpdata)
    els_out, els_GOF, lpvalue = ELS_fluxfitting(elsdata, tempdatetime, alt, els_masses=ELS_smallmasses_dict[flyby],
                                                lpdata=lpdata)


#non_actuating_test("ta", datetime.datetime(2004,10,26,15, 31, 27))
#single_slice_test("t43", slicenumber=3)
#multiple_alongtrackwinds_flybys(["t36"])
#multiple_alongtrackwinds_flybys(['t27'])
# multiple_alongtrackwinds_flybys(
#     ['t16', 't17', 't20', 't21', 't25', 't26', 't27', 't28', 't29', 't30', 't32', 't42', 't46'])
# multiple_alongtrackwinds_flybys(
#     ['t36','t48','t49','t50','t51','t71','t83'])


multiple_alongtrackwinds_flybys(['t16', 't17', 't19', 't21', 't23', 't25', 't26', 't27', 't28', 't29', 't30', 't32', 't36', 't39', 't40',
              't41', 't42', 't43', 't48','t49','t50','t51','t71','t83'])
# plt.close("all")
plt.show()
