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

from symfit.core.objectives import HessianObjective
from symfit.core.minimizers import LBFGSB
from symfit.core.support import keywordonly
from symfit import parameters, variables, Fit, Model, Parameter, exp
from symfit.distributions import Gaussian

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

import datetime
from util import *

tstart = time.time()

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


class LeastSquaresPercentageObjective(HessianObjective):
    @keywordonly(flatten_components=True)
    def __call__(self, ordered_parameters=[], **parameters):
        """
        :param ordered_parameters: See ``parameters``.
        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`S` at.
        :param flatten_components: if `True`, return the total :math:`S`. If
            `False`, return the :math:`S` per component of the
            :class:`~symfit.core.models.BaseModel`.
        :return: scalar or list of scalars depending on the value of `flatten_components`.
        """
        flatten_components = parameters.pop('flatten_components')
        evaluated_func = super(LeastSquaresPercentageObjective, self).__call__(
            ordered_parameters, **parameters
        )

        chi2 = [0 for _ in evaluated_func]
        for index, (dep_var, dep_var_value) in enumerate(zip(self.model.dependent_vars, evaluated_func)):
            dep_data = self.dependent_data.get(dep_var, None)

            if dep_data is not None:

                sigma = self.sigma_data[self.model.sigmas[dep_var]]
                #print((dep_var_value - dep_data), (dep_var_value - dep_data) / dep_data)
                chi2[index] += np.sum(abs((dep_var_value - dep_data) / dep_data) / abs(sigma))
                #chi2[index] += np.sum((dep_var_value - dep_data) ** 2 / sigma ** 2)
                #if len(dep_var_value) == 17:
                    # print("normal",((dep_var_value - dep_data) ** 2 / sigma ** 2))
                    # print("Percent",((dep_var_value - dep_data) / dep_data) ** 2 / sigma ** 2)
        chi2 = np.sum(chi2) if flatten_components else chi2
        return chi2 / 2

    def eval_jacobian(self, ordered_parameters=[], **parameters):
        """
        Jacobian of :math:`S` in the
        :class:`~symfit.core.argument.Parameter`'s (:math:`\\nabla_\\vec{p} S`).

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\\nabla_\\vec{p} S` at.
        :return: ``np.array`` of length equal to the number of parameters..
        """
        evaluated_func = super(LeastSquaresPercentageObjective, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(LeastSquaresPercentageObjective, self).eval_jacobian(
            ordered_parameters, **parameters
        )

        result = 0
        for var, f, jac_comp in zip(self.model.dependent_vars, evaluated_func,
                                    evaluated_jac):
            y = self.dependent_data.get(var, None)
            sigma_var = self.model.sigmas[var]
            if y is not None:
                sigma = self.sigma_data[sigma_var]
                pre_sum = jac_comp * ((y - f) / sigma ** 2)[np.newaxis, ...]
                axes = tuple(range(1, len(pre_sum.shape)))
                result -= np.sum(pre_sum, axis=axes, keepdims=False)
        return np.atleast_1d(np.squeeze(np.array(result)))

    def eval_hessian(self, ordered_parameters=[], **parameters):
        """
        Hessian of :math:`S` in the
        :class:`~symfit.core.argument.Parameter`'s (:math:`\\nabla_\\vec{p}^2 S`).

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\\nabla_\\vec{p} S` at.
        :return: ``np.array`` of length equal to the number of parameters..
        """
        evaluated_func = super(LeastSquaresPercentageObjective, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(LeastSquaresPercentageObjective, self).eval_jacobian(
            ordered_parameters, **parameters
        )
        evaluated_hess = super(LeastSquaresPercentageObjective, self).eval_hessian(
            ordered_parameters, **parameters
        )

        result = 0
        for var, f, jac_comp, hess_comp in zip(self.model.dependent_vars,
                                               evaluated_func, evaluated_jac,
                                               evaluated_hess):
            y = self.dependent_data.get(var, None)
            sigma_var = self.model.sigmas[var]
            if y is not None:
                sigma = self.sigma_data[sigma_var]
                p1 = hess_comp * ((y - f) / sigma ** 2)[np.newaxis, np.newaxis, ...]
                # Outer product
                p2 = np.einsum('i...,j...->ij...', jac_comp, jac_comp)
                p2 = p2 / sigma[np.newaxis, np.newaxis, ...] ** 2
                # We sum away everything except the matrices in the axes 0 & 1.
                axes = tuple(range(2, len(p2.shape)))
                result += np.sum(p2 - p1, axis=axes, keepdims=False)
        return np.atleast_2d(np.squeeze(np.array(result)))


def IBS_ELS_gaussian(ibs_x, ibs_dataslice, els_x, els_dataslice, cassini_speed, lpvalue, temperature):
    x_1, x_2, y_1, y_2 = variables('x_1, x_2, y_1, y_2')

    k_e = Parameter("k_e", value=scipy.constants.physical_constants['Boltzmann constant'][0] /
                                 scipy.constants.physical_constants['atomic unit of charge'][0])
    k_e.fixed = True
    AMU_e = Parameter("AMU_e", value=scipy.constants.physical_constants['atomic mass constant'][0] /
                                     scipy.constants.physical_constants['atomic unit of charge'][0])
    AMU_e.fixed = True

    # Physical parameters
    sc_velocity = Parameter("sc_velocity", value=cassini_speed)
    sc_velocity.fixed = True
    temp_eV = Parameter("temp_eV",
                        value=(8 * scipy.constants.physical_constants['Boltzmann constant'][0] * temperature) /
                              scipy.constants.physical_constants['atomic unit of charge'][0])
    temp_eV.fixed = True
    lp_pot = Parameter("lp_pot", value=lpvalue - 0.5, max=lpvalue, min=-2.5)

    ionvelocity = Parameter("ionvelocity", value=-347)

    # Negative Ion Parameters
    mass26_neg_amp = Parameter("mass26_neg_amp", value=0.5)
    mass50_neg_amp = Parameter("mass50_neg_amp", value=0.5)
    mass74_neg_amp = Parameter("mass74_neg_amp", value=0.5)
    mass117_neg_amp = Parameter("mass117_neg_amp", value=0.5)

    mass26_neg_sig = Parameter("mass26_neg_sig", value=0.3, min=0.2, max=0.7)
    mass50_neg_sig = Parameter("mass50_neg_sig", value=0.62, min=0.4, max=1)
    mass74_neg_sig = Parameter("mass74_neg_sig", value=0.9, min=0.7, max=1.5)
    mass117_neg_sig = Parameter("mass117_neg_sig", value=1.5, min=0.8, max=1.8)

    # Positive Ion Parameters
    mass28_amp = Parameter("mass28_amp", value=0.5)
    mass41_amp = Parameter("mass41_amp", value=0.5)
    mass53_amp = Parameter("mass53_amp", value=0.5)
    mass66_amp = Parameter("mass66_amp", value=0.5)
    mass78_amp = Parameter("mass78_amp", value=0.5)
    mass91_amp = Parameter("mass91_amp", value=0.5)

    mass28_sig = Parameter("mass28_sig", value=0.4, min=0.2, max=0.6)
    mass41_sig = Parameter("mass41_sig", value=0.5, min=0.2, max=0.6)
    mass53_sig = Parameter("mass53_sig", value=0.5, min=0.2, max=0.6)
    mass66_sig = Parameter("mass66_sig", value=0.5, min=0.2, max=0.6)
    mass78_sig = Parameter("mass78_sig", value=0.5, min=0.2, max=0.7)
    mass91_sig = Parameter("mass91_sig", value=0.5, min=0.2, max=0.7)

    model = Model({
        y_1: (mass26_neg_amp * exp(
            -(x_1 - (13 * AMU_e * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                    2. * mass26_neg_sig ** 2))) +
             (mass50_neg_amp * exp(
                 -(x_1 - (25 * AMU_e * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                         2. * mass50_neg_sig ** 2))) +
             (mass74_neg_amp * exp(
                 -(x_1 - (37 * AMU_e * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                         2. * mass74_neg_sig ** 2))) +
             (mass117_neg_amp * exp(
                 -(x_1 - (58.5 * AMU_e * ((sc_velocity + ionvelocity) ** 2) + lp_pot + temp_eV)) ** 2 / (
                         2. * mass117_neg_sig ** 2))),
        y_2: (mass28_amp * exp(
            -(x_2 - (14 * AMU_e * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                    2. * mass28_sig ** 2))) +
             (mass41_amp * exp(
                 -(x_2 - (20.5 * AMU_e * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                         2. * mass41_sig ** 2))) +
             (mass53_amp * exp(
                 -(x_2 - (26.5 * AMU_e * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                         2. * mass53_sig ** 2))) +
             (mass66_amp * exp(
                 -(x_2 - (33 * AMU_e * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                         2. * mass66_sig ** 2))) +
             (mass78_amp * exp(
                 -(x_2 - (39 * AMU_e * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                         2. * mass78_sig ** 2))) +
             (mass91_amp * exp(
                 -(x_2 - (45.5 * AMU_e * ((sc_velocity + ionvelocity) ** 2) - lp_pot + temp_eV)) ** 2 / (
                         2. * mass91_sig ** 2)))
    })

    # init_y  = model(x_1=els_x, x_2=ibs_x, **fit_result.params)
    # print(els_dataslice, ibs_dataslice)
    minimize_options = {"options":{"disp": True}}
    fit = Fit(model, x_1=els_x, x_2=ibs_x, y_1=els_dataslice, y_2=ibs_dataslice, objective=LeastSquaresPercentageObjective)

    fit_result = fit.execute(**minimize_options)
    print(fit_result)
    tend = time.time()
    print("Fit Run Time", tend - tstart)

    y = model(x_1=els_x, x_2=ibs_x, **fit_result.params)

    print("Negative Ions")
    for i in [26, 50, 74, 117]:
        negionenergy = (0.5 * (i * AMU_e.value) * ((sc_velocity.value + fit_result.params['ionvelocity']) ** 2) +
                        fit_result.params['lp_pot'] + temp_eV.value)
        print("mass" + str(i), "%2.2f" % negionenergy, "%2.2f" % fit_result.params['mass' + str(i) + '_neg_sig'],
              "%2.2f" % fit_result.params['mass' + str(i) + '_neg_amp'])

    print("Positive Ions")
    for i in [28, 41, 53, 66, 78, 91]:
        posionenergy = (0.5 * (i * AMU_e.value) * ((sc_velocity.value + fit_result.params['ionvelocity']) ** 2) -
                        fit_result.params['lp_pot'] + temp_eV.value)
        print("mass" + str(i), "%2.2f" % posionenergy, "%2.2f" % fit_result.params['mass' + str(i) + '_sig'],
              "%2.2f" % fit_result.params['mass' + str(i) + '_amp'])

    return fit_result, y, ionvelocity, lp_pot


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
    print("interp lpvalue", lpvalue)
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

    ibs_dataslice = np.array(ibsdata['ibsdata'][ibs_lowerenergyslice:ibs_upperenergyslice, 1, ibs_slicenumber])
    ibs_dataslice[ibs_dataslice <= 0] = 1
    ibs_x = ibscalib['ibsearray'][ibs_lowerenergyslice:ibs_upperenergyslice]
    scaled_ibs_dataslice = ibs_dataslice / max(ibs_dataslice)

    anode = ELS_maxflux_anode(elsdata, tempdatetime - datetime.timedelta(seconds=10),
                              tempdatetime + datetime.timedelta(seconds=10))
    print("anode", anode)
    els_dataslice = np.float32(ELS_backgroundremoval(elsdata, els_slicenumber, els_slicenumber + 1, datatype="data")[
                               els_lowerenergyslice:els_upperenergyslice, anode, 0])
    els_dataslice[els_dataslice <= 0] = 1
    print(elsdata['flyby'], "Cassini velocity", cassini_speed, "Altitude", titanaltitude)
    els_x = elscalib['earray'][els_lowerenergyslice:els_upperenergyslice]
    scaled_els_dataslice = els_dataslice / max(els_dataslice)

    fit_result, y, ionvelocity, lp_pot = IBS_ELS_gaussian(ibs_x, scaled_ibs_dataslice, els_x, scaled_els_dataslice, cassini_speed,
                                                          lpvalue,
                                                          temperature)

    stepplotfig, (elsax, ibsax) = plt.subplots(2, sharex='all')
    elsax.plot(els_x, y[0], color='r', label="ELS fit")
    elsax.step(els_x, scaled_els_dataslice, where='mid', color='k', label="ELS data")
    elsax.set_ylim(min(scaled_els_dataslice), max(scaled_els_dataslice) * 1.5)
    elsax.set_yscale("log")
    elsax.set_ylabel("Normalised to max flux bin \n in reduced energy range", fontsize=15)
    elsax.set_title("ELS data from " + elsdata['flyby'] + " " + str(tempdatetime), fontsize=24)
    elsax.legend()
    elsax.text(10, 0.015, "Ion velocity = %2.2f ± %2.2f m/s" % (fit_result.value(ionvelocity),
                                                                fit_result.stdev(ionvelocity)))
    elsax.text(10, 0.02, "CAPS S/C potential = %2.2f ± %2.2f V" % (
        fit_result.value(lp_pot), fit_result.stdev(lp_pot)))
    elsax.text(10, 0.025, "LP S/C potential = %2.2f V" % lpvalue)

    ibsax.plot(ibs_x, y[1], color='r', label="IBS fit")
    ibsax.step(ibs_x, scaled_ibs_dataslice, where='mid', color='k', label="IBS data")
    ibsax.set_yscale("log")
    ibsax.set_ylim(min(scaled_ibs_dataslice), max(scaled_ibs_dataslice) * 1.5)
    ibsax.set_xlabel("Energy (ev/q)", fontsize=20)
    ibsax.set_ylabel("Normalised to max flux bin \n in reduced energy range", fontsize=15)
    ibsax.set_title("IBS data from " + elsdata['flyby'] + " " + str(tempdatetime), fontsize=24)
    ibsax.legend()

    return fit_result, lpvalue, ionvelocity, lp_pot, cassini_speed


windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])
windsdf['Negative Peak Time'] = pd.to_datetime(windsdf['Negative Peak Time'])


def multiple_alongtrackwinds_flybys(usedflybys):
    times = []
    fit_results, lpvalues, ionwindspeeds, ionwindspeeds_errors, scps, scps_errors, cassini_speeds = [], [], [], [], [], [], []
    for flyby in usedflybys:
        tempdf = windsdf[windsdf['Flyby'] == flyby.lower()]
        times = times + list(tempdf['Positive Peak Time'])
        elsdata = readsav("data/els/elsres_" + filedates[flyby] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")
        ibsdata = readsav("data/ibs/ibsres_" + filedates[flyby] + ".dat")
        generate_aligned_ibsdata(ibsdata, elsdata, flyby)
        for (i, j) in zip(tempdf['Positive Peak Time'], tempdf['Altitude']):
            fit_result, lpvalue, ionwind_param, scp_pot_param, cassini_speed = ELS_IBS_fluxfitting(elsdata, ibsdata,
                                                                                                   i, j)
            fit_results.append(fit_result)
            ionwindspeeds.append(fit_result.value(ionwind_param))
            ionwindspeeds_errors.append(fit_result.stdev(ionwind_param))
            scps.append(fit_result.value(scp_pot_param))
            scps_errors.append(fit_result.stdev(scp_pot_param))
            lpvalues.append(lpvalue)
            cassini_speeds.append(cassini_speed)

    outputdf = pd.DataFrame()
    outputdf['Positive Peak Time'] = times
    outputdf['Alongtrack velocity'] = ionwindspeeds
    # outputdf['IBS residuals'] = ibs_residuals
    outputdf['CAPS spacecraft potentials'] = scps
    outputdf['Actual spacecraft velocity'] = cassini_speeds
    outputdf.to_csv("testalongtrackvelocity.csv")
    # print(outputdf['Positive Peak Time'])

    fig5, axes = plt.subplots(2)
    axes[0].errorbar(tempdf['Positive Peak Time'], ionwindspeeds, yerr=np.array(ionwindspeeds_errors),
                     label="IBS - Ion Wind Speeds", linestyle='--', capsize=5)
    axes[0].set_ylabel("Ion Wind Speed (m/s)")
    # for counter, x in enumerate(ibs_fits):
    #     axes[0].text(tempdf['Positive Peak Time'].iloc[counter], ibs_ionwindspeeds[counter], "Chi-Sqr =  %.1E" % x.chisqr)
    axes[0].legend()

    axes[1].errorbar(tempdf['Positive Peak Time'], scps, yerr=np.array(scps_errors), color='C1',
                     label="S/C potential, IBS derived", capsize=5)
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


# multiple_alongtrackwinds_flybys(['t16'])
single_slice_test(flyby="t16", slicenumber=6)

plt.show()
