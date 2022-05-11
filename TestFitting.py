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
def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



def total_fluxgaussian(xvalues, yvalues, mu, sigma, amp):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    mod = GaussianModel()
    pars.update(mod.make_params())

    pars['center'].set(value=mu)
    pars['sigma'].set(value=sigma)
    pars['amplitude'].set(value=amp)

    init = mod.eval(pars, x=xvalues)
    out = mod.fit(yvalues, pars, x=xvalues)

    print(out.fit_report(min_correl=0.7))

    # Calculating CI's
    # print(out.ci_report(p_names=["scp","windspeed"],sigmas=[1],verbose=True,with_offset=False,ndigits=2))

    return out, init


x_lin = np.linspace(10, 100, 5)
x_log = np.logspace(1, 2, 5)
mu = 70
sig = 5
y_lin = gaussian(x_lin, mu, sig, 1)
y_log = gaussian(x_log, mu, sig, 1)

out_lin, init_lin = total_fluxgaussian(x_lin, y_lin, mu*1.2, sig*1.2, 1)
out_log, init_log = total_fluxgaussian(x_log, y_log, mu*1.2, sig*1.2, 1)


percent_dif = out_log.params['center']/out_lin.params['center']
print("Log Center/Lin Center", percent_dif)

fig, ax = plt.subplots()
ax.step(x_lin, y_lin, color='k', label="Linear")
ax.plot(x_lin, out_lin.best_fit, 'k-.', label='Linear Fit')
ax.step(x_log, y_log,  color='r', label="Log")
ax.plot(x_log, out_log.best_fit, 'r-.', label='Log Fit')
ax.legend()
ax.vlines(70,0,1,label="Original",color='C0')
ax.vlines(out_log.params['center'],0,1,label="Log Centre",color='C1')
ax.vlines(out_lin.params['center'],0,1,label="Lin Centre",color='C2')
ax.set_title("Log/Linear = " + str(percent_dif))

plt.show()