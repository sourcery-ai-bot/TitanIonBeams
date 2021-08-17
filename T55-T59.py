from scipy.io import readsav
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from util import generate_mass_bins
from sunpy.time import TimeRange
import datetime
from heliopy.data.cassini import caps_ibs
from cassinipy.caps.mssl import *
from cassinipy.caps.util import *
from util import *

from lmfit import CompositeModel, Model
from lmfit.models import GaussianModel
from lmfit import Parameters
from lmfit import Minimizer
matplotlib.rcParams.update({'font.size': 15})

ibscalib = readsav('calib/ibsdisplaycalib.dat')
sngcalib = readsav('calib/sngdisplaycalib.dat')

filedates_times = {"t55": ["21-may-2009", "21:27:00"],
                   "t56": ["06-jun-2009", "20:00:00"],
                   "t57": ["22-jun-2009", "18:33:00"],
                   "t58": ["08-jul-2009", "17:04:00"],
                   "t59": ["24-jul-2009", "15:31:00"],
                   }

flyby_datetimes = {"t55": [datetime.datetime(2009, 5, 21, 21, 23), datetime.datetime(2009, 5, 21, 21, 30)],
                   "t56": [datetime.datetime(2009, 6, 6,  19, 56), datetime.datetime(2009, 6, 6, 20, 3)],
                   "t57": [datetime.datetime(2009, 6, 22,  18, 29, 20), datetime.datetime(2009, 6, 22, 18, 37)],
                   "t58": [datetime.datetime(2009, 7, 8, 17,0, 57), datetime.datetime(2009, 7, 8, 17, 8, 20)],
                   "t59": [datetime.datetime(2009, 7, 24, 15, 30), datetime.datetime(2009, 7, 24, 15, 38)],

                   }

IBS_fluxfitting_dict = {"mass15_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass28_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass40_": {"sigma": [0.2, 0.3, 0.6], "amplitude": []},
                        "mass53_": {"sigma": [0.3, 0.5, 0.65], "amplitude": []},
                        "mass66_": {"sigma": [0.4, 0.6, 0.7], "amplitude": []}, \
                        "mass78_": {"sigma": [0.5, 0.7, 0.8], "amplitude": []}, \
                        "mass91_": {"sigma": [0.6, 0.8, 1], "amplitude": []}}
if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

def total_fluxgaussian(xvalues, yvalues, masses, cassini_speed, windspeed, LPvalue, lpoffset, temperature, charge,
                       FWHM):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    if charge == 1:
        pars.add('scp', value=LPvalue+0.25, min=LPvalue - 2, max=0.25)
        # pars.add('scp', value=LPvalue+lpoffset)
        #pars['scp'].vary = False
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
    #print(yvalues,pars,xvalues)
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

    return out

IBS_FWHM = 0.014
flyby = "t59"


elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
generate_mass_bins(elsdata, flyby, "els")
ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
generate_mass_bins(ibsdata, flyby, "ibs")

start_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes[flyby][0])
end_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes[flyby][1])
lowerenergy_ibs = 4.1
upperenergy_ibs = 19
lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy_ibs , lowerenergy_ibs)[0]
upperenergyslice_ibs = CAPS_energyslice("ibs", upperenergy_ibs, upperenergy_ibs)[0]

start_els = CAPS_slicenumber(elsdata,flyby_datetimes[flyby][0])
end_els = CAPS_slicenumber(elsdata,flyby_datetimes[flyby][1])
lowerenergy_els = 4.1
upperenergy_els = 19
lowerenergyslice_els = CAPS_energyslice("els", lowerenergy_els , lowerenergy_els)[0]
upperenergyslice_els = CAPS_energyslice("els", upperenergy_els, upperenergy_els)[0]

ibs_outs = []
altitudes = []
for i in np.arange(start_ibs,end_ibs):
    tempdatetime = ibsdata['times_utc'][i]
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    altitudes.append(np.sqrt((state[0]) ** 2 + (state[1]) ** 2 + (state[2]) ** 2) - 2574.7)
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs,1,i],
                                 masses=[28, 40, 53, 66, 78, 91], cassini_speed=cassini_speed, windspeed=0, LPvalue=-0.6, lpoffset=0,
                                 temperature=150, charge=1, FWHM=IBS_FWHM)
    ibs_outs.append(out_ibs)

fig, (elsax, ibsax, altax, velax, scpax) = plt.subplots(5,sharex='col')
elsax.pcolormesh(elsdata['times_utc'][start_els:end_els], elscalib['earray'], elsdata['data'][:,3,start_els:end_els], norm=LogNorm(vmin=1e3, vmax=5e5), cmap='viridis')
ibsax.pcolormesh(ibsdata['times_utc'][start_ibs:end_ibs], ibscalib['ibsearray'], ibsdata['ibsdata'][:,1,start_ibs:end_ibs], norm=LogNorm(vmin=1e3, vmax=5e5), cmap='viridis')
ibsax.set_ylabel("IBS Fan 2 \n ev/q")
ibsax.set_yscale("log")
elsax.set_yscale("log")
ibsax.set_ylim(3,100)
elsax.set_ylim(1,400)
velax.plot(ibsdata['times_utc'][start_ibs:end_ibs], [i.params['ionvelocity'].value for i in ibs_outs])
altax.plot(ibsdata['times_utc'][start_ibs:end_ibs], altitudes)
scpax.plot(ibsdata['times_utc'][start_ibs:end_ibs], [i.params['scp'].value for i in ibs_outs])
velax.set_ylabel("IBS-derived \n Ion Velocity")
altax.set_ylabel("Altitudes \n [km]")
scpax.set_ylabel("IBS-derived \n S/C potential")

masslists = []
for mass in [28, 40, 53, 66, 78, 91]:
    tempmasses =  [i.params['mass'+str(mass)+'_center'].value for i in ibs_outs]
    masslists.append(tempmasses)

fig2, axes = plt.subplots(3,sharex='all')
for masslist,masses in zip(masslists,[28, 40, 53, 66, 78, 91]):
    axes[0].plot(ibsdata['times_utc'][start_ibs:end_ibs],np.array(masslist)-masslist[0],label="Mass " +str(masses))
    axes[1].plot(ibsdata['times_utc'][start_ibs:end_ibs],
                 scipy.signal.savgol_filter(np.array(masslist) - masslist[0], 15, 1), label="Mass " + str(masses))
    axes[2].plot(ibsdata['times_utc'][start_ibs:end_ibs],
                 np.gradient(scipy.signal.savgol_filter(np.array(masslist) - masslist[0], 11, 1)), label="Mass " + str(masses))

axes[0].set_ylabel("Observed difference \n from first \n timeslice energy [eV]")
axes[1].set_ylabel("Smoothed [eV]")
axes[2].set_ylabel("dy/dx of smoothed [eV]")
axes[0].legend()

plt.show()