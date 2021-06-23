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

IBS_fluxfitting_dict = {"mass15_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass28_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass40_": {"sigma": [0.2, 0.3, 0.6], "amplitude": []},
                        "mass53_": {"sigma": [0.3, 0.5, 0.65], "amplitude": []},
                        "mass66_": {"sigma": [0.4, 0.6, 0.7], "amplitude": []}, \
                        "mass78_": {"sigma": [0.5, 0.7, 0.8], "amplitude": []}, \
                        "mass91_": {"sigma": [0.6, 0.8, 1], "amplitude": []}}

ELS_fluxfitting_dict = {"mass26_": {"sigma": [0.1, 0.2, 0.7], "amplitude": [5]},
                        "mass50_": {"sigma": [0.5, 0.6, 0.9], "amplitude": [4]},
                        "mass74_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass77_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass79_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass91_": {"sigma": [0.6, 0.8, 1.6], "amplitude": [3]},
                        "mass117_": {"sigma": [0.8, 0.9, 1.7], "amplitude": [3]}}

def total_fluxgaussian(xvalues, yvalues, masses, cassini_speed, windspeed, LPvalue, lpoffset, temperature, charge,
                       FWHM):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    if charge == 1:
        pars.add('scp', value=LPvalue+0.25, min=LPvalue - 2, max=0)
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
    print(yvalues,pars,xvalues)
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
SNG_FWHM = 0.167

matplotlib.rcParams.update({'font.size': 15})

ibscalib = readsav('calib/ibsdisplaycalib.dat')
sngcalib = readsav('calib/sngdisplaycalib.dat')

TAELSdata = readsav("data/els/elsres_26-oct-2004.dat")
TASNGdata = readsav("data/sng/sngres_26-oct-2004.dat")
TAIBSdata = readsav("data/ibs/ibsres_26-oct-2004.dat")

generate_mass_bins(TAELSdata, "ta", "els")
generate_mass_bins(TASNGdata, "ta", "sng")
generate_aligned_ibsdata(TAIBSdata, TAELSdata, "ta")

start = datetime.datetime(2004, 10, 26, 15, 23)
end = datetime.datetime(2004, 10, 26, 15, 37)
TAELS_startslice = CAPS_slicenumber(TAELSdata, start)
TAELS_endslice = CAPS_slicenumber(TAELSdata, end)
TAIBS_startslice = CAPS_slicenumber(TAIBSdata, start)
TAIBS_endslice = CAPS_slicenumber(TAIBSdata, end)
TASNG_startslice = CAPS_slicenumber(TASNGdata, start)
TASNG_endslice = CAPS_slicenumber(TASNGdata, end)

fig, (elsax, ibsax, sngax) = plt.subplots(3, sharex="all", sharey="all")
elsax.pcolormesh(TAELSdata['times_utc'][TAELS_startslice:TAELS_endslice], elscalib['earray'],
                 TAELSdata['data'][:, 3, TAELS_startslice:TAELS_endslice], norm=LogNorm(vmin=50, vmax=5e4),
                 cmap='viridis')
ibsax.pcolormesh(TAIBSdata['times_utc'][TAIBS_startslice:TAIBS_endslice], ibscalib['ibsearray'],
                 TAIBSdata['ibsdata'][:, 1, TAIBS_startslice:TAIBS_endslice], norm=LogNorm(vmin=50, vmax=5e4),
                 cmap='viridis')
sngax.pcolormesh(TASNGdata['times_utc'][TASNG_startslice:TASNG_endslice], sngcalib['sngearray'],
                 TASNGdata['sngdata'][:, 9, TASNG_startslice:TASNG_endslice], norm=LogNorm(vmin=50, vmax=5e4),
                 cmap='viridis')
sngax.set_yscale("log")
ibsax.set_ylabel("IBS Fan 2 \n ev/q")
sngax.set_ylabel("SNG All Anodes \n ev/q")
# plt.show()

# slicetime = datetime.datetime(2004,10,26,15,29,45)

def intra_calibration(slicetime_sng, plot=True):

    #slicetime_ibs = datetime.datetime(2004, 10, 26, 15, 34, 38)
    slicetime_ibs = slicetime_sng + datetime.timedelta(seconds=2)

    end_slicetime_ibs = slicetime_ibs + datetime.timedelta(seconds=2)
    end_slicetime_sng = slicetime_sng + datetime.timedelta(seconds=4)

    TAIBS_slice = CAPS_slicenumber(TAIBSdata, slicetime_ibs)
    TAIBS_slice_end = CAPS_slicenumber(TAIBSdata, end_slicetime_ibs)
    TASNG_slice = CAPS_slicenumber(TASNGdata, slicetime_sng)
    TASNG_slice_end = CAPS_slicenumber(TASNGdata, end_slicetime_sng)
    TA_IBSslicetime = TAIBSdata['times_utc'][TAIBS_slice]
    TA_IBSslicetime_end = TAIBSdata['times_utc'][TAIBS_slice_end]
    TA_SNGslicetime = TASNGdata['times_utc'][TASNG_slice]
    TA_SNGslicetime_end = TASNGdata['times_utc'][TASNG_slice_end]

    lowerenergy = 2.3
    upperenergy = 10
    lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy, lowerenergy)[0]
    upperenergyslice_ibs  = CAPS_energyslice("ibs", upperenergy, upperenergy)[0]
    lowerenergyslice_sng = CAPS_energyslice("ims", lowerenergy, lowerenergy)[0]
    upperenergyslice_sng  = CAPS_energyslice("ims", upperenergy, upperenergy)[0]

    TAIBS_data = np.mean(TAIBSdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, TAIBS_slice:TAIBS_slice_end],axis=1) / (ibscalib['ibsgeom'] * 1e-4)
    TAIBS_series = pd.Series(TAIBS_data, index=ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs])
    TAIBS_bins = pd.cut(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], sngcalib['sngpolyearray'])
    #print(TAIBS_bins)
    TAIBS_sampledseries = TAIBS_series.groupby(TAIBS_bins).agg(['sum'])
    #print(TAIBS_sampledseries)
    IBS_scaling = 0.06
    IBS_shift = 1e13

    TAIBS_data_binned = TAIBS_sampledseries.to_numpy() * IBS_scaling
    TASNG_data = np.mean(TASNGdata['sngdef'][lowerenergyslice_sng:upperenergyslice_sng, 8, TASNG_slice:TASNG_slice_end],axis=1)



    #---Fitting attempt
    lpdata = read_LP_V1("ta")
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    lpvalue = np.interp(datetime.datetime.timestamp(slicetime_ibs), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    #print(TAIBS_data, TAIBS_data.shape,type(TAIBS_data))
    out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], TAIBS_data, masses=[15,28],cassini_speed=6.1e3,windspeed=-500,LPvalue=lpvalue,lpoffset=0,temperature=150,charge=1,FWHM=IBS_FWHM)
    #print(TAIBS_data_binned.flatten(), TAIBS_data_binned.flatten().shape,type(TAIBS_data_binned))
    #out_ibs_binned = total_fluxgaussian(sngcalib['sngearray'], TAIBS_data_binned.flatten(), masses=[15,28],cassini_speed=6.1e3,windspeed=-500,LPvalue=lpvalue,lpoffset=-0.1,temperature=150,charge=1,FWHM=SNG_FWHM)
    out_sng = total_fluxgaussian(sngcalib['sngearray'][lowerenergyslice_sng:upperenergyslice_sng], TASNG_data , masses=[15,28],cassini_speed=6.1e3,windspeed=-500,LPvalue=lpvalue,lpoffset=-0.1,temperature=150,charge=1,FWHM=SNG_FWHM)

    for i in [out_ibs, out_sng]:
        if i.params['ionvelocity'].stderr is None:
            i.params['ionvelocity'].stderr = np.nan
        if i.params['scp'].stderr is None:
            i.params['scp'].stderr = np.nan

    GOF = np.mean((abs(out_ibs.best_fit - TAIBS_data) / TAIBS_data) * 100)

    if plot==True:
        energyfig, axes = plt.subplots(2, sharex='all')
        axes[0].step(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], TAIBS_data, where='mid',
                label="IBS Data, " + str(TA_IBSslicetime) + " to " + str(TA_IBSslicetime_end))
        axes[0].plot(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], out_ibs.best_fit, 'r-', label='IBS best fit')
        axes[1].step(sngcalib['sngearray'][lowerenergyslice_sng:upperenergyslice_sng], TASNG_data, where='mid',
                label="SNG Data, " + str(TA_SNGslicetime) + " to " + str(TA_SNGslicetime_end))
        axes[1].plot(sngcalib['sngearray'][lowerenergyslice_sng:upperenergyslice_sng], out_sng.best_fit, 'c-', label='SNG best fit')


        for ax, out, instrument in zip(axes, [out_ibs, out_sng], ["IBS","IMS"]):
            ax.set_yscale("log")
            ax.text(0.6, 0.01,
                    "Ion wind = %2.0f ± %2.0f m/s" % (out.params['ionvelocity'], out.params['ionvelocity'].stderr),
                    transform=ax.transAxes)
            ax.text(0.6, .09,
                    instrument+"-derived S/C Potential = %2.2f ± %2.2f V" % (out.params['scp'], out.params['scp'].stderr),
                    transform=ax.transAxes)
            ax.text(0.6, .17, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=ax.transAxes)
            ax.set_ylim(bottom=1e9)
            ax.text(out.params['mass15_center'].value, out.params['mass15_height'].value, "%2.2f eV" % out.params['mass15_center'].value)
            ax.text(out.params['mass28_center'].value, out.params['mass28_height'].value, "%2.2f eV" % out.params['mass28_center'].value)

    # axes[2].step(sngcalib['sngearray'], TAIBS_data_binned , where='mid',
    #         label="IBS data, binned to SNG bins," + str(IBS_scaling) + " scale")
    # axes[2].plot(sngcalib['sngearray'], out_ibs_binned.best_fit, 'b-', label='IBS binned best fit')
    # # ax.step(sngcalib['sngearray'],TAIBS_sampledseries.to_numpy()-IBS_shift,where='mid',label="IBS data, binned to SNG bins, -" + str(IBS_shift) +" shift")
    # axes[2].set_yscale("log")
    # axes[2].text(0.6, 0.01,
    #         "Ion wind = %2.0f ± %2.0f m/s" % (out_ibs_binned.params['ionvelocity'], out_ibs_binned.params['ionvelocity'].stderr),
    #         transform=axes[1].transAxes)
    # axes[2].text(0.6, .09,
    #         "IBS-derived S/C Potential = %2.2f ± %2.2f V" % (out_ibs_binned.params['scp'], out_ibs_binned.params['scp'].stderr),
    #         transform=axes[1].transAxes)
    # axes[2].text(0.6, .17, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=axes[1].transAxes)

        axes[0].set_xscale("log")
        axes[0].set_xlim(1, 100)

        axes[1].set_ylabel("DEF")
        #axes[2].set_xlabel("eV/q")

        axes[0].legend()
        axes[1].legend()
    #axes[2].legend()

    ibs_mass15 = out_ibs.params['mass15_center'].value
    ibs_mass28 = out_ibs.params['mass28_center'].value
    ibs_ionvelocity = out_ibs.params['ionvelocity'].value
    ibs_scp = out_ibs.params['scp'].value
    sng_mass15 = out_sng.params['mass15_center'].value
    sng_mass28 = out_sng.params['mass28_center'].value
    sng_ionvelocity = out_sng.params['ionvelocity'].value
    sng_scp = out_sng.params['scp'].value
    lp_scp = lpvalue

    return ibs_mass15, sng_mass15, ibs_mass28, sng_mass28, ibs_ionvelocity, sng_ionvelocity, ibs_scp, sng_scp, lp_scp

df = pd.DataFrame(columns=["Time","IBS 15u energy", "SNG 15u energy",
                                 "IBS 28u energy", "SNG 28u energy",
                                 "IBS Ion Velocity", "SNG Ion Velocity",
                                 "IBS SCP", "SNG SCP", "LP SCP"])

#slicetime_sng = datetime.datetime(2004, 10, 26, 15, 34, 32)
#slicetimes = [datetime.datetime(2004, 10, 26, 15, 27, 12)]
slicetimes = [datetime.datetime(2004, 10, 26, 15, 25, 56),
              datetime.datetime(2004, 10, 26, 15, 26, 48), datetime.datetime(2004, 10, 26, 15, 34, 36),
              datetime.datetime(2004, 10, 26, 15, 34, 40), datetime.datetime(2004, 10, 26, 15, 34, 44),
              datetime.datetime(2004, 10, 26, 15, 34, 48)]

for slicetime_sng in slicetimes:
    temp = [slicetime_sng] + list(intra_calibration(slicetime_sng))
    df = df.append(pd.Series(temp,index=df.columns),ignore_index=True)

print(df)
df.to_csv("TA_calibration.csv")

plt.show()
