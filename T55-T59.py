from scipy.io import readsav
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from util import generate_mass_bins
from sunpy.time import TimeRange
import datetime
#from heliopy.data.cassini import caps_ibs
from heliopy.data.cassini import mag_1min
from cassinipy.caps.mssl import *
from cassinipy.caps.util import *
from util import *
import time

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

flyby_datetimes_ibs = {
                    "t55": [datetime.datetime(2009, 5, 21, 21, 19, 6, 15), datetime.datetime(2009, 5, 21, 21, 32,46)],
                   "t56": [datetime.datetime(2009, 6, 6,  19, 54, 45), datetime.datetime(2009, 6, 6, 20, 5, 57)],
                   "t57": [datetime.datetime(2009, 6, 22,  18, 26, 13), datetime.datetime(2009, 6, 22, 18, 40)],
                   "t58": [datetime.datetime(2009, 7, 8, 16, 58), datetime.datetime(2009, 7, 8, 17, 11, 30)],
                   "t59": [datetime.datetime(2009, 7, 24, 15, 25, 36), datetime.datetime(2009, 7, 24, 15, 40)],
                   }

# flyby_datetimes_ibs = {"t55": [datetime.datetime(2009, 5, 21, 21, 19, 6), datetime.datetime(2009, 5, 21, 21, 22)],
#                    "t56": [datetime.datetime(2009, 6, 6,  19, 54, 45), datetime.datetime(2009, 6, 6, 19, 56)],
#                    "t57": [datetime.datetime(2009, 6, 22,  18, 26, 13), datetime.datetime(2009, 6, 22, 18, 28)],
#                    "t58": [datetime.datetime(2009, 7, 8, 16, 58), datetime.datetime(2009, 7, 8, 16,59)],
#                    "t59": [datetime.datetime(2009, 7, 24, 15, 25, 36), datetime.datetime(2009, 7, 24, 15, 28, 30)],
#                    }

flyby_datetimes_els = {"t55": [datetime.datetime(2009, 5, 21, 21, 23, 30), datetime.datetime(2009, 5, 21, 21, 29, 48)],
                   #"t56": [datetime.datetime(2009, 6, 6,  19, 57, 20), datetime.datetime(2009, 6, 6, 19, 57, 30)],
                   "t56": [datetime.datetime(2009, 6, 6,  19, 57, 20), datetime.datetime(2009, 6, 6, 20, 3, 6)],
                   "t57": [datetime.datetime(2009, 6, 22,  18, 29, 16), datetime.datetime(2009, 6, 22, 18, 36, 10)],
                   "t58": [datetime.datetime(2009, 7, 8, 17, 0 ,30), datetime.datetime(2009, 7, 8, 17, 7, 12)],
                   "t59": [datetime.datetime(2009, 7, 24, 15, 30), datetime.datetime(2009, 7, 24, 15, 38)],
                   }


flyby_startparams_ibs = {"t55": [[500, 500],1000,25],
                   "t56": [[0, 0],500,0.25],
                   "t57": [[1300, 1300],500,0],
                   "t58": [[250, 250],500,0.6],
                   "t59": [[400, 2400],10000,5]}

flyby_inboundparams_ibs = {"t55": [0,1000,0.25],
                   "t56": [0,500,0.25],
                   "t57": [150,500,0.25],
                   "t58": [0,500,0.25],
                   "t59": [500,100,8]}

flyby_outboundparams_ibs = {"t55": [[500, 500],1000,0],
                   "t56": [[0, 0],500,0],
                   "t57": [[-130, -130],500,0],
                   "t58": [[0, 0],500,0],
                   "t59": [[0, 0],1000,0]}



flyby_startparams_els = {"t55": [0,500,-1],
                   "t56": [-200,500,-1],
                   "t57": [0,500,-0.4],
                   "t58": [-500,500,-0.5],
                   "t59": [-200,500,-1]}

flyby_masses_els = {"t55": [[26, 50],[26, 50, 79],[26, 50, 79]],
                   "t56": [[26, 50],[26, 50, 79],[26, 50, 79, 117]],
                   "t57": [[26, 50],[26, 50, 79],[26, 50, 79]],
                   "t58": [[26, 50],[26, 50, 79],[26, 50, 79, 117]],
                   "t59": [[26, 50],[26, 50, 79],[26, 50, 79, 117]]}


IBS_fluxfitting_dict = {"mass15_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass16_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass17_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass28_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass29_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass40_": {"sigma": [0.2, 0.3, 0.6], "amplitude": []},
                        "mass53_": {"sigma": [0.3, 0.5, 0.65], "amplitude": []},
                        "mass66_": {"sigma": [0.4, 0.6, 0.7], "amplitude": []}, \
                        "mass78_": {"sigma": [0.5, 0.7, 0.9], "amplitude": []}, \
                        "mass91_": {"sigma": [0.5, 0.8, 1.1], "amplitude": []}}

ELS_fluxfitting_dict = {"mass13_": {"sigma": [0.1, 0.2, 0.7], "amplitude": [5]},
                        "mass26_": {"sigma": [0.1, 0.3, 1], "amplitude": [5]},
                        "mass50_": {"sigma": [0.5, 0.6, 1.3], "amplitude": [4]},
                        "mass74_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass77_": {"sigma": [0.5, 0.7, 1.3], "amplitude": [3]},
                        "mass79_": {"sigma": [0.5, 0.7, 2], "amplitude": [3]},
                        "mass91_": {"sigma": [0.6, 0.8, 1.6], "amplitude": [3]},
                        "mass117_": {"sigma": [0.8, 0.9, 3], "amplitude": [3]}}

IBS_energybound_dict = {"t55": [[4, 19],[3, 19],[3, 19]],
                               "t56": [[4, 19],[3, 19],[3, 19]],
                               "t57": [[4, 19],[3, 19],[3, 19]],
                               "t58": [[4, 18.4],[3, 19],[3, 19]],
                               "t59": [[4, 18.2],[3, 19],[3, 55]],}

ELS_energybound_dict = {"t55": [[1, 9],[1, 15],[1, 15]],
                               "t56": [[1, 9],[1, 21],[1, 21]],
                               "t57": [[1, 9],[1, 17],[1, 17]],
                               "t58": [[1, 9],[1, 17],[1, 30]],
                               "t59": [[1, 10],[1, 17],[1, 30]],}


if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

IBS_FWHM = 0.014


def total_fluxgaussian(xvalues, yvalues, masses, cassini_speed, windspeeds, scpvalue, lpvalue, temperature, charge, multipleionvelocity):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    if charge == 1:
        if multipleionvelocity == False:
            if scpvalue-0.5 < lpvalue:
                minscp = lpvalue
            else:
                minscp = scpvalue-0.2
            if scpvalue + 0.3 < 0:
                maxscp = scpvalue + 0.3
            else:
                maxscp = 0
            pars.add('scp', value=scpvalue, min=minscp, max=maxscp)
        if multipleionvelocity == True:
            if scpvalue-1 < lpvalue:
                minscp = lpvalue
            else:
                minscp = scpvalue-1
            if scpvalue + 2 > 0:
                maxscp = 0
            else:
                maxscp = scpvalue+2
            pars.add('scp', value=scpvalue, min=minscp, max=maxscp)

    elif charge == -1:
        if scpvalue - 0.5 < -3:
            minscp = -3
        else:
            minscp = scpvalue - 0.5
        if scpvalue + 0.3 > lpvalue:
            maxscp = lpvalue
        else:
            maxscp = scpvalue + 0.3
        pars.add('scp', value=scpvalue, min=minscp, max=maxscp)
    if temperature > 10000:
        temperature = 10000

    #print(temperature)
    pars.add('temp_eV', value=8 * k * temperature)  # , min=130, max=170)
    pars.add('spacecraftvelocity', value=cassini_speed)
    if multipleionvelocity == False:
        if charge == 1:
            if windspeeds < -500:
                minvel = -500
            else:
                minvel = windspeeds - 100
            if windspeeds > 1500:
                maxvel = 1500
            else:
                maxvel = windspeeds + 100
            pars.add('ionvelocity', value=windspeeds, min=minvel, max=maxvel)

        elif charge == -1:
            if windspeeds - 150 < -600:
                minvel = -600
            else:
                minvel = windspeeds - 150
            if windspeeds + 150 > 200:
                maxvel = 200
            else:
                maxvel = windspeeds + 150
            pars.add('ionvelocity', value=windspeeds, min= minvel, max=maxvel)
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
        #pars.add(tempprefix, value=mass, min=mass-1, max=mass+1)

        if multipleionvelocity == True:
            windmax = windspeeds[masscounter] + 200
            windmin = windspeeds[masscounter] - 200

            pars.add(tempprefix+'ionvelocity', value=windspeeds[masscounter], min=windmin, max=windmax)


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

        temppeakflux = peakflux(mass, pars['spacecraftvelocity'], 0, scpvalue, temperature, charge=charge)
        # print("mass", mass, "Init Flux", temppeakflux)
        if multipleionvelocity == True:
            peakfluxexpr = '(0.5*(' + tempprefix + '*AMU)*((spacecraftvelocity+' + tempprefix + 'ionvelocity)**2) - scp*e*charge + temp_eV)/e'
        else:
            peakfluxexpr = '(0.5*(' + tempprefix + '*AMU)*((spacecraftvelocity+ionvelocity)**2) - scp*e*charge + temp_eV)/e'
        pars[tempprefix + 'center'].set(expr=peakfluxexpr)
        # min=temppeakflux - 2, max=temppeakflux + 2)

        if multipleionvelocity == True:
            energy_fwhm = ((cassini_speed + windspeeds[masscounter]) * np.sqrt(2*k*temperature*mass*AMU))/e
            energy_sigma = energy_fwhm/2.355
            #print("energy fwhm",energy_fwhm)
            pars[tempprefix + 'sigma'].set(value=energy_sigma, min=energy_sigma-0.5, max=energy_sigma+0.1)
        else:
            pars[tempprefix + 'sigma'].set(value=sigmavals[1], min=sigmavals[0], max=sigmavals[2])
        pars[tempprefix + 'amplitude'].set(value=np.max(yvalues) * (1 + sigmavals[1]), min=min(yvalues))

    for counter, model in enumerate(gaussmodels):
        if counter == 0:
            mod = model
        else:
            mod = mod + model

    #print(yvalues)
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

    #print(out.fit_report(min_correl=0.7))

    # Calculating CI's
    # print(out.ci_report(p_names=["scp","windspeed"],sigmas=[1],verbose=True,with_offset=False,ndigits=2))

    return out

def plotting(flyby, multipleionvelocity=False):

    IBS_FWHM = 0.014

    elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(ibsdata, flyby, "ibs")

    start_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes_ibs[flyby][0])
    end_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes_ibs[flyby][1])
    lowerenergy_ibs = 3
    upperenergy_ibs = 55
    lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy_ibs , lowerenergy_ibs)[0]
    upperenergyslice_ibs = CAPS_energyslice("ibs", upperenergy_ibs, upperenergy_ibs)[0]

    start_els = CAPS_slicenumber(elsdata,flyby_datetimes_ibs[flyby][0])
    end_els = CAPS_slicenumber(elsdata,flyby_datetimes_ibs[flyby][1])
    lowerenergy_els = 4.1
    upperenergy_els = 19
    lowerenergyslice_els = CAPS_energyslice("els", lowerenergy_els , lowerenergy_els)[0]
    upperenergyslice_els = CAPS_energyslice("els", upperenergy_els, upperenergy_els)[0]

    lpdata = read_LP_V1(flyby)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]


    ibs_outs = []
    altitudes = []
    lpvalues = []
    # if multipleionvelocity == True:
    #     #previous_windspeeds = [0,0,0,0,0,0]
    #     previous_windspeeds = [1200,1100]
    # else:
    #     previous_windspeeds = 1000
    highalt_slicenumbers = []
    highalt_counters = []
    midalt_slicenumbers = []
    midalt_counters  = []
    lowalt_slicenumbers = []
    lowalt_counters = []
    derived_temps = []

    tempdatetime = ibsdata['times_utc'][start_ibs]
    start_lp = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    used_scp = start_lp + flyby_startparams_ibs[flyby][2]
    last_ion_temp = flyby_startparams_ibs[flyby][1]


    for counter, i in enumerate(np.arange(start_ibs,end_ibs)):
        tempdatetime = ibsdata['times_utc'][i]
        lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
        lpvalues.append(lpvalue)
        et = spice.datetime2et(tempdatetime)
        state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
        alt = np.sqrt((state[0]) ** 2 + (state[1]) ** 2 + (state[2]) ** 2) - 2574.7

        if alt < 1100:
            multipleionvelocity = False
            masses_used = [28, 40, 53, 66, 78, 91]
            if counter == 0 or isinstance(previous_windspeeds,list):
                previous_windspeeds = 0
            last_ion_temp = titan_linearfit_temperature(alt)
            lowalt_slicenumbers.append(i)
            lowalt_counters.append(counter)
        elif alt > 1100 and alt < 1500:
            multipleionvelocity = False
            masses_used = [17, 28, 40, 53, 66, 78, 91]
            if counter == 0:
                previous_windspeeds = flyby_startparams_ibs[flyby][0][0]
            if isinstance(previous_windspeeds,list):
                previous_windspeeds = np.mean(previous_windspeeds)
            last_ion_temp = titan_linearfit_temperature(alt)
            midalt_slicenumbers.append(i)
            midalt_counters.append(counter)
        elif alt > 1500:
            multipleionvelocity = True
            masses_used = [17, 28]
            if counter == 0:
                previous_windspeeds = flyby_startparams_ibs[flyby][0]
            if isinstance(previous_windspeeds,float):
                previous_windspeeds = [0, 0]
            highalt_slicenumbers.append(i)
            highalt_counters.append(counter)

        derived_temps.append(last_ion_temp)


        altitudes.append(alt)
        cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
        out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs,1,i],
                                     masses=masses_used, cassini_speed=cassini_speed, windspeeds=previous_windspeeds, scpvalue=used_scp, lpvalue=lpvalue,  #[17, 28, 40, 53, 66, 78, 91],
                                     temperature=last_ion_temp, charge=1, multipleionvelocity=multipleionvelocity)
        used_scp = out_ibs.params['scp'].value

        if multipleionvelocity == True:
            for counter, j in enumerate(masses_used):
                previous_windspeeds[counter] = out_ibs.params['mass'+str(j)+'_ionvelocity'].value
        else:
                previous_windspeeds = out_ibs.params['ionvelocity'].value

        temp_temps = []
        for counter, j in enumerate(masses_used):
            if multipleionvelocity == True:
                derived_temperature = 4 * (1 / (cassini_speed + previous_windspeeds[counter]) ** 2) * (
                        (out_ibs.params['mass' + str(j) + '_fwhm'].value * e) ** 2) * (1 / (2 * k * j * AMU))
            else:
                derived_temperature = 4 * (1 / (cassini_speed + previous_windspeeds) ** 2) * (
                        (out_ibs.params['mass' + str(j) + '_fwhm'].value * e) ** 2) * (1 / (2 * k * j * AMU))
            temp_temps.append(derived_temperature/4)
            print("derived temperature", derived_temperature)
        last_ion_temp = np.mean(temp_temps)


        ibs_outs.append(out_ibs)

    fig, (elsax, ibsax, altax, velax, scpax, tempax) = plt.subplots(6,sharex='col')
    elsax.pcolormesh(elsdata['times_utc'][start_els:end_els], elscalib['earray'], elsdata['data'][:,3,start_els:end_els], norm=LogNorm(vmin=1e3, vmax=5e5), cmap='viridis')
    elsax.set_ylabel("ELS \n Anode 4 \n ev/q")
    ibsax.pcolormesh(ibsdata['times_utc'][start_ibs:end_ibs], ibscalib['ibsearray'], ibsdata['ibsdata'][:,1,start_ibs:end_ibs], norm=LogNorm(vmin=1e3, vmax=5e5), cmap='viridis')
    ibsax.set_ylabel("IBS  \n Fan 2 \n ev/q")
    ibsax.set_yscale("log")
    elsax.set_yscale("log")
    ibsax.set_ylim(3,100)
    elsax.set_ylim(1,400)

    for masscounter, x in enumerate([17, 28]):
        ibsax.scatter(np.array(ibsdata['times_utc'])[highalt_slicenumbers], [i.params['mass'+str(x)+'_center'].value for i in np.array(ibs_outs)[highalt_counters]],label='mass'+str(x),color='C'+str(masscounter))
        velax.scatter(np.array(ibsdata['times_utc'])[highalt_slicenumbers], [i.params['mass' + str(x) + '_ionvelocity'].value for i in np.array(ibs_outs)[highalt_counters]], label='mass' + str(x),color='C'+str(masscounter))
    for masscounter,x in enumerate([17, 28, 40, 53, 66, 78, 91]):
        ibsax.scatter(np.array(ibsdata['times_utc'])[midalt_slicenumbers], [i.params['mass'+str(x)+'_center'].value for i in np.array(ibs_outs)[midalt_counters]],label='mass'+str(x),color='C'+str(masscounter))

    for masscounter,x in enumerate([28, 40, 53, 66, 78, 91],1):
        ibsax.scatter(np.array(ibsdata['times_utc'])[lowalt_slicenumbers], [i.params['mass'+str(x)+'_center'].value for i in np.array(ibs_outs)[lowalt_counters]],label='mass'+str(x),color='C'+str(masscounter))

    velax.scatter(np.array(ibsdata['times_utc'])[midalt_slicenumbers],  [i.params['ionvelocity'].value for i in np.array(ibs_outs)[midalt_counters]])
    velax.plot(np.array(ibsdata['times_utc'])[lowalt_slicenumbers],  [i.params['ionvelocity'].value for i in np.array(ibs_outs)[lowalt_counters]])


    # if multipleionvelocity == True:
    #     for x in masses_used:
    #         velax.plot(ibsdata['times_utc'][start_ibs:end_ibs], [i.params['mass'+str(x)+'_ionvelocity'].value for i in ibs_outs],label='mass'+str(x))
    # else:
    #     velax.plot(ibsdata['times_utc'][start_ibs:end_ibs],[i.params['ionvelocity'].value for i in ibs_outs])
    altax.plot(ibsdata['times_utc'][start_ibs:end_ibs], altitudes)
    scpax.plot(ibsdata['times_utc'][start_ibs:end_ibs], [i.params['scp'].value for i in ibs_outs])
    scpax.plot(ibsdata['times_utc'][start_ibs:end_ibs], lpvalues,label="LP values")
    tempax.plot(ibsdata['times_utc'][start_ibs:end_ibs], derived_temps,label="Derived Ion Temp")
    #actax.plot(elsdata['times_utc'][start_els:end_els], elsdata['actuator'][start_els:end_els])
    #energyax.set_ylabel("IBS \n ion energies")
    velax.set_ylabel("IBS \n -derived \n Ion Velocity")
    altax.set_ylabel("Altitudes \n [km]")
    scpax.set_ylabel("IBS \n -derived \n S/C potential")
    tempax.set_ylabel("Ion temp")
    velax.hlines(0,ibsdata['times_utc'][start_ibs],ibsdata['times_utc'][end_ibs],color='k')
    # for ax in (elsax, ibsax, altax, energyax, velax, scpax, actax):
    #     ax.legend()

    # masslists = []
    # for mass in masses_used:
    #     tempmasses =  [i.params['mass'+str(mass)+'_center'].value for i in ibs_outs]
    #     masslists.append(tempmasses)
    #
    # fig2, axes = plt.subplots(3,sharex='all')
    # for masslist,masses in zip(masslists,masses_used):
    #     axes[0].plot(ibsdata['times_utc'][start_ibs:end_ibs],np.array(masslist)-masslist[0],label="Mass " +str(masses))
    #     axes[1].plot(ibsdata['times_utc'][start_ibs:end_ibs],
    #                  scipy.signal.savgol_filter(np.array(masslist) - masslist[0], 15, 1), label="Mass " + str(masses))
    #     axes[2].plot(ibsdata['times_utc'][start_ibs:end_ibs],
    #                  np.gradient(scipy.signal.savgol_filter(np.array(masslist) - masslist[0], 11, 1)), label="Mass " + str(masses))

    # axes[0].set_ylabel("Observed difference \n from first \n timeslice energy [eV]")
    # axes[1].set_ylabel("Smoothed [eV]")
    # axes[2].set_ylabel("dy/dx of smoothed [eV]")
    # for ax in axes:
    #     ax.legend()

    plt.show()

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


def non_actuating_alongtrackwinds_flybys_ibs(usedflybys=["t55","t56","t57","t58","t59"],plotting=False):
    outputdf = pd.DataFrame()
    for flyby in usedflybys:
        ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
        generate_mass_bins(ibsdata, flyby, "ibs")

        start_ibs = CAPS_slicenumber(ibsdata, flyby_datetimes_ibs[flyby][0])
        end_ibs = CAPS_slicenumber(ibsdata, flyby_datetimes_ibs[flyby][1])
        # lowerenergy_ibs = 3
        # upperenergy_ibs = 19
        # lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy_ibs, lowerenergy_ibs)[0]
        # upperenergyslice_ibs = CAPS_energyslice("ibs", upperenergy_ibs, upperenergy_ibs)[0]

        highalt_slicenumbers, highalt_counters = [], []
        midalt_slicenumbers, midalt_counters  = [], []
        lowalt_slicenumbers, lowalt_counters  = [], []
        derived_temps = []

        ibs_outputs, ibs_datetimes, ibs_GOFvalues, lpvalues, cassini_speeds = [], [], [], [],[]

        zonalangles, zonalwinds_2008MW, zonalwinds_2016, zonalwinds_2017 = [], [], [], []
        altitudes, longitude, latitude, temps = [], [], [], []
        x_titan, y_titan, z_titan, dx_titan, dy_titan, dz_titan = [], [], [], [], [], []
        ibs_velocities, ibs_velocities_stderr, ibs_scp, ibs_scp_stderr = [], [], [], []
        lowerenergyslices, upperenergyslices = [], []
        lptimes = list(ibsdata['times_utc'][start_ibs:end_ibs])
        lpdata = read_LP_V1(flyby)
        lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
        tempdatetime = ibsdata['times_utc'][start_ibs]
        start_lp = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
        used_scp = start_lp + flyby_startparams_ibs[flyby][2]
        last_ion_temp = flyby_startparams_ibs[flyby][1]

        all_possible_masses = [17, 28, 40, 53, 66, 78, 91]
        mass_energies_dict, mass_energies_dict_els = {}, {}
        for mass in all_possible_masses:
            mass_energies_dict["ibsmass"+str(mass)] = []

        for counter, i in enumerate(np.arange(start_ibs,end_ibs)):
            tempdatetime = ibsdata['times_utc'][i]
            lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps,
                                lpdata['SPACECRAFT_POTENTIAL'])
            lpvalues.append(lpvalue)
            et = spice.datetime2et(tempdatetime)
            state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
            alt, lat, lon = cassini_titan_altlatlon(tempdatetime)

            if alt < 1100:
                multipleionvelocity = False
                if counter == 0 or isinstance(previous_windspeeds, list):
                    previous_windspeeds = 0
                last_ion_temp = titan_linearfit_temperature(alt)
                lowalt_slicenumbers.append(i)
                lowalt_counters.append(counter)
                #lowerenergyslice_ibs = 0
                lowerenergyslice_ibs = CAPS_energyslice("ibs", IBS_energybound_dict[flyby][0][0], IBS_energybound_dict[flyby][0][0])[0]
                upperenergyslice_ibs = CAPS_energyslice("ibs", IBS_energybound_dict[flyby][0][1], IBS_energybound_dict[flyby][0][1])[0]
            elif alt > 1100 and alt < 1500:
                multipleionvelocity = False
                if counter == 0:
                    previous_windspeeds = flyby_startparams_ibs[flyby][0][0]
                elif isinstance(previous_windspeeds, list):
                    previous_windspeeds = flyby_inboundparams_ibs[flyby][0]
                    used_scp = lpvalue + 0.25
                    #used_scp = lpvalue
                last_ion_temp = titan_linearfit_temperature(alt)
                midalt_slicenumbers.append(i)
                midalt_counters.append(counter)
                lowerenergyslice_ibs = CAPS_energyslice("ibs", IBS_energybound_dict[flyby][1][0], IBS_energybound_dict[flyby][1][0])[0]
                upperenergyslice_ibs = CAPS_energyslice("ibs", IBS_energybound_dict[flyby][1][1], IBS_energybound_dict[flyby][1][1])[0]
            elif alt > 1500:
                multipleionvelocity = True
                if counter == 0:
                    previous_windspeeds = flyby_startparams_ibs[flyby][0]
                    print(flyby,previous_windspeeds,used_scp)
                elif isinstance(previous_windspeeds, float):
                    previous_windspeeds = flyby_outboundparams_ibs[flyby][0]
                    print(flyby, previous_windspeeds, used_scp)
                highalt_slicenumbers.append(i)
                highalt_counters.append(counter)
                lowerenergyslice_ibs = CAPS_energyslice("ibs", IBS_energybound_dict[flyby][2][0], IBS_energybound_dict[flyby][2][0])[0]
                upperenergyslice_ibs = CAPS_energyslice("ibs", IBS_energybound_dict[flyby][2][1], IBS_energybound_dict[flyby][2][1])[0]

            #print("ion temp",last_ion_temp)
            derived_temps.append(last_ion_temp)
            cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3

            # while ibsdata['ibsdata'][lowerenergyslice_ibs, 1, i] == 0.:
            #     lowerenergyslice_ibs += 1


            #--------NEW BIT ----------
            #print(lowerenergyslice_ibs,ibscalib['ibsearray'][lowerenergyslice_ibs],upperenergyslice_ibs,ibscalib['ibsearray'][upperenergyslice_ibs])
            if alt < 1500:
                while np.mean(ibsdata['ibsdata'][lowerenergyslice_ibs:lowerenergyslice_ibs+3, 1, i]) < 2200:
                    lowerenergyslice_ibs += 1
                while np.mean(ibsdata['ibsdata'][upperenergyslice_ibs-2:lowerenergyslice_ibs, 1, i]) < 800:
                    upperenergyslice_ibs -= 1
            if alt > 1500:
                while np.mean(ibsdata['ibsdata'][lowerenergyslice_ibs:lowerenergyslice_ibs+3, 1, i]) < 1000:
                    lowerenergyslice_ibs += 1
                while np.mean(ibsdata['ibsdata'][upperenergyslice_ibs-7:upperenergyslice_ibs, 1, i]) < 1000:
                    upperenergyslice_ibs -= 1

            peaks, properties = find_peaks(ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i], distance=5, width=2)

            if upperenergyslice_ibs < lowerenergyslice_ibs and plotting == True:
                print(counter,alt)
                print("wrong")
                print(lowerenergyslice_ibs,ibscalib['ibsearray'][lowerenergyslice_ibs],upperenergyslice_ibs,ibscalib['ibsearray'][upperenergyslice_ibs])

                plt.plot(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs],ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i])
                plt.scatter(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs][peaks],ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i][peaks])
                plt.yscale("log")
                plt.show()

            test_ibs_masses = []
            for peak in peaks:
                energy = ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs][peak]
                if energy < 5:
                    test_ibs_masses = test_ibs_masses + [17]
                if energy > 5 and energy < 6.8:
                    if alt > 1400:
                        test_ibs_masses = test_ibs_masses + [17, 28]
                    else:
                        test_ibs_masses = test_ibs_masses + [28]
                if energy > 6.8 and energy < 9:
                    test_ibs_masses = test_ibs_masses + [28, 40]
                if energy > 9 and energy < 11.5:
                    test_ibs_masses = test_ibs_masses + [28, 40, 53]
                if energy > 11.5 and energy < 14:
                    test_ibs_masses = test_ibs_masses + [28, 40, 53, 66]
                if energy > 14 and energy < 16.4:
                    test_ibs_masses = test_ibs_masses + [28, 40, 53, 66, 78]
                if energy > 16.4:
                    test_ibs_masses = test_ibs_masses + [28, 40, 53, 66, 78, 91]
            test_ibs_masses = sorted(set(test_ibs_masses))[:len(peaks)]

            if len(peaks) == 0 and plotting == True:
                print(counter,alt)
                print("wrong")
                print(lowerenergyslice_ibs,ibscalib['ibsearray'][lowerenergyslice_ibs],upperenergyslice_ibs,ibscalib['ibsearray'][upperenergyslice_ibs])

                plt.plot(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs],ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i])
                plt.scatter(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs][peaks],ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i][peaks])
                plt.yscale("log")
                plt.show()


            #highest_energy_peak = ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs][peaks[-1]]
            slice = ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i]

            if len(peaks) != 0:
                counter = peaks[-1]
                #print(highest_energy_peak, counter, lowerenergyslice_ibs, upperenergyslice_ibs)
                #print(slice[counter+1],slice[counter])
                while slice[counter+1] < slice[counter] or slice[counter+2] < slice[counter]:
                    counter += 1
                    if counter + 2 >= len(slice):
                        counter +=1
                        break
                if counter - peaks[-1] < 5:
                    counter = peaks[-1] + 5
                upperenergyslice_ibs = lowerenergyslice_ibs+counter
            #print(highest_energy_peak, counter, lowerenergyslice_ibs, upperenergyslice_ibs)

            # print("old", masses_used)
            #print("new", test_ibs_masses)



            masses_used = test_ibs_masses


            #Multiple ion velocity exception
            if alt > 1500:
                masses_used = [17, 28]

            # if alt>1500:
            #     print(peaks,masses_used)
            #     plt.plot(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs],ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i])
            #     plt.scatter(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs][peaks],ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i][peaks])
            #     plt.yscale("log")
            #     plt.show()

            #print(alt,masses_used)

            #--------------------

            lowerenergyslices.append(lowerenergyslice_ibs)
            upperenergyslices.append(upperenergyslice_ibs)

            t0 = time.time()
            print(alt, used_scp, previous_windspeeds)
            out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs],
                                         ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i],
                                         masses=masses_used, cassini_speed=cassini_speed,
                                         windspeeds=previous_windspeeds, scpvalue=used_scp, lpvalue=lpvalue,
                                         # [17, 28, 40, 53, 66, 78, 91],
                                         temperature=last_ion_temp, charge=1,
                                         multipleionvelocity=multipleionvelocity)
            used_scp = out_ibs.params['scp'].value

            t1 = time.time()

            if t1-t0 > 10 and plotting == True:
                print("time", t1-t0, "alt", alt)
                print(out_ibs.fit_report(min_correl=0.7))

                x = ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs]
                y = ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i]


                comps = out_ibs.eval_components(x=x)

                stepplotfig_ibs, ax = plt.subplots()
                stepplotfig_ibs.suptitle("Histogram of " + ibsdata['flyby'].upper() + " IBS data", fontsize=32)
                ax.step(x,y, where='mid')
                ax.scatter(x[peaks], y[peaks])
                ax.plot(x, out_ibs.best_fit, 'k-', label='best fit')
                ax.set_yscale("log")
                ax.set_ylabel("Counts [/s]", fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
                ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
                ax.minorticks_on()
                ax.set_xlabel("Energy [eV/q]", fontsize=14)
                for mass in masses_used:
                    ax.plot(x, comps["mass" + str(mass) + '_'], '-', label=str(mass) + " amu/q")
                ax.legend(loc="best")
                ax.set_ylim(min(y),max(y)*1.2)
                plt.show()


            if multipleionvelocity == True:
                tempvelocities = []
                for counter, j in enumerate(masses_used):
                    tempvelocities.append(out_ibs.params['mass' + str(j) + '_ionvelocity'].value)
                    previous_windspeeds[counter] = out_ibs.params['mass' + str(j) + '_ionvelocity'].value
                ibs_velocities.append(np.mean(tempvelocities))
                ibs_velocities_stderr.append(np.std(tempvelocities))
            else:
                ibs_velocities.append(out_ibs.params['ionvelocity'].value)
                if out_ibs.params['ionvelocity'].stderr is None:
                    ibs_velocities_stderr.append(np.nan)
                elif out_ibs.params['ionvelocity'].stderr is not None:
                    ibs_velocities_stderr.append(out_ibs.params['ionvelocity'].stderr)

            ibs_scp.append(out_ibs.params['scp'].value)
            if out_ibs.params['scp'].stderr is None:
                ibs_scp_stderr.append(np.nan)
            elif out_ibs.params['scp'].stderr is not None:
                ibs_scp_stderr.append(out_ibs.params['scp'].stderr)

                if multipleionvelocity == False:
                    previous_windspeeds = out_ibs.params['ionvelocity'].value

            for mass in all_possible_masses:
                if 'mass' + str(mass) + "_center" in out_ibs.params.keys():
                    mass_energies_dict["ibsmass"+str(mass)].append(out_ibs.params['mass' + str(mass) + "_center"].value)
                else:
                    mass_energies_dict["ibsmass" + str(mass)].append(np.NaN)

            temp_temps = []
            for counter, j in enumerate(masses_used):
                if multipleionvelocity == True:
                    derived_temperature = 4 * (1 / (cassini_speed + previous_windspeeds[counter]) ** 2) * (
                            (out_ibs.params['mass' + str(j) + '_fwhm'].value * e) ** 2) * (1 / (2 * k * j * AMU))
                else:
                    derived_temperature = 4 * (1 / (cassini_speed + previous_windspeeds) ** 2) * (
                            (out_ibs.params['mass' + str(j) + '_fwhm'].value * e) ** 2) * (1 / (2 * k * j * AMU))
                temp_temps.append(derived_temperature / 4)
                #print("derived temperature", derived_temperature)
            last_ion_temp = np.mean(temp_temps)

            ibs_outputs.append(out_ibs)
            cassini_speeds.append(cassini_speed)

            x_titan.append(state[0])
            y_titan.append(state[1])
            z_titan.append(state[2])
            dx_titan.append(state[3])
            dy_titan.append(state[4])
            dz_titan.append(state[5])

            altitudes.append(alt)
            longitude.append(lon)
            latitude.append(lat)
            temps.append(last_ion_temp)
            state, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "NONE", "titan")
            ramdir = spice.vhat(state[3:6])
            titandir, state = titan_dir(tempdatetime)
            titandir_unorm = spice.vhat(titandir)
            parallel_to_titan = spice.rotvec(titandir_unorm, 90 * spice.rpd(), 3)
            parallel_to_titan_noZ = [parallel_to_titan[0], parallel_to_titan[1], 0]

            angle_2_zonal = spice.dpr() * spice.vsep(ramdir, parallel_to_titan_noZ)
            zonalangles.append(angle_2_zonal)
            zonalwinds_2008MW.append(-gaussian(lat, 0, 70 / 2.355, 60) * np.cos(angle_2_zonal * spice.rpd()))
            zonalwinds_2016.append(-gaussian(lat, 0, 70 / 2.355, 373)*np.cos(angle_2_zonal*spice.rpd()))
            zonalwinds_2017.append(-gaussian(lat, 3, 101 / 2.355, 196) * np.cos(angle_2_zonal * spice.rpd()))


        tempoutputdf = pd.DataFrame()
        tempoutputdf['Flyby'] = [flyby] * (end_ibs-start_ibs)
        tempoutputdf['Flyby velocity'] = cassini_speeds
        tempoutputdf['Positive Peak Time'] = ibsdata['times_utc'][start_ibs:end_ibs]
        tempoutputdf['X Titan'] = x_titan
        tempoutputdf['Y Titan'] = y_titan
        tempoutputdf['Z Titan'] = z_titan
        tempoutputdf['DX Titan'] = dx_titan
        tempoutputdf['DY Titan'] = dy_titan
        tempoutputdf['DZ Titan'] = dz_titan
        tempoutputdf['Altitude'] = altitudes
        tempoutputdf['Latitude'] = latitude
        tempoutputdf['Longitude'] = longitude
        tempoutputdf['Temperature'] = derived_temps
        tempoutputdf['IBS alongtrack velocity'] = ibs_velocities
        tempoutputdf['IBS alongtrack velocity stderr'] = ibs_velocities_stderr
        tempoutputdf['IBS spacecraft potentials'] = ibs_scp
        tempoutputdf['IBS spacecraft potentials stderr'] = ibs_scp_stderr
        tempoutputdf['LP Potentials'] = lpvalues
        tempoutputdf['Angles to Zonal Wind'] = zonalangles
        tempoutputdf['2008 MullerWodarg Winds'] = zonalwinds_2008MW
        tempoutputdf['2016 Zonal Winds'] = zonalwinds_2016
        tempoutputdf['2017 Zonal Winds'] = zonalwinds_2017
        tempoutputdf['Lower IBS Energy Bin'] = lowerenergyslices
        tempoutputdf['Upper IBS Energy Bin'] = upperenergyslices
        for i in all_possible_masses:
            tempoutputdf['IBS Mass ' + str(i) +' energy'] = mass_energies_dict["ibsmass"+str(i)]
        outputdf = pd.concat([outputdf, tempoutputdf])
    outputdf.reset_index(inplace=True)
    if len(usedflybys) == 1:
        outputdf.to_csv("nonactuatingflybys_alongtrackvelocity_ibs_" + usedflybys[0] + ".csv")
        if plotting == True:
            fig, (ibsax, altax, velax, scpax, tempax) = plt.subplots(5, sharex='col')

            x = ibsdata['times_utc'][start_ibs:end_ibs]
            y = ibscalib['ibsearray']
            z = np.zeros(shape=(len(x),len(y)))
            print("z",z.shape)

            print(tempoutputdf['Lower IBS Energy Bin'].iloc[0])
            for i in range(len(tempoutputdf['Positive Peak Time'])):
                z[i,:tempoutputdf['Lower IBS Energy Bin'].iloc[i]] = [0]*int(tempoutputdf['Lower IBS Energy Bin'].iloc[i])
                z[i, tempoutputdf['Lower IBS Energy Bin'].iloc[i]:tempoutputdf['Upper IBS Energy Bin'].iloc[i]] = ibsdata['ibsdata'][tempoutputdf['Lower IBS Energy Bin'].iloc[i]:tempoutputdf['Upper IBS Energy Bin'].iloc[i], 1, start_ibs+i]
                z[i, tempoutputdf['Upper IBS Energy Bin'].iloc[i]:] = [0]*int(len(y)-tempoutputdf['Upper IBS Energy Bin'].iloc[i])
            print(z)
            ibsax.pcolormesh(x, y, z.T, norm=LogNorm(vmin=1e3, vmax=5e5),
                             cmap='viridis')
            ibsax.set_ylabel("IBS  \n Fan 2 \n ev/q")
            ibsax.set_yscale("log")
            ibsax.set_ylim(3, 100)

            # for masscounter, x in enumerate([17, 28]):
            #     ibsax.scatter(np.array(ibsdata['times_utc'])[highalt_slicenumbers],
            #                   [i.params['mass' + str(x) + '_center'].value for i in
            #                    np.array(ibs_outs)[highalt_counters]], label='mass' + str(x),
            #                   color='C' + str(masscounter))
            #     velax.scatter(np.array(ibsdata['times_utc'])[highalt_slicenumbers],
            #                   [i.params['mass' + str(x) + '_ionvelocity'].value for i in
            #                    np.array(ibs_outs)[highalt_counters]], label='mass' + str(x),
            #                   color='C' + str(masscounter))
            for masscounter, x in enumerate([17, 28, 40, 53, 66, 78, 91]):
                ibsax.scatter(outputdf['Positive Peak Time'], outputdf['IBS Mass ' + str(x) +' energy'], label='mass' + str(x),
                              color='C' + str(masscounter),s=10)

            # for masscounter, x in enumerate([28, 40, 53, 66, 78, 91], 1):
            #     ibsax.scatter(np.array(ibsdata['times_utc'])[lowalt_slicenumbers],
            #                   [i.params['mass' + str(x) + '_center'].value for i in
            #                    np.array(ibs_outs)[lowalt_counters]], label='mass' + str(x),
            #                   color='C' + str(masscounter))

            velax.errorbar(x=outputdf['Positive Peak Time'], y=outputdf['IBS alongtrack velocity'],yerr=outputdf['IBS alongtrack velocity stderr'])

            altax.plot(outputdf['Positive Peak Time'], altitudes)
            scpax.errorbar(outputdf['Positive Peak Time'], outputdf['IBS spacecraft potentials'], yerr=outputdf['IBS spacecraft potentials stderr'])
            scpax.plot(outputdf['Positive Peak Time'], lpvalues, label="LP values")
            scpax.hlines(0, ibsdata['times_utc'][start_ibs], ibsdata['times_utc'][end_ibs], color='k')
            tempax.plot(outputdf['Positive Peak Time'], derived_temps, label="Derived Ion Temp")

            velax.set_ylabel("IBS \n -derived \n Ion Velocity")
            altax.set_ylabel("Altitudes \n [km]")
            scpax.set_ylabel("IBS \n -derived \n S/C potential")
            tempax.set_ylabel("Ion temp")
            velax.hlines(0, ibsdata['times_utc'][start_ibs], ibsdata['times_utc'][end_ibs], color='k')

            velax.set_ylim(-400,400)
            scpax.set_ylim(-1,1)

            plt.show()
    else:
        outputdf.to_csv("nonactuatingflybys_alongtrackvelocity_ibs.csv")

def non_actuating_alongtrackwinds_flybys_els(usedflybys=["t55","t56","t57","t58","t59"],plotting=False):
    outputdf = pd.DataFrame()
    for flyby in usedflybys:
        elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")

        start_els = CAPS_slicenumber(elsdata, flyby_datetimes_els[flyby][0])
        end_els = CAPS_slicenumber(elsdata, flyby_datetimes_els[flyby][1])

        highalt_slicenumbers, highalt_counters = [], []
        midalt_slicenumbers, midalt_counters  = [], []
        lowalt_slicenumbers, lowalt_counters  = [], []
        derived_temps = []

        els_outputs, els_datetimes, els_GOFvalues, els_velocities, els_scp, els_velocities_stderr, els_scp_stderr = [], [], [], [], [], [], []
        lpvalues, cassini_speeds, anode_used = [], [], []

        zonalangles, zonalwinds_2008MW, zonalwinds_2016, zonalwinds_2017 = [], [], [], []
        altitudes, longitude, latitude, temps = [], [], [], []
        x_titan, y_titan, z_titan, dx_titan, dy_titan, dz_titan = [], [], [], [], [], []
        lptimes = list(elsdata['times_utc'][start_els:end_els])
        lpdata = read_LP_V1(flyby)
        lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
        tempdatetime = elsdata['times_utc'][start_els]
        start_lp = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
        used_scp = start_lp + flyby_startparams_els[flyby][2]
        last_ion_temp = flyby_startparams_els[flyby][1]

        all_possible_masses_els = [26, 50, 79, 117]
        mass_energies_dict, mass_energies_dict_els = {}, {}
        for mass in all_possible_masses_els:
            mass_energies_dict_els["elsmass"+str(mass)] = []

        for counter, i in enumerate(np.arange(start_els,end_els)):
            tempdatetime = elsdata['times_utc'][i]
            lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps,
                                lpdata['SPACECRAFT_POTENTIAL'])
            lpvalues.append(lpvalue)
            et = spice.datetime2et(tempdatetime)
            state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
            et = spice.datetime2et(tempdatetime)
            alt, lat, lon = cassini_titan_altlatlon(tempdatetime)

            if alt < 1000:
                #els_masses = flyby_masses_els[flyby][2]
                lowerenergyslice_els = CAPS_energyslice("els", ELS_energybound_dict[flyby][2][0], ELS_energybound_dict[flyby][2][0])[0]
                upperenergyslice_els = CAPS_energyslice("els", ELS_energybound_dict[flyby][2][1], ELS_energybound_dict[flyby][2][1])[0]
                if counter == 0:
                    previous_windspeeds = 0
                last_ion_temp = titan_linearfit_temperature(alt)
                lowalt_slicenumbers.append(i)
                lowalt_counters.append(counter)
            elif alt > 1000 and alt < 1130:
                #els_masses = flyby_masses_els[flyby][1]
                lowerenergyslice_els = CAPS_energyslice("els", ELS_energybound_dict[flyby][1][0], ELS_energybound_dict[flyby][1][0])[0]
                upperenergyslice_els = CAPS_energyslice("els", ELS_energybound_dict[flyby][1][1], ELS_energybound_dict[flyby][1][1])[0]
                if counter == 0:
                    previous_windspeeds = flyby_startparams_els[flyby][0]
                last_ion_temp = titan_linearfit_temperature(alt)
                midalt_slicenumbers.append(i)
                midalt_counters.append(counter)
            elif alt > 1100:
                #els_masses = flyby_masses_els[flyby][0]
                lowerenergyslice_els = CAPS_energyslice("els", ELS_energybound_dict[flyby][0][0], ELS_energybound_dict[flyby][0][0])[0]
                upperenergyslice_els = CAPS_energyslice("els", ELS_energybound_dict[flyby][0][1], ELS_energybound_dict[flyby][0][1])[0]
                if counter == 0:
                    previous_windspeeds = flyby_startparams_els[flyby][0]
                    print(flyby,previous_windspeeds,counter)
                last_ion_temp = titan_linearfit_temperature(alt)
                highalt_slicenumbers.append(i)
                highalt_counters.append(counter)

            anode = ELS_maxflux_anode(elsdata, tempdatetime - datetime.timedelta(seconds=10),
                                      tempdatetime + datetime.timedelta(seconds=10))


            tempdataslice = np.float32(ELS_backgroundremoval(elsdata, start_els+counter, start_els+counter+1, datatype="data"))[
                           lowerenergyslice_els:upperenergyslice_els, anode, :].flatten()


            tempx = elscalib['earray'][lowerenergyslice_els:upperenergyslice_els]

            #print("first", tempdataslice)
            temp_counter_0 = 0
            while tempdataslice[temp_counter_0] == 0.:
                temp_counter_0  += 1
            tempx = tempx[temp_counter_0:]
            tempdataslice = tempdataslice[temp_counter_0:]
            #print("after", tempdataslice)
            tempdataslice[tempdataslice == 0.] = 1.


            peaks, properties = find_peaks(tempdataslice, height=1e3, distance=2)

            test_els_masses = []
            for peak in peaks:
                energy = tempx[peak]
                if energy > 1 and energy < 5:
                    test_els_masses = test_els_masses + [26]
                if energy > 5 and energy < 10:
                    test_els_masses = test_els_masses + [26, 50]
                if energy > 10 and energy < 14:
                    test_els_masses = test_els_masses + [26, 50, 79]
                if energy > 14:
                    test_els_masses = test_els_masses + [26, 50, 79, 117]

            els_masses = set(test_els_masses)
            # print("masses", els_masses)
            # print(alt, lowerenergyslice_els,upperenergyslice_els)
            # print(peaks,tempdataslice[peaks])

            # plt.plot(tempx,elsdata['data'][lowerenergyslice_els:upperenergyslice_els, anode, i])
            # plt.plot(tempx,tempdataslice)
            # plt.scatter(tempx[peaks],tempdataslice[peaks])
            # plt.yscale("log")
            # plt.show()

            highcounter = peaks[-1]
            # print(highest_energy_peak, counter, lowerenergyslice_ibs, upperenergyslice_ibs)
            # print(slice[counter+1],slice[counter])
            while tempdataslice[highcounter + 1] < tempdataslice[highcounter]:
                highcounter += 1
                if highcounter + 1 == len(tempdataslice):
                    break
            lowcounter = 0
            while tempdataslice[lowcounter + 1] < tempdataslice[lowcounter]:
                lowcounter += 1
                if lowcounter + 1 == len(tempdataslice):
                    break
            # if counter - peaks[-1] < 7:
            #     counter = peaks[-1] + 7
            # upperenergyslice_ibs = lowerenergyslice_ibs + counter
            #print(peaks[-1], counter, tempx[counter])
            tempdataslice = tempdataslice[lowcounter:highcounter + 1]
            tempx = tempx[lowcounter:highcounter + 1]

            #print("ion temp",last_ion_temp)
            derived_temps.append(last_ion_temp)
            cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3

            out_els = total_fluxgaussian(tempx,
                                         tempdataslice,
                                         masses=els_masses, cassini_speed=cassini_speed,
                                         windspeeds=previous_windspeeds, scpvalue=used_scp, lpvalue=lpvalue,
                                         temperature=last_ion_temp, charge=-1,
                                         multipleionvelocity=False)
            used_scp = out_els.params['scp'].value

            previous_windspeeds = out_els.params['ionvelocity'].value
            els_velocities.append(out_els.params['ionvelocity'].value)
            els_scp.append(out_els.params['scp'].value)


            if out_els.params['ionvelocity'].stderr is None:
                els_velocities_stderr.append(np.nan)
                print(out_els.fit_report(min_correl=0.7))
            elif out_els.params['ionvelocity'].stderr is not None:
                els_velocities_stderr.append(out_els.params['ionvelocity'].stderr)
            if out_els.params['scp'].stderr is None:
                els_scp_stderr.append(np.nan)
                print(out_els.fit_report(min_correl=0.7))
                stepplotfig_els, ax = plt.subplots()
                stepplotfig_els.suptitle(
                    "Histogram of " + elsdata['flyby'].upper() + " ELS data " + "at " + elsdata['times_utc_strings'][
                        i], fontsize=32)
                ax.step(tempx, tempdataslice, where='mid', label="ELS Data")
                ax.errorbar(tempx, tempdataslice, yerr=[np.sqrt(i) for i in tempdataslice], color='k', fmt='none')
                ax.plot(tempx, out_els.init_fit, 'b-', label='Initial Fit')
                ax.plot(tempx, out_els.best_fit, 'k-', label='Best Fitting')
                ax.set_yscale("log")
                # ax.set_xlim(lowerenergyslice_els, upperenergyslice_els)
                ax.set_ylim(0.9 * min(tempdataslice), 2 * max(tempdataslice))
                ax.set_ylabel("Counts [/s]", fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
                ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
                ax.minorticks_on()
                # ax.set_title(elsdata['times_utc_strings'][actualslice])
                ax.set_xlabel("Energy [eV/q]", fontsize=14)
                ax.legend()

                GOF = np.mean((abs(out_els.best_fit - tempdataslice) / tempdataslice) * 100)
                ax.text(0.6, 0.01,
                        "Ion wind = %2.0f ± %2.0f m/s" % (out_els.params['ionvelocity'], np.nan),
                        transform=ax.transAxes, fontsize=18)
                ax.text(0.6, .05,
                        "ELS-derived S/C Potential = %2.2f ± %2.2f V" % (out_els.params['scp'], np.nan),
                        transform=ax.transAxes, fontsize=18)
                ax.text(0.6, .09, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=ax.transAxes, fontsize=18)
                ax.text(0.6, .13, "My GOF = %2.0f %%" % GOF, transform=ax.transAxes, fontsize=18)

            elif out_els.params['scp'].stderr is not None:
                els_scp_stderr.append(out_els.params['scp'].stderr)

            for mass in all_possible_masses_els:
                if 'mass' + str(mass) + "_center" in out_els.params.keys():
                    mass_energies_dict_els["elsmass"+str(mass)].append(out_els.params['mass' + str(mass) + "_center"].value)
                else:
                    mass_energies_dict_els["elsmass" + str(mass)].append(np.NaN)

            els_outputs.append(out_els)
            cassini_speeds.append(cassini_speed)
            anode_used.append(anode)

            x_titan.append(state[0])
            y_titan.append(state[1])
            z_titan.append(state[2])
            dx_titan.append(state[3])
            dy_titan.append(state[4])
            dz_titan.append(state[5])

            altitudes.append(alt)
            longitude.append(lon)
            latitude.append(lat)
            temps.append(last_ion_temp)
            state, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "NONE", "titan")
            ramdir = spice.vhat(state[3:6])
            titandir, state = titan_dir(tempdatetime)
            titandir_unorm = spice.vhat(titandir)
            parallel_to_titan = spice.rotvec(titandir_unorm, 90 * spice.rpd(), 3)
            parallel_to_titan_noZ = [parallel_to_titan[0], parallel_to_titan[1], 0]

            angle_2_zonal = spice.dpr() * spice.vsep(ramdir, parallel_to_titan_noZ)
            zonalangles.append(angle_2_zonal)
            zonalwinds_2008MW.append(-gaussian(lat, 0, 70 / 2.355, 60) * np.cos(angle_2_zonal * spice.rpd()))
            zonalwinds_2016.append(-gaussian(lat, 0, 70 / 2.355, 373)*np.cos(angle_2_zonal*spice.rpd()))
            zonalwinds_2017.append(-gaussian(lat, 3, 101 / 2.355, 196) * np.cos(angle_2_zonal * spice.rpd()))


        tempoutputdf = pd.DataFrame()
        tempoutputdf['Flyby'] = [flyby] * (end_els-start_els)
        tempoutputdf['Flyby velocity'] = cassini_speeds
        tempoutputdf['Negative Peak Time'] = elsdata['times_utc'][start_els:end_els]
        tempoutputdf['X Titan'] = x_titan
        tempoutputdf['Y Titan'] = y_titan
        tempoutputdf['Z Titan'] = z_titan
        tempoutputdf['DX Titan'] = dx_titan
        tempoutputdf['DY Titan'] = dy_titan
        tempoutputdf['DZ Titan'] = dz_titan
        tempoutputdf['Altitude'] = altitudes
        tempoutputdf['Latitude'] = latitude
        tempoutputdf['Longitude'] = longitude
        tempoutputdf['Temperature'] = derived_temps
        tempoutputdf['Anode'] = anode_used
        tempoutputdf['ELS alongtrack velocity'] = els_velocities
        tempoutputdf['ELS alongtrack velocity stderr'] = els_velocities_stderr
        tempoutputdf['ELS spacecraft potentials'] = els_scp
        tempoutputdf['ELS spacecraft potentials stderr'] = els_scp_stderr
        tempoutputdf['LP Potentials'] = lpvalues
        tempoutputdf['Angles to Zonal Wind'] = zonalangles
        tempoutputdf['2008 MullerWodarg Winds'] = zonalwinds_2008MW
        tempoutputdf['2016 Zonal Winds'] = zonalwinds_2016
        tempoutputdf['2017 Zonal Winds'] = zonalwinds_2017
        for i in all_possible_masses_els:
            tempoutputdf['ELS Mass ' + str(i) +' energy'] = mass_energies_dict_els["elsmass"+str(i)]
        outputdf = pd.concat([outputdf, tempoutputdf])
    outputdf.reset_index(inplace=True)
    if len(usedflybys) == 1:
        outputdf.to_csv("nonactuatingflybys_alongtrackvelocity_els_" + usedflybys[0] + ".csv")
        if plotting == True:
            fig, (elsax0,elsax1, elsax2, altax, velax, scpax, tempax) = plt.subplots(7, sharex='col')
            anode3_backgroundremoved = np.array(list(
                np.float32(ELS_backgroundremoval(elsdata, start_els, end_els, datatype="data")[:, 2, :])))
            anode4_backgroundremoved = np.array(list(
                np.float32(ELS_backgroundremoval(elsdata, start_els, end_els, datatype="data")[:, 3, :])))
            anode5_backgroundremoved = np.array(list(
                np.float32(ELS_backgroundremoval(elsdata, start_els, end_els, datatype="data")[:, 4, :])))
            # anode6_backgroundremoved = np.array(list(
            #     np.float32(ELS_backgroundremoval(elsdata, start_els, end_els, datatype="data")[:, 5, :])))
            elsax0.pcolormesh(elsdata['times_utc'][start_els:end_els], elscalib['earray'],
                             anode3_backgroundremoved, norm=LogNorm(vmin=2e3, vmax=5e5),
                             cmap='viridis')
            elsax1.pcolormesh(elsdata['times_utc'][start_els:end_els], elscalib['earray'],
                             anode4_backgroundremoved, norm=LogNorm(vmin=2e3, vmax=5e5),
                             cmap='viridis')
            elsax2.pcolormesh(elsdata['times_utc'][start_els:end_els], elscalib['earray'],
                             anode5_backgroundremoved, norm=LogNorm(vmin=2e3, vmax=5e5),
                             cmap='viridis')
            elsax0.set_ylabel("ELS  \n Anode 3 \n ev/q")
            elsax0.set_yscale("log")
            elsax0.set_ylim(0.6, 100)
            elsax1.set_ylabel("ELS  \n Anode 4 \n ev/q")
            elsax1.set_yscale("log")
            elsax1.set_ylim(0.6, 100)
            elsax2.set_ylabel("ELS  \n Anode 5 \n ev/q")
            elsax2.set_yscale("log")
            elsax2.set_ylim(0.6, 100)
            # elsax3.set_ylabel("ELS  \n Anode 6 \n ev/q")
            # elsax3.set_yscale("log")
            # elsax3.set_ylim(0.6, 100)

            for masscounter, x in enumerate(all_possible_masses_els):
                elsax0.scatter(outputdf['Negative Peak Time'][outputdf['Anode']==2], outputdf['ELS Mass ' + str(x) + ' energy'][outputdf['Anode']==2],
                           label='mass' + str(x),
                           color='C' + str(masscounter),s=8)
                elsax1.scatter(outputdf['Negative Peak Time'][outputdf['Anode']==3], outputdf['ELS Mass ' + str(x) + ' energy'][outputdf['Anode']==3],
                           label='mass' + str(x),
                           color='C' + str(masscounter),s=8)
                elsax2.scatter(outputdf['Negative Peak Time'][outputdf['Anode']==4], outputdf['ELS Mass ' + str(x) + ' energy'][outputdf['Anode']==4],
                           label='mass' + str(x),
                           color='C' + str(masscounter),s=8)


            velax.errorbar(x=outputdf['Negative Peak Time'], y=outputdf['ELS alongtrack velocity'],
                           yerr=outputdf['ELS alongtrack velocity stderr'])
            altax.plot(outputdf['Negative Peak Time'], altitudes)
            scpax.errorbar(outputdf['Negative Peak Time'], outputdf['ELS spacecraft potentials'],
                           yerr=outputdf['ELS spacecraft potentials stderr'])
            scpax.plot(outputdf['Negative Peak Time'], lpvalues, label="LP values")
            tempax.plot(outputdf['Negative Peak Time'], derived_temps, label="Derived Ion Temp")

            velax.set_ylabel("ELS \n -derived \n Ion Velocity")
            altax.set_ylabel("Altitudes \n [km]")
            scpax.set_ylabel("ELS \n -derived \n S/C potential")
            tempax.set_ylabel("Ion temp")
            velax.hlines(0, elsdata['times_utc'][start_els], elsdata['times_utc'][end_els], color='k')

            velax.set_ylim(-300,200)
            scpax.set_ylim(-2.5,0)

            plt.show()
    else:
        outputdf.to_csv("nonactuatingflybys_alongtrackvelocity_els.csv")


def single_slice_test(flyby, slicenumber,multipleionvelocity=False):
    ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(ibsdata, flyby, "ibs")

    start_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes_ibs[flyby][0])
    actualslice = start_ibs+slicenumber
    lowerenergy_ibs = 4.2
    upperenergy_ibs = 18
    lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy_ibs , lowerenergy_ibs)[0]
    upperenergyslice_ibs = CAPS_energyslice("ibs", upperenergy_ibs, upperenergy_ibs)[0]

    lpdata = read_LP_V1(flyby)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]

    masses_used = [28, 40, 53, 66, 78, 91]
    lpvalues = []

    if multipleionvelocity == True:
        #previous_windspeeds = [0,0,0,0,0,0]
        #previous_windspeeds = [-1800,1900]
        previous_windspeeds = flyby_startparams_ibs[flyby][0]
    else:
        previous_windspeeds = 0

    tempdatetime = ibsdata['times_utc'][actualslice]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    lpvalues.append(lpvalue)
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    dataslice = ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs,1,actualslice]
    out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], dataslice,
                                 masses=masses_used, cassini_speed=cassini_speed, windspeeds=previous_windspeeds, scpvalue=lpvalue+flyby_startparams_ibs[flyby][2], lpvalue=lpvalue,  #[17, 28, 40, 53, 66, 78, 91],
                                 temperature=flyby_startparams_ibs[flyby][1], charge=1, multipleionvelocity=multipleionvelocity)
    previous_scp = out_ibs.params['scp'].value

    print(out_ibs.fit_report(min_correl=0.7))
    if multipleionvelocity == True:
        for counter, j in enumerate(masses_used):
            previous_windspeeds[counter] = out_ibs.params['mass'+str(j)+'_ionvelocity'].value
    else:
        previous_windspeeds = out_ibs.params['ionvelocity'].value

    # for counter, j in enumerate(masses_used):
    #     derived_temperature = 4 * (1 / (cassini_speed + previous_windspeeds[counter]) ** 2) * (
    #                 (out_ibs.params['mass' + str(j) + '_fwhm'].value*e) ** 2) * (1 / (2 * k * j * AMU))
    #     print("derived temperature", derived_temperature/4)

    x = ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs]

    stepplotfig_ibs, ax = plt.subplots()
    stepplotfig_ibs.suptitle("Histogram of " + ibsdata['flyby'].upper() + " IBS data", fontsize=32)
    ax.step(x, dataslice, where='mid')
    ax.errorbar(x, dataslice, yerr=[np.sqrt(i) for i in dataslice], color='k', fmt='none')
    ax.plot(x, out_ibs.init_fit, 'b-', label='init fit')
    ax.plot(x, out_ibs.best_fit, 'k-', label='best fit')
    ax.set_yscale("log")
    ax.set_xlim(lowerenergy_ibs, upperenergy_ibs)
    ax.set_ylim(0.9 * min(dataslice), 1.1 * max(dataslice))
    ax.set_ylabel("Counts [/s]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
    ax.minorticks_on()
    ax.set_title(ibsdata['times_utc_strings'][actualslice])
    ax.set_xlabel("Energy [eV/q]", fontsize=14)

    GOF = np.mean((abs(out_ibs.best_fit - dataslice) / dataslice) * 100)
    ax.text(0.6, 0.01,
                  "Ion wind = %2.0f ± %2.0f m/s" % (out_ibs.params['ionvelocity'], out_ibs.params['ionvelocity'].stderr),
                  transform=ax.transAxes, fontsize=18)
    ax.text(0.6, .05,
                  "IBS-derived S/C Potential = %2.2f ± %2.2f V" % (out_ibs.params['scp'], out_ibs.params['scp'].stderr),
                  transform=ax.transAxes, fontsize=18)
    ax.text(0.6, .09, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=ax.transAxes, fontsize=18)
    # ax.text(0.8, .20, "Temp = %2.2f" % temperature, transform=ax.transAxes)
    ax.text(0.6, .13, "Chi-square = %.2E" % out_ibs.chisqr, transform=ax.transAxes, fontsize=18)
    ax.text(0.6, .17, "My GOF = %2.0f %%" % GOF, transform=ax.transAxes, fontsize=18)

    plt.show()


def single_slice_test_els(flyby, slicenumber):
    elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    start_els = CAPS_slicenumber(elsdata, flyby_datetimes_els[flyby][0]) + slicenumber

    lpdata = read_LP_V1(flyby)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
    tempdatetime = elsdata['times_utc'][start_els]

    all_possible_masses_els = [26, 50, 79, 117]
    mass_energies_dict, mass_energies_dict_els = {}, {}
    for mass in all_possible_masses_els:
        mass_energies_dict_els["elsmass" + str(mass)] = []

    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps,
                        lpdata['SPACECRAFT_POTENTIAL'])
    used_scp = lpvalue - 1
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    alt, lat, lon = cassini_titan_altlatlon(tempdatetime)
    previous_windspeeds = 0

    if alt < 1000:
        lowerenergyslice_els = \
        CAPS_energyslice("els", ELS_energybound_dict[flyby][2][0], ELS_energybound_dict[flyby][2][0])[0]
        upperenergyslice_els = \
        CAPS_energyslice("els", ELS_energybound_dict[flyby][2][1], ELS_energybound_dict[flyby][2][1])[0]
        last_ion_temp = titan_linearfit_temperature(alt)
    elif alt > 1000 and alt < 1130:
        # els_masses = flyby_masses_els[flyby][1]
        lowerenergyslice_els = \
        CAPS_energyslice("els", ELS_energybound_dict[flyby][1][0], ELS_energybound_dict[flyby][1][0])[0]
        upperenergyslice_els = \
        CAPS_energyslice("els", ELS_energybound_dict[flyby][1][1], ELS_energybound_dict[flyby][1][1])[0]
        last_ion_temp = titan_linearfit_temperature(alt)
    elif alt > 1100:
        # els_masses = flyby_masses_els[flyby][0]
        lowerenergyslice_els = \
        CAPS_energyslice("els", ELS_energybound_dict[flyby][0][0], ELS_energybound_dict[flyby][0][0])[0]
        upperenergyslice_els = \
        CAPS_energyslice("els", ELS_energybound_dict[flyby][0][1], ELS_energybound_dict[flyby][0][1])[0]
        last_ion_temp = titan_linearfit_temperature(alt)

    anode = ELS_maxflux_anode(elsdata, tempdatetime - datetime.timedelta(seconds=10),
                              tempdatetime + datetime.timedelta(seconds=10))

    print(ELS_energybound_dict[flyby][0][0],ELS_energybound_dict[flyby][0][1])

    tempdataslice = np.float32(
        ELS_backgroundremoval(elsdata, start_els, start_els + 1, datatype="data"))[
                    lowerenergyslice_els:upperenergyslice_els, anode, :].flatten()

    tempx = elscalib['earray'][lowerenergyslice_els:upperenergyslice_els]

    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3

    temp_counter_0 = 0
    while tempdataslice[temp_counter_0] == 0.:
        temp_counter_0 += 1
    tempx = tempx[temp_counter_0:]
    tempdataslice = tempdataslice[temp_counter_0:]
    tempdataslice[tempdataslice == 0.] = 1.



    peaks, properties = find_peaks(tempdataslice)#, height=1e3, distance=2)
    print(peaks,tempx[peaks])

    test_els_masses = []
    for peak in peaks:
        energy = tempx[peak]
        if energy > 1 and energy < 5:
            test_els_masses = test_els_masses + [26]
        if energy > 5 and energy < 10:
            test_els_masses = test_els_masses + [26, 50]
        if energy > 10 and energy < 14:
            test_els_masses = test_els_masses + [26, 50, 79]
        if energy > 14:
            test_els_masses = test_els_masses + [26, 50, 79, 117]

    els_masses = set(test_els_masses)

    counter = peaks[-1]
    # print(highest_energy_peak, counter, lowerenergyslice_ibs, upperenergyslice_ibs)
    # print(slice[counter+1],slice[counter])
    while tempdataslice[counter + 1] < tempdataslice[counter]:
        counter += 1
        if counter + 1 == len(tempdataslice):
            break
    # if counter - peaks[-1] < 7:
    #     counter = peaks[-1] + 7
    #upperenergyslice_ibs = lowerenergyslice_ibs + counter
    print(peaks[-1],counter,tempx[counter])
    tempdataslice = tempdataslice[:counter+1]
    tempx = tempx[:counter+1]

    print(els_masses,alt,tempx[0],tempx[-1])

    print(used_scp)
    out_els = total_fluxgaussian(tempx,
                                 tempdataslice,
                                 masses=els_masses, cassini_speed=cassini_speed,
                                 windspeeds=previous_windspeeds, scpvalue=used_scp, lpvalue=lpvalue,
                                 temperature=last_ion_temp, charge=-1,
                                 multipleionvelocity=False)

    if out_els.params['ionvelocity'].stderr is None:
        vel_err = np.nan
        print(out_els.fit_report(min_correl=0.7))
    elif out_els.params['ionvelocity'].stderr is not None:
        vel_err = out_els.params['ionvelocity'].stderr
    if out_els.params['scp'].stderr is None:
        scp_err = np.nan
        print(out_els.fit_report(min_correl=0.7))
    elif out_els.params['scp'].stderr is not None:
        scp_err = out_els.params['scp'].stderr

    stepplotfig_els, ax = plt.subplots()
    stepplotfig_els.suptitle("Histogram of " + elsdata['flyby'].upper() + " ELS data " + "at " + elsdata['times_utc_strings'][start_els], fontsize=32)
    ax.step(tempx, tempdataslice, where='mid',label="ELS Data")
    ax.errorbar(tempx, tempdataslice, yerr=[np.sqrt(i) for i in tempdataslice], color='k', fmt='none')
    ax.plot(tempx, out_els.init_fit, 'b-', label='Initial Fit')
    ax.plot(tempx, out_els.best_fit, 'k-', label='Best Fitting')
    ax.set_yscale("log")
    #ax.set_xlim(lowerenergyslice_els, upperenergyslice_els)
    ax.set_ylim(0.9 * min(tempdataslice), 2 * max(tempdataslice))
    ax.set_ylabel("Counts [/s]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.25)
    ax.minorticks_on()
    #ax.set_title(elsdata['times_utc_strings'][actualslice])
    ax.set_xlabel("Energy [eV/q]", fontsize=14)
    ax.legend()

    GOF = np.mean((abs(out_els.best_fit - tempdataslice) / tempdataslice) * 100)
    ax.text(0.6, 0.01,
                  "Ion wind = %2.0f ± %2.0f m/s" % (out_els.params['ionvelocity'], vel_err),
                  transform=ax.transAxes, fontsize=18)
    ax.text(0.6, .05,
                  "ELS-derived S/C Potential = %2.2f ± %2.2f V" % (out_els.params['scp'], scp_err),
                  transform=ax.transAxes, fontsize=18)
    ax.text(0.6, .09, "LP-derived S/C Potential = %2.2f" % lpvalue, transform=ax.transAxes, fontsize=18)
    ax.text(0.6, .13, "My GOF = %2.0f %%" % GOF, transform=ax.transAxes, fontsize=18)
    print(out_els.var_names,out_els.init_vals)

    plt.show()



#non_actuating_alongtrackwinds_flybys_ibs(["t55","t56","t57","t58","t59"])
#non_actuating_alongtrackwinds_flybys_ibs(["t55"],plotting=True)
non_actuating_alongtrackwinds_flybys_els(["t55","t56","t57","t58","t59"])
#non_actuating_alongtrackwinds_flybys_els(["t59"],plotting=True)
#plotting("t55", multipleionvelocity=True)


#non_actuating_alongtrackwinds_flybys_ibs(["t59"],plotting=True)
#single_slice_test_els("t59",15)