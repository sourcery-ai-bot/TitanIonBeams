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

flyby_datetimes = {"t55": [datetime.datetime(2009, 5, 21, 21, 19, 6), datetime.datetime(2009, 5, 21, 21, 33)],
                   "t56": [datetime.datetime(2009, 6, 6,  19, 54, 45), datetime.datetime(2009, 6, 6, 20, 5, 57)],
                   "t57": [datetime.datetime(2009, 6, 22,  18, 26, 13), datetime.datetime(2009, 6, 22, 18, 40)],
                   "t58": [datetime.datetime(2009, 7, 8, 16, 58), datetime.datetime(2009, 7, 8, 17, 12, 10)],
                   "t59": [datetime.datetime(2009, 7, 24, 15, 25, 36), datetime.datetime(2009, 7, 24, 15, 41)],
                   }

flyby_startparams = {"t55": [[500, 500],1000,22],
                   "t56": [[500, 500],500,1],
                   "t57": [[500, 500],500,0.4],
                   "t58": [[500, 500],500,0.4],
                   "t59": [[1000, 4000],10000,10]}


IBS_fluxfitting_dict = {"mass15_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass16_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass17_": {"sigma": [0.1, 0.2, 0.3], "amplitude": []},
                        "mass28_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
                        "mass29_": {"sigma": [0.2, 0.2, 0.4], "amplitude": []},
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

IBS_FWHM = 0.014


def total_fluxgaussian(xvalues, yvalues, masses, cassini_speed, windspeeds, scpvalue, lpvalue, temperature, charge,
                       FWHM, multipleionvelocity=False):
    gaussmodels = []
    pars = Parameters()
    eval_pars = Parameters()

    if charge == 1:

        if multipleionvelocity == True:
            if scpvalue-0.5 < lpvalue:
                minscp = lpvalue + 0.25
            else:
                minscp = scpvalue-0.5
            if scpvalue + 2 < 0:
                maxscp = scpvalue + 2
            else:
                maxscp = 0
            pars.add('scp', value=scpvalue + 0.25, min=minscp, max=maxscp)
        else:
            pars.add('scp', value=scpvalue, min=lpvalue-0.5, max=0)

        # pars.add('scp', value=LPvalue+lpoffset)
        #pars['scp'].vary = False
    elif charge == -1:
        pars.add('scp', value=scpvalue, min=scpvalue - 2, max=0)
        # pars.add('scp', value=LPvalue+lpoffset)
        # pars['scp'].vary = False

    if temperature > 10000:
        temperature = 10000

    #print(temperature)
    pars.add('temp_eV', value=8 * k * temperature)  # , min=130, max=170)
    pars.add('spacecraftvelocity', value=cassini_speed)
    if multipleionvelocity == False:
        pars.add('ionvelocity', value=windspeeds, min=windspeeds-500, max=windspeeds+500)
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

        if multipleionvelocity == True:
            windmax = windspeeds[masscounter] + 500
            windmin = windspeeds[masscounter] - 500

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
        pars[tempprefix + 'amplitude'].set(value=np.max(yvalues) * (1 + sigmavals[1]), min=20*min(yvalues))

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

def plotting(flyby, multipleionvelocity=False):

    IBS_FWHM = 0.014

    elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(ibsdata, flyby, "ibs")

    start_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes[flyby][0])
    end_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes[flyby][1])
    lowerenergy_ibs = 3.5
    upperenergy_ibs = 55
    lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy_ibs , lowerenergy_ibs)[0]
    upperenergyslice_ibs = CAPS_energyslice("ibs", upperenergy_ibs, upperenergy_ibs)[0]

    start_els = CAPS_slicenumber(elsdata,flyby_datetimes[flyby][0])
    end_els = CAPS_slicenumber(elsdata,flyby_datetimes[flyby][1])
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
    used_scp = start_lp + flyby_startparams[flyby][2]
    last_ion_temp = flyby_startparams[flyby][1]


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
            if isinstance(previous_windspeeds,list):
                previous_windspeeds = np.mean(previous_windspeeds)
            if counter == 0:
                previous_windspeeds = flyby_startparams[flyby][0][0]
            last_ion_temp = titan_linearfit_temperature(alt)
            midalt_slicenumbers.append(i)
            midalt_counters.append(counter)
        elif alt > 1500:
            multipleionvelocity = True
            masses_used = [17, 28]
            if counter == 0:
                previous_windspeeds = flyby_startparams[flyby][0]
            if isinstance(previous_windspeeds,float):
                previous_windspeeds = [0, 0]
            highalt_slicenumbers.append(i)
            highalt_counters.append(counter)

        derived_temps.append(last_ion_temp)


        altitudes.append(alt)
        cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
        out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs,1,i],
                                     masses=masses_used, cassini_speed=cassini_speed, windspeeds=previous_windspeeds, scpvalue=used_scp, lpvalue=lpvalue,  #[17, 28, 40, 53, 66, 78, 91],
                                     temperature=last_ion_temp, charge=1, FWHM=IBS_FWHM, multipleionvelocity=multipleionvelocity)
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



def non_actuating_alongtrackwinds_flybys(usedflybys=["t55","t56","t57","t58","t59"]):
    outputdf = pd.DataFrame()
    for flyby in usedflybys:
        elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
        generate_mass_bins(elsdata, flyby, "els")
        ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
        generate_mass_bins(ibsdata, flyby, "ibs")

        start_ibs = CAPS_slicenumber(ibsdata, flyby_datetimes[flyby][0])
        end_ibs = CAPS_slicenumber(ibsdata, flyby_datetimes[flyby][1])
        lowerenergy_ibs = 3.5
        upperenergy_ibs = 55
        lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy_ibs, lowerenergy_ibs)[0]
        upperenergyslice_ibs = CAPS_energyslice("ibs", upperenergy_ibs, upperenergy_ibs)[0]

        start_els = CAPS_slicenumber(elsdata, flyby_datetimes[flyby][0])
        end_els = CAPS_slicenumber(elsdata, flyby_datetimes[flyby][1])
        lowerenergy_els = 4.1
        upperenergy_els = 19
        lowerenergyslice_els = CAPS_energyslice("els", lowerenergy_els, lowerenergy_els)[0]
        upperenergyslice_els = CAPS_energyslice("els", upperenergy_els, upperenergy_els)[0]

        highalt_slicenumbers, highalt_counters = [], []
        midalt_slicenumbers, midalt_counters  = [], []
        lowalt_slicenumbers, lowalt_counters  = [], []
        derived_temps = []

        ibs_outputs, ibs_datetimes, ibs_GOFvalues, lpvalues, cassini_speeds = [], [], [], [],[]
        els_outputs, els_datetimes, els_GOFvalues = [], [], []
        zonalangles, zonalwinds_2008MW, zonalwinds_2016, zonalwinds_2017 = [], [], [], []
        altitudes, longitude, latitude, temps = [], [], [], []
        x_titan, y_titan, z_titan, dx_titan, dy_titan, dz_titan = [], [], [], [], [], []
        ibs_velocities = []
        lptimes = list(ibsdata['times_utc'][start_ibs:end_ibs])
        lpdata = read_LP_V1(flyby)
        lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]
        tempdatetime = ibsdata['times_utc'][start_ibs]
        start_lp = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
        used_scp = start_lp + flyby_startparams[flyby][2]
        last_ion_temp = flyby_startparams[flyby][1]

        for counter, i in enumerate(np.arange(start_ibs,end_ibs)):
            tempdatetime = ibsdata['times_utc'][i]
            lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps,
                                lpdata['SPACECRAFT_POTENTIAL'])
            lpvalues.append(lpvalue)
            et = spice.datetime2et(tempdatetime)
            state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
            et = spice.datetime2et(tempdatetime)
            alt, lat, lon = cassini_titan_altlatlon(tempdatetime)

            if alt < 1100:
                multipleionvelocity = False
                masses_used = [28, 40, 53, 66, 78, 91]
                if counter == 0 or isinstance(previous_windspeeds, list):
                    previous_windspeeds = 0
                last_ion_temp = titan_linearfit_temperature(alt)
                lowalt_slicenumbers.append(i)
                lowalt_counters.append(counter)
            elif alt > 1100 and alt < 1500:
                multipleionvelocity = False
                masses_used = [17, 28, 40, 53, 66, 78, 91]
                if isinstance(previous_windspeeds, list):
                    previous_windspeeds = np.mean(previous_windspeeds)
                    #used_scp = lpvalue
                if counter == 0:
                    previous_windspeeds = flyby_startparams[flyby][0][0]
                last_ion_temp = titan_linearfit_temperature(alt)
                midalt_slicenumbers.append(i)
                midalt_counters.append(counter)
            elif alt > 1500:
                multipleionvelocity = True
                masses_used = [17, 28]
                if counter == 0:
                    previous_windspeeds = flyby_startparams[flyby][0]
                    print(flyby,previous_windspeeds)
                if isinstance(previous_windspeeds, float):
                    previous_windspeeds = [0, 0]
                highalt_slicenumbers.append(i)
                highalt_counters.append(counter)

            print("ion temp",last_ion_temp)
            derived_temps.append(last_ion_temp)
            cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
            out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs],
                                         ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs, 1, i],
                                         masses=masses_used, cassini_speed=cassini_speed,
                                         windspeeds=previous_windspeeds, scpvalue=used_scp, lpvalue=lpvalue,
                                         # [17, 28, 40, 53, 66, 78, 91],
                                         temperature=last_ion_temp, charge=1, FWHM=IBS_FWHM,
                                         multipleionvelocity=multipleionvelocity)
            used_scp = out_ibs.params['scp'].value

            if multipleionvelocity == True:
                tempvelocities = []
                for counter, j in enumerate(masses_used):
                    tempvelocities.append(out_ibs.params['mass' + str(j) + '_ionvelocity'].value)
                    previous_windspeeds[counter] = out_ibs.params['mass' + str(j) + '_ionvelocity'].value
                ibs_velocities.append(np.mean(tempvelocities))
            else:
                previous_windspeeds = out_ibs.params['ionvelocity'].value
                ibs_velocities.append(out_ibs.params['ionvelocity'].value)

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

            #els_outputs.append(els_output)

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
        tempoutputdf['IBS spacecraft potentials'] = [i.params['scp'].value for i in ibs_outputs]
        #tempoutputdf['ELS alongtrack velocity'] = [i.params['ionvelocity'].value for i in els_outputs]
        #tempoutputdf['ELS spacecraft potentials'] = [i.params['scp'].value for i in els_outputs]
        tempoutputdf['LP Potentials'] = lpvalues
        tempoutputdf['Angles to Zonal Wind'] = zonalangles
        tempoutputdf['2008 MullerWodarg Winds'] = zonalwinds_2008MW
        tempoutputdf['2016 Zonal Winds'] = zonalwinds_2016
        tempoutputdf['2017 Zonal Winds'] = zonalwinds_2017
        # tempoutputdf['IBS Mass 28 energy'] = [i.params['mass28_center'].value for i in ibs_outputs]
        # tempoutputdf['IBS Mass 40 energy'] = [i.params['mass40_center'].value for i in ibs_outputs]
        # tempoutputdf['IBS Mass 53 energy'] = [i.params['mass53_center'].value for i in ibs_outputs]
        # tempoutputdf['IBS Mass 66 energy'] = [i.params['mass66_center'].value for i in ibs_outputs]
        # tempoutputdf['IBS Mass 78 energy'] = [i.params['mass78_center'].value for i in ibs_outputs]
        # tempoutputdf['IBS Mass 91 energy'] = [i.params['mass91_center'].value for i in ibs_outputs]
        print(tempoutputdf)
        outputdf = pd.concat([outputdf, tempoutputdf])
    outputdf.reset_index(inplace=True)
    if len(usedflybys) == 1:
        outputdf.to_csv("nonactuatingflybys_alongtrackvelocity_" + usedflybys[0] + ".csv")
    else:
        outputdf.to_csv("nonactuatingflybys_alongtrackvelocity.csv")


def single_slice_test(flyby, slicenumber,multipleionvelocity=False):
    elsdata = readsav("data/els/elsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(elsdata, flyby, "els")
    ibsdata = readsav("data/ibs/ibsres_" + filedates_times[flyby][0] + ".dat")
    generate_mass_bins(ibsdata, flyby, "ibs")

    start_ibs = CAPS_slicenumber(ibsdata,flyby_datetimes[flyby][0])
    actualslice = start_ibs+slicenumber
    lowerenergy_ibs = 4.1
    upperenergy_ibs = 55
    lowerenergyslice_ibs = CAPS_energyslice("ibs", lowerenergy_ibs , lowerenergy_ibs)[0]
    upperenergyslice_ibs = CAPS_energyslice("ibs", upperenergy_ibs, upperenergy_ibs)[0]

    lpdata = read_LP_V1(flyby)
    lp_timestamps = [datetime.datetime.timestamp(d) for d in lpdata['datetime']]

    masses_used = [16, 28] #[17, 28, 40, 53, 66, 78, 91],
    lpvalues = []

    if multipleionvelocity == True:
        #previous_windspeeds = [0,0,0,0,0,0]
        #previous_windspeeds = [-1800,1900]
        previous_windspeeds = flyby_startparams[flyby][0]
    else:
        previous_windspeeds = 1000

    tempdatetime = ibsdata['times_utc'][actualslice]
    lpvalue = np.interp(datetime.datetime.timestamp(tempdatetime), lp_timestamps, lpdata['SPACECRAFT_POTENTIAL'])
    lpvalues.append(lpvalue)
    et = spice.datetime2et(tempdatetime)
    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    cassini_speed = np.sqrt((state[3]) ** 2 + (state[4]) ** 2 + (state[5]) ** 2) * 1e3
    dataslice = ibsdata['ibsdata'][lowerenergyslice_ibs:upperenergyslice_ibs,1,actualslice]
    out_ibs = total_fluxgaussian(ibscalib['ibsearray'][lowerenergyslice_ibs:upperenergyslice_ibs], dataslice,
                                 masses=masses_used, cassini_speed=cassini_speed, windspeeds=previous_windspeeds, scpvalue=lpvalue+flyby_startparams[flyby][2], lpvalue=lpvalue,  #[17, 28, 40, 53, 66, 78, 91],
                                 temperature=flyby_startparams[flyby][1], charge=1, FWHM=IBS_FWHM, multipleionvelocity=multipleionvelocity)
    previous_scp = out_ibs.params['scp'].value

    print(out_ibs.fit_report(min_correl=0.7))
    if multipleionvelocity == True:
        for counter, j in enumerate(masses_used):
            previous_windspeeds[counter] = out_ibs.params['mass'+str(j)+'_ionvelocity'].value
    else:
        previous_windspeeds = out_ibs.params['ionvelocity'].value

    for counter, j in enumerate(masses_used):
        derived_temperature = 4 * (1 / (cassini_speed + previous_windspeeds[counter]) ** 2) * (
                    (out_ibs.params['mass' + str(j) + '_fwhm'].value*e) ** 2) * (1 / (2 * k * j * AMU))
        print("derived temperature", derived_temperature/4)

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

    plt.show()




non_actuating_alongtrackwinds_flybys(["t55","t56","t57","t58","t59"])
#non_actuating_alongtrackwinds_flybys(["t57"])
#plotting("t59", multipleionvelocity=True)
#single_slice_test("t55",0,multipleionvelocity=True)