import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import datetime
from heliopy.data.cassini import mag_1min
import spiceypy as spice
import glob
import scipy
import time

import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.colors as mcolors

############ 3D plot margin fix##################
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
############################################

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

AMU = scipy.constants.physical_constants['atomic mass constant'][0]
AMU_eV = scipy.constants.physical_constants['atomic mass unit-electron volt relationship'][0]
e = scipy.constants.physical_constants['atomic unit of charge'][0]
e_mass = scipy.constants.physical_constants['electron mass'][0]
e_mass_eV = scipy.constants.physical_constants['electron mass energy equivalent in MeV'][0] * 1e6
c = scipy.constants.physical_constants['speed of light in vacuum'][0]
k = scipy.constants.physical_constants['Boltzmann constant'][0]

def predicted_energy(mass_amu,fullionvelocity,lppotential):
    energy = 0.5*mass_amu*(AMU/e)*(fullionvelocity**2) - lppotential
    return energy

if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))





def magdata_magnitude_hires(tempdatetime,coords="KRTP"):
    start = tempdatetime - datetime.timedelta(seconds=31)
    end = tempdatetime + datetime.timedelta(seconds=31)
    magdata = mag_1min(start, end, coords).data['|B|']
    if magdata.size == 0:
        return np.NaN
    else:
        mag_magnitude = magdata[0]

    return mag_magnitude

#--Generating Windsdf----
# alongtrack_windsdf = pd.read_csv("alongtrackvelocity_unconstrained_refinedpeaks.csv", index_col=0, parse_dates=True)
# crosstrack_windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
#alongtrack_windsdf = pd.read_csv("singleflyby_alongtracktest.csv", index_col=0, parse_dates=True)
# alongtrack_windsdf = pd.read_csv("alongtrackvelocity.csv", index_col=0, parse_dates=True)
# crosstrack_windsdf = pd.read_csv("test.csv", index_col=0, parse_dates=True)
# windsdf = pd.concat([alongtrack_windsdf, crosstrack_windsdf], axis=1)
# windsdf = windsdf.loc[:, ~windsdf.columns.duplicated()]


windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)
#windsdf = pd.read_csv("Singleflyby_winds_full.csv", index_col=0, parse_dates=True)
flybyslist = windsdf.Flyby.unique()
print(flybyslist,len(flybyslist))


crary_windsdf = pd.read_csv("crarywinds.csv")
crary_windsdf = crary_windsdf[crary_windsdf['Flyby'].isin([i.upper() for i in flybyslist])]

#windsdf["IBS alongtrack velocity"] = windsdf["IBS alongtrack velocity"] + 180

FullIonVelocity = windsdf["IBS alongtrack velocity"] + windsdf["Flyby velocity"]
windsdf["FullIonVelocity"] = FullIonVelocity
# wind_predicted = windsdf["Flyby velocity"]
# lpvalues_predicted = windsdf["LP Potentials"]
# windsdf["PredictedMass28Energy"] = predicted_energy(28,wind_predicted ,lpvalues_predicted)
# windsdf["PredictedMass40Energy"] = predicted_energy(40,wind_predicted ,lpvalues_predicted)
# windsdf["PredictedMass53Energy"] = predicted_energy(53,wind_predicted ,lpvalues_predicted)
# windsdf["PredictedMass66Energy"] = predicted_energy(66,wind_predicted ,lpvalues_predicted)
# windsdf["PredictedMass78Energy"] = predicted_energy(78,wind_predicted ,lpvalues_predicted)
# windsdf["PredictedMass91Energy"] = predicted_energy(91,wind_predicted ,lpvalues_predicted)

# reduced_windsdf_ibsenergies = windsdf[["IBS alongtrack velocity", "FullIonVelocity", "IBS spacecraft potentials", "IBS Mass 28 energy",\
#                                        "IBS Mass 40 energy","IBS Mass 53 energy","IBS Mass 66 energy","IBS Mass 78 energy",\
#                                        "IBS Mass 91 energy","Flyby","PredictedMass28Energy", "PredictedMass40Energy",\
#                                        "PredictedMass53Energy", "PredictedMass66Energy","PredictedMass78Energy", "PredictedMass91Energy"]]#, "BT","Solar Zenith Angle",]]
# weirdflyby_list = ["t16", "t17", "t19", "t21", "t23", "t71", "t83"]
# weirdflybys_df = reduced_windsdf_ibsenergies[reduced_windsdf_ibsenergies['Flyby'].isin(weirdflyby_list)]
# normalflybys_df = reduced_windsdf_ibsenergies.loc[~reduced_windsdf_ibsenergies['Flyby'].isin(weirdflyby_list)]

def add_magdata(windsdf):
    mag_magnitudes = []
    for i in windsdf['Positive Peak Time']:
        temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")
        tempmag = magdata_magnitude_hires(temp)
        print("datetime",temp,"B Mag", tempmag)
        mag_magnitudes.append(tempmag)
    windsdf["BT"] = mag_magnitudes

def add_angle_to_corot(windsdf):
    angles_from_corot = []
    for i in windsdf['Positive Peak Time']:
        temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")
        et = spice.datetime2et(temp)
        state, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "NONE", "titan")
        ramdir = spice.vhat(state[3:6])
        #print("ramdir", ramdir)
        #print("Angle to Corot Direction", spice.dpr() * spice.vsepg(ramdir, [0, 1], 2))
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, [0, 1, 0]))
    windsdf["Angle2Corot"] = angles_from_corot

# def add_electricfield(windsdf):
#     ELS_crosstrack_Efield, ELS_alongtrack_Efield, IBS_crosstrack_Efield, IBS_alongtrack_Efield = [],[],[],[]
#     for i in windsdf['BT']:
#         temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")
#         tempmag = magdata_magnitude_hires(temp)
#         print("datetime",temp,"B Mag", tempmag)
#         mag_magnitudes.append(tempmag)
#     windsdf["BT"] = mag_magnitudes

# add_magdata(windsdf)
# add_angle_to_corot(windsdf)
# #windsdf.to_csv("singleflyby_winds_full.csv")
# windsdf.to_csv("winds_full.csv")

#------ All processing above here--------

southern_hemisphere_flybys = ['t36','t39','t40', 't41', 't42', 't48', 't49', 't50', 't51', 't71']

northern_flybys_df = windsdf.loc[~windsdf['Flyby'].isin(southern_hemisphere_flybys)]
southern_flybys_df = windsdf.loc[windsdf['Flyby'].isin(southern_hemisphere_flybys)]



def pairplots():
    lonlatfig, lonlatax = plt.subplots()
    sns.scatterplot(x=windsdf["Longitude"], y=windsdf["Latitude"], hue=windsdf["Flyby"], ax=lonlatax)
    lonlatax.set_xlabel("Longitude")
    lonlatax.set_ylabel("Latitude")


    reduced_windsdf1 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "IBS crosstrack velocity","ELS crosstrack velocity","Flyby", "BT"]]#, "BT","Solar Zenith Angle",]]
    reduced_windsdf2 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "Altitude", "Longitude", "Latitude","Flyby"]]
    reduced_windsdf3 = windsdf[["IBS spacecraft potentials", "ELS spacecraft potentials", "LP Potentials", "Actuation Direction","Flyby"]]#,"Solar Zenith Angle", ]]
    reduced_windsdf4 = windsdf[["Positive Peak Energy", "Negative Peak Energy", "IBS crosstrack velocity","ELS crosstrack velocity", "Actuation Direction"]]
    reduced_windsdf5 = windsdf[["IBS alongtrack velocity", "Altitude", "Flyby", "Angle2Corot" ]]# "Angle2Corot",,"Solar Zenith Angle",]]

    sns.pairplot(reduced_windsdf1,hue="Flyby")
    sns.pairplot(reduced_windsdf2,hue="Flyby")
    sns.pairplot(reduced_windsdf3,hue="Flyby")
    sns.pairplot(reduced_windsdf4)
    sns.pairplot(reduced_windsdf5,hue="Flyby")

# print(windsdf)
maxwind = 500

#------Alongtrack----
# alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
# alongtrack_ionvelocity_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
# sns.histplot(data=windsdf, x="IBS alongtrack velocity", bins=np.arange(-maxwind, maxwind, 50),
#              ax=alongtrack_ibs_ionvelocity_axdist, element="step",
#              kde=True,hue="Actuation Direction")
# sns.histplot(data=windsdf, x="ELS alongtrack velocity", bins=np.arange(-maxwind, maxwind, 50),
#              ax=alongtrack_els_ionvelocity_axdist, element="step",
#              kde=True,hue="Actuation Direction")

#------Crosstrack----
# alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
# alongtrack_ionvelocity_figdist.suptitle(str(crosstrack_windsdf.Flyby.unique()))
# sns.histplot(data=windsdf, x="IBS crosstrack velocity", bins=np.arange(-maxwind, maxwind, 50),
#              ax=alongtrack_ibs_ionvelocity_axdist, element="step",
#              kde=True,hue="Actuation Direction")
# sns.histplot(data=windsdf, x="ELS crosstrack velocity", bins=np.arange(-maxwind, maxwind, 50),
#              ax=alongtrack_els_ionvelocity_axdist, element="step",
#              kde=True,hue="Actuation Direction")


#----Regression Plots ----

def regplots():
    bothdf = windsdf.dropna(subset=["ELS alongtrack velocity"])
    regfig, (alongax,crossax) = plt.subplots(2)
    sns.regplot(data=bothdf, x="IBS alongtrack velocity", y="ELS alongtrack velocity",ax=alongax)
    z1 = np.polyfit(bothdf["IBS alongtrack velocity"],bothdf["ELS alongtrack velocity"],1)
    p1 = np.poly1d(z1)
    print(stats.pearsonr(bothdf["IBS alongtrack velocity"], bothdf["ELS alongtrack velocity"]))
    print(type(stats.pearsonr(bothdf["IBS alongtrack velocity"], bothdf["ELS alongtrack velocity"])))
    alongax.text(-300,250,str(stats.pearsonr(bothdf["IBS alongtrack velocity"], bothdf["ELS alongtrack velocity"])))
    alongax.text(-300,200,str(p1))

    testtuple = stats.pearsonr(bothdf["IBS crosstrack velocity"], bothdf["ELS crosstrack velocity"])
    sns.regplot(data=bothdf, x="IBS crosstrack velocity", y="ELS crosstrack velocity",ax=crossax)
    z2 = np.polyfit(bothdf["IBS crosstrack velocity"],bothdf["ELS crosstrack velocity"],1)
    p2 = np.poly1d(z2)
    crossax.text(-300,250,"({0[0]:.2f},{0[1]:.2f})".format(testtuple))
    crossax.text(-300,200,str(p2))

    alongax.set_xlim(-maxwind,maxwind)
    alongax.set_ylim(-maxwind,maxwind)
    crossax.set_xlim(-maxwind,maxwind)
    crossax.set_ylim(-maxwind,maxwind)

# regfig2, (ax2) = plt.subplots(1)
# sns.regplot(data=windsdf, x="Solar Zenith Angle", y="IBS spacecraft potentials",ax=ax2,label="IBS")
# sns.regplot(data=windsdf, x="Solar Zenith Angle", y="ELS spacecraft potentials",ax=ax2,label="ELS")
# ax2.set_ylabel("Spacecraft potential")
# ax2.legend()
# regfig3, (ax3) = plt.subplots(1)
# sns.regplot(data=windsdf, x="Solar Zenith Angle", y="IBS alongtrack velocity",ax=ax3,label="IBS")
# sns.regplot(data=windsdf, x="Solar Zenith Angle", y="ELS alongtrack velocity",ax=ax3,label="ELS")
# ax3.set_ylabel("Alongtrack Velocity")
# ax3.legend()

#
# regfig4, (ax5, ax6) = plt.subplots(2)
# sns.regplot(data=northern_flybys_df, x="Longitude", y="IBS alongtrack velocity",order=2,ax=ax5,label="IBS")
# sns.regplot(data=northern_flybys_df, x="Longitude", y="ELS alongtrack velocity",order=2,ax=ax5,label="ELS")
# ax5.set_xlabel(" ")
# ax5.set_ylabel("Alongtrack Velocity")
# ax5.legend()
# ax5.set_xlim(0,360)
# ax5.set_title("Northern Flybys")
#
# sns.regplot(data=southern_flybys_df, x="Longitude", y="IBS alongtrack velocity",order=2,ax=ax6,label="IBS")
# sns.regplot(data=southern_flybys_df, x="Longitude", y="ELS alongtrack velocity",order=2,ax=ax6,label="ELS")
# ax6.set_ylabel("Alongtrack Velocity")
# ax6.legend()
# ax6.set_xlim(0,360)
# ax6.set_title("Southern Flybys")
#
# regfig5, ax8 = plt.subplots()
# sns.regplot(data=windsdf, x="IBS alongtrack velocity", y="IBS crosstrack velocity",ax=ax8)
# ax5.set_xlabel(" ")
# ax5.set_ylabel("Alongtrack Velocity")

#
# dist_fig, (northdist_ax, southdist_ax) = plt.subplots(2)
# sns.histplot(data=northern_flybys_df, x="ELS alongtrack velocity", ax=northdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
# sns.histplot(data=northern_flybys_df, x="IBS alongtrack velocity", ax=northdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")
# sns.histplot(data=southern_flybys_df, x="ELS alongtrack velocity", ax=southdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
# sns.histplot(data=southern_flybys_df, x="IBS alongtrack velocity", ax=southdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")
#
# northdist_ax.set_xlim(-maxwind,maxwind)
# southdist_ax.set_xlim(-maxwind,maxwind)
# northdist_ax.legend()
# southdist_ax.legend()
# northdist_ax.set_title("Northern Flybys")
# southdist_ax.set_title("Southern Flybys")
# northdist_ax.set_xlabel("")
# southdist_ax.set_xlabel("Alongtrack Velocity")

def histplots():

    dist_fig2, (alongdist_ax, crossdist_ax) = plt.subplots(2)
    sns.histplot(data=windsdf, x="ELS alongtrack velocity", ax=alongdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
    sns.histplot(data=windsdf, x="IBS alongtrack velocity", ax=alongdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")
    sns.histplot(data=windsdf, x="ELS crosstrack velocity", ax=crossdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
    sns.histplot(data=windsdf, x="IBS crosstrack velocity", ax=crossdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")

    alongdist_ax.set_xlim(-maxwind,maxwind)
    crossdist_ax.set_xlim(-maxwind,maxwind)
    alongdist_ax.legend()
    crossdist_ax.legend()
    alongdist_ax.set_xlabel("Alongtrack Velocity")
    crossdist_ax.set_xlabel("Crosstrack Velocity")

##----Alongtrack Actuation direction
# neg_fig, neg_ax = plt.subplots()
# sns.stripplot(data=windsdf, x="Flyby", y="ELS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=neg_ax)
# pos_fig, pos_ax = plt.subplots()
# sns.stripplot(data=windsdf, x="Flyby", y="IBS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=pos_ax)

def vel_plot_byflyby():
    vel_fig, vel_ax = plt.subplots()
    sns.stripplot(data=windsdf, x="Flyby", y="ELS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=vel_ax)
    sns.stripplot(data=windsdf, x="Flyby", y="IBS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=vel_ax,marker="X")
    vel_ax.set_ylabel("Alongtrack Velocity")

def scp_plot_byflyby():
    scp_fig, scp_ax = plt.subplots()
    sns.stripplot(data=windsdf, x="Flyby", y="ELS spacecraft potentials", hue="Actuation Direction", dodge=False,ax=scp_ax)
    sns.stripplot(data=windsdf, x="Flyby", y="IBS spacecraft potentials", hue="Actuation Direction", dodge=False,ax=scp_ax,marker="X")
    scp_ax.set_ylabel("Derived Spacecraft Potential")


def lat_dist():
    northlatitude_df = windsdf[windsdf['Latitude'] > 30]
    lowlatitude_df = windsdf[(windsdf['Latitude'] > -30) & (windsdf['Latitude'] < 30)]
    southlatitude_df = windsdf[windsdf['Latitude'] < -30]

    dist_fig2, latitude_ax = plt.subplots()
    sns.histplot(data=northlatitude_df, x="IBS alongtrack velocity", ax=latitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="North Latitudes")
    sns.histplot(data=lowlatitude_df, x="IBS alongtrack velocity", ax=latitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="Low Latitudes")
    sns.histplot(data=southlatitude_df, x="IBS alongtrack velocity", ax=latitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C2',label="South Latitudes")
    latitude_ax.legend()

def lon_dist():
    flybys_315_45 = windsdf[(windsdf['Longitude'] > 315) | (windsdf['Longitude'] < 45)]
    flybys_45_135 = windsdf[(windsdf['Longitude'] > 45) & (windsdf['Longitude'] < 135)]
    flybys_135_225 = windsdf[(windsdf['Longitude'] > 135) & (windsdf['Longitude'] < 225)]
    flybys_225_315 = windsdf[(windsdf['Longitude'] > 225) & (windsdf['Longitude'] < 315)]

    dist_fig2, longitude_ax = plt.subplots()
    sns.histplot(data=flybys_315_45, x="IBS alongtrack velocity", ax=longitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="Saturn \n (Long 315-45 deg) ")
    sns.histplot(data=flybys_45_135, x="IBS alongtrack velocity", ax=longitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="Wake \n (Long 45-135 deg) ")
    sns.histplot(data=flybys_135_225, x="IBS alongtrack velocity", ax=longitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C2',label="Anti-Saturn \n (Long 135-225 deg) ")
    sns.histplot(data=flybys_225_315, x="IBS alongtrack velocity", ax=longitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C3',label="Ram \n (Long 225-315 deg) ")
    longitude_ax.legend()

def latitude_regplot():
    northlatitude_df = windsdf[windsdf['Latitude'] > 30]
    lowlatitude_df = windsdf[(windsdf['Latitude'] > -30) & (windsdf['Latitude'] < 30)]
    southlatitude_df = windsdf[windsdf['Latitude'] < -30]

    latitude_velocityfig, latitude_velocityaxes = plt.subplots(nrows=3,ncols=2,sharex='all',sharey='all')
    plt.subplots_adjust(hspace=0.3)

    testx= np.linspace(-200,200,400)
    sns.regplot(x="IBS alongtrack velocity", y="2016 Zonal Winds", data=northlatitude_df ,color='C0',ax=latitude_velocityaxes[0,0],label="2016 Winds")
    sns.regplot(x="IBS alongtrack velocity", y="2017 Zonal Winds", data=northlatitude_df ,color='C1',ax=latitude_velocityaxes[0,0],label="2017 Winds")
    sns.regplot(x="ELS alongtrack velocity", y="2016 Zonal Winds", data=northlatitude_df ,color='C0', marker='x', ax=latitude_velocityaxes[0,1],label="2016 Winds")
    sns.regplot(x="ELS alongtrack velocity", y="2017 Zonal Winds", data=northlatitude_df ,color='C1', marker='x', ax=latitude_velocityaxes[0,1],label="2017 Winds")

    sns.regplot(x="IBS alongtrack velocity", y="2016 Zonal Winds", data=lowlatitude_df ,color='C0',ax=latitude_velocityaxes[1,0],label="2016 Winds")
    sns.regplot(x="IBS alongtrack velocity", y="2017 Zonal Winds", data=lowlatitude_df ,color='C1',ax=latitude_velocityaxes[1,0],label="2017 Winds")
    sns.regplot(x="ELS alongtrack velocity", y="2016 Zonal Winds", data=lowlatitude_df ,color='C0', marker='x', ax=latitude_velocityaxes[1,1],label="2016 Winds")
    sns.regplot(x="ELS alongtrack velocity", y="2017 Zonal Winds", data=lowlatitude_df ,color='C1', marker='x', ax=latitude_velocityaxes[1,1],label="2017 Winds")

    # for counter, j in enumerate(["IBS alongtrack velocity", "ELS alongtrack velocity"]):
    #     for innercounter, i in enumerate(["2016 Zonal Winds", "2017 Zonal Winds"]):
    #         z2 = np.polyfit(lowlatitude_df[j],lowlatitude_df[i],1)
    #         p2 = np.poly1d(z2)
    #         print(j,i,"R = %.2f" % stats.pearsonr(lowlatitude_df[j], lowlatitude_df[i])[0],str(p2))
    #         #latitude_velocityaxes[1, counter].plot(testx,p2(testx),color='C2')
    #         latitude_velocityaxes[1, counter].text(0.6+(innercounter*0.2), 0.075, i + "\nR = %.2f\n" % stats.pearsonr(lowlatitude_df[j], lowlatitude_df[i])[0] + "y = %.2fx + %2.0f" % (p2[1],p2[0]),transform=latitude_velocityaxes[1, counter].transAxes,fontsize=10)
    #         #latitude_velocityaxes[1, counter].text(0.6, 0.15+(innercounter)*0.2, i + " " + "%.2f x + %2.2f" % (p2[1],p2[0]),transform=latitude_velocityaxes[1, counter].transAxes,fontsize=10)

    sns.regplot(x="IBS alongtrack velocity", y="2016 Zonal Winds", data=southlatitude_df ,color='C0',ax=latitude_velocityaxes[2,0],label="2016 Winds")
    sns.regplot(x="IBS alongtrack velocity", y="2017 Zonal Winds", data=southlatitude_df ,color='C1',ax=latitude_velocityaxes[2,0],label="2017 Winds")
    sns.regplot(x="ELS alongtrack velocity", y="2016 Zonal Winds", data=southlatitude_df ,color='C0', marker='x',ax=latitude_velocityaxes[2,1],label="2016 Winds")
    sns.regplot(x="ELS alongtrack velocity", y="2017 Zonal Winds", data=southlatitude_df ,color='C1', marker='x',ax=latitude_velocityaxes[2,1],label="2017 Winds")


    for ax in latitude_velocityaxes.flatten():
        # ax.set_xlabel(" ")
        # ax.set_ylabel("Cordiner+, 2020; \n Zonal Winds")
        ax.legend(loc=2)
    latitude_velocityaxes[1,0].plot([-200,200],[-200,200],color='k')
    latitude_velocityaxes[1,1].plot([-200,200],[-200,200],color='k')

    latitude_velocityaxes[0,0].set_title("Northern Latitudes, >30 degrees")
    latitude_velocityaxes[1,0].set_title("Low Latitudes, -30 to 30 degrees")
    latitude_velocityaxes[2,0].set_title("Southern Latitudes, <-30 degrees")
    latitude_velocityaxes[0,1].set_title("Northern Latitudes, >30 degrees")
    latitude_velocityaxes[1,1].set_title("Low Latitudes, -30 to 30 degrees")
    latitude_velocityaxes[2,1].set_title("Southern Latitudes, <-30 degrees")

    latitude_velocityaxes[2,0].set_xlabel("Alongtrack Ion Velocity")

# latitude_dist_ibs_fig, latitude_dist_ibs_ax = plt.subplots()
# sns.histplot(data=lowlatitude_df, x="IBS alongtrack velocity", ax=latitude_dist_ibs_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")
# sns.histplot(data=lowlatitude_df, x="2016 Zonal Winds", ax=latitude_dist_ibs_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C4',label="2016 Zonal Winds")
# sns.histplot(data=lowlatitude_df, x="2017 Zonal Winds", ax=latitude_dist_ibs_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C5',label="2017 Zonal Winds")
#
# latitude_dist_els_fig, latitude_dist_els_ax = plt.subplots()
# sns.histplot(data=lowlatitude_df, x="ELS alongtrack velocity", ax=latitude_dist_els_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")
# sns.histplot(data=lowlatitude_df, x="2016 Zonal Winds", ax=latitude_dist_els_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C4',label="2016 Zonal Winds")
# sns.histplot(data=lowlatitude_df, x="2017 Zonal Winds", ax=latitude_dist_els_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C5',label="2017 Zonal Winds")
#



# massenergy_fig, massenergy_axes = plt.subplots(nrows=6,ncols=2,sharex='col')
# for counter, i in enumerate([28, 40, 53, 66, 78, 91]):
#     sns.scatterplot(data=normalflybys_df ,x="FullIonVelocity",y="IBS Mass "+str(i)+" energy",hue="Flyby",ax=massenergy_axes[counter,0],legend=False)
#     sns.scatterplot(data=normalflybys_df , x="IBS spacecraft potentials", y="IBS Mass " + str(i) + " energy", hue="Flyby",
#                     ax=massenergy_axes[counter, 1], legend=False)
#     sns.scatterplot(data=weirdflybys_df,x="FullIonVelocity",y="IBS Mass "+str(i)+" energy",hue="Flyby",ax=massenergy_axes[counter,0],marker='x',legend=False)
#     sns.scatterplot(data=weirdflybys_df, x="IBS spacecraft potentials", y="IBS Mass " + str(i) + " energy", hue="Flyby",
#                     ax=massenergy_axes[counter, 1], marker='x', legend=False)
#     massenergy_axes[counter, 0].set_ylabel("IBS Mass\n"+str(i)+" energy")
#     massenergy_axes[counter, 1].set_ylabel("IBS Mass\n" + str(i) + " energy")

def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def regplots2():

    slow_flybys = ['t16', 't17', 't19', 't21', 't23', 't71', 't83']

    normal_df = windsdf.loc[~windsdf['Flyby'].isin(slow_flybys)]
    anomalous_df = windsdf.loc[windsdf['Flyby'].isin(slow_flybys)]
    massenergy_fig2, massenergy_axes2 = plt.subplots(nrows=6,ncols=2,sharex='col',sharey='row')
    for counter, i in enumerate([28, 40, 53, 66, 78, 91]):
        sns.regplot(data=normal_df,x="Angles to Zonal Wind",y="IBS Mass "+str(i)+" energy",color='k',ax=massenergy_axes2[counter, 0])
        sns.regplot(data=anomalous_df, x="Angles to Zonal Wind", y="IBS Mass " + str(i) + " energy", color='r',
                    ax=massenergy_axes2[counter, 1])
        massenergy_axes2[counter, 0].set_ylabel("IBS Mass\n"+str(i)+" energy")
        massenergy_axes2[counter, 1].set_ylabel("IBS Mass\n" + str(i) + " energy")


    latitudedist_fig, latitudedist_ax = plt.subplots()
    x = np.linspace(-90, 90, 1000)

    y_2016 = gaussian(x,0,70/2.355,373)
    y_2017 = gaussian(x,3,101/2.355,196)
    y_MW2006 = gaussian(x,0,70/2.355,60)

    windsdf["Absolute IBS Zonal Velocity"] = abs(windsdf['IBS alongtrack velocity'] * np.cos(windsdf['Angles to Zonal Wind']*spice.rpd()))
    sns.scatterplot(data=windsdf, x="Latitude", y="Absolute IBS Zonal Velocity",ax=latitudedist_ax,label="Absolute IBS Zonal Velocity")
    latitudedist_ax.plot(x, y_2016, color='r', label="August 2016")
    latitudedist_ax.fill_between(x, y_2016-26.6, y_2016+26.6, color='r',alpha=0.25)
    latitudedist_ax.plot(x, y_2017, color='b', label="May 2017")
    latitudedist_ax.fill_between(x, y_2017-14.5, y_2017+14.5, color='b',alpha=0.25)
    latitudedist_ax.plot(x, y_MW2006, color='g', label="Muller-Wodarg 2006 Model")
    latitudedist_ax.legend()
    latitudedist_ax.set_ylabel("Zonal Wind Speeds (m/s)")

    anglefig, angleaxes = plt.subplots(2)
    windsdf['Absolute IBS crosstrack velocity'] = abs(windsdf['IBS crosstrack velocity'])

    sns.regplot(data=windsdf,x="Angles to Zonal Wind", y="IBS alongtrack velocity", ax=angleaxes[0],color='C0')
    sns.regplot(data=windsdf,x="Angles to Zonal Wind", y="ELS alongtrack velocity", ax=angleaxes[0],color='C1')
    # z3 = np.polyfit(windsdf["Angles to Zonal Wind"], windsdf["IBS alongtrack velocity"], 1)
    # p3 = np.poly1d(z3)
    # angleaxes[0].text(0.8, 0.075,"R = %.2f\n" % stats.pearsonr(windsdf["Angles to Zonal Wind"], windsdf["IBS alongtrack velocity"])[
    #                                            0] + "y = %.2fx + %2.0f" % (p3[1], p3[0]),
    #                                        transform=angleaxes[0].transAxes, fontsize=14)

    sns.regplot(data=windsdf,x="Angles to Zonal Wind", y='IBS crosstrack velocity', ax=angleaxes[1],color='C0')
    sns.regplot(data=windsdf,x="Angles to Zonal Wind", y='ELS crosstrack velocity', ax=angleaxes[1],color='C1')
    angleaxes[0].set_ylabel("Alongtrack Velocity")
    angleaxes[1].set_ylabel("Crosstrack Velocity")


    # quickfig, quickaxes = plt.subplots()
    # sns.histplot(data=windsdf, x="Positive Deflection from Ram Angle", ax=quickaxes, bins=np.arange(-4, 4, 0.5), kde=True,color='C0',label="IBS")

    # quickfig2, quickaxes2 = plt.subplots()
    # markerlist = [".",",","x","v","^","<"]
    # for counter, i in enumerate([28, 40, 53, 66, 78, 91]):
    #     windsdf["IBS Mass "+str(i)+" energy normalised"] = windsdf["IBS Mass "+str(i)+" energy"]/max(windsdf["IBS Mass "+str(i)+" energy"])
    #     sns.scatterplot(data=windsdf, x="Angles to Zonal Wind", y="IBS Mass "+str(i)+" energy normalised", ax=quickaxes2,marker=markerlist[counter], hue="Flyby", label="IBS Mass "+str(i)+" energy")
    # quickaxes2.set_ylabel("IBS Energies - Normalised")
    # quickaxes2.get_legend().remove()

flybys_disagreeing = ["t16","t21","t36","t39","t40","t41","t42","t48","t49","t51","t83"]
flybys_agreeing = sorted(list(set(flybyslist) - set(flybys_disagreeing)))
flybys_disagreeing_midlatitude = ["t40","t41","t42","t48"]
#print(flybys_agreeing)

def alt_vel_plot_agree_disagree(flybyslist):

    fig, axes = plt.subplots(ncols=2,sharey='all')

    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS_disagree = len(flybys_disagreeing)
    NUM_COLORS_agree = len(flybys_agreeing)

    flybys_disagreeing_df = windsdf[windsdf['Flyby'].isin(flybys_disagreeing)]
    flybys_agreeing_df = windsdf[~windsdf['Flyby'].isin(flybys_disagreeing)]
    #print(flybys_disagreeing_df,flybys_agreeing_df)

   # print(pd.cut(flybys_disagreeing_df['IBS alongtrack velocity'], np.arange(950, 1850, 100)))
    avgVel_disagreeing_mean = flybys_disagreeing_df.groupby(pd.cut(flybys_disagreeing_df['Altitude'], np.arange(950, 1850, 100))).mean()
    avgVel_disagreeing_std = flybys_disagreeing_df.groupby(
        pd.cut(flybys_disagreeing_df['Altitude'], np.arange(950, 1850, 100))).std()
    avgVel_agreeing_mean = flybys_agreeing_df.groupby(
        pd.cut(flybys_agreeing_df['Altitude'], np.arange(950, 1850, 100))).mean()
    avgVel_agreeing_std = flybys_agreeing_df.groupby(
        pd.cut(flybys_agreeing_df['Altitude'], np.arange(950, 1850, 100))).std()

    print(avgVel_disagreeing_mean['IBS alongtrack velocity'],avgVel_disagreeing_std['IBS alongtrack velocity'])
    print(avgVel_agreeing_mean['IBS alongtrack velocity'], avgVel_agreeing_std['IBS alongtrack velocity'])

    axes[0].set_ylabel("Altitude [km]")
    for flybycounter, flyby in enumerate(flybys_agreeing):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)

            #print(flyby)
            minalt_index = tempdf["Altitude"].idxmin()
            #print(minalt_index)
            tempcolor = cm(1. * flybycounter / NUM_COLORS_agree)
            #print("here, agree")
            # sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="IBS alongtrack velocity", y="Altitude", ax=axes[0],color=tempcolor)
            # sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="IBS alongtrack velocity", y="Altitude", ax=axes[0],color=tempcolor)


            for counter in np.arange(0,minalt_index):
                axes[0].errorbar([tempdf['IBS alongtrack velocity'].iloc[counter], tempdf['IBS alongtrack velocity'].iloc[counter+1]],
                                 [tempdf['Altitude'].iloc[counter],tempdf['Altitude'].iloc[counter+1]],
                                 xerr=[tempdf['IBS alongtrack velocity stderr'].iloc[counter],tempdf['IBS alongtrack velocity stderr'].iloc[counter+1]],
                                 linestyle="-",color=tempcolor,label=flyby,alpha=0.35)
            for counter in np.arange(minalt_index,  tempdf.shape[0]-1):
                axes[0].errorbar([tempdf['IBS alongtrack velocity'].iloc[counter],tempdf['IBS alongtrack velocity'].iloc[counter + 1]],
                        [tempdf['Altitude'].iloc[counter], tempdf['Altitude'].iloc[counter + 1]],
                                 xerr=[tempdf['IBS alongtrack velocity stderr'].iloc[counter],
                                       tempdf['IBS alongtrack velocity stderr'].iloc[counter + 1]],
                                 linestyle="--",color=tempcolor,alpha=0.35)

    for flybycounter, flyby in enumerate(flybys_disagreeing):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)
            #print(flyby)
            minalt_index = tempdf["Altitude"].idxmin()
            #print(minalt_index)
            tempcolor = cm(1. * flybycounter / NUM_COLORS_disagree)
            #print("here, disagree")
            sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="IBS alongtrack velocity", y="Altitude", ax=axes[1],color=tempcolor,markers=['x'])
            sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="IBS alongtrack velocity", y="Altitude", ax=axes[1],color=tempcolor,markers=['x'])

            for counter in np.arange(0,minalt_index):
                axes[1].errorbar([tempdf['IBS alongtrack velocity'].iloc[counter],tempdf['IBS alongtrack velocity'].iloc[counter+1]],
                                 [tempdf['Altitude'].iloc[counter],tempdf['Altitude'].iloc[counter+1]],
                                 xerr=[tempdf['IBS alongtrack velocity stderr'].iloc[counter],
                                       tempdf['IBS alongtrack velocity stderr'].iloc[counter + 1]],
                                 linestyle="-",color=tempcolor,label=flyby,alpha=0.35)
            for counter in np.arange(minalt_index,  tempdf.shape[0]-1):
                axes[1].errorbar([tempdf['IBS alongtrack velocity'].iloc[counter],
                         tempdf['IBS alongtrack velocity'].iloc[counter + 1]],
                             [tempdf['Altitude'].iloc[counter], tempdf['Altitude'].iloc[counter + 1]],
                                 xerr=[tempdf['IBS alongtrack velocity stderr'].iloc[counter],
                                       tempdf['IBS alongtrack velocity stderr'].iloc[counter + 1]],
                                 linestyle="--",
                             color=tempcolor,alpha=0.35)
    print(len(avgVel_disagreeing_mean['IBS alongtrack velocity']),len(np.arange(1000, 1800, 100)))
    print(avgVel_disagreeing_mean['IBS alongtrack velocity'], np.arange(1000, 1800, 100),avgVel_disagreeing_std['IBS alongtrack velocity'])
    axes[0].errorbar(avgVel_disagreeing_mean['IBS alongtrack velocity'], np.arange(1000, 1800, 100),
                                 xerr=avgVel_disagreeing_std['IBS alongtrack velocity'],
                                 linestyle="-",color="k")
    axes[1].errorbar(avgVel_agreeing_mean['IBS alongtrack velocity'], np.arange(1000, 1800, 100),
                                 xerr=avgVel_agreeing_std['IBS alongtrack velocity'],
                                 linestyle="-",color="k")

    axes[0].set_title("Agreeing Flybys")
    axes[1].set_title("Disagreeing Flybys")
    axes[0].set_xlabel("Alongtrack Velocities [m/s]")
    minvel = -350
    maxvel= 350
    axes[0].set_xlim(minvel,maxvel)
    axes[1].set_xlim(minvel, maxvel)
    axes[0].legend()
    axes[1].legend()

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    #ax.set_ylim(bottom=950,top=1300)


def alt_vel_plot_region_positive(flybyslist,windsdf,lowlat=False,compare=False):

    if compare == False:
        fig, axes = plt.subplots(ncols=4,sharey='all',sharex='all')
    if compare == True:
        fig, axes = plt.subplots(ncols=5, sharey='all', sharex='all')

    if lowlat == True:
        windsdf = windsdf[(windsdf['Latitude'] > -50.5) & (windsdf['Latitude'] < 50.5)]

    flybys_315_45 = windsdf[(windsdf['Longitude'] > 315) | (windsdf['Longitude'] < 45)]
    flybys_45_135 = windsdf[(windsdf['Longitude'] > 45) & (windsdf['Longitude'] < 135)]
    flybys_135_225 = windsdf[(windsdf['Longitude'] > 135) & (windsdf['Longitude'] < 225)]
    flybys_225_315 = windsdf[(windsdf['Longitude'] > 225) & (windsdf['Longitude'] < 315)]

    #0 is Saturn
    #90 is Wake
    #180 is Anti-Saturn Facing
    #270 is Ram

    startalt = 950
    endalt = 1850
    intervalalt = 100


    # # print(pd.cut(flybys_disagreeing_df['IBS alongtrack velocity'], np.arange(950, 1850, 100)))
    avgVel_flybys_315_45_mean = flybys_315_45.groupby(pd.cut(flybys_315_45['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_315_45_std = flybys_315_45.groupby(pd.cut(flybys_315_45['Altitude'], np.arange(startalt, endalt, intervalalt))).std()
    avgVel_flybys_45_135_mean = flybys_45_135.groupby(pd.cut(flybys_45_135['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_45_135_std = flybys_45_135.groupby(pd.cut(flybys_45_135['Altitude'], np.arange(startalt, endalt, intervalalt))).std()
    avgVel_flybys_135_225_mean = flybys_135_225.groupby(pd.cut(flybys_135_225['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_135_225_std = flybys_135_225.groupby(pd.cut(flybys_135_225['Altitude'], np.arange(startalt, endalt, intervalalt))).std()
    avgVel_flybys_225_315_mean = flybys_225_315.groupby(pd.cut(flybys_225_315['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_225_315_std = flybys_225_315.groupby(pd.cut(flybys_225_315['Altitude'], np.arange(startalt, endalt, intervalalt))).std()


    for axescounter, flybyset in enumerate([flybys_315_45, flybys_45_135, flybys_135_225, flybys_225_315]):
        print(axescounter,sorted(set(flybyset['Flyby'])))
        for flybycounter, flyby in enumerate(sorted(set(flybyset['Flyby']))):

            tempdf = flybyset[flybyset['Flyby']==flyby]
            tempdf.reset_index(inplace=True)

            minalt_index = tempdf["Altitude"].idxmin()
            axes[axescounter].errorbar(tempdf['IBS alongtrack velocity'],tempdf['Altitude'],
                             xerr=tempdf['IBS alongtrack velocity stderr'],label=flyby,alpha=0.35)

    for axescounter, (mean,std) in enumerate(zip([avgVel_flybys_315_45_mean, avgVel_flybys_45_135_mean, avgVel_flybys_135_225_mean, avgVel_flybys_225_315_mean],
                                                 [avgVel_flybys_315_45_std, avgVel_flybys_45_135_std, avgVel_flybys_135_225_std, avgVel_flybys_225_315_std])):
        axes[axescounter].errorbar(mean['IBS alongtrack velocity'], np.arange(startalt+intervalalt/2, endalt-intervalalt/2, intervalalt),
                         xerr=std['IBS alongtrack velocity'],
                         linestyle="-", color="k")

    # for (mean,std,label) in zip([avgVel_flybys_315_45_mean, avgVel_flybys_45_135_mean, avgVel_flybys_135_225_mean, avgVel_flybys_225_315_mean],
    #                                              [avgVel_flybys_315_45_std, avgVel_flybys_45_135_std, avgVel_flybys_135_225_std, avgVel_flybys_225_315_std],
    #                             ["Saturn Facing", "Wake", "Anti-Saturn Facing", "Ram"]):
    if compare == True:
        for (mean, std, label) in zip([avgVel_flybys_315_45_mean, avgVel_flybys_135_225_mean],
            [avgVel_flybys_315_45_std, avgVel_flybys_135_225_std],
                                       ["Saturn Facing","Anti-Saturn Facing"]):
                axes[4].errorbar(mean['IBS alongtrack velocity'], np.arange(startalt+intervalalt/2, endalt-intervalalt/2, intervalalt),
                                 xerr=std['IBS alongtrack velocity'],
                                 linestyle="-",label=label)
                axes[4].set_xlabel("IBS Alongtrack \n Velocities [m/s]")

    axes[0].set_ylabel("Altitude [km]")
    axes[0].set_title("Saturn \n (Long 315-45 deg) ")
    axes[1].set_title("Wake \n (Long 45-135 deg) ")
    axes[2].set_title("Anti-Saturn \n (Long 135-225 deg) ")
    axes[3].set_title("Ram \n (Long 225-315 deg) ")
    axes[0].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    axes[1].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    axes[2].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    axes[3].set_xlabel("IBS Alongtrack \n Velocities [m/s]")

    minvel = -400
    maxvel= 400
    axes[0].set_xlim(minvel,maxvel)
    axes[0].legend()
    axes[1].legend()

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    #ax.set_ylim(bottom=950,top=1300)

def alt_vel_plot_region_negative(flybyslist,windsdf,lowlat=False,compare=False):

    if compare == False:
        fig, axes = plt.subplots(ncols=4,sharey='all',sharex='all')
    if compare == True:
        fig, axes = plt.subplots(ncols=5, sharey='all', sharex='all')

    if lowlat == True:
        windsdf = windsdf[(windsdf['Latitude'] > -50.5) & (windsdf['Latitude'] < 50.5)]

    flybys_315_45 = windsdf[(windsdf['Longitude'] > 315) | (windsdf['Longitude'] < 45)]
    flybys_45_135 = windsdf[(windsdf['Longitude'] > 45) & (windsdf['Longitude'] < 135)]
    flybys_135_225 = windsdf[(windsdf['Longitude'] > 135) & (windsdf['Longitude'] < 225)]
    flybys_225_315 = windsdf[(windsdf['Longitude'] > 225) & (windsdf['Longitude'] < 315)]

    #0 is Saturn
    #90 is Wake
    #180 is Anti-Saturn Facing
    #270 is Ram

    startalt = 950
    endalt = 1850
    intervalalt = 100


    # # print(pd.cut(flybys_disagreeing_df['IBS alongtrack velocity'], np.arange(950, 1850, 100)))
    avgVel_flybys_315_45_mean = flybys_315_45.groupby(pd.cut(flybys_315_45['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_315_45_std = flybys_315_45.groupby(pd.cut(flybys_315_45['Altitude'], np.arange(startalt, endalt, intervalalt))).std()
    avgVel_flybys_45_135_mean = flybys_45_135.groupby(pd.cut(flybys_45_135['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_45_135_std = flybys_45_135.groupby(pd.cut(flybys_45_135['Altitude'], np.arange(startalt, endalt, intervalalt))).std()
    avgVel_flybys_135_225_mean = flybys_135_225.groupby(pd.cut(flybys_135_225['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_135_225_std = flybys_135_225.groupby(pd.cut(flybys_135_225['Altitude'], np.arange(startalt, endalt, intervalalt))).std()
    avgVel_flybys_225_315_mean = flybys_225_315.groupby(pd.cut(flybys_225_315['Altitude'], np.arange(startalt, endalt, intervalalt))).mean()
    avgVel_flybys_225_315_std = flybys_225_315.groupby(pd.cut(flybys_225_315['Altitude'], np.arange(startalt, endalt, intervalalt))).std()


    for axescounter, flybyset in enumerate([flybys_315_45, flybys_45_135, flybys_135_225, flybys_225_315]):
        print(axescounter,sorted(set(flybyset['Flyby'])))
        for flybycounter, flyby in enumerate(sorted(set(flybyset['Flyby']))):

            tempdf = flybyset[flybyset['Flyby']==flyby]
            tempdf.reset_index(inplace=True)

            minalt_index = tempdf["Altitude"].idxmin()
            axes[axescounter].errorbar(tempdf['ELS alongtrack velocity'],tempdf['Altitude'],
                             xerr=tempdf['ELS alongtrack velocity stderr'],label=flyby,alpha=0.35)

    for axescounter, (mean,std) in enumerate(zip([avgVel_flybys_315_45_mean, avgVel_flybys_45_135_mean, avgVel_flybys_135_225_mean, avgVel_flybys_225_315_mean],
                                                 [avgVel_flybys_315_45_std, avgVel_flybys_45_135_std, avgVel_flybys_135_225_std, avgVel_flybys_225_315_std])):
        axes[axescounter].errorbar(mean['ELS alongtrack velocity'], np.arange(startalt+intervalalt/2, endalt-intervalalt/2, intervalalt),
                         xerr=std['ELS alongtrack velocity'],
                         linestyle="-", color="k")

    # for (mean,std,label) in zip([avgVel_flybys_315_45_mean, avgVel_flybys_45_135_mean, avgVel_flybys_135_225_mean, avgVel_flybys_225_315_mean],
    #                                              [avgVel_flybys_315_45_std, avgVel_flybys_45_135_std, avgVel_flybys_135_225_std, avgVel_flybys_225_315_std],
    #                             ["Saturn Facing", "Wake", "Anti-Saturn Facing", "Ram"]):
    if compare == True:
        for (mean, std, label) in zip([avgVel_flybys_315_45_mean, avgVel_flybys_135_225_mean],
            [avgVel_flybys_315_45_std, avgVel_flybys_135_225_std],
                                       ["Saturn Facing","Anti-Saturn Facing"]):
                axes[4].errorbar(mean['ELS alongtrack velocity'], np.arange(startalt+intervalalt/2, endalt-intervalalt/2, intervalalt),
                                 xerr=std['ELS alongtrack velocity'],
                                 linestyle="-",label=label)
                axes[4].set_xlabel("ELS Alongtrack \n Velocities [m/s]")

    axes[0].set_ylabel("Altitude [km]")
    axes[0].set_title("Saturn \n (Long 315-45 deg) ")
    axes[1].set_title("Wake \n (Long 45-135 deg) ")
    axes[2].set_title("Anti-Saturn \n (Long 135-225 deg) ")
    axes[3].set_title("Ram \n (Long 225-315 deg) ")
    axes[0].set_xlabel("ELS Alongtrack \n Velocities [m/s]")
    axes[1].set_xlabel("ELS Alongtrack \n Velocities [m/s]")
    axes[2].set_xlabel("ELS Alongtrack \n Velocities [m/s]")
    axes[3].set_xlabel("ELS Alongtrack \n Velocities [m/s]")

    minvel = -400
    maxvel= 400
    axes[0].set_xlim(minvel,maxvel)
    axes[0].legend()
    axes[1].legend()

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    #ax.set_ylim(bottom=950,top=1300)

def alt_vel_plot_long_sza(flybyslist, windsdf):
    fig, axes = plt.subplots(ncols=4, nrows=3, sharex='all',sharey='row')

    windsdf = windsdf[(windsdf['Latitude'] > -50.5) & (windsdf['Latitude'] < 50.5)]

    flybys_315_45 = windsdf[(windsdf['Longitude'] > 315) | (windsdf['Longitude'] < 45)] #Saturn
    flybys_45_135 = windsdf[(windsdf['Longitude'] > 45) & (windsdf['Longitude'] < 135)]
    flybys_135_225 = windsdf[(windsdf['Longitude'] > 135) & (windsdf['Longitude'] < 225)]#Anti-Saturn
    flybys_225_315 = windsdf[(windsdf['Longitude'] > 225) & (windsdf['Longitude'] < 315)]

    for axescounter, flybyset in enumerate([flybys_315_45, flybys_45_135, flybys_135_225, flybys_225_315]):
        print(axescounter, sorted(set(flybyset['Flyby'])))
        for flybycounter, flyby in enumerate(sorted(set(flybyset['Flyby']))):
            tempdf = flybyset[flybyset['Flyby'] == flyby]
            tempdf.reset_index(inplace=True)

            axes[0, axescounter].scatter(tempdf['Longitude'].iloc[0], tempdf['Latitude'].iloc[0], color='k', label=flyby, alpha=1,marker='x')
            axes[0,axescounter].scatter(tempdf['Longitude'], tempdf['Latitude'], label=flyby, alpha=0.5)
            axes[1, axescounter].scatter(tempdf['Longitude'].iloc[0], tempdf['Altitude'].iloc[0], color='k', label=flyby, alpha=1,marker='x')
            axes[1,axescounter].scatter(tempdf['Longitude'], tempdf['Altitude'], label=flyby, alpha=0.5)
            axes[2, axescounter].scatter(tempdf['Longitude'].iloc[0], tempdf['Solar Zenith Angle'].iloc[0], color='k',
                                         label=flyby, alpha=1,marker='x')
            axes[2,axescounter].scatter(tempdf['Longitude'], tempdf['Solar Zenith Angle'], label=flyby, alpha=0.5)

    #
    axes[0,0].set_title("Saturn \n (Long 315-45 deg) ")
    axes[0,1].set_title("Wake \n (Long 45-135 deg) ")
    axes[0,2].set_title("Anti-Saturn \n (Long 135-225 deg) ")
    axes[0,3].set_title("Ram \n (Long 225-315 deg) ")

    axes[0,0].set_ylabel("Latitude [km]")
    axes[1,0].set_ylabel("Altitude [km]")
    axes[2, 0].set_ylabel("Solar Zenith Angle [deg]")
    axes[2, 0].set_xlabel("Longitude [deg]")
    axes[2, 1].set_xlabel("Longitude [deg]")
    axes[2, 2].set_xlabel("Longitude [deg]")
    axes[2, 3].set_xlabel("Longitude [deg]")
    # axes[0].set_title("Saturn \n (Long 315-45 deg) ")
    # axes[1].set_title("Wake \n (Long 45-135 deg) ")
    # axes[2].set_title("Anti-Saturn \n (Long 135-225 deg) ")
    # axes[3].set_title("Ram \n (Long 225-315 deg) ")
    # axes[0].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    # axes[1].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    # axes[2].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    # axes[3].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    # axes[4].set_xlabel("IBS Alongtrack \n Velocities [m/s]")
    # minvel = -400
    # maxvel = 400
    # axes[0].set_xlim(minvel, maxvel)
    # axes[0].legend()
    # axes[1].legend()

    axes[0,0].invert_xaxis()


    for ax in [axes[0,0],axes[0,1],axes[0,2],axes[0,3]]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),ncol=2,fontsize=10)
    # ax.set_ylim(bottom=950,top=1300)

def box_titan_trajectory_plot(flybys,wake=True,sun=False,CA=False):
    '''
    Plots Cassini Trajectory on 2d plots in 3d box
    '''

    boundingvalue = 4500
    dotinterval = 60
    radius = 2575.15
    exobase = 1500

    lower_layer_alt = 1030
    middle_layer_alt = 1300
    upper_layer_alt = 1500


    t0 = time.time()
    # ADD Switch for planes

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d',box_aspect=(1,1,1))

    i = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(i)
    art3d.pathpatch_2d_to_3d(i, z=boundingvalue, zdir="x")
    j = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(j)
    art3d.pathpatch_2d_to_3d(j, z=boundingvalue, zdir="y")
    k = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(k)
    art3d.pathpatch_2d_to_3d(k, z=-boundingvalue, zdir="z")

    m = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(m)
    art3d.pathpatch_2d_to_3d(m, z=boundingvalue, zdir="x")
    n = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(n)
    art3d.pathpatch_2d_to_3d(n, z=boundingvalue, zdir="y")
    o = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(o)
    art3d.pathpatch_2d_to_3d(o, z=-boundingvalue, zdir="z")

    for number, flyby in enumerate(flybys):
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::, :]
        # ax.plot((-(singleflyby_df['X Titan'] ** 2 + singleflyby_df['Z Titan'] ** 2) ** 0.5), singleflyby_df['Y Titan'],
        #         zs=-boundingvalue, zdir='z', color='C' + str(number), label=flyby)

        ax.plot(singleflyby_df['X Titan'],singleflyby_df['Y Titan'],zs=-boundingvalue, zdir='z',color='C'+str(number),label=flyby)
        ax.plot(singleflyby_df['X Titan'],singleflyby_df['Z Titan'],zs=boundingvalue, zdir='y',color='C'+str(number))
        ax.plot(singleflyby_df['Y Titan'],singleflyby_df['Z Titan'],zs=boundingvalue, zdir='x',color='C'+str(number))

        if CA == True:
            ax.plot(singleflyby_df['X Titan'].iloc[0], singleflyby_df['Y Titan'].iloc[0], zs=-boundingvalue, zdir='z',
                    color='k', marker="^")
            ax.plot(singleflyby_df['X Titan'].iloc[0], singleflyby_df['Z Titan'].iloc[0], zs=boundingvalue, zdir='y',
                    color='k', marker="^")
            ax.plot(singleflyby_df['Y Titan'].iloc[0], singleflyby_df['Z Titan'].iloc[0], zs=boundingvalue, zdir='x',
                    color='k', marker="^")


    # if timedots == True:
    #     for et in np.arange(lower_et, upper_et, dotinterval):
    #         dotstate = cassini_phase(spice.et2utc(et, "ISOC", 0))
    #         ax.scatter(dotstate[0] / 252, dotstate[1] / 252, zs=-boundingvalue, zdir='z', c='k', marker=".", s=15)
    #         ax.scatter(dotstate[0] / 252, dotstate[2] / 252, zs=boundingvalue, zdir='y', c='k', marker=".", s=15)
    #         ax.scatter(dotstate[1] / 252, dotstate[2] / 252, zs=boundingvalue, zdir='x', c='k', marker=".", s=15)
    #
    #     ax.text2D(0.65, 0.97, r'$\cdot$ 1 minute markers  ', fontsize=12, transform=ax.transAxes)
    #
    if wake == True:
        # XY Wake
        ax.plot([-radius, -radius], [-boundingvalue, 0], zs=-boundingvalue, zdir='z', c='g')
        ax.plot([radius, radius], [-boundingvalue, 0], zs=-boundingvalue, zdir='z', c='g')

        wake_rect_xy = mpatches.Rectangle((-radius, -boundingvalue), 2*radius, boundingvalue, fill=True, facecolor='green', alpha=0.1)
        ax.add_patch(wake_rect_xy)
        art3d.pathpatch_2d_to_3d(wake_rect_xy, z=-boundingvalue, zdir="z")

        # YZ Wake
        ax.plot([-boundingvalue, 0], [-radius, -radius], zs=boundingvalue, zdir='x', c='g')
        ax.plot([-boundingvalue, 0], [radius, radius], zs=boundingvalue, zdir='x', c='g')

        wake_rect_yz = mpatches.Rectangle((-boundingvalue, -radius), boundingvalue, 2*radius, fill=True, facecolor='green', alpha=0.1)
        ax.add_patch(wake_rect_yz)
        art3d.pathpatch_2d_to_3d(wake_rect_yz, z=boundingvalue, zdir="x")

    if len(flybys) == 1 and sun == True:
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::, :]
        i = pd.to_datetime(singleflyby_df['Positive Peak Time']).iloc[0]

        et = spice.datetime2et(i)
        sundir, ltime = spice.spkpos('SUN', et, 'IAU_TITAN', "LT+S", 'TITAN')
        satdir, ltime = spice.spkpos('SATURN', et, 'IAU_TITAN', "LT+S", 'TITAN')
        sundir_unorm = spice.unorm(sundir)[0]
        satdir_unorm = spice.unorm(satdir)[0]
        print("sundir", sundir_unorm)
        print("satdir", satdir_unorm)
        #
        # ax.plot([0, satdir_unorm[0] * 1000], [0, satdir_unorm[1] * 1000],zs=-boundingvalue, zdir='z',color='r')
        # ax.plot([0, satdir_unorm[0] * 1000], [0, satdir_unorm[2] * 1000], zs=boundingvalue, zdir='y', color='r')
        # ax.plot([0, satdir_unorm[1] * 1000], [0, satdir_unorm[2] * 1000], zs=boundingvalue, zdir='x', color='r')

        # ax.plot([0, sundir_unorm[0] * 1000], [0, sundir_unorm[1] * 1000],zs=-boundingvalue, zdir='z',color='b')
        # ax.plot([0, sundir_unorm[0] * 1000], [0, sundir_unorm[2] * 1000], zs=boundingvalue, zdir='y', color='b')
        # ax.plot([0, sundir_unorm[1] * 1000], [0, sundir_unorm[2] * 1000], zs=boundingvalue, zdir='x', color='b')

        if sundir_unorm[1] < 0:
            angle = np.degrees(spice.vsepg([-1, 0], [sundir_unorm[0], sundir_unorm[1]]))
        else:
            angle = -np.degrees(spice.vsepg([-1, 0], [sundir_unorm[0], sundir_unorm[1]]))

        print("angle",angle)
        startangle = -90 + angle
        endangle = 90 + angle

        c = mpatches.Wedge((0, 0), radius, startangle, endangle, fill=True, color='k', alpha=0.4)
        ax.add_patch(c)
        art3d.pathpatch_2d_to_3d(c, z=-boundingvalue, zdir="z")

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    ax.view_init(elev=30., azim=-135)

    fig.legend()

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))

    ax.set_xlim(-boundingvalue, boundingvalue)
    ax.set_ylim(-boundingvalue, boundingvalue)
    ax.set_zlim(-boundingvalue, boundingvalue)

    t1 = time.time()
    print("Time Elapsed", t1 - t0)

def box_titan_trajectory_plot_cylindrical(flybys,wake=True,StartPoint=False):
    '''
    Plots Cassini Trajectory on 2d cylindrical plots in 3d box
    '''

    boundingvalue = 4500
    dotinterval = 60
    radius = 2575.15
    exobase = 1500

    lower_layer_alt = 1030
    middle_layer_alt = 1300
    upper_layer_alt = 1500


    t0 = time.time()
    # ADD Switch for planes

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d',box_aspect=(1,1,1))

    i = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(i)
    art3d.pathpatch_2d_to_3d(i, z=boundingvalue, zdir="x")
    j = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(j)
    art3d.pathpatch_2d_to_3d(j, z=boundingvalue, zdir="y")
    k = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(k)
    art3d.pathpatch_2d_to_3d(k, z=-boundingvalue, zdir="z")

    m = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(m)
    art3d.pathpatch_2d_to_3d(m, z=boundingvalue, zdir="x")
    n = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(n)
    art3d.pathpatch_2d_to_3d(n, z=boundingvalue, zdir="y")
    o = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(o)
    art3d.pathpatch_2d_to_3d(o, z=-boundingvalue, zdir="z")

    for number, flyby in enumerate(flybys):
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::, :]

        # ax.plot(singleflyby_df['X Titan'],singleflyby_df['Y Titan'],zs=-boundingvalue, zdir='z',color='C'+str(number),label=flyby,linestyle='--')
        # ax.plot(singleflyby_df['X Titan'],singleflyby_df['Z Titan'],zs=boundingvalue, zdir='y',color='C'+str(number),linestyle='--')
        ax.plot(singleflyby_df['Y Titan'],singleflyby_df['Z Titan'],zs=boundingvalue, zdir='x',color='C'+str(number))

        if singleflyby_df['X Titan'].iloc[int(len(singleflyby_df['X Titan'])/2)] < 0:
            ax.plot(-np.sqrt(singleflyby_df['X Titan'] ** 2 + singleflyby_df['Z Titan'] ** 2), singleflyby_df['Y Titan'],
                    zs=-boundingvalue, zdir='z', color='C' + str(number), label=flyby)
        else:
            ax.plot(np.sqrt(singleflyby_df['X Titan']**2 + singleflyby_df['Z Titan']**2),singleflyby_df['Y Titan'],zs=-boundingvalue, zdir='z',color='C'+str(number),label=flyby)

        if singleflyby_df['X Titan'].iloc[int(len(singleflyby_df['X Titan'])/2)] < 0:
            ax.plot(-np.sqrt(singleflyby_df['X Titan'] ** 2 + singleflyby_df['Z Titan'] ** 2), singleflyby_df['Z Titan']
                    ,zs=boundingvalue, zdir='y',color='C'+str(number))
        else:
            ax.plot(np.sqrt(singleflyby_df['X Titan']**2 + singleflyby_df['Z Titan']**2),singleflyby_df['Z Titan'],zs=boundingvalue, zdir='y',color='C'+str(number))

        # ax.plot(singleflyby_df['Y Titan'], singleflyby_df['Z Titan'], zs=boundingvalue, zdir='x',
        #         color='C' + str(number), linestyle='--')

        if StartPoint == True:
            ax.plot(singleflyby_df['X Titan'].iloc[0], singleflyby_df['Y Titan'].iloc[0], zs=-boundingvalue, zdir='z',
                    color='k', marker="^")
            ax.plot(singleflyby_df['X Titan'].iloc[0], singleflyby_df['Z Titan'].iloc[0], zs=boundingvalue, zdir='y',
                    color='k', marker="^")
            ax.plot(singleflyby_df['Y Titan'].iloc[0], singleflyby_df['Z Titan'].iloc[0], zs=boundingvalue, zdir='x',
                    color='k', marker="^")

    if wake == True:
        # XY Wake
        ax.plot([-radius, -radius], [-boundingvalue, 0], zs=-boundingvalue, zdir='z', c='g')
        ax.plot([radius, radius], [-boundingvalue, 0], zs=-boundingvalue, zdir='z', c='g')

        wake_rect_xy = mpatches.Rectangle((-radius, -boundingvalue), 2*radius, boundingvalue, fill=True, facecolor='green', alpha=0.1)
        ax.add_patch(wake_rect_xy)
        art3d.pathpatch_2d_to_3d(wake_rect_xy, z=-boundingvalue, zdir="z")

        # YZ Wake
        ax.plot([-boundingvalue, 0], [-radius, -radius], zs=boundingvalue, zdir='x', c='g')
        ax.plot([-boundingvalue, 0], [radius, radius], zs=boundingvalue, zdir='x', c='g')

        wake_rect_yz = mpatches.Rectangle((-boundingvalue, -radius), boundingvalue, 2*radius, fill=True, facecolor='green', alpha=0.1)
        ax.add_patch(wake_rect_yz)
        art3d.pathpatch_2d_to_3d(wake_rect_yz, z=boundingvalue, zdir="x")

    ax.set_xlabel('SqRt(X^2 + Z^2) (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=30., azim=-135)

    fig.legend()

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))


    ax.set_xlim(-boundingvalue, boundingvalue)
    ax.set_ylim(-boundingvalue, boundingvalue)
    ax.set_zlim(-boundingvalue, boundingvalue)


    t1 = time.time()
    print("Time Elapsed", t1 - t0)


def Titan_frame_plot(flybys):

    # average_latitude = np.mean(singleflyby_df['Latitude'])
    # print(average_latitude,np.cos(average_latitude*spice.rpd())*2575.15)
    # Titan_current_radius = np.cos(average_latitude*spice.rpd())*2575.15

    Titan_current_radius = 2575.15
    lower_layer_alt = 1030
    middle_layer_alt = 1300
    upper_layer_alt = 0


    TitanBody_xy = plt.Circle((0, 0), Titan_current_radius, color='y', alpha=0.5)
    TitanLower_xy = plt.Circle((0, 0), Titan_current_radius + lower_layer_alt, color='k', fill=False,
                               linestyle='--')
    TitanMiddle_xy = plt.Circle((0, 0), Titan_current_radius + middle_layer_alt, color='k', fill=False,
                                linestyle='-.')
    TitanExobase_xy = plt.Circle((0, 0), Titan_current_radius + upper_layer_alt, color='k', fill=False)
    TitanBody_yz = plt.Circle((0, 0), Titan_current_radius, color='y', alpha=0.5)
    TitanLower_yz = plt.Circle((0, 0), Titan_current_radius + lower_layer_alt, color='k', fill=False,
                               linestyle='--')
    TitanMiddle_yz = plt.Circle((0, 0), Titan_current_radius + middle_layer_alt, color='k', fill=False,
                                linestyle='-.')
    TitanExobase_yz = plt.Circle((0, 0), Titan_current_radius + upper_layer_alt, color='k', fill=False)
    TitanBody_xz = plt.Circle((0, 0), Titan_current_radius, color='y', alpha=0.5)
    TitanLower_xz = plt.Circle((0, 0), Titan_current_radius + lower_layer_alt, color='k', fill=False,
                               linestyle='--')
    TitanMiddle_xz = plt.Circle((0, 0), Titan_current_radius + middle_layer_alt, color='k', fill=False,
                                linestyle='-.')
    TitanExobase_xz = plt.Circle((0, 0), Titan_current_radius + upper_layer_alt, color='k', fill=False)
    TitanBody_cyl = plt.Circle((0, 0), Titan_current_radius, color='y', alpha=0.5, zorder=3)
    TitanLower_cyl = plt.Circle((0, 0), Titan_current_radius + lower_layer_alt, color='w', fill=True, alpha=0.5,
                                zorder=2)
    TitanMiddle_cyl = plt.Circle((0, 0), Titan_current_radius + middle_layer_alt, color='k', fill=True, alpha=0.5,
                                 zorder=1)
    TitanExobase_cyl = plt.Circle((0, 0), Titan_current_radius + upper_layer_alt, color='g', fill=False, zorder=0)

    figxy, axxy = plt.subplots(figsize=(8, 8), tight_layout=True)
    axxy.set_xlabel("X")
    axxy.set_ylabel("Y")
    axxy.set_xlim(-5000, 5000)
    axxy.set_ylim(-5000, 5000)
    axxy.add_artist(TitanBody_xy)
    axxy.add_artist(TitanLower_xy)
    axxy.add_artist(TitanMiddle_xy)
    axxy.add_artist(TitanExobase_xy)
    axxy.set_aspect("equal")
    figyz, axyz = plt.subplots(figsize=(8, 8), tight_layout=True)
    axyz.set_xlabel("Y")
    axyz.set_ylabel("Z")
    axyz.set_xlim(-5000, 5000)
    axyz.set_ylim(-5000, 5000)
    axyz.add_artist(TitanBody_yz)
    axyz.add_artist(TitanLower_yz)
    axyz.add_artist(TitanMiddle_yz)
    axyz.add_artist(TitanExobase_yz)
    axyz.set_aspect("equal")
    figxz, axxz = plt.subplots(figsize=(8, 8), tight_layout=True)
    axxz.set_xlabel("X")
    axxz.set_ylabel("Z")
    axxz.set_xlim(-5000, 5000)
    axxz.set_ylim(-5000, 5000)

    axxz.add_artist(TitanExobase_xz)
    axxz.add_artist(TitanMiddle_xz)
    axxz.add_artist(TitanLower_xz)
    axxz.add_artist(TitanBody_xz)
    axxz.set_aspect("equal")

    figcyl, axcyl = plt.subplots(figsize=(8, 8), tight_layout=True)
    axcyl.set_xlabel("Y")
    axcyl.set_ylabel("(X^1/2 + Z^1/2)^2")
    axcyl.set_xlim(-5000, 5000)
    axcyl.set_ylim(0, 5000)
    axcyl.add_artist(TitanBody_cyl)
    axcyl.add_artist(TitanLower_cyl)
    axcyl.add_artist(TitanMiddle_cyl)
    axcyl.add_artist(TitanExobase_cyl)
    axcyl.set_aspect("equal")

    axxy.plot([-2575.15, -2575.15], [0, -5000], color='k')
    axxy.plot([2575.15, 2575.15], [0, -5000], color='k')
    axyz.plot([0, -5000], [-2575.15, -2575.15], color='k')
    axyz.plot([0, -5000], [2575.15, 2575.15], color='k')
    axcyl.plot([0, -5000], [-2575.15, -2575.15], color='k', zorder=8)
    axcyl.plot([0, -5000], [2575.15, 2575.15], color='k', zorder=8)


    x1, y1, z1 = spice.pgrrec('TITAN', 90 * spice.rpd(), 30 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)
    x2, y2, z1 = spice.pgrrec('TITAN', 270 * spice.rpd(), 30 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    x1, y3, z2 = spice.pgrrec('TITAN', 90 * spice.rpd(), 50.5 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)
    x2, y4, z2 = spice.pgrrec('TITAN', 270 * spice.rpd(), 50.5 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    print(y1,y2,z1)
    print(y3,y4,z2)


    axyz.plot([y1, y2], [z1, z1], color='r')
    axyz.plot([y1, y2], [-z1, -z1], color='r')

    axyz.plot([y3, y4], [z2, z2], color='b')
    axyz.plot([y3, y4], [-z2, -z2], color='b')

    #
    # axyz.plot([-2575.15 * np.cos(np.radians(50.5)), 2575.15 * np.cos(np.radians(50.5))], [2575.15 * np.sin(np.radians(50.5)), 2575.15 * np.sin(np.radians(50.5))], color='b')
    # axyz.plot([-2575.15 * np.cos(np.radians(50.5)), 2575.15 * np.cos(np.radians(50.5))],
    #           [-2575.15 * np.sin(np.radians(50.5)), -2575.15 * np.sin(np.radians(50.5))], color='b')

    hue_norm = mcolors.CenteredNorm(vcenter=0, halfrange=150)
    for flyby in flybys:
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::, :]

        sns.scatterplot(x='X Titan', y='Y Titan', hue='IBS alongtrack velocity', hue_norm=hue_norm, label=flyby,
                        ax=axxy, palette='bwr', data=singleflyby_df, legend=False)
        sns.scatterplot(x='Y Titan', y='Z Titan', hue='IBS alongtrack velocity', hue_norm=hue_norm, label=flyby,
                        ax=axyz, palette='bwr', data=singleflyby_df, legend=False)
        sns.scatterplot(x='X Titan', y='Z Titan', hue='IBS alongtrack velocity', hue_norm=hue_norm, label=flyby,
                        ax=axxz, palette='bwr', data=singleflyby_df, legend=False)

        # axyz.plot(singleflyby_df['Y Titan'],singleflyby_df['Z Titan'],label=flyby)
        # axxz.plot(singleflyby_df['X Titan'],singleflyby_df['Z Titan'],label=flyby)
        # axcyl.plot(singleflyby_df['Y Titan'],(singleflyby_df['X Titan']**2 + singleflyby_df['Z Titan']**2)**0.5,label=flyby)
        sns.scatterplot(x=singleflyby_df['Y Titan'],
                        y=(singleflyby_df['X Titan'] ** 2 + singleflyby_df['Z Titan'] ** 2) ** 0.5,
                        hue=singleflyby_df['IBS alongtrack velocity'],
                        hue_norm=mcolors.CenteredNorm(vcenter=0, halfrange=50), label=flyby, ax=axcyl,
                        palette='bwr', legend=False, alpha=0.7, zorder=5)

        last_counter = 0
        for counter, (index, row) in enumerate(singleflyby_df.iterrows()):
            alongtrackvelocity = row['IBS alongtrack velocity']
            normed_vector = spice.unorm(np.array([row['DX Titan'], row['DY Titan'], row['DZ Titan']]))
            scale = 0.1
            scaled_dx = normed_vector[0][0] * -alongtrackvelocity * scale
            scaled_dy = normed_vector[0][1] * -alongtrackvelocity * scale
            scaled_dz = normed_vector[0][2] * -alongtrackvelocity * scale
            axxy.arrow(row['X Titan'], row['Y Titan'], scaled_dx, scaled_dy, head_width=30, width=2.5)
            axyz.arrow(row['Y Titan'], row['Z Titan'], scaled_dy, scaled_dz, head_width=30, width=2.5)
            axxz.arrow(row['X Titan'], row['Z Titan'], scaled_dx, scaled_dz, head_width=30, width=2.5)
            # axcyl.arrow(row['Y Titan'], (row['X Titan']**2 + row['Z Titan']**2)**0.5, scaled_dy, (scaled_dx**2 + scaled_dz**2)**0.5, head_width=50, width=2.5)

    for ax in [axxy, axxz, axyz]:
        ax.legend()


def Titan_cylindrical_plot(flybys):

    Titan_current_radius = 2575.15
    lower_layer_alt = 1030
    middle_layer_alt =  1300
    upper_layer_alt =  0

    figcyl, (axescyl_saturn,axescyl_antisaturn) = plt.subplots(nrows=2, sharex='all')

    axescyl_antisaturn.set_xlabel("y")
    axescyl_saturn.set_ylabel(r'$(x^{\frac{1}{2}} + z^{\frac{1}{2}})^{2}$')


    for axcyl in [axescyl_saturn,axescyl_antisaturn]:
        TitanBody_cyl = plt.Circle((0, 0), Titan_current_radius, color='y', alpha=0.5, zorder=3)
        TitanLower_cyl = plt.Circle((0, 0), Titan_current_radius + lower_layer_alt, color='w', fill=True, alpha=0.5,
                                    zorder=2)
        TitanMiddle_cyl = plt.Circle((0, 0), Titan_current_radius + middle_layer_alt, color='k', fill=True, alpha=0.5,
                                     zorder=1)
        TitanExobase_cyl = plt.Circle((0, 0), Titan_current_radius + upper_layer_alt, color='g', fill=False, zorder=0)
        axcyl.set_ylabel(r'$(x^{\frac{1}{2}} + z^{\frac{1}{2}})^{2}$')
        axcyl.set_xlim(-4000, 4000)
        axcyl.set_ylim(0, 4000)
        axcyl.add_artist(TitanBody_cyl)
        axcyl.add_artist(TitanLower_cyl)
        axcyl.add_artist(TitanMiddle_cyl)
        axcyl.add_artist(TitanExobase_cyl)
        axcyl.set_aspect("equal")
        axcyl.plot([0, -5000], [-2575.15, -2575.15], color='g', zorder=8)
        axcyl.plot([0, -5000], [2575.15, 2575.15], color='g', zorder=8)

    for flyby in flybys:
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[:, :]
        singleflyby_df = singleflyby_df[singleflyby_df['Altitude'] < 1500]

        ydata = (singleflyby_df['X Titan'] ** 2 + singleflyby_df['Z Titan'] ** 2) ** 0.5

        if singleflyby_df['X Titan'].iloc[0] < 0:
            axescyl_antisaturn.plot(singleflyby_df['Y Titan'], ydata, marker='', color='k')
            sns.scatterplot(x=singleflyby_df['Y Titan'],
                            y=ydata,
                            hue=singleflyby_df['IBS alongtrack velocity'],
                            hue_norm=mcolors.CenteredNorm(vcenter=0, halfrange=50), label=flyby, ax=axescyl_antisaturn, palette='bwr',
                            legend=False, alpha=0.7, zorder=5)

        if singleflyby_df['X Titan'].iloc[0] > 0:
            axescyl_saturn.plot(singleflyby_df['Y Titan'], ydata, marker='', color='k')
            sns.scatterplot(x=singleflyby_df['Y Titan'],
                            y=ydata,
                            hue=singleflyby_df['IBS alongtrack velocity'],
                            hue_norm=mcolors.CenteredNorm(vcenter=0, halfrange=50), label=flyby, ax=axescyl_saturn, palette='bwr',
                            legend=False, alpha=0.7, zorder=5)
    axescyl_saturn.arrow(0,500,0,500,width=25,head_width=100,zorder=9,fc='r')
    axescyl_antisaturn.arrow(0, 1000, 0, -500,width=25,head_width=100,zorder=9,fc='r')
    axescyl_saturn.text(-50,500,"Saturn",zorder=10,color='r')
    axescyl_antisaturn.text(-50, 1000, "Saturn",zorder=10,color='r')

    axescyl_antisaturn.invert_yaxis()
    axescyl_antisaturn.invert_xaxis()
    figcyl.subplots_adjust(wspace=0, hspace=0)

def regplot_crosstrack():
    limit = 600

    bothdf = windsdf.dropna(subset=["ELS alongtrack velocity"])
    #regfig, crossax = plt.subplots(1,figsize=(8,8))




    #sns.regplot(data=bothdf, x="IBS crosstrack velocity", y="ELS crosstrack velocity",ax=crossax)
    reggrid = sns.jointplot(data=bothdf, x="IBS crosstrack velocity", y="ELS crosstrack velocity",kind='reg',xlim=(-limit,limit),ylim=(-limit,limit))

    testtuple = stats.pearsonr(bothdf["IBS crosstrack velocity"], bothdf["ELS crosstrack velocity"])
    z2 = np.polyfit(bothdf["IBS crosstrack velocity"],bothdf["ELS crosstrack velocity"],1)
    p2 = "y = " + str(np.poly1d(z2))[2:]
    print(p2)
    reggrid.ax_joint.text(-100,-300,"Pearson's r = ({0[0]:.2f},{0[1]:.2f})".format(testtuple))
    reggrid.ax_joint.text(-100,-350,p2)
    plt.gcf().set_size_inches(8, 8)

def regplot_alongtrack():
    limit = 400

    bothdf = windsdf.dropna(subset=["ELS alongtrack velocity"])
    regfig, alongax = plt.subplots(1,figsize=(8,8))

    testtuple = stats.pearsonr(bothdf["IBS alongtrack velocity"], bothdf["ELS alongtrack velocity"])
    #sns.regplot(data=bothdf, x="IBS crosstrack velocity", y="ELS crosstrack velocity",ax=crossax)
    sns.jointplot(data=bothdf, x="IBS alongtrack velocity", y="ELS alongtrack velocity",ax=alongax,kind='reg',xlim=(-limit,limit),ylim=(-limit,limit))

    z2 = np.polyfit(bothdf["IBS alongtrack velocity"],bothdf["ELS alongtrack velocity"],1)
    p2 = np.poly1d(z2)
    alongax.text(-300,250,"({0[0]:.2f},{0[1]:.2f})".format(testtuple))
    alongax.text(-300,200,str(p2))


#scp_plot_byflyby()
#alt_vel_plot_region_positive(flybyslist,windsdf,lowlat=True,compare=True)
#alt_vel_plot_region_negative(flybyslist,windsdf,lowlat=True,compare=True)
#alt_vel_plot_agree_disgaree(flybyslist)
#alt_vel_plot_long_sza(flybyslist,windsdf)
# Titan_frame_plot(flybys_agreeing)
# Titan_frame_plot(flybys_disagreeing)
# Titan_frame_plot(flybys_disagreeing_midlatitude)

#Titan_cylindrical_plot(['t23','t25','t27','t50','t40','t41','t43','t48'])

regplot_crosstrack()
# regplot_alongtrack()

#box_titan_trajectory_plot(['t23','t25','t27','t50','t71','t40','t41','t42','t43','t48'],sun=False,CA=True)
#box_titan_trajectory_plot_cylindrical(['t23','t25','t27','t50','t71','t40','t41','t42','t43','t48'],StartPoint=False)

plt.show()
