import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import datetime
from heliopy.data.cassini import mag_1min
import spiceypy as spice
import glob
import scipy

from matplotlib.lines import Line2D

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
alongtrack_windsdf = pd.read_csv("alongtrackvelocity_unconstrained_2dfitting_2peaks.csv", index_col=0, parse_dates=True)
crosstrack_windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
windsdf = pd.concat([alongtrack_windsdf, crosstrack_windsdf], axis=1)
windsdf = windsdf.loc[:, ~windsdf.columns.duplicated()]

# windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)
flybyslist = windsdf.Flyby.unique()
print(flybyslist)


crary_windsdf = pd.read_csv("crarywinds.csv")
crary_windsdf = crary_windsdf[crary_windsdf['Flyby'].isin([i.upper() for i in flybyslist])]

FullIonVelocity = windsdf["IBS alongtrack velocity"] + windsdf["Flyby velocity"]
windsdf["FullIonVelocity"] = FullIonVelocity
wind_predicted = windsdf["Flyby velocity"]
lpvalues_predicted = windsdf["LP Potentials"]
windsdf["PredictedMass28Energy"] = predicted_energy(28,wind_predicted ,lpvalues_predicted)
windsdf["PredictedMass40Energy"] = predicted_energy(40,wind_predicted ,lpvalues_predicted)
windsdf["PredictedMass53Energy"] = predicted_energy(53,wind_predicted ,lpvalues_predicted)
windsdf["PredictedMass66Energy"] = predicted_energy(66,wind_predicted ,lpvalues_predicted)
windsdf["PredictedMass78Energy"] = predicted_energy(78,wind_predicted ,lpvalues_predicted)
windsdf["PredictedMass91Energy"] = predicted_energy(91,wind_predicted ,lpvalues_predicted)

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
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, [0, -1, 0]))
    windsdf["Angle2Corot"] = angles_from_corot

# def add_electricfield(windsdf):
#     ELS_crosstrack_Efield, ELS_alongtrack_Efield, IBS_crosstrack_Efield, IBS_alongtrack_Efield = [],[],[],[]
#     for i in windsdf['BT']:
#         temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")
#         tempmag = magdata_magnitude_hires(temp)
#         print("datetime",temp,"B Mag", tempmag)
#         mag_magnitudes.append(tempmag)
#     windsdf["BT"] = mag_magnitudes

#add_magdata(windsdf)
add_angle_to_corot(windsdf)
#windsdf.to_csv("winds_full.csv")

#------ All processing above here--------

southern_hemisphere_flybys = ['t36','t39','t40', 't41', 't42', 't48', 't49', 't50', 't51', 't71']

northern_flybys_df = windsdf.loc[~windsdf['Flyby'].isin(southern_hemisphere_flybys)]
southern_flybys_df = windsdf.loc[windsdf['Flyby'].isin(southern_hemisphere_flybys)]


lonlatfig, lonlatax = plt.subplots()
sns.scatterplot(x=windsdf["Longitude"], y=windsdf["Latitude"], hue=windsdf["Flyby"], ax=lonlatax)
lonlatax.set_xlabel("Longitude")
lonlatax.set_ylabel("Latitude")


reduced_windsdf1 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "IBS crosstrack velocity","ELS crosstrack velocity","Flyby"]]#, "BT","Solar Zenith Angle",]]
reduced_windsdf2 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "Altitude", "Longitude", "Latitude","Flyby"]]
reduced_windsdf3 = windsdf[["IBS spacecraft potentials", "ELS spacecraft potentials", "LP Potentials", "Actuation Direction","Flyby"]]#,"Solar Zenith Angle", ]]
reduced_windsdf4 = windsdf[["Positive Peak Energy", "Negative Peak Energy", "IBS crosstrack velocity","ELS crosstrack velocity", "Actuation Direction"]]
reduced_windsdf5 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "Flyby" ]]# "Angle2Corot",,"Solar Zenith Angle",]]

sns.pairplot(reduced_windsdf1,hue="Flyby",corner=True)
sns.pairplot(reduced_windsdf2,hue="Flyby",corner=True)
sns.pairplot(reduced_windsdf3,hue="Flyby",corner=True)
sns.pairplot(reduced_windsdf4,corner=True)
sns.pairplot(reduced_windsdf5,hue="Flyby",corner=True)

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

regfig, (alongax,crossax) = plt.subplots(2)
sns.regplot(data=windsdf, x="IBS alongtrack velocity", y="ELS alongtrack velocity",ax=alongax)
z1 = np.polyfit(windsdf["IBS alongtrack velocity"],windsdf["ELS alongtrack velocity"],1)
p1 = np.poly1d(z1)
alongax.text(-300,250,str(stats.pearsonr(windsdf["IBS alongtrack velocity"], windsdf["ELS alongtrack velocity"])))
alongax.text(-300,200,str(p1))

sns.regplot(data=windsdf, x="IBS crosstrack velocity", y="ELS crosstrack velocity",ax=crossax)
z2 = np.polyfit(windsdf["IBS crosstrack velocity"],windsdf["ELS crosstrack velocity"],1)
p2 = np.poly1d(z2)
crossax.text(-300,250,str(stats.pearsonr(windsdf["IBS crosstrack velocity"], windsdf["ELS crosstrack velocity"])))
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


regfig4, (ax5, ax6) = plt.subplots(2)
sns.regplot(data=northern_flybys_df, x="Longitude", y="IBS alongtrack velocity",order=2,ax=ax5,label="IBS")
sns.regplot(data=northern_flybys_df, x="Longitude", y="ELS alongtrack velocity",order=2,ax=ax5,label="ELS")
ax5.set_xlabel(" ")
ax5.set_ylabel("Alongtrack Velocity")
ax5.legend()
ax5.set_xlim(0,360)
ax5.set_title("Northern Flybys")

sns.regplot(data=southern_flybys_df, x="Longitude", y="IBS alongtrack velocity",order=2,ax=ax6,label="IBS")
sns.regplot(data=southern_flybys_df, x="Longitude", y="ELS alongtrack velocity",order=2,ax=ax6,label="ELS")
ax6.set_ylabel("Alongtrack Velocity")
ax6.legend()
ax6.set_xlim(0,360)
ax6.set_title("Southern Flybys")

regfig5, ax8 = plt.subplots()
sns.regplot(data=windsdf, x="IBS alongtrack velocity", y="IBS crosstrack velocity",ax=ax8)
# ax5.set_xlabel(" ")
# ax5.set_ylabel("Alongtrack Velocity")


dist_fig, (northdist_ax, southdist_ax) = plt.subplots(2)
sns.histplot(data=northern_flybys_df, x="ELS alongtrack velocity", ax=northdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
sns.histplot(data=northern_flybys_df, x="IBS alongtrack velocity", ax=northdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")
sns.histplot(data=southern_flybys_df, x="ELS alongtrack velocity", ax=southdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
sns.histplot(data=southern_flybys_df, x="IBS alongtrack velocity", ax=southdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")

northdist_ax.set_xlim(-maxwind,maxwind)
southdist_ax.set_xlim(-maxwind,maxwind)
northdist_ax.legend()
southdist_ax.legend()
northdist_ax.set_title("Northern Flybys")
southdist_ax.set_title("Southern Flybys")
northdist_ax.set_xlabel("")
southdist_ax.set_xlabel("Alongtrack Velocity")

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

northlatitude_df = windsdf[windsdf['Latitude'] > 30]
lowlatitude_df = windsdf[(windsdf['Latitude'] > -30) & (windsdf['Latitude'] < 30)]
southlatitude_df = windsdf[windsdf['Latitude'] < -30]

dist_fig2, latitude_ax = plt.subplots()
sns.histplot(data=northlatitude_df, x="IBS alongtrack velocity", ax=latitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="North Latitudes")
sns.histplot(data=lowlatitude_df, x="IBS alongtrack velocity", ax=latitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="Low Latitudes")
sns.histplot(data=southlatitude_df, x="IBS alongtrack velocity", ax=latitude_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C2',label="South Latitudes")
latitude_ax.legend()

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

for counter, j in enumerate(["IBS alongtrack velocity", "ELS alongtrack velocity"]):
    for innercounter, i in enumerate(["2016 Zonal Winds", "2017 Zonal Winds"]):
        z2 = np.polyfit(lowlatitude_df[j],lowlatitude_df[i],1)
        p2 = np.poly1d(z2)
        print(j,i,"R = %.2f" % stats.pearsonr(lowlatitude_df[j], lowlatitude_df[i])[0],str(p2))
        #latitude_velocityaxes[1, counter].plot(testx,p2(testx),color='C2')
        latitude_velocityaxes[1, counter].text(0.6+(innercounter*0.2), 0.075, i + "\nR = %.2f\n" % stats.pearsonr(lowlatitude_df[j], lowlatitude_df[i])[0] + "y = %.2fx + %2.0f" % (p2[1],p2[0]),transform=latitude_velocityaxes[1, counter].transAxes,fontsize=10)
        #latitude_velocityaxes[1, counter].text(0.6, 0.15+(innercounter)*0.2, i + " " + "%.2f x + %2.2f" % (p2[1],p2[0]),transform=latitude_velocityaxes[1, counter].transAxes,fontsize=10)

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

# massenergy_fig2, massenergy_axes2 = plt.subplots(nrows=6,ncols=2,sharex='col')
# sns.regplot(data=windsdf,x="FullIonVelocity",y="PredictedMass28Energy",color='k',ax=massenergy_axes2[0,0])
# sns.regplot(data=windsdf,x="FullIonVelocity",y="PredictedMass40Energy",color='k',ax=massenergy_axes2[1,0])
# sns.regplot(data=windsdf,x="FullIonVelocity",y="PredictedMass53Energy",color='k',ax=massenergy_axes2[2,0])
# sns.regplot(data=windsdf,x="FullIonVelocity",y="PredictedMass66Energy",color='k',ax=massenergy_axes2[3,0])
# sns.regplot(data=windsdf,x="FullIonVelocity",y="PredictedMass78Energy",color='k',ax=massenergy_axes2[4,0])
# sns.regplot(data=windsdf,x="FullIonVelocity",y="PredictedMass91Energy",color='k',ax=massenergy_axes2[5,0])
# for counter, i in enumerate([28, 40, 53, 66, 78, 91]):
#     sns.regplot(data=windsdf,x="FullIonVelocity",y="IBS Mass "+str(i)+" energy",color='C0',ax=massenergy_axes2[counter,0])
#     sns.regplot(data=windsdf, x="IBS spacecraft potentials", y="IBS Mass " + str(i) + " energy", color='C0',ax=massenergy_axes2[counter, 1])
#     massenergy_axes2[counter, 0].set_ylabel("IBS Mass\n"+str(i)+" energy")
#     massenergy_axes2[counter, 1].set_ylabel("IBS Mass\n" + str(i) + " energy")

def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
latitudedist_fig, latitudedist_ax = plt.subplots()
x = np.linspace(-90, 90, 1000)

y_2016 = gaussian(x,0,70/2.355,373)
y_2017 = gaussian(x,3,101/2.355,196)
y_MW2006 = gaussian(x,0,70/2.355,60)
windsdf['Absolute IBS Zonal Velocity'] = abs(windsdf['IBS alongtrack velocity'] * np.cos(windsdf['Angles to Zonal Wind']*spice.rpd()))
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
sns.regplot(data=windsdf,x="Angles to Zonal Wind", y="IBS alongtrack velocity", ax=angleaxes[0])
sns.regplot(data=windsdf,x="Angles to Zonal Wind", y='IBS crosstrack velocity', ax=angleaxes[1])
plt.show()
