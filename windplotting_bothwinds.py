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

from matplotlib.lines import Line2D

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

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
#alongtrack_windsdf = pd.read_csv("alongtrackvelocity_unconstrained_2peaks.csv", index_col=0, parse_dates=True)
#crosstrack_windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
# windsdf = pd.concat([alongtrack_windsdf, crosstrack_windsdf], axis=1)
# windsdf = windsdf.loc[:, ~windsdf.columns.duplicated()]

windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)
flybyslist = windsdf.Flyby.unique()
# print(flybyslist)


crary_windsdf = pd.read_csv("crarywinds.csv")
crary_windsdf = crary_windsdf[crary_windsdf['Flyby'].isin([i.upper() for i in flybyslist])]

def add_magdata(windsdf):
    mag_magnitudes = []
    for i in windsdf['Positive Peak Time']:
        temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")
        tempmag = magdata_magnitude_hires(temp)
        print("datetime",temp,"B Mag", tempmag)
        mag_magnitudes.append(tempmag)
    windsdf["BT"] = mag_magnitudes

# def add_electricfield(windsdf):
#     ELS_crosstrack_Efield, ELS_alongtrack_Efield, IBS_crosstrack_Efield, IBS_alongtrack_Efield = [],[],[],[]
#     for i in windsdf['BT']:
#         temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")
#         tempmag = magdata_magnitude_hires(temp)
#         print("datetime",temp,"B Mag", tempmag)
#         mag_magnitudes.append(tempmag)
#     windsdf["BT"] = mag_magnitudes

#add_magdata(windsdf)
#windsdf.to_csv("winds_full.csv")

#------ All processing above here--------

southern_hemisphere_flybys = ['t36','t39','t40', 't41', 't42', 't48', 't49', 't50', 't51', 't71']

northern_flybys_df = windsdf.loc[~windsdf['Flyby'].isin(southern_hemisphere_flybys)]
southern_flybys_df = windsdf.loc[windsdf['Flyby'].isin(southern_hemisphere_flybys)]


lonlatfig, lonlatax = plt.subplots()
sns.scatterplot(x=windsdf["Longitude"], y=windsdf["Latitude"], hue=windsdf["Flyby"], ax=lonlatax)
lonlatax.set_xlabel("Longitude")
lonlatax.set_ylabel("Latitude")


reduced_windsdf1 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "IBS crosstrack velocity","ELS crosstrack velocity","BT","Solar Zenith Angle","Flyby"]]
reduced_windsdf2 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "Altitude", "Longitude", "Latitude","Flyby"]]
reduced_windsdf3 = windsdf[["IBS spacecraft potentials", "ELS spacecraft potentials", "LP Potentials", "Solar Zenith Angle", "Actuation Direction","Flyby"]]
reduced_windsdf4 = windsdf[["Positive Peak Energy", "Negative Peak Energy", "IBS crosstrack velocity","ELS crosstrack velocity", "Actuation Direction"]]
reduced_windsdf5 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "Solar Zenith Angle","Flyby"]]

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

regfig2, (ax2) = plt.subplots(1)
sns.regplot(data=windsdf, x="Solar Zenith Angle", y="IBS spacecraft potentials",ax=ax2,label="IBS")
sns.regplot(data=windsdf, x="Solar Zenith Angle", y="ELS spacecraft potentials",ax=ax2,label="ELS")
ax2.set_ylabel("Spacecraft potential")
ax2.legend()
regfig3, (ax3) = plt.subplots(1)
sns.regplot(data=windsdf, x="Solar Zenith Angle", y="IBS alongtrack velocity",ax=ax3,label="IBS")
sns.regplot(data=windsdf, x="Solar Zenith Angle", y="ELS alongtrack velocity",ax=ax3,label="ELS")
ax3.set_ylabel("Alongtrack Velocity")
ax3.legend()


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

dist_fig, (northdist_ax, southdist_ax) = plt.subplots(2)
sns.histplot(data=northern_flybys_df, x="ELS alongtrack velocity", ax=northdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
sns.histplot(data=northern_flybys_df, x="IBS alongtrack velocity", ax=northdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")
sns.histplot(data=southern_flybys_df, x="IBS alongtrack velocity", ax=southdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C0',label="ELS")
sns.histplot(data=southern_flybys_df, x="ELS alongtrack velocity", ax=southdist_ax, bins=np.arange(-maxwind, maxwind, 50), kde=True,color='C1',label="IBS")

northdist_ax.set_xlim(-maxwind,maxwind)
southdist_ax.set_xlim(-maxwind,maxwind)
northdist_ax.legend()
southdist_ax.legend()
northdist_ax.set_title("Northern Flybys")
southdist_ax.set_title("Southern Flybys")
northdist_ax.set_xlabel("")
southdist_ax.set_xlabel("Alongtrack Velocity")

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




plt.show()
