import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm

from matplotlib.lines import Line2D

#matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

# ---Alongtrack stats-----
alongtrack_windsdf = pd.read_csv("alongtrackvelocity_unconstrained_2peaks_lpoffset.csv", index_col=0, parse_dates=True)
crosstrack_windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
crary_windsdf = pd.read_csv("crarywinds.csv")
flybyslist = alongtrack_windsdf.Flyby.unique()
# print(flybyslist)
crary_windsdf = crary_windsdf[crary_windsdf['Flyby'].isin([i.upper() for i in flybyslist])]
# print(crary_windsdf)

# print(alongtrack_windsdf)
# print(crosstrack_windsdf)

windsdf = pd.concat([alongtrack_windsdf, crosstrack_windsdf], axis=1)
windsdf = windsdf.loc[:, ~windsdf.columns.duplicated()]
#windsdf.to_csv("winds_full.csv")

reduced_windsdf1 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "IBS crosstrack velocity","ELS crosstrack velocity"]]
reduced_windsdf2 = windsdf[["IBS alongtrack velocity", "ELS alongtrack velocity", "Altitude", "Longitude", "Latitude"]]
reduced_windsdf3 = windsdf[["IBS spacecraft potentials", "ELS spacecraft potentials", "LP Potentials", "Actuation Direction","Flyby"]]
reduced_windsdf4 = windsdf[["Positive Peak Energy", "Negative Peak Energy", "IBS crosstrack velocity","ELS crosstrack velocity", "Actuation Direction"]]

sns.pairplot(reduced_windsdf1,corner=True)
sns.pairplot(reduced_windsdf2,corner=True)
sns.pairplot(reduced_windsdf3,hue="Flyby",corner=True)
sns.pairplot(reduced_windsdf4,corner=True)

# print(windsdf)
maxwind = 400

#------Alongtrack----
alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
alongtrack_ionvelocity_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=windsdf, x="IBS alongtrack velocity", bins=np.arange(-maxwind, maxwind, 50),
             ax=alongtrack_ibs_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")
sns.histplot(data=windsdf, x="ELS alongtrack velocity", bins=np.arange(-maxwind, maxwind, 50),
             ax=alongtrack_els_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")

#------Crosstrack----
alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
alongtrack_ionvelocity_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=windsdf, x="IBS crosstrack velocity", bins=np.arange(-maxwind, maxwind, 50),
             ax=alongtrack_ibs_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")
sns.histplot(data=windsdf, x="ELS crosstrack velocity", bins=np.arange(-maxwind, maxwind, 50),
             ax=alongtrack_els_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")


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

##----Alongtrack Actuation direction
# neg_fig, neg_ax = plt.subplots()
# sns.stripplot(data=windsdf, x="Flyby", y="ELS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=neg_ax)
# pos_fig, pos_ax = plt.subplots()
# sns.stripplot(data=windsdf, x="Flyby", y="IBS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=pos_ax)

vel_fig, vel_ax = plt.subplots()
sns.stripplot(data=windsdf, x="Flyby", y="ELS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=vel_ax)
sns.stripplot(data=windsdf, x="Flyby", y="IBS alongtrack velocity", hue="Actuation Direction", dodge=False,ax=vel_ax,marker="X")
vel_ax.set_ylabel("Alongtrack Velocity")

scp_fig, scp_ax = plt.subplots()
sns.stripplot(data=windsdf, x="Flyby", y="ELS spacecraft potentials", hue="Actuation Direction", dodge=False,ax=scp_ax)
sns.stripplot(data=windsdf, x="Flyby", y="IBS spacecraft potentials", hue="Actuation Direction", dodge=False,ax=scp_ax,marker="X")
scp_ax.set_ylabel("Derived Spacecraft Potential")




plt.show()
