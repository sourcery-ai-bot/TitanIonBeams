import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)


flybyslist = windsdf.Flyby.unique()
print("Number of Flybys", len(flybyslist))

# for counter, flyby in enumerate(flybyslist):
#     tempdf = windsdf[windsdf['Flyby'] == flyby]
#
#     fig, ax = plt.subplots()
#     sns.scatterplot(x='Peak Time', y="Crosstrack velocity", style="Actuation Direction", markers = {"positive":"X","negative":"o"}, data=tempdf, ax=ax, s=200,
#                     label=flyby)
#     fig.autofmt_xdate()
#     ax.set_xlabel("Time")
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     ax.set_ylabel("Crosstrack Wind Velocity")
# fig.legend()
# fig.autofmt_xdate()

# -----------------------------------------------
lonlatfig, lonlatax = plt.subplots()
sns.scatterplot(x=windsdf["Longitude"], y=windsdf["Latitude"], hue=windsdf["Flyby"], ax=lonlatax,legend=False)
lonlatax.set_xlabel("Longitude")
lonlatax.set_ylabel("Latitude")

# -----------------------------------------------
lonfig, lonax = plt.subplots()
lonax.errorbar(x=windsdf["Longitude"], y=windsdf["Crosstrack velocity"], fmt='o',
               yerr=[windsdf["IBS crosstrack velocity"] - windsdf["Crosstrack velocity"]
                   , windsdf["Crosstrack velocity"] - windsdf["ELS crosstrack velocity"]])
lonax.set_xlabel("Longitude")
lonax.set_ylabel("Crosstrack velocity [m/s]")

latfig, latax = plt.subplots()
latax.errorbar(x=windsdf["Latitude"], y=windsdf["Crosstrack velocity"], fmt='o',
               yerr=[windsdf["IBS crosstrack velocity"] - windsdf["Crosstrack velocity"]
                   , windsdf["Crosstrack velocity"] - windsdf["ELS crosstrack velocity"]])
latax.set_xlabel("Latitude")
latax.set_ylabel("Crosstrack velocity [m/s]")

altfig, altax = plt.subplots()
altax.errorbar(x=windsdf["Altitude"], y=windsdf["Crosstrack velocity"], fmt='o',
               yerr=[windsdf["IBS crosstrack velocity"] - windsdf["Crosstrack velocity"]
                   , windsdf["Crosstrack velocity"] - windsdf["ELS crosstrack velocity"]])
altax.set_xlabel("Altitude [km]")
altax.set_ylabel("Crosstrack velocity [m/s]")

altfig2, altax2 = plt.subplots()
sns.scatterplot(data=windsdf, x="Altitude", y="Absolute Crosstrack velocity", ax=altax2)
altax2.set_xlabel("Altitude [km]")
altax2.set_ylabel("Absolute Crosstrack velocity [m/s]")

#---------------------------------

# crosstrack_fig, crosstrack_ax = plt.subplots()
# reduced_windsdf = windsdf[["IBS crosstrack velocity","ELS crosstrack velocity"]]
# sns.regplot(data=windsdf, x="IBS crosstrack velocity", y="ELS crosstrack velocity",ax=crosstrack_ax)
# z = np.polyfit(windsdf["IBS crosstrack velocity"],windsdf["ELS crosstrack velocity"],1)
# p = np.poly1d(z)
# print("Coefficients", p )
# x = np.linspace(-400,400,10)
# crosstrack_ax.plot(x,x,color='k')

#-----------------------------------

crosstrack_dist_fig, (crosstrack_ibsdist_ax, crosstrack_elsdist_ax) = plt.subplots(2)

sns.histplot(data=windsdf, x="ELS crosstrack velocity", ax=crosstrack_elsdist_ax, bins=np.arange(-600, 600, 50), kde=True)
sns.histplot(data=windsdf, x="IBS crosstrack velocity", ax=crosstrack_ibsdist_ax, bins=np.arange(-600, 600, 50), kde=True)

crosstrack_elsdist_ax.set_xlim(-600,600)
crosstrack_ibsdist_ax.set_xlim(-600,600)
# -----------------------------------------------

figdist, axdist = plt.subplots()
sns.histplot(data=windsdf, x="Crosstrack velocity", bins=np.arange(-1000, 1000, 100), ax=axdist, kde=True)
axdist.set_xlim(-1000,1000)
# sns.kdeplot(data=windsdf, x="Crosstrack velocity", ax=axdist)
figdist.legend()

g = sns.FacetGrid(windsdf, row="Flyby", aspect=8, height=2, sharey=False)
g.map(sns.kdeplot, "ELS crosstrack velocity", color='r', fill=True, clip=[-1000, 1000],label='ELS')
g.map(sns.rugplot, "ELS crosstrack velocity", color='r', height=0.4)
g.map(sns.kdeplot, "IBS crosstrack velocity", color='b', fill=True, clip=[-1000, 1000],label='IBS')
g.map(sns.rugplot, "IBS crosstrack velocity", color='b', height=0.4)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

for flyby, ax in zip(flybyslist,g.axes):
    ax[0].set_ylabel(flyby,rotation="horizontal")
    ax[0].set_xlim(-1000,1000)

g.axes[-1][0].set_xlabel("Crosstrack velocity [m/s]")
# def label(label,color):
#     ax = plt.gca()
#     ax.y_label(label,color=color)
#
# for flyby in flybyslist:
#     g.map(flyby,color='k')

g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
plt.subplots_adjust(bottom=0.10)

handles, labels = g.axes[0][0].get_legend_handles_labels()
plt.gcf().legend(handles, labels)

# -----------------------------------------------

fig, ax = plt.subplots()
sns.stripplot(data=windsdf, x="Flyby", y="ELS crosstrack velocity", hue="Actuation Direction", dodge=False,ax=ax)
sns.stripplot(data=windsdf, x="Flyby", y="IBS crosstrack velocity", hue="Actuation Direction", marker="X", dodge=False,ax=ax)

plt.show()
