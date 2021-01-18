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
sns.scatterplot(x=windsdf["Longitude"], y=windsdf["Latitude"], size=windsdf["Absolute Crosstrack velocity"],
                hue=windsdf["Flyby"], ax=lonlatax)
lonlatax.set_xlabel("Longitude")
lonlatax.set_ylabel("Latitude")

# -----------------------------------------------
lonfig, lonax = plt.subplots()
lonax.errorbar(x=windsdf["Longitude"], y=windsdf["Crosstrack velocity"], fmt='o',
               yerr=[windsdf["Positive crosstrack velocity"] - windsdf["Crosstrack velocity"]
                   ,windsdf["Crosstrack velocity"] - windsdf["Negative crosstrack velocity"]])
lonax.set_xlabel("Longitude")
lonax.set_ylabel("Crosstrack velocity [m/s]")

latfig, latax = plt.subplots()
latax.errorbar(x=windsdf["Latitude"], y=windsdf["Crosstrack velocity"], fmt='o',
               yerr=[windsdf["Positive crosstrack velocity"] - windsdf["Crosstrack velocity"]
                   ,windsdf["Crosstrack velocity"] - windsdf["Negative crosstrack velocity"]])
latax.set_xlabel("Latitude")
latax.set_ylabel("Crosstrack velocity [m/s]")

altfig, altax = plt.subplots()
altax.errorbar(x=windsdf["Altitude"], y=windsdf["Crosstrack velocity"], fmt='o',
               yerr=[windsdf["Positive crosstrack velocity"] - windsdf["Crosstrack velocity"]
                   ,windsdf["Crosstrack velocity"] - windsdf["Negative crosstrack velocity"]])
altax.set_xlabel("Altitude [km]")
altax.set_ylabel("Crosstrack velocity [m/s]")

# -----------------------------------------------

figdist, axdist = plt.subplots()
sns.histplot(data=windsdf, x="Crosstrack velocity", bins=np.arange(-550, 550, 50), ax=axdist, element="step",
             stat="probability")
# sns.kdeplot(data=windsdf, x="Crosstrack velocity", ax=axdist)
figdist.legend()

g = sns.FacetGrid(windsdf, row="Flyby", hue="Flyby", aspect=8, height=2, sharey=False)
g.map(sns.kdeplot, "Crosstrack velocity", fill=True, clip=[-500, 500])
g.map(sns.rugplot, "Crosstrack velocity", height=0.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "Crosstrack velocity")
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
plt.subplots_adjust(bottom=0.10)

# -----------------------------------------------

fig, ax = plt.subplots()
sns.stripplot(data=windsdf, x="Flyby", y="Crosstrack velocity", hue="Actuation Direction", dodge=False)

plt.show()
