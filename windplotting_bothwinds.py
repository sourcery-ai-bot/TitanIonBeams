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
alongtrack_windsdf = pd.read_csv("alongtrackvelocity_1.073energy.csv", index_col=0, parse_dates=True)
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
reduced_windsdf3 = windsdf[["IBS spacecraft potentials", "ELS spacecraft potentials", "Actuation Direction"]]
reduced_windsdf4 = windsdf[["Positive Peak Energy", "Negative Peak Energy", "IBS crosstrack velocity","ELS crosstrack velocity", "Actuation Direction"]]

sns.pairplot(reduced_windsdf1,corner=True)
sns.pairplot(reduced_windsdf2,corner=True)
sns.pairplot(reduced_windsdf3,hue="Actuation Direction",corner=True)
sns.pairplot(reduced_windsdf4,hue="Actuation Direction",corner=True)

# print(windsdf)

#------Alongtrack----
alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
alongtrack_ionvelocity_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=windsdf, x="IBS alongtrack velocity", bins=np.arange(-400, 400, 50),
             ax=alongtrack_ibs_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")
sns.histplot(data=windsdf, x="ELS alongtrack velocity", bins=np.arange(-400, 400, 50),
             ax=alongtrack_els_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")

#------Crosstrack----
alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
alongtrack_ionvelocity_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=windsdf, x="IBS crosstrack velocity", bins=np.arange(-400, 400, 50),
             ax=alongtrack_ibs_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")
sns.histplot(data=windsdf, x="ELS crosstrack velocity", bins=np.arange(-400, 400, 50),
             ax=alongtrack_els_ionvelocity_axdist, element="step",
             kde=True,hue="Actuation Direction")


plt.show()
