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
alongtrack_windsdf = pd.read_csv("alongtrackvelocity_unconstrained.csv", index_col=0, parse_dates=True)
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

reduced_els_windsdf = windsdf[["ELS alongtrack velocity", "ELS spacecraft potentials","Negative crosstrack velocity"]]
reduced_ibs_windsdf = windsdf[["IBS alongtrack velocity", "IBS spacecraft potentials", "Positive crosstrack velocity"]]
reduced_ibs_windsdf2 = windsdf[["IBS alongtrack velocity", "Positive Deflection from Ram Angle", "Positive crosstrack velocity"]]

sns.pairplot(reduced_els_windsdf,corner=True)
sns.pairplot(reduced_ibs_windsdf,corner=True)
sns.pairplot(reduced_ibs_windsdf2,corner=True)

# print(windsdf)

plt.show()
