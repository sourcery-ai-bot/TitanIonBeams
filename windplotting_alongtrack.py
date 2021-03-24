import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm

from matplotlib.lines import Line2D

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

# ---Alongtrack stats-----
alongtrack_windsdf = pd.read_csv("alongtrackvelocity_unconstrained.csv", index_col=0, parse_dates=True)
crary_windsdf = pd.read_csv("crarywinds.csv")
flybyslist = alongtrack_windsdf.Flyby.unique()
print(flybyslist)
crary_windsdf = crary_windsdf[crary_windsdf['Flyby'].isin([i.upper() for i in flybyslist])]
print(crary_windsdf)

alongtrack_figdist, alongtrack_ax = plt.subplots()
alongtrack_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))

sns.scatterplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials", ax=alongtrack_ax,
                color='C0')
sns.kdeplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials", ax=alongtrack_ax,
            levels=5, color='C0')
sns.scatterplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", y="ELS spacecraft potentials", ax=alongtrack_ax,
                color='C1')
sns.kdeplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", y="ELS spacecraft potentials", ax=alongtrack_ax,
            levels=5, color='C1')
# sns.kdeplot(data=windsdf, x="Crosstrack velocity", ax=axdist)
alongtrack_ax.set_xlabel("Alongtrack velocity [m/s]")
alongtrack_ax.set_ylabel("Derived S/c potential [V]")
alongtrack_ax.legend()

alongtrack_scp_figdist, (alongtrack_ibs_scp_axdist, alongtrack_els_scp_axdist) = plt.subplots(2)
alongtrack_scp_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=alongtrack_windsdf, x="IBS spacecraft potentials", bins=np.arange(-3, 0, 0.15),
             ax=alongtrack_ibs_scp_axdist, element="step",
             stat="probability", color='C0')
sns.histplot(data=alongtrack_windsdf, x="ELS spacecraft potentials", bins=np.arange(-3, 0, 0.15),
             ax=alongtrack_els_scp_axdist, element="step",
             stat="probability", color='C1')

alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
alongtrack_ionvelocity_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", bins=np.arange(-400, 400, 50),
             ax=alongtrack_ibs_ionvelocity_axdist, element="step",
             stat="probability", color='C0')
sns.histplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", bins=np.arange(-400, 400, 50),
             ax=alongtrack_els_ionvelocity_axdist, element="step",
             stat="probability", color='C1')
# sns.kdeplot(data=windsdf, x="Crosstrack velocity", ax=axdist)


ibs_regfig, ibs_regax = plt.subplots()
sns.regplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="ELS alongtrack velocity",ax=ibs_regax)
print(stats.pearsonr(alongtrack_windsdf["IBS alongtrack velocity"], alongtrack_windsdf["ELS alongtrack velocity"]))

# testfig, testax = plt.subplots()
# sns.regplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials",ax=testax)
# print(stats.pearsonr(alongtrack_windsdf["IBS alongtrack velocity"], alongtrack_windsdf["IBS spacecraft potentials"]))
#
# testfig2, testax2 = plt.subplots()
# sns.regplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", y="ELS spacecraft potentials",ax=testax2)
# print(stats.pearsonr(alongtrack_windsdf["ELS alongtrack velocity"], alongtrack_windsdf["ELS spacecraft potentials"]))

els_regfig, els_regax = plt.subplots()
sns.regplot(data=alongtrack_windsdf, x="IBS spacecraft potentials", y="ELS spacecraft potentials",ax=els_regax)

flyby_potentialfig, flyby_potentialax = plt.subplots()
sns.pointplot(x="Flyby", y="IBS spacecraft potentials", data=alongtrack_windsdf,join=False,color='C0',ax=flyby_potentialax,capsize=.2)
sns.pointplot(x="Flyby", y="LP Potentials", data=alongtrack_windsdf,join=False,color='r',ax=flyby_potentialax,capsize=.2)
sns.pointplot(x="Flyby", y="ELS spacecraft potentials", data=alongtrack_windsdf,join=False,color='C1',ax=flyby_potentialax,capsize=.2)
flyby_potentialax.set_ylabel("Derived S/C Potential")
flyby_potentialax.legend(handles=[Line2D([0], [0], marker='o', color='C0', label='IBS',
                          markerfacecolor='C0', markersize=8),
          Line2D([0], [0], marker='o', color='C1', label='ELS',
                          markerfacecolor='C1', markersize=8),
          Line2D([0], [0], marker='o', color='r', label='LP',
               markerfacecolor='r', markersize=8)])
maxlp = alongtrack_windsdf.groupby('Flyby', as_index=False)["LP Potentials"].max()
minlp = alongtrack_windsdf.groupby('Flyby', as_index=False)["LP Potentials"].min()

flyby_velocityfig, flyby_velocityax = plt.subplots()
sns.pointplot(x="Flyby", y="Ion velocity", data=crary_windsdf,join=False,color='g',ax=flyby_velocityax,capsize=.2)
sns.pointplot(x="Flyby", y="IBS alongtrack velocity", data=alongtrack_windsdf,join=False,color='C0',ax=flyby_velocityax,capsize=.2)
sns.pointplot(x="Flyby", y="ELS alongtrack velocity", data=alongtrack_windsdf,join=False,color='C1',ax=flyby_velocityax,capsize=.2)
flyby_velocityax.set_ylabel("Derived Ion Velocities")
flyby_velocityax.legend(handles=[Line2D([0], [0], marker='o', color='C0', label='IBS',
                          markerfacecolor='C0', markersize=8),
          Line2D([0], [0], marker='o', color='C1', label='ELS',
                          markerfacecolor='C1', markersize=8),
          Line2D([0], [0], marker='o', color='g', label='Crary+, 2009; IBS',
                          markerfacecolor='g', markersize=8)]
)

#----------------Unconstrained----------------
flyby_potentialax.plot(flybyslist,minlp["LP Potentials"]-2,color='k',linestyle='--')
flyby_potentialax.plot(flybyslist,[0]*len(flybyslist),color='k',linestyle='--')
flyby_potentialax.set_title("Derived s/c potential bounds, [LPvalue-2,0] \n Derived ion velocity bounds, [-500,500]")
flyby_velocityax.plot(flybyslist,[500]*len(flybyslist),color='k',linestyle='--')
flyby_velocityax.plot(flybyslist,[-500]*len(flybyslist),color='k',linestyle='--')
flyby_velocityax.set_title("Derived s/c potential bounds, [LPvalue-2,0] \n Derived ion velocity bounds, [-500,500]")

#----------------Constrained----------------
# flyby_potentialax.plot(flybyslist,minlp["LP Potentials"]-0.3,color='k',linestyle='--')
# flyby_potentialax.plot(flybyslist,maxlp["LP Potentials"]+0.3,color='k',linestyle='--')
# flyby_potentialax.set_title("Derived s/c potential bounds, [LPvalue-0.3,LPvalue+0.3] \n Derived ion velocity bounds, [-500,500]")
# flyby_velocityax.plot(flybyslist,[500]*len(flybyslist),color='k',linestyle='--')
# flyby_velocityax.plot(flybyslist,[-500]*len(flybyslist),color='k',linestyle='--')
# flyby_velocityax.set_title("Derived s/c potential bounds, [LPvalue-0.3,LPvalue+0.3] \n Derived ion velocity bounds, [-500,500]")


plt.show()
