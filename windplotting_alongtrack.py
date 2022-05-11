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
#alongtrack_windsdf = pd.read_csv("alongtrackvelocity_unconstrained_refinedpeaks.csv", index_col=0, parse_dates=True)
alongtrack_windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)
crary_windsdf = pd.read_csv("crarywinds.csv")
original_crary_winds = ["t16","t17","t18","t21","t23","t25","t26","t28","t29","t30","t32","t36","t39","t40"]

alongtrack_windsdf['Crary Potentials Estimated'] = np.where(alongtrack_windsdf['Flyby'].isin(original_crary_winds),alongtrack_windsdf["LP Potentials"]+0.25,np.nan)
alongtrack_windsdf['Crary Potentials Extrapolated'] = np.where(~alongtrack_windsdf['Flyby'].isin(original_crary_winds),alongtrack_windsdf["LP Potentials"]+0.25,np.nan)


desai_windsdf = pd.read_csv("DesaiPotentials.csv")
flybyslist = alongtrack_windsdf.Flyby.unique()
crary_windsdf = crary_windsdf[crary_windsdf['Flyby'].isin([i.upper() for i in flybyslist])]



alongtrack_figdist, alongtrack_ax = plt.subplots()
#alongtrack_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))

# sns.scatterplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials", ax=alongtrack_ax,
#                 color='C0')
# sns.kdeplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials", ax=alongtrack_ax,
#             levels=5, color='C0')
sns.regplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials", ax=alongtrack_ax,
            color='C0',line_kws={"linestyle":'--'},scatter_kws={"alpha":0.4},ci=95)
# sns.scatterplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", y="ELS spacecraft potentials", ax=alongtrack_ax,
#                 color='C1')
sns.regplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", y="ELS spacecraft potentials", ax=alongtrack_ax,
            color='C1',line_kws={"linestyle":'--'},scatter_kws={"alpha":0.4},ci=95)
# sns.kdeplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", y="ELS spacecraft potentials", ax=alongtrack_ax,
#             levels=5, color='C1')
# sns.kdeplot(data=windsdf, x="Crosstrack velocity", ax=axdist)
alongtrack_ax.set_xlabel("Alongtrack velocity [m/s]")
alongtrack_ax.set_ylabel("Derived S/c potential [V]")
#alongtrack_ax.legend()

alongtrack_scp_figdist, (alongtrack_ibs_scp_axdist, alongtrack_els_scp_axdist) = plt.subplots(2)
alongtrack_scp_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=alongtrack_windsdf, x="IBS spacecraft potentials", bins=np.arange(-3, 0, 0.15),
             ax=alongtrack_ibs_scp_axdist, element="step",
             color='C0')
sns.histplot(data=alongtrack_windsdf, x="ELS spacecraft potentials", bins=np.arange(-3, 0, 0.15),
             ax=alongtrack_els_scp_axdist, element="step",
             color='C1')

maxwind = 800

alongtrack_ionvelocity_figdist, (alongtrack_ibs_ionvelocity_axdist, alongtrack_els_ionvelocity_axdist) = plt.subplots(2)
alongtrack_ionvelocity_figdist.suptitle(str(alongtrack_windsdf.Flyby.unique()))
sns.histplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", bins=np.arange(-maxwind, maxwind, 50),
             ax=alongtrack_ibs_ionvelocity_axdist, element="step",
             color='C0',kde=True)
sns.histplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", bins=np.arange(-maxwind, maxwind, 50),
             ax=alongtrack_els_ionvelocity_axdist, element="step",
             color='C1',kde=True)
# sns.kdeplot(data=windsdf, x="Crosstrack velocity", ax=axdist)
alongtrack_ibs_ionvelocity_axdist.set_xlim(-maxwind,maxwind)
alongtrack_els_ionvelocity_axdist.set_xlim(-maxwind,maxwind)


bothdf = alongtrack_windsdf.dropna(subset=["ELS alongtrack velocity"])

ibs_regfig, ibs_regax = plt.subplots()
sns.regplot(data=bothdf , x="IBS alongtrack velocity", y="ELS alongtrack velocity",ax=ibs_regax)
ibs_regax.set_xlim(-maxwind,maxwind)
ibs_regax.set_ylim(-maxwind,maxwind)
print(stats.pearsonr(bothdf["IBS alongtrack velocity"], bothdf ["ELS alongtrack velocity"]))

# testfig, testax = plt.subplots()
# sns.regplot(data=alongtrack_windsdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials",ax=testax)
# print(stats.pearsonr(alongtrack_windsdf["IBS alongtrack velocity"], alongtrack_windsdf["IBS spacecraft potentials"]))
#
# testfig2, testax2 = plt.subplots()
# sns.regplot(data=alongtrack_windsdf, x="ELS alongtrack velocity", y="ELS spacecraft potentials",ax=testax2)
# print(stats.pearsonr(alongtrack_windsdf["ELS alongtrack velocity"], alongtrack_windsdf["ELS spacecraft potentials"]))

els_regfig, els_regax = plt.subplots()
sns.regplot(data=alongtrack_windsdf, x="IBS spacecraft potentials", y="ELS spacecraft potentials",ax=els_regax)

print(desai_windsdf)
alongtrack_windsdf_lowalt = alongtrack_windsdf[alongtrack_windsdf['Altitude'] < 1500]

flyby_potentialfig, flyby_potentialax = plt.subplots()
#sns.pointplot(x="Flyby", y="ELS Potential", data=desai_windsdf,join=False,color='g',ax=flyby_potentialax,capsize=.2)
sns.pointplot(x="Flyby", y="IBS spacecraft potentials", data=bothdf ,join=False,color='C0',ax=flyby_potentialax,capsize=.2,markers='.')
sns.pointplot(x="Flyby", y='Crary Potentials Estimated', data=bothdf ,join=False,color='g',ax=flyby_potentialax,capsize=.2,markers='.')
sns.pointplot(x="Flyby", y='Crary Potentials Extrapolated', data=bothdf ,join=False,color='y',ax=flyby_potentialax,capsize=.2,markers='.')
sns.pointplot(x="Flyby", y="LP Potentials", data=bothdf ,join=False,color='r',ax=flyby_potentialax,capsize=.2,markers='.')
sns.pointplot(x="Flyby", y="ELS spacecraft potentials", data=bothdf  ,join=False,color='C1',ax=flyby_potentialax,capsize=.2,markers='.')
sns.pointplot(x="Flyby", y='ELS Potential', data=desai_windsdf,join=False, color='b',ax=flyby_potentialax,capsize=.2)

flyby_potentialax.set_ylabel("Derived S/C Potential")
flyby_potentialax.legend(handles=[Line2D([0], [0], marker='.', color='C0', label='IBS',
                                       markerfacecolor='C0', markersize=8),
                                  Line2D([0], [0], marker='.', color='C1', label='ELS',
                                       markerfacecolor='C1', markersize=8),
                                  Line2D([0], [0], marker='.', color='r', label='LP',
                                       markerfacecolor='r', markersize=8),
                                  Line2D([0], [0], marker='o', color='b', label='Desai+, 2018; ELS',
                                       markerfacecolor='b', markersize=8),
                                  Line2D([0], [0], marker='.', color='g', label='Crary+, 2009; IBS estimated',
                                       markerfacecolor='g', markersize=8),
                                  Line2D([0], [0], marker='.', color='y', label='Crary+, 2009; IBS extrapolated',
                                         markerfacecolor='y', markersize=8)
                                  ],loc=8
                )
flyby_potentialax.set_ylabel("Derived spacecraft potential [V]")

maxlp = alongtrack_windsdf.groupby('Flyby', as_index=False)["LP Potentials"].max()
minlp = alongtrack_windsdf.groupby('Flyby', as_index=False)["LP Potentials"].min()



flyby_velocityfig, flyby_velocityax = plt.subplots()
temp_melt_df = alongtrack_windsdf_lowalt.melt(id_vars='Flyby', value_vars=["IBS alongtrack velocity","ELS alongtrack velocity"], var_name="Alongtrack velocity")
#sns.pointplot(x="Positive Peak Time", y="value", hue="Alongtrack velocity", ci="sd", dodge=0.2, data=temp_melt_df,join=False,ax=flyby_velocityax,capsize=.2,zorder=-1)
sns.stripplot(x="Flyby", y="value", hue="Alongtrack velocity", dodge=0.0001, data=temp_melt_df,ax=flyby_velocityax,zorder=-1)

# sns.pointplot(x="Flyby", y="IBS alongtrack velocity", data=alongtrack_windsdf,join=False,color='C0',ax=flyby_velocityax,capsize=.2)
# sns.pointplot(x="Flyby", y="ELS alongtrack velocity", data=alongtrack_windsdf,join=False,color='C1',ax=flyby_velocityax,capsize=.2)

# violins = sns.violinplot(x="Flyby", y="value", hue="Alongtrack velocity", inner="stick", dodge=True, split=True,
#                          data=temp_melt_df, ax=flyby_velocityax,kdeprops={"zorder":0},rugprops={"zorder":1})
#points = sns.pointplot(x="Flyby", y="Ion velocity", data=crary_windsdf,join=False,color='g',ax=flyby_velocityax,capsize=.2,zorder=4,markers='x')
# sns.stripplot(x="Flyby", y="IBS alongtrack velocity", data=alongtrack_windsdf, color='C0', ax=flyby_velocityax)
# sns.stripplot(x="Flyby", y="ELS alongtrack velocity", data=alongtrack_windsdf, color='C1', ax=flyby_velocityax)
# sns.pointplot(x="Flyby", y="2016 Zonal Winds", data=alongtrack_windsdf,join=False,color='C4',ax=flyby_velocityax,capsize=.2)
# sns.pointplot(x="Flyby", y="2017 Zonal Winds", data=alongtrack_windsdf,join=False,color='C5',ax=flyby_velocityax,capsize=.2)
flyby_velocityax.set_ylabel("Derived Ion Velocities")
flyby_velocityax.legend(handles=[Line2D([0], [0], marker='o', color='C0', label='IBS',
                          markerfacecolor='C0', markersize=8),
          Line2D([0], [0], marker='o', color='C1', label='ELS',
                          markerfacecolor='C1', markersize=8),
          Line2D([0], [0], marker='x', color='g', label='Crary+, 2009; IBS',
                          markerfacecolor='g', markersize=8),
         # Line2D([0], [0], marker='o', color='C4', label='Cordiner, 2016 winds',
         #        markerfacecolor='C4', markersize=8),
         # Line2D([0], [0], marker='o', color='C5', label='Cordiner, 2017 winds',
         #        markerfacecolor='C5', markersize=8)
                                 ]
)
flyby_velocityax.grid(True,axis='both')
flyby_velocityax.set_ylabel("Ion Velocities [m/s]")


#----------------Unconstrained----------------
#flyby_potentialax.plot(flybyslist,minlp["LP Potentials"]-2,color='k',linestyle='--')
#flyby_potentialax.plot(flybyslist,[0]*len(flybyslist),color='k',linestyle='--')
#flyby_potentialax.set_title("Derived s/c potential bounds, [LPvalue-2,0] \n Derived ion velocity bounds, [-500,500]")
# flyby_velocityax.plot(flybyslist,[500]*len(flybyslist),color='k',linestyle='--')
# flyby_velocityax.plot(flybyslist,[-500]*len(flybyslist),color='k',linestyle='--')
# flyby_velocityax.set_title("Derived s/c potential bounds, [LPvalue-2,0] \n Derived ion velocity bounds, [-500,500]")

#----------------Constrained----------------
# flyby_potentialax.plot(flybyslist,minlp["LP Potentials"]-0.3,color='k',linestyle='--')
# flyby_potentialax.plot(flybyslist,maxlp["LP Potentials"]+0.3,color='k',linestyle='--')
# flyby_potentialax.set_title("Derived s/c potential bounds, [LPvalue-0.3,LPvalue+0.3] \n Derived ion velocity bounds, [-500,500]")
# flyby_velocityax.plot(flybyslist,[500]*len(flybyslist),color='k',linestyle='--')
# flyby_velocityax.plot(flybyslist,[-500]*len(flybyslist),color='k',linestyle='--')
# flyby_velocityax.set_title("Derived s/c potential bounds, [LPvalue-0.3,LPvalue+0.3] \n Derived ion velocity bounds, [-500,500]")




plt.show()
