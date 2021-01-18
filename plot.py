import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

CAPS_df = pd.read_csv("Beams_database.csv")


data = CAPS_df


#Overview of parameters
fig, axes = plt.subplots(1,3)
g = sns.scatterplot(x="Peak Energy",y="Azimuthal Angle to Ram",
                    hue="Flyby",style="Instrument",data=data, ax=axes[0])
handles, labels = g.get_legend_handles_labels()
g.get_legend().remove()
sns.scatterplot(x="Peak Energy",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=data, ax=axes[1],legend=False)
sns.scatterplot(x="Peak Energy",y="Deflection from Bulk",
                hue="Flyby",style="Instrument",data=data, ax=axes[2],legend=False)
fig.legend(handles,labels)


#Overview of parameters
fig, axes = plt.subplots(2,3)
g = sns.scatterplot(x="Peak Energy",y="Azimuthal Angle to Heavy Ion",
                    hue="Flyby",style="Instrument",data=data, ax=axes[0,0])
handles, labels = g.get_legend_handles_labels()
g.get_legend().remove()
sns.scatterplot(x="Azimuthal Ram Angle",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=data, ax=axes[0,1],legend=False)
sns.scatterplot(x="Peak Elevation",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=data, ax=axes[1,0],legend=False)
sns.scatterplot(x="Spacecraft Potential (LP)",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=data, ax=axes[1,1],legend=False)
sns.scatterplot(x="Bulk Azimuth",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=data, ax=axes[0,2],legend=False)
sns.scatterplot(x="Bulk Azimuth",y="Azimuthal Ram Angle",
                hue="Flyby",style="Instrument",data=data, ax=axes[1,2],legend=False)
fig.legend(handles,labels)
#plt.tight_layout()

#Mass separated plotting
fig, axes = plt.subplots(3,1,sharex=True,sharey=True)
g = sns.scatterplot(x="Azimuthal Ram Angle",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=CAPS_df[(CAPS_df['Peak Energy'] < 6)], ax=axes[0])
handles, labels = g.get_legend_handles_labels()
g.get_legend().remove()
sns.scatterplot(x="Azimuthal Ram Angle",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=CAPS_df[(CAPS_df['Peak Energy'] > 6) & (CAPS_df['Peak Energy'] < 15)], ax=axes[1],legend=False)

sns.scatterplot(x="Azimuthal Ram Angle",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=CAPS_df[CAPS_df['Peak Energy'] > 15], ax=axes[2],legend=False)
axes[0].set_title("Ions below 6 eV/q")
axes[1].set_title("Ions between 6 and 15 eV/q")
axes[2].set_title("Ions above 15 eV/q")
fig.legend(handles,labels)
plt.subplots_adjust(hspace=0.4)

#Mass separated plotting - LP
fig, axes = plt.subplots(3,1,sharex=True,sharey=True)
g = sns.scatterplot(x="Spacecraft Potential (LP)",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=CAPS_df[(CAPS_df['Peak Energy'] < 6)], ax=axes[0])
handles, labels = g.get_legend_handles_labels()
g.get_legend().remove()
sns.scatterplot(x="Spacecraft Potential (LP)",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=CAPS_df[(CAPS_df['Peak Energy'] > 6) & (CAPS_df['Peak Energy'] < 15)], ax=axes[1],legend=False)

sns.scatterplot(x="Spacecraft Potential (LP)",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=CAPS_df[CAPS_df['Peak Energy'] > 15], ax=axes[2],legend=False)
axes[0].set_title("Ions below 6 eV/q")
axes[1].set_title("Ions between 6 and 15 eV/q")
axes[2].set_title("Ions above 15 eV/q")
fig.legend(handles,labels)
plt.subplots_adjust(hspace=0.4)


data = CAPS_df

fig, axes = plt.subplots(3,1)
g = sns.scatterplot(x="Peak Energy",y="Azimuthal Angle to Heavy Ion",
                    hue="Flyby",style="Instrument",data=data, ax=axes[0])
handles, labels = g.get_legend_handles_labels()
g.get_legend().remove()
sns.scatterplot(x="Azimuthal Ram Angle",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=data, ax=axes[1],legend=False)
sns.scatterplot(x="Spacecraft Potential (LP)",y="Azimuthal Angle to Heavy Ion",
                hue="Flyby",style="Instrument",data=data, ax=axes[2],legend=False)
fig.legend(handles,labels)
plt.subplots_adjust(hspace=0.4)
#plt.tight_layout()

plt.show()