import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import datetime
from heliopy.data.cassini import mag_1min, mag_hires
import spiceypy as spice
import glob
import scipy

from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

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

def magdata_magnitude_hires(tempdatetime):
    start = tempdatetime - datetime.timedelta(seconds=31)
    end = tempdatetime + datetime.timedelta(seconds=31)
    magdata = mag_hires(start, end).data
    print(magdata)
    if magdata.size == 0:
        return np.NaN
    else:
        mag_magnitude = magdata[0]

    return mag_magnitude

#--Generating Windsdf----
alongtrack_windsdf = pd.read_csv("nonactuatingflybys_alongtrackvelocity.csv", index_col=0, parse_dates=True)
windsdf = alongtrack_windsdf
#windsdf = windsdf.loc[:, ~windsdf.columns.duplicated()]

# windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)
flybyslist = windsdf.Flyby.unique()
print(flybyslist)

#windsdf["IBS alongtrack velocity"] = windsdf["IBS alongtrack velocity"] + 180

FullIonVelocity = windsdf["IBS alongtrack velocity"] + windsdf["Flyby velocity"]
windsdf["FullIonVelocity"] = FullIonVelocity

def add_magdata(windsdf,flybys):

    mag_x, mag_y, mag_z = [],[],[]
    mag_x_titan, mag_y_titan, mag_z_titan = [], [], []
    for flyby in flybys:

        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])]
        startdatetime = datetime.datetime.strptime(singleflyby_df['Positive Peak Time'].iloc[0], "%Y-%m-%d %H:%M:%S")
        enddatetime = datetime.datetime.strptime(singleflyby_df['Positive Peak Time'].iloc[-1], "%Y-%m-%d %H:%M:%S")

        magdata = mag_hires(startdatetime, enddatetime).to_dataframe()
        singleflybymag_x, singleflybymag_y, singleflybymag_z = [],[],[]
        singleflybymag_x_titan, singleflybymag_y_titan, singleflybymag_z_titan = [], [], []
        mag_timestamps = [datetime.datetime.timestamp(d) for d in magdata.index]
        for i in singleflyby_df['Positive Peak Time']:
            Bx = np.interp(datetime.datetime.timestamp(datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")), mag_timestamps, magdata['Bx'])
            By = np.interp(datetime.datetime.timestamp(datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")), mag_timestamps,
                          magdata['By'])
            Bz = np.interp(datetime.datetime.timestamp(datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")), mag_timestamps,
                          magdata['Bz'])

            singleflybymag_x.append(Bx)
            singleflybymag_y.append(By)
            singleflybymag_z.append(Bz)

            temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
            et = spice.datetime2et(temp)
            Cassini2Titan = spice.pxform('CASSINI_SC_COORD', 'IAU_TITAN', et)
            magdir = spice.mxv(Cassini2Titan, [Bx,By,Bz])
            singleflybymag_x_titan.append(magdir[0])
            singleflybymag_y_titan.append(magdir[1])
            singleflybymag_z_titan.append(magdir[2])

        mag_x += singleflybymag_x
        mag_y += singleflybymag_y
        mag_z += singleflybymag_z
        mag_x_titan += singleflybymag_x_titan
        mag_y_titan += singleflybymag_y_titan
        mag_z_titan += singleflybymag_z_titan
    print(mag_x,len(mag_x))
    windsdf['Bx_sc'] = mag_x
    windsdf['By_sc'] = mag_y
    windsdf['Bz_sc'] = mag_z
    windsdf['Bx_Titan'] = mag_x_titan
    windsdf['By_Titan'] = mag_y_titan
    windsdf['Bz_Titan'] = mag_z_titan

    #     temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
    #     tempmag = magdata_magnitude_hires(temp)
    #     print("datetime",temp,"B Mag", tempmag)
    #     mag_magnitudes.append(tempmag)
    # windsdf["BT"] = mag_magnitudes

def add_angle_to_corot(windsdf):
    angles_from_corot = []
    for i in windsdf['Positive Peak Time']:
        temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        et = spice.datetime2et(temp)
        state, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "NONE", "titan")
        ramdir = spice.vhat(state[3:6])
        #print("ramdir", ramdir)
        #print("Angle to Corot Direction", spice.dpr() * spice.vsepg(ramdir, [0, 1], 2))
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, [0, 1, 0]))
    windsdf["Angle2Corot"] = angles_from_corot

def add_angle_to_corot_SaturnFrame(windsdf):
    angles_from_corot = []
    for i in windsdf['Positive Peak Time']:
        temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        et = spice.datetime2et(temp)
        state, ltime = spice.spkezr("CASSINI", et, "IAU_SATURN", "NONE", "titan")
        ramdir = spice.vhat(state[3:6])
        #print("ramdir", ramdir)
        #print("Angle to Corot Direction", spice.dpr() * spice.vsepg(ramdir, [0, 1], 2))
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, [0, 1, 0]))
    windsdf["Angle2Corot"] = angles_from_corot

# def add_electricfield(windsdf):
#     ELS_crosstrack_Efield, ELS_alongtrack_Efield, IBS_crosstrack_Efield, IBS_alongtrack_Efield = [],[],[],[]
#     for i in windsdf['BT']:
#         temp = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")
#         tempmag = magdata_magnitude_hires(temp)
#         print("datetime",temp,"B Mag", tempmag)
#         mag_magnitudes.append(tempmag)
#     windsdf["BT"] = mag_magnitudes

add_magdata(windsdf,flybyslist)
add_angle_to_corot(windsdf)
#windsdf.to_csv("nonactuating_winds.csv")
windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])

def pairplots():
    reduced_windsdf = windsdf[["IBS alongtrack velocity", "Altitude", "Flyby"]]
    reduced_windsdf2 = windsdf[["IBS alongtrack velocity", "Altitude", "Longitude", "Latitude","Flyby"]]
    reduced_windsdf3 = windsdf[["IBS spacecraft potentials", "LP Potentials", "Flyby"]]#,"Solar Zenith Angle", ]]
    reduced_windsdf5 = windsdf[["IBS alongtrack velocity", "Altitude", "Flyby", "Angle2Corot" ]]
    print(reduced_windsdf2)
    sns.pairplot(reduced_windsdf,hue="Flyby",corner=True)
    sns.pairplot(reduced_windsdf2,hue="Flyby",corner=True)
    sns.pairplot(reduced_windsdf3,hue="Flyby",corner=True)
    sns.pairplot(reduced_windsdf5,hue="Flyby",corner=True)

def flyby_timeplot():
    for flyby in flybyslist:
        tempdf = windsdf[windsdf['Flyby']==flyby]
        fig, axes = plt.subplots(2)
        sns.lineplot(data=tempdf, x="Positive Peak Time", y="IBS alongtrack velocity",ax=axes[0],label="IBS alongtrack Velocity")
        sns.lineplot(data=tempdf, x="Positive Peak Time", y="2008 MullerWodarg Winds", ax=axes[0], label="2008 Muller Wodarg - Zonal Winds")
        sns.lineplot(data=tempdf, x="Positive Peak Time", y="2016 Zonal Winds",ax=axes[0],label="2016 Zonal Winds")
        sns.lineplot(data=tempdf, x="Positive Peak Time", y="2017 Zonal Winds",ax=axes[0],label="2017 Zonal Winds")
        sns.lineplot(data=tempdf, x="Positive Peak Time", y="Altitudes",ax=axes[1],label="Altitude")
        axes[0].set_title(flyby)
        axes[1].set_xlabel("Time")
        axes[0].set_ylabel("Alongtrack Velocities [m/s]")
        axes[1].set_ylabel("Altitude [km]")
        axes[0].legend(loc=4,fontsize=12)
        fig.autofmt_xdate()
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

def Titan_frame_plot(flybys):

    # average_latitude = np.mean(singleflyby_df['Latitude'])
    # print(average_latitude,np.cos(average_latitude*spice.rpd())*2575.15)
    # Titan_current_radius = np.cos(average_latitude*spice.rpd())*2575.15

    Titan_current_radius = 2575.15
    lower_layer_alt = 1030
    middle_layer_alt =  1300
    upper_layer_alt =  0

    TitanBody_xy = plt.Circle((0, 0), Titan_current_radius, color='y',alpha=0.5)
    TitanLower_xy = plt.Circle((0, 0), Titan_current_radius+lower_layer_alt, color='k', fill=False,linestyle='--')
    TitanMiddle_xy = plt.Circle((0, 0), Titan_current_radius+middle_layer_alt, color='k', fill=False,linestyle='-.')
    TitanExobase_xy = plt.Circle((0, 0), Titan_current_radius+upper_layer_alt, color='k', fill=False)
    TitanBody_yz = plt.Circle((0, 0), Titan_current_radius, color='y',alpha=0.5)
    TitanLower_yz = plt.Circle((0, 0), Titan_current_radius+lower_layer_alt, color='k', fill=False,linestyle='--')
    TitanMiddle_yz = plt.Circle((0, 0), Titan_current_radius+middle_layer_alt, color='k', fill=False,linestyle='-.')
    TitanExobase_yz = plt.Circle((0, 0), Titan_current_radius+upper_layer_alt, color='k', fill=False)
    TitanBody_xz = plt.Circle((0, 0), Titan_current_radius, color='y',alpha=0.5)
    TitanLower_xz = plt.Circle((0, 0), Titan_current_radius+lower_layer_alt, color='k', fill=False,linestyle='--')
    TitanMiddle_xz = plt.Circle((0, 0), Titan_current_radius+middle_layer_alt, color='k', fill=False,linestyle='-.')
    TitanExobase_xz = plt.Circle((0, 0), Titan_current_radius+upper_layer_alt, color='k', fill=False)
    TitanBody_cyl = plt.Circle((0, 0), Titan_current_radius, color='y',alpha=0.5,zorder=3)
    TitanLower_cyl = plt.Circle((0, 0), Titan_current_radius+lower_layer_alt, color='w', fill=True,alpha=0.5,zorder=2)
    TitanMiddle_cyl = plt.Circle((0, 0), Titan_current_radius+middle_layer_alt, color='k', fill=True,alpha=0.5,zorder=1)
    TitanExobase_cyl = plt.Circle((0, 0), Titan_current_radius+upper_layer_alt, color='g', fill=False,zorder=0)

    figxy, axxy = plt.subplots(figsize=(8,8), tight_layout=True)
    axxy.set_xlabel("X")
    axxy.set_ylabel("Y")
    axxy.set_xlim(-5000, 5000)
    axxy.set_ylim(-5000, 5000)
    axxy.add_artist(TitanBody_xy)
    axxy.add_artist(TitanLower_xy)
    axxy.add_artist(TitanMiddle_xy)
    axxy.add_artist(TitanExobase_xy)
    axxy.set_aspect("equal")
    figyz, axyz = plt.subplots(figsize=(8,8), tight_layout=True)
    axyz.set_xlabel("Y")
    axyz.set_ylabel("Z")
    axyz.set_xlim(-5000, 5000)
    axyz.set_ylim(-5000, 5000)
    axyz.add_artist(TitanBody_yz)
    axyz.add_artist(TitanLower_yz)
    axyz.add_artist(TitanMiddle_yz)
    axyz.add_artist(TitanExobase_yz)
    axyz.set_aspect("equal")
    figxz, axxz = plt.subplots(figsize=(8,8), tight_layout=True)
    axxz.set_xlabel("X")
    axxz.set_ylabel("Z")
    axxz.set_xlim(-5000, 5000)
    axxz.set_ylim(-5000, 5000)

    axxz.add_artist(TitanExobase_xz)
    axxz.add_artist(TitanMiddle_xz)
    axxz.add_artist(TitanLower_xz)
    axxz.add_artist(TitanBody_xz)
    axxz.set_aspect("equal")

    figcyl, axcyl = plt.subplots(figsize=(8,8), tight_layout=True)
    axcyl.set_xlabel("Y")
    axcyl.set_ylabel("(X^1/2 + Z^1/2)^2")
    axcyl.set_xlim(-5000, 5000)
    axcyl.set_ylim(0, 5000)
    axcyl.add_artist(TitanBody_cyl)
    axcyl.add_artist(TitanLower_cyl)
    axcyl.add_artist(TitanMiddle_cyl)
    axcyl.add_artist(TitanExobase_cyl)
    axcyl.set_aspect("equal")

    axxy.plot([-2575.15,-2575.15], [0,-5000],color='k')
    axxy.plot([2575.15,2575.15], [0,-5000],color='k')
    axyz.plot([0,-5000],[-2575.15,-2575.15],color='k')
    axyz.plot([0,-5000], [2575.15, 2575.15],color='k')
    axcyl.plot([0,-5000],[-2575.15,-2575.15],color='k',zorder=8)
    axcyl.plot([0,-5000], [2575.15, 2575.15],color='k',zorder=8)

    hue_norm = mcolors.CenteredNorm(vcenter=0, halfrange=150)
    for flyby in flybys:
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::10, :]

        sns.scatterplot(x='X Titan',y='Y Titan',hue='IBS alongtrack velocity',hue_norm=hue_norm,label=flyby,ax=axxy,palette='bwr',data=singleflyby_df,legend=False)
        sns.scatterplot(x='Y Titan', y='Z Titan', hue='IBS alongtrack velocity', hue_norm=hue_norm, label=flyby,
                        ax=axyz, palette='bwr', data=singleflyby_df, legend=False)
        sns.scatterplot(x='X Titan', y='Z Titan', hue='IBS alongtrack velocity', hue_norm=hue_norm, label=flyby,
                        ax=axxz, palette='bwr', data=singleflyby_df, legend=False)


        # axyz.plot(singleflyby_df['Y Titan'],singleflyby_df['Z Titan'],label=flyby)
        # axxz.plot(singleflyby_df['X Titan'],singleflyby_df['Z Titan'],label=flyby)
        #axcyl.plot(singleflyby_df['Y Titan'],(singleflyby_df['X Titan']**2 + singleflyby_df['Z Titan']**2)**0.5,label=flyby)
        sns.scatterplot(x=singleflyby_df['Y Titan'],y=(singleflyby_df['X Titan']**2 + singleflyby_df['Z Titan']**2)**0.5,hue=singleflyby_df['IBS alongtrack velocity'],
                        hue_norm=mcolors.CenteredNorm(vcenter=0,halfrange=50),label=flyby,ax=axcyl,palette='bwr',legend=False,alpha=0.7,zorder=5)

        last_counter=0
        for counter, (index, row) in enumerate(singleflyby_df.iterrows()):
            alongtrackvelocity = row['IBS alongtrack velocity']
            if counter == last_counter + 2:
                normed_vector = spice.unorm(np.array([row['DX Titan'], row['DY Titan'],row['DZ Titan']]))
                scaled_dx = normed_vector[0][0] * -alongtrackvelocity * 0.05
                scaled_dy = normed_vector[0][1] * -alongtrackvelocity * 0.05
                scaled_dz = normed_vector[0][2] * -alongtrackvelocity * 0.05
                axxy.arrow(row['X Titan'], row['Y Titan'], scaled_dx, scaled_dy, head_width=30,width=2.5)
                axyz.arrow(row['Y Titan'], row['Z Titan'], scaled_dy, scaled_dz, head_width=30, width=2.5)
                axxz.arrow(row['X Titan'], row['Z Titan'], scaled_dx, scaled_dz, head_width=30, width=2.5)
                #axcyl.arrow(row['Y Titan'], (row['X Titan']**2 + row['Z Titan']**2)**0.5, scaled_dy, (scaled_dx**2 + scaled_dz**2)**0.5, head_width=50, width=2.5)
                last_counter = counter


    for ax in [axxy,axxz,axyz]:
        ax.legend()

def alt_vel_plot(flybyslist):

    flyby_levels = {"t55":[[1231,1202],[1127,1138],[1041,1059]],
                    "t56":[[1297,1378],[1108,1094],[1015,1033]],
                    "t57":[[1379,0],[1180,0],[1028,998]],
                    "t58":[[0,1162],[0,1073],[0,0]],
                    "t59":[[1296,1184],[1074,1100],[0,0]]}

    num_of_flybys = len(flybyslist)
    print(num_of_flybys)
    fig, axes = plt.subplots(ncols=num_of_flybys,sharex='all',sharey='all')

    axes[0].set_ylabel("Altitude [km]")
    print(flybyslist,axes)
    for flyby, ax in zip(flybyslist,axes):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)
            print(flyby, ax)

            minalt_index = tempdf["Altitude"].idxmin()
            print(minalt_index)

            sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C0')
            sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C1')

    for flyby, ax in zip(flybyslist, axes):
        ax.set_title(flyby)
        ax.set_xlabel("Alongtrack Velocities [m/s]")
        minvel = -250
        maxvel= 250
        ax.set_xlim(minvel,maxvel)
        ax.set_ylim(bottom=950,top=1800)
        ax.hlines(flyby_levels[flyby][0][0],minvel,maxvel,color='C0')
        ax.hlines(flyby_levels[flyby][0][1],minvel,maxvel,color='C1')
        ax.hlines(flyby_levels[flyby][1][0],minvel,maxvel,color='C0',linestyles='dashed')
        ax.hlines(flyby_levels[flyby][1][1],minvel,maxvel,color='C1',linestyles='dashed')
        ax.hlines(flyby_levels[flyby][2][0],minvel,maxvel,color='C0',linestyles='dashdot')
        ax.hlines(flyby_levels[flyby][2][1],minvel,maxvel,color='C1',linestyles='dashdot')
        ax.hlines(1500,minvel,maxvel,color='k',linestyles='dotted')

    # for i in np.arange(1,5):
    #     axes[i].yaxis.set_ticklabels([])
    #     axes[i].set_ylabel("")

def alt_mag_plot(flybyslist):

    flyby_levels = {"t55":[[1231,1202],[1127,1138],[1041,1059]],
                    "t56":[[1297,1378],[1108,1094],[1015,1033]],
                    "t57":[[1379,0],[1180,0],[1028,998]],
                    "t58":[[0,1162],[0,1073],[0,0]],
                    "t59":[[1296,1184],[1074,1100],[0,0]]}

    num_of_flybys = len(flybyslist)
    print(num_of_flybys)
    fig, axes = plt.subplots(ncols=num_of_flybys,sharex='all',sharey='all')

    axes[0].set_ylabel("Altitude [km]")
    print(flybyslist,axes)
    for flyby, ax in zip(flybyslist,axes):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)
            print(flyby, ax)

            minalt_index = tempdf["Altitude"].idxmin()
            print(minalt_index)

            sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="Bx_Titan", y="Altitude", ax=ax,color='r')
            sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="Bx_Titan", y="Altitude", ax=ax,color='r',marker='x')
            sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="By_Titan", y="Altitude", ax=ax,color='g')
            sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="By_Titan", y="Altitude", ax=ax,color='g',marker='x')
            sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="Bz_Titan", y="Altitude", ax=ax,color='b')
            sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="Bz_Titan", y="Altitude", ax=ax,color='b',marker='x')

    for flyby, ax in zip(flybyslist, axes):
        ax.set_title(flyby)
        ax.set_xlabel("B [nT]")
        minmag = -10
        maxmag= 10
        ax.set_xlim(minmag,maxmag)
        ax.set_ylim(bottom=950,top=1800)
        ax.hlines(flyby_levels[flyby][0][0],minmag,maxmag,color='C0')
        ax.hlines(flyby_levels[flyby][0][1],minmag,maxmag,color='C1')
        ax.hlines(flyby_levels[flyby][1][0],minmag,maxmag,color='C0',linestyles='dashed')
        ax.hlines(flyby_levels[flyby][1][1],minmag,maxmag,color='C1',linestyles='dashed')
        ax.hlines(flyby_levels[flyby][2][0],minmag,maxmag,color='C0',linestyles='dashdot')
        ax.hlines(flyby_levels[flyby][2][1],minmag,maxmag,color='C1',linestyles='dashdot')
        ax.hlines(1500,minmag,maxmag,color='k',linestyles='dotted')

    # for i in np.arange(1,5):
    #     axes[i].yaxis.set_ticklabels([])
    #     axes[i].set_ylabel("")


Titan_frame_plot(["t55","t56","t57","t58","t59"])
alt_vel_plot(["t55","t56","t57","t58","t59"])
#alt_mag_plot(["t55","t56","t57","t58","t59"])

plt.show()
