import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from heliopy.data.cassini import mag_1min, mag_hires
import spiceypy as spice
import glob
import scipy
import time
from scipy import stats

import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d

import matplotlib.colors as mcolors

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

from matplotlib.lines import Line2D

############ 3D plot margin fix##################
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
############################################

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
alongtrack_windsdf = pd.read_csv("nonactuatingflybys_alongtrackvelocity_ibs.csv", index_col=0, parse_dates=True)

alongtrack_windsdf["Positive Peak Time"] = pd.to_datetime(alongtrack_windsdf["Positive Peak Time"])
print(type(alongtrack_windsdf["Positive Peak Time"].iloc[0]))

alongtrack_windsdf["Positive Peak Time"][alongtrack_windsdf['Flyby'] == "t56"] = alongtrack_windsdf["Positive Peak Time"][alongtrack_windsdf['Flyby'] == "t56"] - datetime.timedelta(seconds=1)
alongtrack_windsdf["Positive Peak Time"][alongtrack_windsdf['Flyby'] == "t57"] = alongtrack_windsdf["Positive Peak Time"][alongtrack_windsdf['Flyby'] == "t57"] - datetime.timedelta(seconds=1)

windsdf = alongtrack_windsdf
windsdf = windsdf.loc[:, ~windsdf.columns.duplicated()]
alongtrack_windsdf_negative = pd.read_csv("nonactuatingflybys_alongtrackvelocity_els.csv", index_col=0, parse_dates=True)
windsdf_negative = alongtrack_windsdf_negative
westlake_inbound_scp = pd.read_csv("data/westlake2011/SCP-Inbound.csv",names=['Potential','Altitude'])
westlake_outbound_scp = pd.read_csv("data/westlake2011/SCP-Outbound.csv",names=['Potential','Altitude'])
westlake_inbound_vel = pd.read_csv("data/westlake2011/ionwind-Inbound.csv",names=['Velocity','Altitude'])
westlake_outbound_vel = pd.read_csv("data/westlake2011/ionwind-Outbound.csv",names=['Velocity','Altitude'])


alongtrack_windsdf.rename(columns = {'Positive Peak Time':'Time'}, inplace = True)
alongtrack_windsdf_negative.rename(columns = {'Negative Peak Time':'Time'}, inplace = True)
alongtrack_windsdf['Time'] = alongtrack_windsdf['Time'].astype(str)
alongtrack_windsdf_negative['Time'] = alongtrack_windsdf_negative['Time'].astype(str)
bothwindsdf = alongtrack_windsdf.merge(alongtrack_windsdf_negative,on="Time")

flybyslist = bothwindsdf.Flyby_x.unique()
print(flybyslist)

bothwindsdf.to_csv("nonactuating_bothwinds.csv")

# windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)

#windsdf["IBS alongtrack velocity"] = windsdf["IBS alongtrack velocity"] + 180

FullIonVelocity = windsdf["IBS alongtrack velocity"] + windsdf["Flyby velocity"]
windsdf["FullIonVelocity"] = FullIonVelocity

def add_magdata(windsdf,flybys):

    mag_x, mag_y, mag_z = [],[],[]
    mag_x_titan, mag_y_titan, mag_z_titan = [], [], []
    for flyby in flybys:

        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])]
        startdatetime = singleflyby_df['Positive Peak Time'].iloc[0]
        enddatetime = singleflyby_df['Positive Peak Time'].iloc[-1]

        magdata = mag_hires(startdatetime, enddatetime).to_dataframe()
        singleflybymag_x, singleflybymag_y, singleflybymag_z = [],[],[]
        singleflybymag_x_titan, singleflybymag_y_titan, singleflybymag_z_titan = [], [], []
        mag_timestamps = [datetime.datetime.timestamp(d) for d in magdata.index]
        for i in singleflyby_df['Positive Peak Time']:
            Bx = np.interp(datetime.datetime.timestamp(i), mag_timestamps, magdata['Bx'])
            By = np.interp(datetime.datetime.timestamp(i), mag_timestamps, magdata['By'])
            Bz = np.interp(datetime.datetime.timestamp(i), mag_timestamps, magdata['Bz'])

            singleflybymag_x.append(Bx)
            singleflybymag_y.append(By)
            singleflybymag_z.append(Bz)

            et = spice.datetime2et(i)
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
        et = spice.datetime2et(i)
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
#windsdf['Positive Peak Time'] = pd.to_datetime(windsdf['Positive Peak Time'])
#windsdf_negative['Negative Peak Time'] = pd.to_datetime(windsdf_negative['Negative Peak Time'])

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
    upper_layer_alt =  1500

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

    figcyl, axescyl = plt.subplots(3,2,tight_layout=True, sharex='all',sharey='all')
    figcyl.delaxes(axescyl.flatten()[-1])
    for axcyl, flyby in zip(axescyl.flatten(),flybys):
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::10, :]
        singleflyby_df = singleflyby_df[singleflyby_df['Altitude'] < 1500]
        TitanBody_cyl = plt.Circle((0, 0), Titan_current_radius, color='y', alpha=0.5, zorder=3)
        TitanLower_cyl = plt.Circle((0, 0), Titan_current_radius + lower_layer_alt, color="0.8", fill=True, zorder=2)
        TitanMiddle_cyl = plt.Circle((0, 0), Titan_current_radius + middle_layer_alt, color="0.5", fill=True, zorder=1)
        TitanExobase_cyl = plt.Circle((0, 0), Titan_current_radius + upper_layer_alt, color="0.8", fill=True, zorder=0)
        axcyl.set_xlabel("y")
        axcyl.set_ylabel(r'$(x^{\frac{1}{2}} + z^{\frac{1}{2}})^{2}$')
        axcyl.set_xlim(-2000, 2000)
        axcyl.set_ylim(2000, 4500)
        axcyl.add_artist(TitanBody_cyl)
        axcyl.add_artist(TitanLower_cyl)
        axcyl.add_artist(TitanMiddle_cyl)
        axcyl.add_artist(TitanExobase_cyl)
        axcyl.set_aspect("equal")
        axcyl.plot([0, -5000], [-2575.15, -2575.15], color='k', zorder=8)
        axcyl.plot([0, -5000], [2575.15, 2575.15], color='k', zorder=8)
        sns.scatterplot(x=singleflyby_df['Y Titan'],
                        y=(singleflyby_df['X Titan'] ** 2 + singleflyby_df['Z Titan'] ** 2) ** 0.5,
                        hue=singleflyby_df['IBS alongtrack velocity'],
                        hue_norm=mcolors.CenteredNorm(vcenter=0, halfrange=50), label=flyby, ax=axcyl, palette='bwr',
                        legend=False, alpha=0.7, zorder=5)
        axcyl.set_title(flyby)
    figcyl.subplots_adjust(wspace=0, hspace=0)

    axxy.plot([-2575.15,-2575.15], [0,-5000],color='k')
    axxy.plot([2575.15,2575.15], [0,-5000],color='k')
    axyz.plot([0,-5000],[-2575.15,-2575.15],color='k')
    axyz.plot([0,-5000], [2575.15, 2575.15],color='k')

    x1, y1, z1 = spice.pgrrec('TITAN', 90 * spice.rpd(), 30 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)
    x2, y2, z1 = spice.pgrrec('TITAN', 270 * spice.rpd(), 30 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    x1, y3, z2 = spice.pgrrec('TITAN', 90 * spice.rpd(), 50.5 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)
    x2, y4, z2 = spice.pgrrec('TITAN', 270 * spice.rpd(), 50.5 * spice.rpd(), lower_layer_alt,
                           spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    print(y1,y2,z1)
    print(y3,y4,z2)


    axyz.plot([y1, y2], [z1, z1], color='r')
    axyz.plot([y1, y2], [-z1, -z1], color='r')

    axyz.plot([y3, y4], [z2, z2], color='b')
    axyz.plot([y3, y4], [-z2, -z2], color='b')

    hue_norm = mcolors.CenteredNorm(vcenter=0, halfrange=150)
    for flyby in flybys:
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::10, :]
        singleflyby_df = singleflyby_df[singleflyby_df['Altitude'] < 1500]

        sns.scatterplot(x='X Titan',y='Y Titan',hue='IBS alongtrack velocity',hue_norm=hue_norm,label=flyby,ax=axxy,palette='bwr',data=singleflyby_df,legend=False)
        sns.scatterplot(x='Y Titan', y='Z Titan', hue='IBS alongtrack velocity', hue_norm=hue_norm, label=flyby,
                        ax=axyz, palette='bwr', data=singleflyby_df, legend=False)
        sns.scatterplot(x='X Titan', y='Z Titan', hue='IBS alongtrack velocity', hue_norm=hue_norm, label=flyby,
                        ax=axxz, palette='bwr', data=singleflyby_df, legend=False)


        # axyz.plot(singleflyby_df['Y Titan'],singleflyby_df['Z Titan'],label=flyby)
        # axxz.plot(singleflyby_df['X Titan'],singleflyby_df['Z Titan'],label=flyby)
        #axcyl.plot(singleflyby_df['Y Titan'],(singleflyby_df['X Titan']**2 + singleflyby_df['Z Titan']**2)**0.5,label=flyby)
        # sns.scatterplot(x=singleflyby_df['Y Titan'],y=(singleflyby_df['X Titan']**2 + singleflyby_df['Z Titan']**2)**0.5,hue=singleflyby_df['IBS alongtrack velocity'],
        #                 hue_norm=mcolors.CenteredNorm(vcenter=0,halfrange=50),label=flyby,ax=axcyl,palette='bwr',legend=False,alpha=0.7,zorder=5)

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

def alt_vel_plot_positive(flybyslist):

    flyby_levels = {"t55":[[1231,1202],[1127,1138],[1041,1059]],
                    "t56":[[1297,1378],[1108,1094],[1015,1033]],
                    "t57":[[1379,0],[1180,0],[1028,998]],
                    "t58":[[1462,1162],[0,1073],[969,966]],
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

            # sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C0')
            # sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C1')
            ax.errorbar(tempdf["IBS alongtrack velocity"].iloc[:minalt_index], tempdf["Altitude"].iloc[:minalt_index], xerr=tempdf["IBS alongtrack velocity stderr"].iloc[:minalt_index], color='C0',ms=5,alpha=0.7)
            ax.errorbar(tempdf["IBS alongtrack velocity"].iloc[minalt_index:], tempdf["Altitude"].iloc[minalt_index:], xerr=tempdf["IBS alongtrack velocity stderr"].iloc[minalt_index:], color='C1',ms=5,alpha=0.7)

    for flyby, ax in zip(flybyslist, axes):
        ax.set_title(flyby)
        ax.set_xlabel("IBS Alongtrack \n Velocities [m/s]")
        minvel = -400
        maxvel= 400
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

def alt_vel_plot_negative(flybyslist):

    flyby_levels = {"t55":[[1231,1202],[1127,1138],[1041,1059]],
                    "t56":[[1297,1378],[1108,1094],[1015,1033]],
                    "t57":[[1379,0],[1180,0],[1028,998]],
                    "t58":[[1462,1162],[0,1073],[969,966]],
                    "t59":[[1296,1184],[1074,1100],[0,0]]}

    num_of_flybys = len(flybyslist)
    print(num_of_flybys)
    fig, axes = plt.subplots(ncols=num_of_flybys,sharex='all',sharey='all')

    axes[0].set_ylabel("Altitude [km]")
    print(flybyslist,axes)
    for flyby, ax in zip(flybyslist,axes):
        if flyby in list(windsdf_negative['Flyby']):
            tempdf = windsdf_negative[windsdf_negative['Flyby']==flyby]
            tempdf.reset_index(inplace=True)
            print(flyby, ax)

            minalt_index = tempdf["Altitude"].idxmin()
            print(minalt_index)

            sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="ELS alongtrack velocity", y="Altitude", ax=ax,color='C0')
            sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="ELS alongtrack velocity", y="Altitude", ax=ax,color='C1')

    for flyby, ax in zip(flybyslist, axes):
        ax.set_title(flyby)
        ax.set_xlabel("ELS Alongtrack \n Velocities [m/s]")
        minvel = -400
        maxvel= 400
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



def alt_vel_plot(flybyslist):

    flyby_levels = {"t55":[[1231,1202],[1127,1138],[1041,1059]],
                    "t56":[[1297,1378],[1108,1094],[1015,1033]],
                    "t57":[[1379,0],[1180,0],[1028,998]],
                    "t58":[[1462,1162],[0,1073],[969,966]],
                    "t59":[[1296,1184],[1074,1100],[0,0]]}

    num_of_flybys = len(flybyslist)
    print(num_of_flybys)
    fig, axes = plt.subplots(ncols=num_of_flybys,nrows=2,sharex='all',gridspec_kw={"hspace":0})

    axes[0,0].set_ylabel("Inbound")
    axes[1,0].set_ylabel("Outbound")

    print(flybyslist,axes)
    for flyby, ax in zip(flybyslist,axes[0,:]):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)

            minalt_index = tempdf["Altitude"].idxmin()

            # sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C0')
            # sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C1')
            ax.errorbar(tempdf["IBS alongtrack velocity"].iloc[:minalt_index], tempdf["Altitude"].iloc[:minalt_index], xerr=tempdf["IBS alongtrack velocity stderr"].iloc[:minalt_index], color='C0',ms=5,alpha=0.7)

            tempdf_negative = windsdf_negative[windsdf_negative['Flyby']==flyby]
            tempdf_negative .reset_index(inplace=True)

            minalt_index = tempdf_negative["Altitude"].idxmin()

            ax.errorbar(tempdf_negative["ELS alongtrack velocity"].iloc[:minalt_index], tempdf_negative["Altitude"].iloc[:minalt_index],
                        xerr=tempdf_negative["ELS alongtrack velocity stderr"].iloc[:minalt_index], color='C2', ms=5,
                        alpha=0.7)

    for flyby, ax in zip(flybyslist,axes[1,:]):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)

            minalt_index = tempdf["Altitude"].idxmin()

            ax.errorbar(tempdf["IBS alongtrack velocity"].iloc[minalt_index:], tempdf["Altitude"].iloc[minalt_index:], xerr=tempdf["IBS alongtrack velocity stderr"].iloc[minalt_index:], color='C0',ms=5,alpha=0.5)

            tempdf_negative = windsdf_negative[windsdf_negative['Flyby']==flyby]
            tempdf_negative .reset_index(inplace=True)

            minalt_index = tempdf_negative["Altitude"].idxmin()
            ax.errorbar(tempdf_negative["ELS alongtrack velocity"].iloc[minalt_index:], tempdf_negative["Altitude"].iloc[minalt_index:],
                        xerr=tempdf_negative["ELS alongtrack velocity stderr"].iloc[minalt_index:], color='C2', ms=5,
                        alpha=0.5)

    axes[0,2].plot(westlake_inbound_vel['Velocity'],westlake_inbound_vel['Altitude'],color='C3')
    axes[1, 2].plot(westlake_outbound_vel['Velocity'], westlake_outbound_vel['Altitude'], color='C3')

    minvel = -200
    maxvel = 200

    top = 1100

    lower_layer_alt = 1030
    middle_layer_alt =  1300
    upper_layer_alt =  1500

    for flyby, ax in zip(flybyslist, axes[0,:]):
        ax.set_title(flyby)
        ax.set_xlim(minvel,maxvel)
        ax.set_ylim(bottom=950,top=top)

    for flyby, ax in zip(flybyslist, axes[1,:]):
        ax.set_xlim(minvel,maxvel)
        ax.set_ylim(bottom=950,top=top)
        ax.invert_yaxis()

    upperchange = mpatches.Rectangle((minvel, 1225), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    lowerchange = mpatches.Rectangle((minvel, 1025), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    axes[0,0].add_patch(upperchange)
    axes[0,0].add_patch(lowerchange)
    upperchange = mpatches.Rectangle((minvel, 1280), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    lowerchange = mpatches.Rectangle((minvel, 990), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    axes[0,1].add_patch(upperchange)
    axes[0,1].add_patch(lowerchange)
    upperchange = mpatches.Rectangle((minvel, 1350), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    lowerchange = mpatches.Rectangle((minvel, 1015), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    axes[0,2].add_patch(upperchange)
    axes[0,2].add_patch(lowerchange)
    upperchange = mpatches.Rectangle((minvel, 1480), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    lowerchange = mpatches.Rectangle((minvel, 955), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    axes[0,3].add_patch(upperchange)
    axes[0,3].add_patch(lowerchange)
    upperchange = mpatches.Rectangle((minvel, 1265), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    axes[0,4].add_patch(upperchange)

    upperchange = mpatches.Rectangle((minvel, 1165), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    lowerchange = mpatches.Rectangle((minvel, 1035), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    axes[1,0].add_patch(upperchange)
    axes[1,0].add_patch(lowerchange)
    #upperchange = mpatches.Rectangle((minvel, 1280), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    lowerchange = mpatches.Rectangle((minvel, 1015), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    #axes[1,1].add_patch(upperchange)
    axes[1,1].add_patch(lowerchange)
    #upperchange = mpatches.Rectangle((minvel, 1350), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    lowerchange = mpatches.Rectangle((minvel, 975), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    #axes[1,2].add_patch(upperchange)
    axes[1,2].add_patch(lowerchange)
    upperchange = mpatches.Rectangle((minvel, 1150), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    #lowerchange = mpatches.Rectangle((minvel, 955), maxvel - minvel, 30, fill=True, facecolor='y', alpha=0.25)
    axes[1,3].add_patch(upperchange)
    #axes[1,3].add_patch(lowerchange)
    upperchange = mpatches.Rectangle((minvel, 1165), maxvel - minvel, 30, fill=True, facecolor='r', alpha=0.25)
    axes[1,4].add_patch(upperchange)






    for ax, ax2 in zip(axes[0,1:],axes[1,1:]):
        ax.set_yticklabels([])
        ax2.set_yticklabels([])

    fig.text(0.5, 0.05, 'Alongtrack Velocity [m/s]', ha='center',fontsize=20)
    fig.text(0.06, 0.5, 'Altitude [km]', va='center', rotation='vertical',fontsize=20)


    fig.legend(handles=[Line2D([0], [0], color='C0', label='IBS', ls='-'),
                        Line2D([0], [0], color='C2', label='ELS', ls='-'),
                        Line2D([0], [0], color='C3', label='Westlake+, 2014', ls='-'),
                        ]
                             )


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

def alt_scp_plot(flybyslist):

    flyby_levels = {"t55":[[1231,1202],[1127,1138],[1041,1059]],
                    "t56":[[1297,1378],[1108,1094],[1015,1033]],
                    "t57":[[1379,0],[1180,0],[1028,998]],
                    "t58":[[1462,1162],[0,1073],[969,966]],
                    "t59":[[1296,1184],[1074,1100],[0,0]]}

    num_of_flybys = len(flybyslist)
    print(num_of_flybys)
    fig, axes = plt.subplots(ncols=num_of_flybys,nrows=2,sharex='all',gridspec_kw={"hspace":0})

    axes[0,0].set_ylabel("Inbound")
    axes[1,0].set_ylabel("Outbound")

    print(flybyslist,axes)
    for flyby, ax in zip(flybyslist,axes[0,:]):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)

            minalt_index = tempdf["Altitude"].idxmin()

            # sns.scatterplot(data=tempdf.iloc[:minalt_index,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C0')
            # sns.scatterplot(data=tempdf.iloc[minalt_index:,:], x="IBS alongtrack velocity", y="Altitude", ax=ax,color='C1')
            ax.errorbar(tempdf["IBS spacecraft potentials"].iloc[:minalt_index], tempdf["Altitude"].iloc[:minalt_index], xerr=tempdf["IBS spacecraft potentials stderr"].iloc[:minalt_index], color='C0',ms=5,alpha=0.7)
            ax.plot(tempdf["LP Potentials"].iloc[:minalt_index], tempdf["Altitude"].iloc[:minalt_index], color='C1',ms=5,alpha=0.7)
            ax.plot(tempdf["LP Potentials"].iloc[:minalt_index]+0.25, tempdf["Altitude"].iloc[:minalt_index], color='C4',ms=5,alpha=0.7)

            tempdf_negative = windsdf_negative[windsdf_negative['Flyby']==flyby]
            tempdf_negative .reset_index(inplace=True)

            minalt_index = tempdf_negative["Altitude"].idxmin()

            ax.errorbar(tempdf_negative["ELS spacecraft potentials"].iloc[:minalt_index], tempdf_negative["Altitude"].iloc[:minalt_index],
                        xerr=tempdf_negative["ELS spacecraft potentials stderr"].iloc[:minalt_index], color='C2', ms=5,
                        alpha=0.7)

    for flyby, ax in zip(flybyslist,axes[1,:]):
        if flyby in list(windsdf['Flyby']):
            tempdf = windsdf[windsdf['Flyby']==flyby]
            tempdf.reset_index(inplace=True)

            minalt_index = tempdf["Altitude"].idxmin()

            ax.errorbar(tempdf["IBS spacecraft potentials"].iloc[minalt_index:], tempdf["Altitude"].iloc[minalt_index:], xerr=tempdf["IBS spacecraft potentials stderr"].iloc[minalt_index:], color='C0',ms=5,alpha=0.7)
            ax.plot(tempdf["LP Potentials"].iloc[minalt_index:], tempdf["Altitude"].iloc[minalt_index:], color='C1',ms=5,alpha=0.7)
            ax.plot(tempdf["LP Potentials"].iloc[minalt_index:]+0.25, tempdf["Altitude"].iloc[minalt_index:], color='C4',ms=5,alpha=0.7)

            tempdf_negative = windsdf_negative[windsdf_negative['Flyby']==flyby]
            tempdf_negative .reset_index(inplace=True)

            minalt_index = tempdf_negative["Altitude"].idxmin()
            ax.errorbar(tempdf_negative["ELS spacecraft potentials"].iloc[minalt_index:], tempdf_negative["Altitude"].iloc[minalt_index:],
                        xerr=tempdf_negative["ELS spacecraft potentials stderr"].iloc[minalt_index:], color='C2', ms=5,
                        alpha=0.7)

    axes[0,2].plot(westlake_inbound_scp['Potential'],westlake_inbound_scp['Altitude'],color='C3')
    axes[1, 2].plot(westlake_outbound_scp['Potential'], westlake_outbound_scp['Altitude'], color='C3')

    for flyby, ax in zip(flybyslist, axes[0,:]):
        ax.set_title(flyby)
        minscp = -2
        maxscp = 0.25
        ax.set_xlim(minscp,maxscp)
        ax.set_ylim(bottom=950,top=1500)

    for flyby, ax in zip(flybyslist, axes[1,:]):
        minscp = -2
        maxscp = 0.25
        ax.set_xlim(minscp,maxscp)
        ax.set_ylim(bottom=950,top=1500)
        ax.invert_yaxis()

    for ax, ax2 in zip(axes[0,1:],axes[1,1:]):
        ax.set_yticklabels([])
        ax2.set_yticklabels([])

    fig.text(0.5, 0.05, 'Spacecraft Potential [V]', ha='center',fontsize=20)
    fig.text(0.06, 0.5, 'Altitude [km]', va='center', rotation='vertical',fontsize=20)


    fig.legend(handles=[Line2D([0], [0], color='C0', label='IBS', ls='-'),
                        Line2D([0], [0], color='C1', label='LP', ls='-'),
                        Line2D([0], [0], color='C2', label='ELS', ls='-'),
                        Line2D([0], [0], color='C3', label='W2014 [T57]', ls='-'),
                        Line2D([0], [0], color='C4', label='LP + 0.25V', ls='-'),
                        ]
                             )

def box_titan_trajectory_plot(flybys,wake=True,sun=False,StartPoint=False):
    '''
    Plots Cassini Trajectory on 2d plots in 3d box
    '''

    boundingvalue = 4500
    dotinterval = 60
    radius = 2575.15
    exobase = 1500

    lower_layer_alt = 1030
    middle_layer_alt = 1300
    upper_layer_alt = 1500


    t0 = time.time()
    # ADD Switch for planes

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d',box_aspect=(1,1,1))

    i = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(i)
    art3d.pathpatch_2d_to_3d(i, z=boundingvalue, zdir="x")
    j = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(j)
    art3d.pathpatch_2d_to_3d(j, z=boundingvalue, zdir="y")
    k = mpatches.Circle((0, 0), radius, fill=False, ec='k', lw=1)
    ax.add_patch(k)
    art3d.pathpatch_2d_to_3d(k, z=-boundingvalue, zdir="z")

    m = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(m)
    art3d.pathpatch_2d_to_3d(m, z=boundingvalue, zdir="x")
    n = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(n)
    art3d.pathpatch_2d_to_3d(n, z=boundingvalue, zdir="y")
    o = mpatches.Circle((0, 0), radius+upper_layer_alt, fill=False, ec='grey', lw=1)
    ax.add_patch(o)
    art3d.pathpatch_2d_to_3d(o, z=-boundingvalue, zdir="z")

    for number, flyby in enumerate(flybys):
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::, :]
        # ax.plot((-(singleflyby_df['X Titan'] ** 2 + singleflyby_df['Z Titan'] ** 2) ** 0.5), singleflyby_df['Y Titan'],
        #         zs=-boundingvalue, zdir='z', color='C' + str(number), label=flyby)

        ax.plot(singleflyby_df['X Titan'],singleflyby_df['Y Titan'],zs=-boundingvalue, zdir='z',color='C'+str(number),label=flyby)
        ax.plot(singleflyby_df['X Titan'],singleflyby_df['Z Titan'],zs=boundingvalue, zdir='y',color='C'+str(number))
        ax.plot(singleflyby_df['Y Titan'],singleflyby_df['Z Titan'],zs=boundingvalue, zdir='x',color='C'+str(number))

        if StartPoint == True:
            ax.plot(singleflyby_df['X Titan'].iloc[0], singleflyby_df['Y Titan'].iloc[0], zs=-boundingvalue, zdir='z',
                    color='k', marker="^")
            ax.plot(singleflyby_df['X Titan'].iloc[0], singleflyby_df['Z Titan'].iloc[0], zs=boundingvalue, zdir='y',
                    color='k', marker="^")
            ax.plot(singleflyby_df['Y Titan'].iloc[0], singleflyby_df['Z Titan'].iloc[0], zs=boundingvalue, zdir='x',
                    color='k', marker="^")


    # if timedots == True:
    #     for et in np.arange(lower_et, upper_et, dotinterval):
    #         dotstate = cassini_phase(spice.et2utc(et, "ISOC", 0))
    #         ax.scatter(dotstate[0] / 252, dotstate[1] / 252, zs=-boundingvalue, zdir='z', c='k', marker=".", s=15)
    #         ax.scatter(dotstate[0] / 252, dotstate[2] / 252, zs=boundingvalue, zdir='y', c='k', marker=".", s=15)
    #         ax.scatter(dotstate[1] / 252, dotstate[2] / 252, zs=boundingvalue, zdir='x', c='k', marker=".", s=15)
    #
    #     ax.text2D(0.65, 0.97, r'$\cdot$ 1 minute markers  ', fontsize=12, transform=ax.transAxes)
    #
    if wake == True:
        # XY Wake
        ax.plot([-radius, -radius], [-boundingvalue, 0], zs=-boundingvalue, zdir='z', c='g')
        ax.plot([radius, radius], [-boundingvalue, 0], zs=-boundingvalue, zdir='z', c='g')

        wake_rect_xy = mpatches.Rectangle((-radius, -boundingvalue), 2*radius, boundingvalue, fill=True, facecolor='green', alpha=0.1)
        ax.add_patch(wake_rect_xy)
        art3d.pathpatch_2d_to_3d(wake_rect_xy, z=-boundingvalue, zdir="z")

        # YZ Wake
        ax.plot([-boundingvalue, 0], [-radius, -radius], zs=boundingvalue, zdir='x', c='g')
        ax.plot([-boundingvalue, 0], [radius, radius], zs=boundingvalue, zdir='x', c='g')

        wake_rect_yz = mpatches.Rectangle((-boundingvalue, -radius), boundingvalue, 2*radius, fill=True, facecolor='green', alpha=0.1)
        ax.add_patch(wake_rect_yz)
        art3d.pathpatch_2d_to_3d(wake_rect_yz, z=boundingvalue, zdir="x")

    if len(flybys) == 1 and sun == True:
        singleflyby_df = windsdf.loc[windsdf['Flyby'].isin([flyby])].iloc[::, :]
        i = pd.to_datetime(singleflyby_df['Positive Peak Time']).iloc[0]

        et = spice.datetime2et(i)
        sundir, ltime = spice.spkpos('SUN', et, 'IAU_TITAN', "LT+S", 'TITAN')
        satdir, ltime = spice.spkpos('SATURN', et, 'IAU_TITAN', "LT+S", 'TITAN')
        sundir_unorm = spice.unorm(sundir)[0]
        satdir_unorm = spice.unorm(satdir)[0]
        print("sundir", sundir_unorm)
        print("satdir", satdir_unorm)
        #
        # ax.plot([0, satdir_unorm[0] * 1000], [0, satdir_unorm[1] * 1000],zs=-boundingvalue, zdir='z',color='r')
        # ax.plot([0, satdir_unorm[0] * 1000], [0, satdir_unorm[2] * 1000], zs=boundingvalue, zdir='y', color='r')
        # ax.plot([0, satdir_unorm[1] * 1000], [0, satdir_unorm[2] * 1000], zs=boundingvalue, zdir='x', color='r')

        # ax.plot([0, sundir_unorm[0] * 1000], [0, sundir_unorm[1] * 1000],zs=-boundingvalue, zdir='z',color='b')
        # ax.plot([0, sundir_unorm[0] * 1000], [0, sundir_unorm[2] * 1000], zs=boundingvalue, zdir='y', color='b')
        # ax.plot([0, sundir_unorm[1] * 1000], [0, sundir_unorm[2] * 1000], zs=boundingvalue, zdir='x', color='b')

        if sundir_unorm[1] < 0:
            angle = np.degrees(spice.vsepg([-1, 0], [sundir_unorm[0], sundir_unorm[1]]))
        else:
            angle = -np.degrees(spice.vsepg([-1, 0], [sundir_unorm[0], sundir_unorm[1]]))

        print("angle",angle)
        startangle = -90 + angle
        endangle = 90 + angle

        c = mpatches.Wedge((0, 0), radius, startangle, endangle, fill=True, color='k', alpha=0.4)
        ax.add_patch(c)
        art3d.pathpatch_2d_to_3d(c, z=-boundingvalue, zdir="z")

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    ax.view_init(elev=30., azim=-135)

    fig.legend()

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))

    ax.set_xlim(-boundingvalue, boundingvalue)
    ax.set_ylim(-boundingvalue, boundingvalue)
    ax.set_zlim(-boundingvalue, boundingvalue)

    t1 = time.time()
    print("Time Elapsed", t1 - t0)



def regplot_alongtrack():
    limit = 400

    bothdf = bothwindsdf.dropna(subset=["ELS alongtrack velocity"])

    testtuple = stats.pearsonr(bothdf["IBS alongtrack velocity"], bothdf["ELS alongtrack velocity"])
    testjointgrid = sns.jointplot(data=bothdf, x="IBS alongtrack velocity", y="ELS alongtrack velocity",
                  kind='reg',xlim=(-limit,limit),ylim=(-limit,limit),marginal_kws=dict(bins=range(-limit,limit,50)))

    z2 = np.polyfit(bothdf["IBS alongtrack velocity"],bothdf["ELS alongtrack velocity"],1)
    p2 = "y = " + str(np.poly1d(z2))[2:]
    testjointgrid.ax_joint.text(0,-200,"Pearson's r = ({0[0]:.2f},{0[1]:.2f})".format(testtuple))
    testjointgrid.ax_joint.text(0,-250,str(p2))
    testjointgrid.ax_joint.set_xlabel("IBS alongtrack velocity [m/s]")
    testjointgrid.ax_joint.set_ylabel("ELS alongtrack velocity [m/s]")

    x = np.linspace(-400,400,10)
    testjointgrid.ax_joint.plot(x,x,color='k')
    plt.gcf().set_size_inches(8,8)


def regplot_alongtrack_alt():
    limit = 400

    bothdf = bothwindsdf.dropna(subset=["ELS alongtrack velocity"])


    bothdf_highvel = bothdf[bothwindsdf["IBS alongtrack velocity"] > -20]
    bothdf_lowvel = bothdf[bothwindsdf["IBS alongtrack velocity"] < -80]
    bothdf_midvel = bothdf[(bothwindsdf["IBS alongtrack velocity"] < -20) & (bothdf["IBS alongtrack velocity"] > -80)]


    dfs = [bothdf_lowvel, bothdf_midvel,bothdf_highvel]
    fig, ax = plt.subplots()
    for df in dfs:
        sns.scatterplot(data=df, x="IBS alongtrack velocity", y="ELS alongtrack velocity",hue="Flyby_x",ax=ax)

    ax.set_xlabel("IBS alongtrack velocity [m/s]")
    ax.set_ylabel("ELS alongtrack velocity [m/s]")

    x = np.linspace(-400,400,10)
    ax.plot(x,x,color='k')
    plt.gcf().set_size_inches(8,8)


def regplot_alongtrack_scp():
    bothdf = bothwindsdf.dropna(subset=["ELS alongtrack velocity"])

    print(bothwindsdf.keys())
    #sns.regplot(data=bothdf, x="IBS crosstrack velocity", y="ELS crosstrack velocity",ax=crossax)
    ax = sns.scatterplot(data=bothdf, x="IBS alongtrack velocity", y="IBS spacecraft potentials",hue="Flyby_x")

    ax.set_xlabel("IBS alongtrack velocity [m/s]")
    ax.set_ylabel("IBS spacecraft potentials")

    plt.gcf().set_size_inches(8,8)





Titan_frame_plot(["t55","t56","t57","t58","t59"])
#alt_vel_plot(["t55","t56","t57","t58","t59"])
#alt_scp_plot(["t55","t56","t57","t58","t59"])
#alt_vel_plot_negative(["t55","t56","t57","t58","t59"])
#alt_mag_plot(["t55","t56","t57","t58","t59"])
#box_titan_trajectory_plot(["t57"],StartPoint=True,sun=True)

#regplot_alongtrack()

plt.show()
