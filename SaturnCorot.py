import spiceypy as spice
import glob
import datetime
import matplotlib.pyplot as plt
from util import generate_mass_bins
from scipy.io import readsav
import pandas as pd
import matplotlib
from mpl_toolkits import mplot3d

from cassinipy.caps.mssl import *
from cassinipy.caps.spice import *
from cassinipy.caps.util import *
from cassinipy.misc import *
from cassinipy.spice import *

matplotlib.rcParams.update({'font.size': 15})

# Loading Kernels
if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

def add_angle_to_corot_TitanFrame(datetimes):
    angles_from_corot, cassinistates, titanstates = [],[],[]
    for i in datetimes:
        et = spice.datetime2et(i)


        cassinistate, ltime = spice.spkezr("CASSINI", et, "IAU_SATURN", "LT+S", "Saturn")
        titanstate, ltime = spice.spkezr("TITAN", et, "IAU_SATURN", "LT+S", "Saturn")
        Saturn2TitanFrame = spice.sxform("IAU_SATURN", "IAU_TITAN", et-ltime)

        cassinistate_tform = spice.mxvg(Saturn2TitanFrame, cassinistate, 6, 6)
        titanstate_tform = spice.mxvg(Saturn2TitanFrame, titanstate, 6, 6)
        cassinistates.append(cassinistate_tform)
        titanstates.append(titanstate_tform)

        ramdir = spice.vhat(cassinistate_tform[3:6])
        titandir = spice.vhat(titanstate_tform[3:6])
        print(cassinistate_tform,titanstate_tform)

        #print("ramdir", ramdir)
        #print("Angle to Corot Direction", spice.dpr() * spice.vsepg(ramdir, [0, 1], 2))
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, titandir))
    return angles_from_corot, np.array(cassinistates), np.array(titanstates)

def add_angle_to_corot_SaturnFrame(datetimes):
    angles_from_corot, cassinistates, titanstates = [], [], []
    for i in datetimes:
        et = spice.datetime2et(i)
        state, ltime = spice.spkezr("CASSINI", et, "IAU_SATURN", "LT+S", "Saturn")
        ramdir = spice.vhat(state[3:6])
        titanstate, ltime = spice.spkezr("TITAN", et, "IAU_SATURN", "LT+S", "Saturn")
        titandir = spice.vhat(titanstate[3:6])
        cassinistates.append(state)
        titanstates.append(titanstate)
        print(ramdir,titandir)
        #print("ramdir", ramdir)
        #print("Angle to Corot Direction", spice.dpr() * spice.vsepg(ramdir, [0, 1], 2))
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, titandir))
    return angles_from_corot, np.array(cassinistates), np.array(titanstates)

def add_angle_to_corot_TitanObvs(datetimes):
    angles_from_corot, cassinistates = [],[]
    for i in datetimes:
        et = spice.datetime2et(i)

        cassinistate, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "LT+S", "Titan")
        cassinistates.append(cassinistate)

        ramdir = spice.vhat(cassinistate[3:6])
        print(cassinistate)

        #print("ramdir", ramdir)
        #print("Angle to Corot Direction", spice.dpr() * spice.vsepg(ramdir, [0, 1], 2))
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, [0, 1, 0]))
    return angles_from_corot, np.array(cassinistates)

def add_angle_to_corot_CassiniFrame(datetimes):
    angles_from_corot, titanstates, cassinistates = [], [], []
    for i in datetimes:
        et = spice.datetime2et(i)

        titanstate_saturnframe, ltime = spice.spkezr("TITAN", et, "IAU_SATURN", "LT+S", "Cassini")
        Saturn2TitanFrame = spice.sxform("IAU_SATURN", "IAU_TITAN", et - ltime)
        titanstate_tform = spice.mxvg(Saturn2TitanFrame, titanstate_saturnframe, 6, 6)

        print("saturn frame", titanstate_saturnframe)
        print("saturn frame - tform", titanstate_tform)

        cassinistate, ltime = spice.spkezr("Cassini", et, "IAU_TITAN", "LT+S", "Cassini")
        cassinistates.append(cassinistate)
        titanstate, ltime = spice.spkezr("Titan", et, "IAU_TITAN", "LT+S", "Cassini")
        titanstates.append(titanstate)

        ramdir = spice.vhat(titanstate[3:6])
        print("titan frame", titanstate)
        print("titan frame - cassini", cassinistate)

        #print("ramdir", ramdir)
        #print("Angle to Corot Direction", spice.dpr() * spice.vsepg(ramdir, [0, 1], 2))
        angles_from_corot.append(spice.dpr() * spice.vsep(ramdir, titanstate[3:6]))
    return angles_from_corot, np.array(titanstates)


def corot_data(datetimes):

    cassinipositions = []
    titanpositions = []

    for i in datetimes:
        et = spice.datetime2et(i)
        titanpos, ltime = spice.spkpos('TITAN', et, 'IAU_SATURN', "LT+S", 'SATURN')
        cassinipos, ltime = spice.spkpos('CASSINI', et, 'IAU_SATURN', "LT+S", 'SATURN')
        cassinipositions.append(cassinipos)
        titanpositions.append(titanpos)

    return np.array(cassinipositions), np.array(titanpositions)


angles_dataframe = pd.DataFrame()


# flybys = ['t55']
# windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)


interval =  datetime.timedelta(minutes=1)
startdatetime = datetime.datetime(2009,5,21,21,18)
numberofintervals = 12
datetimes = [startdatetime + interval*i for i in range(numberofintervals)]

saturn_corot_angles, cassinistates_saturnframe, titanstates_saturnframe = add_angle_to_corot_SaturnFrame(datetimes)
print(saturn_corot_angles)
titan_corot_angles, cassinistates_titanframe, titanstates_titanframe = add_angle_to_corot_TitanFrame(datetimes)
print(titan_corot_angles)
titan_corot_angles_Cassini, cassinistates_titanframe_Cassini = add_angle_to_corot_CassiniFrame(datetimes)
print(titan_corot_angles_Cassini)

#cassinipositions, titanpositions = corot_data(datetimes)
# print(cassinipositions,titanpositions)
# print(cassinipositions[:,0])

def plot_saturnframe_xy(cassinistates,titanstates):
    figxy, axxy = plt.subplots(figsize=(6.4, 6.4), tight_layout=True)
    axxy.scatter(0, 0, color='r', label='Saturn')
    for time,cassinipos,titanpos in zip(datetimes,cassinistates[:,:3],titanstates[:,:3]):
        axxy.text(cassinipos[0],cassinipos[1],time.strftime("%H:%M"),color='k',fontsize=8)
        axxy.text(titanpos[0], titanpos[1], time.strftime("%H:%M"), color='y',fontsize=8)
        circle1 = plt.Circle((titanpos[0], titanpos[1]), 2575.15, color='y', fill=False)
        axxy.add_artist(circle1)
    axxy.plot(titanstates[:,0], titanstates[:,1], color='y', label='Titan', marker='o')
    axxy.plot(cassinistates[:,0], cassinistates[:,1], color='k', label='Cassini', marker='x')
    axxy.set_xlabel("X")
    axxy.set_ylabel("Y")
    # axxy.set_xlim(-75000, 3500)
    # axxy.set_ylim(-75000, 3500)
    axxy.set_box_aspect(1)
    # axxy.plot([-2575.15, -2575.15], [0, -5000], color='k')
    # axxy.plot([2575.15, 2575.15], [0, -5000], color='k')
    axxy.legend()

def plot_titanframe_xy(titanstates):
    figxy, axxy = plt.subplots(figsize=(6.4, 6.4), tight_layout=True)

    for time,titanpos in zip(datetimes,titanstates[:,:3]):
        axxy.text(titanpos[0], titanpos[1], time.strftime("%H:%M"), color='y',fontsize=8)
        circle1 = plt.Circle((titanpos[0], titanpos[1]), 2575.15, color='y', fill=False)
        axxy.add_artist(circle1)
    axxy.plot(titanstates[:,0], titanstates[:,1], color='y', label='Titan', marker='o')
    axxy.set_xlabel("X")
    axxy.set_ylabel("Y")
    # axxy.set_xlim(-75000, 3500)
    # axxy.set_ylim(-75000, 3500)
    axxy.set_box_aspect(1)
    # axxy.plot([-2575.15, -2575.15], [0, -5000], color='k')
    # axxy.plot([2575.15, 2575.15], [0, -5000], color='k')
    axxy.legend()

def plot_saturnframe_yz():
    figyz, axyz = plt.subplots(figsize=(6.4, 6.4), tight_layout=True)
    #axyz.scatter(0, 0, color='r', label='Saturn')
    for time,cassinipos,titanpos in zip(datetimes,cassinipositions,titanpositions):
        axyz.text(cassinipos[1],cassinipos[2],time.strftime("%H:%M"))
        circle1 = plt.Circle((titanpos[1], titanpos[2]), 2575.15, color='k', fill=False)
        axyz.add_artist(circle1)
    axyz.plot(titanpositions[:,1], titanpositions[:,2], color='y', label='Titan', marker='o')
    axyz.plot(cassinipositions[:,1], cassinipositions[:,2], color='k', label='Cassini', marker='x')
    axyz.set_xlabel("Y")
    axyz.set_ylabel("Z")
    # axyz.set_xlim(-3500, 3500)
    # axyz.set_ylim(-3500, 3500)
    axyz.set_box_aspect(1)
    # axyz.plot([0, -5000], [-2575.15, -2575.15], color='k')
    # axyz.plot([0, -5000], [2575.15, 2575.15], color='k')
    axyz.legend()

def plot_saturnframe_xz():
    figxz, axxz = plt.subplots(figsize=(6.4, 6.4), tight_layout=True)
    #axxz.scatter(0, 0, color='r', label='Saturn')
    for time,cassinipos,titanpos in zip(datetimes,cassinipositions,titanpositions):
        axxz.text(cassinipos[0],cassinipos[2],time.strftime("%H:%M"))
        circle1 = plt.Circle((titanpos[0], titanpos[2]), 2575.15, color='k', fill=False)
        axxz.add_artist(circle1)
    axxz.plot(titanpositions[:,0], titanpositions[:,2], color='y', label='Titan', marker='o')
    axxz.plot(cassinipositions[:,0], cassinipositions[:,2], color='k', label='Cassini', marker='x')
    axxz.set_xlabel("X")
    axxz.set_ylabel("Z")
    # axxz.set_xlim(-3500, 3500)
    # axxz.set_ylim(-3500, 3500)
    axxz.set_box_aspect(1)
    axxz.legend()

def plot_saturnframe_3d():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(titanpositions[0,0], titanpositions[0,1], titanpositions[0,2], color='r')
    ax.scatter(cassinipositions[0,0], cassinipositions[0,1], cassinipositions[0,2],  color='r')
    ax.plot(titanpositions[:,0], titanpositions[:,1], titanpositions[:,2], color='y', label='Titan', marker='o')
    ax.plot(cassinipositions[:,0], cassinipositions[:,1], cassinipositions[:,2], color='k', label='Cassini', marker='x')


plot_saturnframe_xy(cassinistates_saturnframe, titanstates_saturnframe)
plot_saturnframe_xy(cassinistates_titanframe, titanstates_titanframe)
plot_titanframe_xy(cassinistates_titanframe_Cassini)
plt.show()
