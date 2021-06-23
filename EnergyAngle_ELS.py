import glob
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import normalize
from scipy.signal import convolve2d
import matplotlib.rcsetup as rcsetup

sweepvalues_5eV = np.arange(0.66, 1.02, 0.03) #12 ESA voltages, 8 anode angles, 11 elevation angles, 8 anodes
sweepvalues_5eV_edges = np.arange(0.645, 1.02, 0.03)
elevations_5eV = np.arange(-10, 12, 2)
elevations_5eV_edges = np.arange(-11, 12, 2)

sweepvalues_30eV = np.arange(4.05, 5.8, 0.17)
sweepvalues_30eV_edges = np.arange(3.965, 6., 0.17)
elevations_30eV = np.arange(-8, 9, 1)
elevations_30eV_edges = np.arange(-8.5, 9, 1)



filelist_30eV = ["ELSC030B.DAT","ELSC030C.DAT","ELSC030D.DAT","ELSC030E.DAT","ELSC030F.DAT",
                 "ELSC030G.DAT","ELSC030H.DAT","ELSC030J.DAT","ELSC030K.DAT","ELSC030L.DAT","ELSC030M.DAT"] #Only half the files?

def parse_ELS_energyangle():

    #datafiles = glob.glob("ELS_energyangle_data/30eV/*")
    datafiles = ["ELS_energyangle_data/30eV/" + i for i in filelist_30eV]
    print(datafiles)
    dataarray = np.zeros(shape=(len(datafiles), 8, 17, 8)) #11 ESA voltages, 8 anode angles, 17 elevation angles, 8 anodes

    filecounter = 0
    for filename in datafiles:
        print(filename)
        with open(filename, 'r') as file:
            anodeangle_counter = 0
            elevationcounter = 0
            for x in file:
                if elevationcounter == 17:
                    anodeangle_counter += 1
                    elevationcounter = 0
                if x[:7] == "    -99":
                    values_str = list(filter(None,x[7:-1].split(" ")))
                    values = [int(i) for i in values_str]
                    #print(filename,anodeangle_counter,elevationcounter,x,values)
                    dataarray[filecounter,anodeangle_counter,elevationcounter,:] = values
                    elevationcounter += 1
        filecounter +=1
    return dataarray


def plot_ELS_energyangle_anodes(data,anodeangle):
    fig, axes = plt.subplots(4, 2, sharex='all', sharey='all')
    anodecounter = 0
    for i in range(4):
        for j in range(2):
            # print(sweepvalues,elevations)
            axes[i, j].pcolormesh(sweepvalues_30eV_edges, elevations_30eV_edges, data[:, 7 - (anodeangle - 1), :, anodecounter].T,
                                  norm=LogNorm(vmin=1, vmax=2e4),
                                  cmap='jet', shading="flat")
            axes[i, j].set_title("Anode " + str(anodecounter+1))
            anodecounter += 1

    plt.show()

def plot_ELS_1D_elevationsweep(data,anodeangle = 3 , anode = 4):
    singleanode = data[:, anodeangle, :, anode]
    max_index = np.unravel_index(np.argmax(singleanode, axis=None), singleanode.shape)
    print(singleanode,singleanode.shape)
    print(max_index)
    print("Peak Elevation", elevations_30eV[max_index[1]])
    print("Peak Sweep Energy", sweepvalues_30eV[max_index[0]])
    print(singleanode[max_index])

    fig, axes = plt.subplots(2)
    axes[0].plot(elevations_30eV, singleanode[max_index[1], :])
    axes[0].set_xlabel("Elevation Angle")
    axes[0].set_ylabel("Counts")
    axes[1].plot(sweepvalues_30eV, singleanode[:, max_index[0]])
    axes[1].set_xlabel("Hemisphere sweep voltage")
    axes[1].set_ylabel("Counts")


# Mean vector and covariance matrix
mu = np.array([4.83, 0.])
Sigma = np.array([[0.01, 0], [0,  1]])

X, Y = np.meshgrid(sweepvalues_30eV, elevations_30eV)
# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N
# The distribution on the variables X, Y packed into pos.

def multi_elevation_plot(elsdata):
    fig, ax = plt.subplots(1)
    jet = plt.cm.jet
    N = elsdata.shape[2]
    idx = np.linspace(0, 1, N)
    ax.set_prop_cycle(rcsetup.cycler('color', jet(idx)))

    for i in range(elsdata.shape[2]):
        ax.plot(sweepvalues_30eV,elsdata[:, 3, i, 4],label=str("Elev = " + str(elevations_30eV[i])))
    ax.set_xlabel("Sweep Voltage")
    ax.set_xlabel("Counts")
    ax.legend()

def convolution_2d(elsdata):
    Z = multivariate_gaussian(pos, mu, Sigma)
    norm_Z = normalize(Z, axis=0, norm='l1')
    print("Z Sum", np.sum(Z))
    print("Norm Z sum", np.sum(norm_Z))




    singleanode = els_30eV_data[:, 3, :, 4]
    print(singleanode.shape)
    normed_matrix = normalize(singleanode, axis=0, norm='l1')

    convolved = normalize(convolve2d(normed_matrix,Z,mode="full"), axis=1, norm='l1')
    print(convolved.shape)

    fig, axes = plt.subplots(4,sharex="all",sharey="all")

    axes[0].pcolormesh(sweepvalues_30eV_edges, elevations_30eV_edges, singleanode.T,norm=LogNorm(vmin=1, vmax=1e4), cmap='jet', shading="flat")
    axes[0].set_title("30eV Anode 4")

    axes[1].pcolormesh(sweepvalues_30eV_edges, elevations_30eV_edges, normed_matrix.T,norm=LogNorm(vmin=0.01, vmax=1), cmap='jet', shading="flat")
    axes[1].set_title("30eV Anode 4 - Normalised")

    axes[2].pcolormesh(sweepvalues_30eV_edges, elevations_30eV_edges, Z,norm=LogNorm(vmin=0.01, vmax=1), cmap='jet', shading="flat")
    axes[2].set_title("Test Ion Beam")

    #axes[3].pcolormesh(sweepvalues_30eV_edges, elevations_30eV_edges, convolved,norm=LogNorm(vmin=0.01, vmax=1), cmap='jet', shading="flat")
    axes[3].set_xlabel("Sweep Energy",fontsize=25)
    axes[3].set_title("Convolved")

    fig.text(0.08, 0.5, 'Elevation', va='center', rotation='vertical',fontsize=25)

els_30eV_data = parse_ELS_energyangle()

multi_elevation_plot(els_30eV_data)

plt.show()