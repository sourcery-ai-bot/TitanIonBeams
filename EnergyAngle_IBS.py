import glob
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import normalize
from scipy.signal import convolve2d
import matplotlib.rcsetup as rcsetup
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

relativeenergies = np.arange(-6.628, 0.1950, 0.1949) #12 ESA voltages, 8 anode angles, 11 elevation angles, 8 anodes
relativeenergies_edges = np.arange(-6.72545, 0.3897, 0.1949)
azimuths = np.arange(-21.6, 14.1, 0.2)
azimuth_edges = np.arange(-21.7, 14.2, 0.2)

#print(relativeenergies,azimuths)
#print(relativeenergies_edges,azimuth_edges)

def plot_IBS_energyangle(data,anodeangle):
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


arr=plt.imread('IBS_energyangle_0polar_tidied.png')
#print(arr.shape)
arr_crop = arr[172:-126,326:-425,0]
#print(arr_crop.shape)
horizontal_tile_limits = list(np.arange(48,470,26))
vertical_tile_limits  = list(np.arange(114,480,24))

masked_arr = np.zeros(shape=arr_crop.shape)
masked_arr = np.where(arr_crop<0.08, np.nan, arr_crop)

averaged_arr = np.zeros(shape=arr_crop.shape)
new_arr = np.zeros(shape=(15,16))
for i in np.arange(0,16,1):
    for j in np.arange(0, 15, 1):
        if (114+(24*j) > 232 and 114+(24*j) < 254) or (48+(26*i) > 228 and 48+(26*i) < 253):
            small_arr = arr_crop[114+(24*j):114+(24*(j+1)),48+(26*i):48+(26*(i+1))]
            if 48 + (26 * i) < 80:
                threshold = 0.09
            elif 48 + (26 * i) > 305:
                threshold = 0.07
            elif 114+(24*j) < 200:
                threshold = 0.1
            elif 114+(24*j) > 350:
                threshold = 0.1
            else:
                threshold = 0.65
            small_masked_arr = np.where(small_arr<threshold , np.nan, small_arr)
            masked_arr[114+(24*j):114+(24*(j+1)),48+(26*i):48+(26*(i+1))] = small_masked_arr
            averaged_arr[114+(24*j):114+(24*(j+1)),48+(26*i):48+(26*(i+1))] = np.nanmean(small_masked_arr)
            new_arr[j, i] = np.nanmean(small_masked_arr)
        else:
            small_arr = arr_crop[114+(24*j):114+(24*(j+1)),48+(26*i):48+(26*(i+1))]
            masked_arr[114+(24*j):114+(24*(j+1)),48+(26*i):48+(26*(i+1))] = small_arr
            averaged_arr[114+(24*j):114+(24*(j+1)),48+(26*i):48+(26*(i+1))] = np.nanmean(small_arr)
            new_arr[j,i] = np.nanmean(small_arr)
fig, axs = plt.subplots(ncols=3,sharex='all',sharey='all')
axs[0].imshow(arr_crop,cmap="gray")
axs[0].hlines(vertical_tile_limits,47,470)
axs[0].vlines(horizontal_tile_limits,113,480)

axs[1].imshow(masked_arr,cmap="gray")
axs[2].imshow(averaged_arr,cmap="gray")

fig2, axs2 = plt.subplots()
energies_small = np.arange(-4.8739, -1.8, 0.1949)
energies_small_edges = np.arange(-4.97135, -1.8, 0.1949)
azimuths_small = np.arange(-1.8,1.2,0.2)
azimuths_small_edges = np.arange(-1.9,1.2,0.2)
print(new_arr)
axs2.pcolormesh(energies_small_edges,azimuths_small_edges,np.flip(new_arr,axis=0),cmap="gray")
divider = make_axes_locatable(axs2)
cax = divider.append_axes("right", size="1.5%", pad=.05)
cbar = fig.colorbar(cm.ScalarMappable(cmap='gray'), ax=axs2, cax=cax)
axs2.set_xlabel("Relative Energy (/%)",fontsize=20)
axs2.set_ylabel("Azimuth",fontsize=20)

plt.show()