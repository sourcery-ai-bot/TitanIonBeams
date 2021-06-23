from scipy.io import readsav
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from util import generate_mass_bins
from sunpy.time import TimeRange
import datetime
from heliopy.data.cassini import caps_ibs
from cassinipy.caps.mssl import *
from cassinipy.caps.util import *
from util import *
import seaborn as sns

matplotlib.rcParams.update({'font.size': 18})

df = pd.read_csv("TA_calibration.csv")

reduced_windsdf1 = df[["IBS 15u energy", "SNG 15u energy", "IBS 28u energy", "SNG 28u energy",]]#, "BT","Solar Zenith Angle",]]

dist_fig, (ax1, ax2) = plt.subplots(ncols=2)
sns.regplot(data=df, x="IBS 15u energy", y="SNG 15u energy", ax=ax1, color='C0')
sns.regplot(data=df, x="IBS 28u energy", y="SNG 28u energy", ax=ax2, color='C0')

line = np.arange(1,10,1)
ax1.plot(line,line,color='k',linestyle='--')
ax2.plot(line,line,color='k',linestyle='--')

ax1.set_xlim(3,3.7)
ax1.set_ylim(3,3.7)

ax2.set_xlim(5.1,5.7)
ax2.set_ylim(5.1,5.7)

ax1.legend()
ax2.legend()
plt.show()