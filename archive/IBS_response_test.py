import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.5


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


IBS_calib = pd.read_csv("IBS_calib.csv")

x = np.linspace(-10, 10, 1000)
IBS_calib_interp = np.interp(x, IBS_calib['Azimuth'], IBS_calib['Intensity'])
IBS_calib_interp = IBS_calib_interp / max(IBS_calib_interp)
x2 = np.linspace(-10, 10, 1999)

print(x[:5], x2[:5])
mu = 0
sig = 0.5

y_ionbeam = gaussian(x, mu, sig)
convolved_y = np.convolve(y_ionbeam, IBS_calib_interp, 'full')
convolved_y = convolved_y / max(convolved_y)
print(x[IBS_calib_interp.argmax()], x2[convolved_y.argmax()])

fig, ax = plt.subplots()
ax.plot(x, IBS_calib_interp, label="IBS azimuth response")
ax.plot(x, y_ionbeam, label="Ion Beam")
ax.plot(x2, convolved_y, label="Convolution")
ax.set_xlabel("Azimuth Angle")
ax.set_ylabel("Normalised Intensity")
ax.legend()

plt.show()
