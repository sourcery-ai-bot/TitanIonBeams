import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)

def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



x = np.linspace(-90, 90, 1000)
mu_2016 = 0
mu_2017 = 3
mu_MW2006 = 0
FWHM_2016 = 70
FWHM_2017 = 101
FWHM_MW2006 = 70
sig_2016 = FWHM_2016/2.355
sig_2017 = FWHM_2017/2.355
sig_MW2006 = FWHM_MW2006/2.355
amp_2016 = 373
amp_2017 = 196
amp_MW2006 = 60


y_2016 = gaussian(x,mu_2016,sig_2016,amp_2016)
y_2017 = gaussian(x,mu_2017,sig_2017,amp_2017)
y_MW2006 = gaussian(x,mu_MW2006,sig_MW2006,amp_MW2006)

fig, axes = plt.subplots(2,sharex='all')

axes[0].plot(x, y_2016, label="August 2016")
axes[0].plot(x, y_2017, label="May 2017")
axes[0].plot(x, y_MW2006, label="Muller-Wodarg 2006 Model")
axes[0].set_xlabel("Latitude (degrees)",fontsize=18)
axes[0].set_ylabel("Zonal Wind Speed (m/s)",fontsize=18)
axes[0].legend()
axes[0].set_xlim(-90,90)

sns.kdeplot(data=windsdf,x="Latitude",hue="Flyby",ax=axes[1],legend=True)
axes[1].legend()

for i in windsdf["Latitude"].values:
    print("Latitude",i,gaussian(i,mu_2016,sig_2016,amp_2016),gaussian(i,mu_2017,sig_2017,amp_2017))

plt.show()