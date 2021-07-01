import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)

def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x = np.linspace(-90, 90, 1000)

y_2016 = gaussian(x,0,70/2.355,373)
y_2017 = gaussian(x,3,101/2.355,196)
y_MW2006 = gaussian(x,0,70/2.355,60)

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

# for flyby, i in zip(windsdf["Flyby"].values, windsdf["Latitude"].values):
#     print(flyby,"Latitude",i,gaussian(i,0,70/2.355,373),gaussian(i,3,101/2.355,196))

plt.show()