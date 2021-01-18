from __future__ import division
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import scipy.integrate

matplotlib.rcParams.update({'font.size': 15})

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x = np.linspace(-15, 15, 1000)
x_convolve = np.linspace(-15, 15, 1999)
mu = 0
sig = 1
y_ionbeam = gaussian(x, mu, sig)

def calc_prod(ELS_act, x=x, y_ionbeam=y_ionbeam):
    ELS_mu = 0.41  # 0.41 at 125eV
    ELS_sig = 2.74
    y_ELS = gaussian(x, ELS_mu + ELS_act, ELS_sig)

    prod = y_ionbeam * y_ELS
    integrated = scipy.integrate.simps(prod, x)

    return y_ELS, prod, integrated

fig = plt.figure()
ax1 = plt.axes(xlim=(-10, 10), ylim=(0,1))
ax1.plot(x, y_ionbeam , label="Ion Beam")
textobj = ax1.text(0.8,0.8,s="")
plotlays, plotcols = [2], ["black","red"]
frame_num = 128
x1,y1 = [],[]
x2,y2 = [],[]

lines = []
for index in range(2):
    lobj = ax1.plot([],[],lw=1,color=plotcols[index])[0]
    lines.append(lobj)
lines.append(textobj)

def init():
    for line in lines:
        if type(line) == matplotlib.lines.Line2D:
            line.set_data([],[])
    return lines

def animate(i,x,y_ionbeam):
    y_ELS, convolved_y, integrated = calc_prod((i-64)/8, x=x, y_ionbeam=y_ionbeam)

    xlist = [x, x]
    ylist = [y_ELS, convolved_y]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        if type(line) == matplotlib.lines.Line2D:
            line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.
        elif type(line) == matplotlib.text.Text:
            textobj.set_text('{:04.4f}\n{:04.4f}'.format((i-64)/8,integrated))

    return lines

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frame_num, fargs=[x, y_ionbeam], interval=250, blit=True)

plt.show()
