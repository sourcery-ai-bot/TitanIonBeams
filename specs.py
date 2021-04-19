from util import generate_mass_bins
from scipy.io import readsav
from cassinipy.caps.mssl import ELS_spectrogram, IBS_spectrogram
import matplotlib.pyplot as plt

plt.rcParams['date.epoch'] = '0000-12-31'

# T16ELSdata = readsav("data/els/elsres_22-jul-2006.dat")
# generate_mass_bins(T16ELSdata, "t16", "els")
# ELS_spectrogram(T16ELSdata, [2,3,4,5], "00:20:00", 600)
#
# T16IBSdata = readsav("data/ibs/ibsres_22-jul-2006.dat")
# generate_mass_bins(T16IBSdata, "t16", "ibs")
# IBS_spectrogram(T16IBSdata, [0,1,2], "00:20:00", 600)

# T17ELSdata = readsav("data/els/elsres_07-sep-2006.dat")
# generate_mass_bins(T17ELSdata, "t17", "els")
# ELS_spectrogram(T17ELSdata, [2,3,4,5], "20:13:00", 600)

# T17IBSdata = readsav("data/ibs/ibsres_07-sep-2006.dat")
# generate_mass_bins(T17IBSdata, "t17", "ibs")
# IBS_spectrogram(T17IBSdata, [0,1,2], "20:13:00", 600)

# T18ELSdata = readsav("data/els/elsres_23-sep-2006.dat")
# generate_mass_bins(T18ELSdata, "t18", "els")
# ELS_spectrogram(T18ELSdata, [2,3,4,5], "18:55:00", 600)
#
# T18IBSdata = readsav("data/ibs/ibsres_23-sep-2006.dat")
# generate_mass_bins(T18IBSdata, "t18", "ibs")
# IBS_spectrogram(T18IBSdata, [0,1,2], "18:55:00", 600)

# T19ELSdata = readsav("data/els/elsres_09-oct-2006.dat")
# generate_mass_bins(T19ELSdata, "t19", "els")
# ELS_spectrogram(T19ELSdata, [2,3,4,5], "17:26:00", 600)
#
# T19IBSdata = readsav("data/ibs/ibsres_09-oct-2006.dat")
# generate_mass_bins(T19IBSdata, "t19", "ibs")
# IBS_spectrogram(T19IBSdata, [0,1,2], "17:23:00", 600)

# T20ELSdata = readsav("data/els/elsres_25-oct-2006.dat")
# generate_mass_bins(T20ELSdata, "t20", "els")
# ELS_spectrogram(T20ELSdata, [3, 4], "15:56:00", 600)

# T20IBSdata = readsav("data/ibs/ibsres_25-oct-2006.dat")
# generate_mass_bins(T20IBSdata, "t20", "ibs")
# IBS_spectrogram(T20IBSdata, [1], "15:56:00", 600)

# T21IBSdata = readsav("data/ibs/ibsres_12-dec-2006.dat")
# generate_mass_bins(T21IBSdata, "t21", "ibs")
# IBS_spectrogram(T21IBSdata, [1], "11:37:00", 600)

# T23ELSdata = readsav("data/els/elsres_13-jan-2007.dat")
# generate_mass_bins(T23ELSdata, "t23", "els")
# ELS_spectrogram(T23ELSdata, [3, 4], "08:34:00", 600)
#
# T23IBSdata = readsav("data/ibs/ibsres_13-jan-2007.dat")
# generate_mass_bins(T23IBSdata, "t23", "ibs")
# IBS_spectrogram(T23IBSdata, [1], "08:34:00", 600)

# T25ELSdata = readsav("data/els/elsres_22-feb-2007.dat")
# generate_mass_bins(T25ELSdata, "t25", "els")
# ELS_spectrogram(T25ELSdata, [2, 3, 4, 5], "03:08:00", 600)
#
# T25IBSdata = readsav("data/ibs/ibsres_22-feb-2007.dat")
# generate_mass_bins(T25IBSdata, "t25", "ibs")
# IBS_spectrogram(T25IBSdata, [1], "03:08:00", 600)

# T26ELSdata = readsav("data/els/elsres_10-mar-2007.dat")
# generate_mass_bins(T26ELSdata, "t26", "els")
# ELS_spectrogram(T26ELSdata, [3, 4], "01:45:30", 480)
#
# T26IBSdata = readsav("data/ibs/ibsres_10-mar-2007.dat")
# generate_mass_bins(T26IBSdata, "t26", "ibs")
# IBS_spectrogram(T26IBSdata, [1], "01:45:30", 480)

# T27ELSdata = readsav("data/els/elsres_26-mar-2007.dat")
# generate_mass_bins(T27ELSdata, "t27", "els")
# ELS_spectrogram(T27ELSdata, [4,5,6,7], "00:18:00", 750)

# T28ELSdata = readsav("data/els/elsres_10-apr-2007.dat")
# generate_mass_bins(T28ELSdata, "t28", "els")
# ELS_spectrogram(T28ELSdata, [3, 4], "22:54:00", 480)
#
# T28IBSdata = readsav("data/ibs/ibsres_10-apr-2007.dat")
# generate_mass_bins(T28IBSdata, "t28", "ibs")
# IBS_spectrogram(T28IBSdata, [1], "22:54:00", 480)

# T29ELSdata = readsav("data/els/elsres_26-apr-2007.dat")
# generate_mass_bins(T29ELSdata, "t29", "els")
# ELS_spectrogram(T29ELSdata, [3, 4, 5], "21:29:00", 600)
#
# T29IBSdata = readsav("data/ibs/ibsres_26-apr-2007.dat")
# generate_mass_bins(T29IBSdata, "t29", "ibs")
# IBS_spectrogram(T29IBSdata, [0, 1, 2], "21:29:00", 600)

# T30ELSdata = readsav("data/els/elsres_12-may-2007.dat")
# generate_mass_bins(T30ELSdata, "t30", "els")
# ELS_spectrogram(T30ELSdata, [3, 4, 5], "20:05:00", 600)
#
# T30IBSdata = readsav("data/ibs/ibsres_12-may-2007.dat")
# generate_mass_bins(T30IBSdata, "t30", "ibs")
# IBS_spectrogram(T30IBSdata, [1], "20:05:00", 600)

# T32ELSdata = readsav("data/els/elsres_13-jun-2007.dat")
# generate_mass_bins(T32ELSdata, "t32", "els")
# ELS_spectrogram(T32ELSdata, [2, 3, 4], "17:43:00", 600)
#
# T32IBSdata = readsav("data/ibs/ibsres_13-jun-2007.dat")
# generate_mass_bins(T32IBSdata, "t32", "ibs")
# IBS_spectrogram(T32IBSdata, [1], "17:43:00", 600)

# T32ELSdata = readsav("data/els/elsres_13-jun-2007.dat")
# generate_mass_bins(T32ELSdata, "t32", "els")
# ELS_spectrogram(T32ELSdata, [2, 3, 4], "17:43:00", 600)
#
# T32IBSdata = readsav("data/ibs/ibsres_13-jun-2007.dat")
# generate_mass_bins(T32IBSdata, "t32", "ibs")
# IBS_spectrogram(T32IBSdata, [1], "17:43:00", 600)

# T42ELSdata = readsav("data/els/elsres_25-mar-2008.dat")
# generate_mass_bins(T42ELSdata, "t42", "els")
# ELS_spectrogram(T42ELSdata, [3,4], "14:24:00", 600)

# T42IBSdata = readsav("data/ibs/ibsres_25-mar-2008.dat")
# generate_mass_bins(T42IBSdata, "t42", "ibs")
# IBS_spectrogram(T42IBSdata, [1], "14:24:00", 600)

# T47ELSdata = readsav("data/els/elsres_19-nov-2008.dat")
# generate_mass_bins(T47ELSdata, "t47", "els")
# ELS_spectrogram(T47ELSdata, [0,1], "15:52:00", 300)

plt.show()
