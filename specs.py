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
# IBS_spectrogram(T16IBSdata, [0,1,2], "00:18:00", 1080)

# T17ELSdata = readsav("data/els/elsres_07-sep-2006.dat")
# generate_mass_bins(T17ELSdata, "t17", "els")
# ELS_spectrogram(T17ELSdata, [2,3,4,5], "20:13:00", 600)

# T17IBSdata = readsav("data/ibs/ibsres_07-sep-2006.dat")
# generate_mass_bins(T17IBSdata, "t17", "ibs")
# IBS_spectrogram(T17IBSdata, [0,1,2], "20:08:00", 1080)

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
# IBS_spectrogram(T19IBSdata, [0,1,2], "17:21:00", 1080)

# T20ELSdata = readsav("data/els/elsres_25-oct-2006.dat")
# generate_mass_bins(T20ELSdata, "t20", "els")
# ELS_spectrogram(T20ELSdata, [3, 4], "15:56:00", 600)

# T20IBSdata = readsav("data/ibs/ibsres_25-oct-2006.dat")
# generate_mass_bins(T20IBSdata, "t20", "ibs")
# IBS_spectrogram(T20IBSdata, [0,1,2], "15:56:00", 600)

# T21IBSdata = readsav("data/ibs/ibsres_12-dec-2006.dat")
# generate_mass_bins(T21IBSdata, "t21", "ibs")
# IBS_spectrogram(T21IBSdata, [0,1,2], "11:33:00", 1080)

# T23ELSdata = readsav("data/els/elsres_13-jan-2007.dat")
# generate_mass_bins(T23ELSdata, "t23", "els")
# ELS_spectrogram(T23ELSdata, [3, 4], "08:34:00", 600)
#
# T23IBSdata = readsav("data/ibs/ibsres_13-jan-2007.dat")
# generate_mass_bins(T23IBSdata, "t23", "ibs")
# IBS_spectrogram(T23IBSdata, [1], "08:30:00", 1080)

# T25ELSdata = readsav("data/els/elsres_22-feb-2007.dat")
# generate_mass_bins(T25ELSdata, "t25", "els")
# ELS_spectrogram(T25ELSdata, [2, 3, 4, 5], "03:08:00", 600)
#
# T25IBSdata = readsav("data/ibs/ibsres_22-feb-2007.dat")
# generate_mass_bins(T25IBSdata, "t25", "ibs")
# IBS_spectrogram(T25IBSdata, [0,1,2,3], "03:03:00", 1080)

# T26ELSdata = readsav("data/els/elsres_10-mar-2007.dat")
# generate_mass_bins(T26ELSdata, "t26", "els")
# ELS_spectrogram(T26ELSdata, [3, 4], "01:45:30", 480)
#
# T26IBSdata = readsav("data/ibs/ibsres_10-mar-2007.dat")
# generate_mass_bins(T26IBSdata, "t26", "ibs")
# IBS_spectrogram(T26IBSdata, [1], "01:41:00", 1080)

# T27ELSdata = readsav("data/els/elsres_26-mar-2007.dat")
# generate_mass_bins(T27ELSdata, "t27", "els")
# ELS_spectrogram(T27ELSdata, [4,5,6,7], "00:18:00", 750)

# T27IBSdata = readsav("data/ibs/ibsres_26-mar-2007.dat")
# generate_mass_bins(T27IBSdata, "t27", "ibs")
# IBS_spectrogram(T27IBSdata, [0,1,2], "00:15:00", 1080)

# T28ELSdata = readsav("data/els/elsres_10-apr-2007.dat")
# generate_mass_bins(T28ELSdata, "t28", "els")
# ELS_spectrogram(T28ELSdata, [3, 4], "22:54:00", 480)
#
# T28IBSdata = readsav("data/ibs/ibsres_10-apr-2007.dat")
# generate_mass_bins(T28IBSdata, "t28", "ibs")
# IBS_spectrogram(T28IBSdata, [0, 1, 2], "22:48:00", 1080)

# T29ELSdata = readsav("data/els/elsres_26-apr-2007.dat")
# generate_mass_bins(T29ELSdata, "t29", "els")
# ELS_spectrogram(T29ELSdata, [3, 4, 5], "21:29:00", 600)
#
# T29IBSdata = readsav("data/ibs/ibsres_26-apr-2007.dat")
# generate_mass_bins(T29IBSdata, "t29", "ibs")
# IBS_spectrogram(T29IBSdata, [0, 1, 2], "21:25:00", 1080)

# T30ELSdata = readsav("data/els/elsres_12-may-2007.dat")
# generate_mass_bins(T30ELSdata, "t30", "els")
# ELS_spectrogram(T30ELSdata, [3, 4, 5], "20:05:00", 600)
#
# T30IBSdata = readsav("data/ibs/ibsres_12-may-2007.dat")
# generate_mass_bins(T30IBSdata, "t30", "ibs")
# IBS_spectrogram(T30IBSdata, [0,1,2], "20:02:00", 1080)

# T32ELSdata = readsav("data/els/elsres_13-jun-2007.dat")
# generate_mass_bins(T32ELSdata, "t32", "els")
# ELS_spectrogram(T32ELSdata, [2, 3, 4], "17:43:00", 600)
#
# T32IBSdata = readsav("data/ibs/ibsres_13-jun-2007.dat")
# generate_mass_bins(T32IBSdata, "t32", "ibs")
# IBS_spectrogram(T32IBSdata, [1], "17:39:00", 1080)

# T32ELSdata = readsav("data/els/elsres_13-jun-2007.dat")
# generate_mass_bins(T32ELSdata, "t32", "els")
# ELS_spectrogram(T32ELSdata, [2, 3, 4], "17:43:00", 600)
#
# T32IBSdata = readsav("data/ibs/ibsres_13-jun-2007.dat")
# generate_mass_bins(T32IBSdata, "t32", "ibs")
# IBS_spectrogram(T32IBSdata, [1], "17:43:00", 600)
#
# T40IBSdata = readsav("data/ibs/ibsres_05-jan-2008.dat")
# generate_mass_bins(T40IBSdata, "t40", "ibs")
# IBS_spectrogram(T40IBSdata, [0, 1, 2, 3], "21:22:00", 1020)

# T42ELSdata = readsav("data/els/elsres_25-mar-2008.dat")
# generate_mass_bins(T42ELSdata, "t42", "els")
# ELS_spectrogram(T42ELSdata, [3,4], "14:24:00", 600)

# T42IBSdata = readsav("data/ibs/ibsres_25-mar-2008.dat")
# generate_mass_bins(T42IBSdata, "t42", "ibs")
# IBS_spectrogram(T42IBSdata, [1], "14:24:00", 600)

# T43IBSdata = readsav("data/ibs/ibsres_12-may-2008.dat")
# generate_mass_bins(T43IBSdata, "t43", "ibs")
# IBS_spectrogram(T43IBSdata, [0, 1, 2, 3], "09:55:00", 900)

# T47ELSdata = readsav("data/els/elsres_19-nov-2008.dat")
# generate_mass_bins(T47ELSdata, "t47", "els")
# ELS_spectrogram(T47ELSdata, [0,1], "15:52:00", 300)

# T48IBSdata = readsav("data/ibs/ibsres_05-dec-2008.dat")
# generate_mass_bins(T48IBSdata, "t48", "ibs")
# IBS_spectrogram(T48IBSdata, [1], "14:17:00", 1080)

#
# T49IBSdata = readsav("data/ibs/ibsres_21-dec-2008.dat")
# generate_mass_bins(T49IBSdata, "t49", "ibs")
# IBS_spectrogram(T49IBSdata, [0, 1, 2, 3], "12:53:00", 1080)

# T50IBSdata = readsav("data/ibs/ibsres_07-feb-2009.dat")
# generate_mass_bins(T50IBSdata, "t50", "ibs")
# IBS_spectrogram(T50IBSdata, [1], "08:40:00", 1080)

# T51IBSdata = readsav("data/ibs/ibsres_27-mar-2009.dat")
# generate_mass_bins(T51IBSdata, "t50", "ibs")
# IBS_spectrogram(T51IBSdata, [1], "04:35:00", 1080)

T55ELSdata = readsav("data/els/elsres_21-may-2009.dat")
generate_mass_bins(T55ELSdata, "t55", "els")
ELS_spectrogram(T55ELSdata, [2,3,4,5], "21:20:00", 720)


# T57IBSdata = readsav("data/ibs/ibsres_22-jun-2009.dat")
# generate_mass_bins(T57IBSdata, "t57", "ibs")
# IBS_spectrogram(T57IBSdata, [0, 1, 2, 3], "18:26:10", 1080)

# T71IBSdata = readsav("data/ibs/ibsres_07-jul-2010.dat")
# generate_mass_bins(T71IBSdata, "t71", "ibs")
# IBS_spectrogram(T71IBSdata, [1], "00:16:00", 1080)

# T83IBSdata = readsav("data/ibs/ibsres_22-may-2012.dat")
# generate_mass_bins(T83IBSdata, "t83", "ibs")
# IBS_spectrogram(T83IBSdata, [1], "01:00:00", 1080)

plt.show()
