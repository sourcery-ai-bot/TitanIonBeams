import datetime
import spiceypy as spice
import numpy as np
import glob
import pandas as pd

if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

def cassini_SZA(tempdatetime,moon="TITAN"):
    et = spice.datetime2et(tempdatetime)
    cassinidir, ltime = spice.spkpos('CASSINI', et, 'IAU_TITAN', "LT+S", 'TITAN')
    sundir, ltime = spice.spkpos('SUN', et, 'IAU_TITAN', "LT+S", 'TITAN')
    print(cassinidir,sundir)
    SZA = spice.vsep(cassinidir,sundir) * spice.dpr()
    print("SZA", SZA)
    return SZA

windsdf = pd.read_csv("winds_full.csv", index_col=0, parse_dates=True)


SZAs = []
for tempdatetime in windsdf['Positive Peak Time']:
    SZA = cassini_SZA(datetime.datetime.strptime(tempdatetime, "%Y-%m-%d %H:%M:%S.%f"))
    SZAs.append(SZA)
windsdf['Solar Zenith Angle'] = SZAs

windsdf.to_csv("winds_full.csv")

