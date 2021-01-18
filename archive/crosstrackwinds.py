import pandas as pd
import numpy as np
import spiceypy as spice
import matplotlib
import glob

# Loading Kernels
if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.5

titan_flybyvelocities = {'t16': 6e3, 't17': 6e3,
                         't20': 6e3, 't21': 5.9e3, 't25': 6.2e3, 't26': 6.2e3, 't27': 6.2e3, 't28': 6.2e3, 't29': 6.2e3,
                         't30': 6.2e3, 't32': 6.2e3,
                         't40': 6.3e3, 't42': 6.3e3, 't46': 6.3e3, 't47': 6.3e3,
                         't83': 5.9e3}


def cassini_titan_altlatlon(tempdatetime):
    et = spice.datetime2et(tempdatetime)

    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    lon, lat, alt = spice.recpgr('TITAN', state[:3], spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    return alt, lat * spice.dpr(), lon * spice.dpr()


CAPS_readin_df = pd.read_csv("Beams_database.csv", index_col=0, parse_dates=True)
CAPS_readin_df['Peak Time'] = pd.to_datetime(CAPS_readin_df.index)
data = pd.DataFrame()
for counter, flyby in enumerate(CAPS_readin_df.Flyby.unique()):

    single_flyby_df_grouped = CAPS_readin_df[CAPS_readin_df['Flyby'] == flyby].groupby(pd.Grouper(freq='30s'))
    group_list = [(index, group) for index, group in single_flyby_df_grouped if len(group) > 0]
    tempdata = pd.DataFrame(
        [x[1].iloc[x[1]['Peak Energy'].argmax()] for x in group_list])

    data = pd.concat([data,tempdata])
    #print(flyby, data)

alts, lats, lons = [], [], []
for tempdatetime in data.index:
    alt, lat, lon = cassini_titan_altlatlon(tempdatetime)
    alts.append(alt)
    lats.append(lat)
    lons.append(lon)
data['Altitude'] = alts
data['Longitude'] = lons
data['Latitude'] = lats

data = data[data['Peak Energy'] > 15]
data.dropna(subset=['Bulk Azimuth'], inplace=True)
data = data[(data["Actuation Direction"]=="positive") | (data["Actuation Direction"]=="negative")]

velocityseries = pd.Series(dtype=float)
flybyslist = data.Flyby.unique()

for counter, flyby in enumerate(data.Flyby.unique()):
    tempdf = data[data['Flyby'] == flyby]
    tempvelocities = [np.sin(x * spice.rpd()) * titan_flybyvelocities[flyby] for x in
                      tempdf["Bulk Deflection from Ram Angle"]]

    tempseries = pd.Series(data=tempvelocities, name=flyby)
    tempseries.reset_index(drop=True, inplace=True)
    velocityseries = velocityseries.append(tempseries)

data.reset_index(drop=True, inplace=True)
velocityseries.reset_index(drop=True, inplace=True)
velocityseries.name = "Crosstrack velocity"

absvelocityseries = abs(velocityseries)
absvelocityseries.name = "Absolute Crosstrack velocity"
data = pd.concat([data, velocityseries, absvelocityseries], axis=1)
data["Azimuthal Ram Time temp"] = pd.to_datetime(data["Azimuthal Ram Time"]).apply(lambda x: x.replace(microsecond=0))
data.drop_duplicates(subset="Azimuthal Ram Time temp",inplace=True)
data.drop(columns=['Azimuthal Ram Time temp'],inplace=True)
print(data['Crosstrack velocity'])
data.to_csv("crosswinds_database.csv")
