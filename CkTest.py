import spiceypy as spice
import glob
import datetime
import matplotlib.pyplot as plt
from util import generate_mass_bins
from scipy.io import readsav
import pandas as pd
import matplotlib

from cassinipy.caps.mssl import *
from cassinipy.caps.spice import *
from cassinipy.caps.util import *
from cassinipy.misc import *
from cassinipy.spice import *

matplotlib.rcParams.update({'font.size': 15})

# Loading Kernels
if spice.ktotal('spk') == 0:
    for file in glob.glob("spice/**/*.*", recursive=True):
        spice.spiceypy.furnsh(file)
    count = spice.ktotal('ALL')
    print('Kernel count after load:        {0}\n'.format(count))


def ELS_beamanodes(elevation):
    beamanodes = [0, 0]
    anoderanges = np.arange(-70, 90, 20)
    if elevation < -70:
        return [0]
    if elevation > 70:
        return [7]

    for counter, elv in enumerate(anoderanges):
        # print(counter,elv, beamanodes)
        if elevation >= elv and elevation <= anoderanges[counter + 1]:
            beamanodes[1] += 1
            return beamanodes
        else:
            beamanodes[0] += 1
            beamanodes[1] += 1


def ELS_ramanodes(tempdatetime):
    ramelv = caps_ramdirection_azielv(tempdatetime)[1]
    return ELS_beamanodes(ramelv)


def cassini_titan_altlatlon(tempdatetime):
    et = spice.datetime2et(tempdatetime)

    state, ltime = spice.spkezr('CASSINI', et, 'IAU_TITAN', 'NONE', 'TITAN')
    lon, lat, alt = spice.recpgr('TITAN', state[:3], spice.bodvrd('TITAN', 'RADII', 3)[1][0], 2.64e-4)

    return alt, lat * spice.dpr(), lon * spice.dpr()


def caps_all_anodes(tempdatetime):
    et = spice.datetime2et(tempdatetime)
    sclkdp = spice.sce2c(-82, et)  # converts an et to a continuous encoded sc clock (ticks)

    caps_els_anode_vecs = []
    for x in np.arange(70, -90, -20):
        # print(anodenumber, x)
        rotationmatrix_anode = spice.spiceypy.axisar(np.array([1, 0, 0]),
                                                     x * spice.rpd())  # Get angles for different anodes
        # print("rotationmatrix_anode", rotationmatrix_anode)
        postanode_rotation = spice.vhat(
            spice.mxv(rotationmatrix_anode, -spice.spiceypy.getfov(-82821, 20)[2]))  # Apply rotation for anodes
        # print("postanode_rotation", postanode_rotation)

        # print("caps_els_boresight", caps_els_boresight)
        cassini_caps_mat = spice.ckgp(-82821, sclkdp, 0, 'CASSINI_CAPS_BASE')[0]  # Get actuation angle
        # print("cassini_caps_mat", cassini_caps_mat)
        cassini_caps_act_vec = spice.mxv(cassini_caps_mat, postanode_rotation)  # Rotate with actuator
        # print("Actuating frame", cassini_caps_act_vec)

        CAPS_act_2_titan_cmat = spice.ckgp(-82000, sclkdp, 0, 'IAU_TITAN')[
            0]  # Find matrix to transform to IAU_TITAN frame
        CAPS_act_2_titan_cmat_transpose = spice.xpose(CAPS_act_2_titan_cmat)  # Tranpose matrix
        rotated_vec = spice.mxv(CAPS_act_2_titan_cmat_transpose, cassini_caps_act_vec)  # Apply Matrix
        # print("rotated_vec ", rotated_vec)
        caps_els_anode_vecs.append(rotated_vec)

    return caps_els_anode_vecs


def caps_crosstrack(tempdatetime, windspeed):
    et = spice.datetime2et(tempdatetime)

    state, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "NONE", "titan")
    ramdir = spice.vhat(state[3:6])
    # print("ramdir",ramdir)

    # Gets Attitude
    sclkdp = spice.sce2c(-82, et)  # converts an et to a continuous encoded sc clock (ticks)
    ckgp_output = spice.ckgp(-82000, sclkdp, 0, "IAU_TITAN")
    cmat = ckgp_output[0]

    spacecraft_axis = np.array([0, 0, 1])
    rotated_spacecraft_axis = spice.mxv(cmat, spacecraft_axis)
    # print("cmat", cmat)
    # print("rotated spacecraft axis",rotated_spacecraft_axis)
    ram_unit = spice.mxv(cmat, -ramdir)  # Ram Unit in SC coords
    # print("ram_unit",ram_unit)

    if windspeed < 0:
        rotationmatrix = spice.axisar(np.array([0, 0, -1]), 90 * spice.rpd())
    if windspeed > 0:
        rotationmatrix = spice.axisar(np.array([0, 0, -1]), -90 * spice.rpd())

    # print(rotationmatrix)
    crossvec = spice.mxv(rotationmatrix, ram_unit)  # Rotate ram unit to find crosstrack velocity vector
    # print("temp crossvec",crossvec)
    # print("vsep SC Frame",spice.vsep(ram_unit,crossvec)*spice.dpr())
    cmat_t = spice.xpose(cmat)
    return spice.mxv(cmat_t, crossvec)


def caps_crosstrack_spice(tempdatetime, windspeed):
    et = spice.datetime2et(tempdatetime)
    sclkdp = spice.sce2c(-82, et)  # converts an et to a continuous encoded sc clock (ticks)

    state, ltime = spice.spkezr("CASSINI", et, "IAU_TITAN", "NONE", "titan")
    ramdir = spice.vhat(state[3:6])
    # print("ramdir",ramdir)

    # Gets Attitude
    sclkdp = spice.sce2c(-82, et)  # converts an et to a continuous encoded sc clock (ticks)
    ckgp_output = spice.ckgp(-82000, sclkdp, 0, "IAU_TITAN")
    cmat = ckgp_output[0]
    print("cmat", cmat)

    ram_unit = spice.mxv(cmat, ramdir)  # Ram Unit in SC coords
    # print("ram_unit", ram_unit)
    anglediff = spice.vsepg(ram_unit[:2], np.array([0, 1, 0]),
                            2)  # Find azimuthal angle between normal boresight and ram direction
    # print("anglediff", anglediff * spice.dpr())
    cassini_ram_mat = spice.rotate(-anglediff, 3)
    # print("cassini_ram_mat", cassini_ram_mat)
    # Rotates rotational axis with actuation
    # cassini_caps_mat = spice.ckgp(-82821, sclkdp, 0, 'CASSINI_CAPS_BASE')[0]  # Rotation matrix of actuation
    # print("cassini_caps_mat", cassini_caps_mat)
    anode_rotational_axis = spice.mxv(cassini_ram_mat, np.array([1, 0, 0]))  # Rotate with actuator
    print("Rotational Axis", anode_rotational_axis)

    rotationmatrix_1 = spice.spiceypy.axisar(anode_rotational_axis,
                                             -70 * spice.rpd())
    rotationmatrix_2 = spice.spiceypy.axisar(anode_rotational_axis,
                                             70 * spice.rpd())

    ram_unit_rotated1 = spice.mxv(rotationmatrix_1, ram_unit)
    ram_unit_rotated2 = spice.mxv(rotationmatrix_2, ram_unit)
    scframe_spiceplane = spice.psv2pl([0, 0, 0], ram_unit_rotated1, ram_unit_rotated2)
    print("ram_unit", ram_unit, ram_unit_rotated1, ram_unit_rotated2)
    print("SC frame spice normal", spice.psv2pl([0, 0, 0], ram_unit_rotated1, ram_unit_rotated2))
    cmat_t = spice.xpose(cmat)
    ram_unit_rotated1_titan = spice.mxv(cmat_t, ram_unit_rotated1)  # Transform back to IAU Titan Frame
    ram_unit_rotated2_titan = spice.mxv(cmat_t, ram_unit_rotated2)  # Transform back to IAU Titan Frame
    spiceplanenormal = spice.mxv(cmat_t, spice.pl2nvp(scframe_spiceplane)[0])

    # Old method finding normal in titan frame
    # spiceplane = spice.psv2pl(state[:3], ram_unit_rotated1_titan, ram_unit_rotated2_titan)
    # spiceplanenormal = spice.pl2nvp(spiceplane)[0]

    print("SPICE NORMAL", spiceplanenormal)
    # print("Spice normal, sc frame", scframe_spicenormal_titan)

    if windspeed > 0:
        spiceplanenormal = -1 * spiceplanenormal
        print("spice plane fipped", windspeed, spiceplanenormal)

    print("vsep titan frame", spice.vsep(ramdir, spiceplanenormal) * spice.dpr())

    return spiceplanenormal, ram_unit_rotated1_titan, ram_unit_rotated2_titan


def cassini_titan_test(flyby, anodes=False):
    times = []
    states = []
    lons, lats, alts = [], [], []
    crossvecs_lonlatalts = []
    crossvecs_lonlatalts_spicenormal = []
    cmats = []
    vecs = []
    anode_vecs = []
    anode_seps = [[], [], [], [], [], [], [], []]
    anodes1, anodes8 = [], []
    crossvecs = []
    angularseparations = []
    beamanodes = []
    spiceplanenormals = []

    windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
    tempdf = windsdf[windsdf['Flyby'] == flyby]
    for tempdatetime, negwindspeed, poswindspeed in zip(pd.to_datetime(tempdf['Bulk Time']),
                                                        tempdf["Negative crosstrack velocity"],
                                                        tempdf["Positive crosstrack velocity"]):
        print("---------")
        print(tempdatetime)
        times.append(tempdatetime)
        beamanodes.append(np.mean(ELS_ramanodes(tempdatetime)) + 1)
        states.append(cassini_phase(tempdatetime.strftime('%Y-%m-%dT%H:%M:%S')))
        # print(states[-1])
        lon, lat, alt = spice.recpgr("TITAN", states[-1][:3], spice.bodvrd("TITAN", 'RADII', 3)[1][0], 1.44e-4)
        lons.append(lon * spice.dpr())
        lats.append(lat * spice.dpr())
        alts.append(alt)
        # vecs.append(cassini_act_2_titan(tempdatetime))
        crossvec = caps_crosstrack(tempdatetime, np.mean([negwindspeed, poswindspeed]))
        print("crossvec", crossvec)
        testspicenormal, anode1, anode8 = caps_crosstrack_spice(tempdatetime, np.mean([negwindspeed, poswindspeed]))
        anodes1.append(anode1)
        anodes8.append(anode8)
        spiceplanenormals.append(testspicenormal)
        print("test spice normal", testspicenormal)
        jacobian = spice.dpgrdr("TITAN", states[-1][0], states[-1][1], states[-1][2],
                                spice.bodvrd('TITAN', 'RADII', 3)[1][0],
                                1.44e-4)
        # print("jacobian", jacobian)
        crossvec_lonlatalt = spice.mxv(jacobian, spice.vhat(crossvec))
        crossvec_lonlatalt_spicenormal = spice.mxv(jacobian, testspicenormal)

        # print("recpgr", lon, lat, alt)
        # print("crossvec latlon", crossvec_lonlatalt)
        # print("crossvec latlon vhat", spice.vhat(crossvec_latlon))
        crossvecs.append(crossvec)
        crossvecs_lonlatalts.append(crossvec_lonlatalt)
        crossvecs_lonlatalts_spicenormal.append(crossvec_lonlatalt_spicenormal)
        # print("Time", tempdatetime)
        # print("position", states[-1][:3])
        # print("velocity", spice.vhat(states[-1][3:]))
        # print("direction", spice.vhat(vecs[-1]))

        # if anodes:
        #     anode_vecs.append(caps_all_anodes(tempdatetime))
        #     print("anode vecs 1 & 8", anode_vecs[-1][0], anode_vecs[-1][7])
        #     # spiceplanenormal = spice.psv2pl(states[-1][:3],anode_vecs[-1][0],anode_vecs[-1][7])
        #     # print("SPICE NORMAL", spice.pl2nvp(spiceplanenormal))
        #     #
        #     # spiceplanenormals.append(-1*spice.pl2nvp(spiceplanenormal)[0])
        #     # print("Crossvec", crossvec)
        #     for anodecounter, i in enumerate(anode_vecs[-1]):
        #         # print(anodecounter,anode_vecs[-1][anodecounter])
        #         anode_seps[anodecounter].append(
        #             spice.vsep(spice.vhat(states[-1][3:]), spice.vhat(anode_vecs[-1][anodecounter])) * spice.dpr())
        # print("anodeseps",anode_seps)
        # print("Angular Separation", spice.vsep(spice.vhat(states[-1][3:]), spice.vhat(vecs[-1])) * spice.dpr())

    x, y, z, u, v, w = [], [], [], [], [], []

    for i in states:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    # CAPS direction
    for i in vecs:
        u.append(i[0])
        v.append(i[1])
        w.append(i[2])

    # Crosstrack
    u2, v2, w2 = [], [], []
    for j in crossvecs:
        u2.append(j[0])
        v2.append(j[1])
        w2.append(j[2])

    # SPICE plane normal
    u3, v3, w3 = [], [], []
    for j in spiceplanenormals:
        u3.append(j[0])
        v3.append(j[1])
        w3.append(j[2])

    # Ram Direction
    u1, v1, w1 = [], [], []
    for i in states:
        u1.append(i[3])
        v1.append(i[4])
        w1.append(i[5])

    fig = plt.figure()

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = 2574.7 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 2574.7 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 2574.7 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    # ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='b')
    # ax.plot(x, y, z, alpha=0.5, color='k')
    if anodes:
        for timecounter, (i, j) in enumerate(zip(anodes1, anodes8)):
            X = x[timecounter]
            Y = y[timecounter]
            Z = z[timecounter]
            # print(i)
            # for anodecounter, j in enumerate(i):
            #     if anodecounter in [0, 7]:
            #         ax.quiver(X, Y, Z, j[0], j[1], j[2], length=20, color='C' + str(anodecounter))
            # print(timecounter, i, j)
            ax.quiver(X, Y, Z, i[0], i[1], i[2], length=30, color='C1')
            ax.quiver(X, Y, Z, j[0], j[1], j[2], length=30, color='C2')

    ax.quiver(x, y, z, u2, v2, w2, length=30, color='m')
    ax.quiver(x, y, z, u1, v1, w1, length=5, color='k')
    ax.quiver(x, y, z, u3, v3, w3, length=30, color='r')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))

    dlat, dlon = [], []
    for i in crossvecs_lonlatalts:
        dlat.append(i[1])
        dlon.append(i[0])

    dlat_spicenormal, dlon_spicenormal = [], []
    for i in crossvecs_lonlatalts_spicenormal:
        dlat_spicenormal.append(i[1])
        dlon_spicenormal.append(i[0])

    fig2, ax2 = plt.subplots()
    ax2.plot(lons, lats)
    ax2.quiver(lons, lats, dlon, dlat)
    ax2.quiver(lons, lats, dlon_spicenormal, dlat_spicenormal, color='r')
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid()


def caps_crosstrack_latlon(time, negwindspeed, poswindspeed, anodes=False):
    beamanodes = [np.mean(ELS_ramanodes(time)) + 1]

    state = cassini_phase(time.strftime('%Y-%m-%dT%H:%M:%S'))
    # crossvec = caps_crosstrack(time, windspeed) * abs(windspeed) * 1e-3 #Old Method
    crossvec, anode1, anode8 = caps_crosstrack_spice(time, np.mean([negwindspeed, poswindspeed]))  # SPICE Plane method
    crossvec_neg = crossvec * 1e-3
    crossvec_pos = crossvec * 1e-3

    print("crossvec", crossvec_neg, crossvec_pos)
    newstate_neg = list(state[:3]) + list(crossvec_neg)
    newstate_pos = list(state[:3]) + list(crossvec_pos)
    transformed_state_neg = spice.xfmsta(newstate_neg, 'RECTANGULAR', 'Latitudinal', "TITAN")
    transformed_state_pos = spice.xfmsta(newstate_pos, 'RECTANGULAR', 'Latitudinal', "TITAN")
    print("test state", newstate_neg, newstate_pos)
    print("test xfmsta", transformed_state_neg, transformed_state_pos)
    alt = transformed_state_neg[0]
    lon = transformed_state_neg[1] * spice.dpr()
    lat = transformed_state_neg[2] * spice.dpr()

    if anodes:
        anode_vecs = [caps_all_anodes(time)]
        anode_seps = [[], [], [], [], [], [], [], []]
        for anodecounter, i in enumerate(anode_vecs[-1]):
            anode_seps[anodecounter].append(
                spice.vsep(spice.vhat(state[3:]), spice.vhat(anode_vecs[-1][anodecounter])) * spice.dpr())
        print("anode_vecs", anode_vecs)
        print("anode_seps", anode_seps)
    dvec_neg = [transformed_state_neg[4], transformed_state_neg[5], transformed_state_neg[3]]
    dvec_pos = [transformed_state_pos[4], transformed_state_pos[5], transformed_state_pos[3]]
    print("dvec xfmsta", dvec_neg, dvec_pos)
    print(lon, lat, alt)

    titanrad = spice.bodvrd('TITAN', 'RADII', 3)[1][0]  # Get Titan Radius
    mag_test_neg = spice.unorm(
        [dvec_neg[0] * (alt + titanrad) * 1e3, dvec_neg[1] * (alt + titanrad) * 1e3, dvec_neg[2] * 1e3])
    mag_test_pos = spice.unorm(
        [dvec_pos[0] * (alt + titanrad) * 1e3, dvec_pos[1] * (alt + titanrad) * 1e3, dvec_pos[2] * 1e3])
    # Convert Lat/Lon from rad/s to m/s, convert Alt from km/s to m/s. Forms Unit Vector here
    print("mag test", mag_test_pos)

    dlon_neg = mag_test_neg[0][0] * abs(negwindspeed)
    dlat_neg = mag_test_neg[0][1] * abs(negwindspeed)
    dalt_neg = mag_test_neg[0][2] * abs(negwindspeed)

    dlon_pos = mag_test_pos[0][0] * abs(poswindspeed)
    dlat_pos = mag_test_pos[0][1] * abs(poswindspeed)
    dalt_pos = mag_test_pos[0][2] * abs(poswindspeed)

    return lon, lat, alt, dlon_neg, dlat_neg, dalt_neg, dlon_pos, dlat_pos, dalt_pos


def caps_crosstrack_xyz(time, windspeed, anodes=False):
    state = cassini_phase(time.strftime('%Y-%m-%dT%H:%M:%S'))
    crossvec, temp1, temp2 = caps_crosstrack_spice(time, windspeed)
    return list(state[:3]) + list(crossvec * abs(windspeed))




def crosstrack_latlon_plot():
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    lons, lats, alts, magnitudes_neg, magnitudes_pos, windspeeds, flybyslist, flybycolors = [], [], [], [], [], [], [], []
    dlons_neg, dlats_neg, dalts_neg, dlons_pos, dlats_pos, dalts_pos, = [], [], [], [], [], []
    altplane = spice.nvc2pl((0, 0, 1), 1)
    windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)
    for counter, flyby in enumerate(flybys):
        tempdf = windsdf[windsdf['Flyby'] == flyby]
        for i, negwindspeed, poswindspeed in zip(pd.to_datetime(tempdf['Bulk Time']),
                                                 tempdf["Negative crosstrack velocity"],
                                                 tempdf["Positive crosstrack velocity"]):
            print("------------------------------------------------------------------------")
            # print("Windspeed",windspeed)
            temp = caps_crosstrack_latlon(i, negwindspeed, poswindspeed)
            # print("temp",temp)
            lons.append(temp[0])
            lats.append(temp[1])
            alts.append(temp[2])
            dlonlatalt_neg = [temp[3], temp[4], temp[5]]
            dlonlatalt_pos = [temp[6], temp[7], temp[8]]
            print("dlonlatalt", dlonlatalt_neg, dlonlatalt_pos)
            projectedvector_neg = spice.vprjp(dlonlatalt_neg, altplane)
            projectedvector_pos = spice.vprjp(dlonlatalt_pos, altplane)
            print("projected scaled", projectedvector_neg, spice.unorm(projectedvector_neg),
                  np.mean([negwindspeed, poswindspeed]))
            dlons_neg.append(projectedvector_neg[0])
            dlats_neg.append(projectedvector_neg[1])
            dalts_neg.append(projectedvector_neg[2])
            dlons_pos.append(projectedvector_pos[0])
            dlats_pos.append(projectedvector_pos[1])
            dalts_pos.append(projectedvector_pos[2])
            windspeeds.append(np.mean([negwindspeed, poswindspeed]))
            magnitudes_neg.append(np.sqrt(projectedvector_neg[0] ** 2 + projectedvector_neg[1] ** 2))
            magnitudes_pos.append(np.sqrt(projectedvector_pos[0] ** 2 + projectedvector_pos[1] ** 2))
            flybyslist.append(flyby)
            flybycolors.append(f"C{str(counter)}")

    for lon, lat, dlon_neg, dlat_neg, dlon_pos, dlat_pos, windspeed_neg, windspeed_pos, flyby, flybycolor in zip(lons,
                                                                                                                 lats,
                                                                                                                 dlons_neg,
                                                                                                                 dlats_neg,
                                                                                                                 dlons_pos,
                                                                                                                 dlats_pos,
                                                                                                                 magnitudes_neg,
                                                                                                                 magnitudes_pos,
                                                                                                                 flybyslist,
                                                                                                                 flybycolors):
        # print(lon,lat,dlon,dlat,flybycolor)
        if windspeed_neg < windspeed_pos:
            ax2.arrow(lon, lat, dlon_neg * 0.05, dlat_neg * 0.05, color=flybycolor, hatch='-', width=1, label=flyby,
                      alpha=0.5)
            ax2.text(lon, lat, '{:.0f} m/s'.format(windspeed_neg))
            ax3.arrow(lon, lat, dlon_pos * 0.05, dlat_pos * 0.05, color=flybycolor, hatch='|', width=1, label=flyby,
                      alpha=0.5)
            ax3.text(lon, lat, '{:.0f} m/s'.format(windspeed_pos))
        if windspeed_neg > windspeed_pos:
            ax3.arrow(lon, lat, dlon_neg * 0.05, dlat_neg * 0.05, color=flybycolor, hatch='-', width=1, label=flyby,
                      alpha=0.5)
            ax3.text(lon, lat, '{:.0f} m/s'.format(windspeed_neg))
            ax2.arrow(lon, lat, dlon_pos * 0.05, dlat_pos * 0.05, color=flybycolor, hatch='|', width=1, label=flyby,
                      alpha=0.5)
            ax2.text(lon, lat, '{:.0f} m/s'.format(windspeed_pos))
    # tempquiver = ax2.quiver(lon, lat, dlon, dlat, pivot="middle",color='C'+str(counter),label=flyby)

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid()
    ax2.set_title("Minimum Wind Vectors")
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.grid()
    ax3.set_title("Maximum Wind Vectors")
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys())


def crosstrack_xyz_plot():
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot the surface
    # u = np.linspace(0, 2 * np.pi, 50)
    # v = np.linspace(0, np.pi, 50)
    # x_sphere = 2574.7 * np.outer(np.cos(u), np.sin(v))
    # y_sphere = 2574.7 * np.outer(np.sin(u), np.sin(v))
    # z_sphere = 2574.7 * np.outer(np.ones(np.size(u)), np.cos(v))
    # ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b')

    circle1 = plt.Circle((0, 0), 2575.15, color='k', fill=False)
    circle2 = plt.Circle((0, 0), 2575.15, color='k', fill=False)
    circle3 = plt.Circle((0, 0), 2575.15, color='k', fill=False)

    figxy, axxy = plt.subplots()
    axxy.set_xlabel("X")
    axxy.set_ylabel("Y")
    axxy.set_xlim(-3500, 3500)
    axxy.set_ylim(-3500, 3500)
    axxy.add_artist(circle1)
    figyz, axyz = plt.subplots()
    axyz.set_xlabel("Y")
    axyz.set_ylabel("Z")
    axyz.set_xlim(-3500, 3500)
    axyz.set_ylim(-3500, 3500)
    axyz.add_artist(circle2)
    figxz, axxz = plt.subplots()
    axxz.set_xlabel("X")
    axxz.set_ylabel("Z")
    axxz.add_artist(circle3)
    axxz.set_xlim(-3500, 3500)
    axxz.set_ylim(-3500, 3500)

    magnitudes, windspeeds, flybyslist, flybycolors = [], [], [], []
    crosstrack_states = []
    for counter, flyby in enumerate(flybys):
        tempdf = windsdf[windsdf['Flyby'] == flyby]
        crosstrack_states = []
        for i, windspeed in zip(pd.to_datetime(tempdf['Bulk Time']), tempdf["Crosstrack velocity"]):
            print("------------------------------------------------------------------------")
            # print("Windspeed",windspeed)
            crosstrack_state_vector = caps_crosstrack_xyz(i, windspeed)
            crosstrack_states.append(crosstrack_state_vector)
            windspeeds.append(windspeed)
            magnitudes.append(spice.unorm(crosstrack_state_vector[3:])[1])
            flybyslist.append(flyby)
        crosstrack_states = np.array(crosstrack_states)
        # ax.quiver(crosstrack_states[:, 0], crosstrack_states[:, 1], crosstrack_states[:, 2],
        #          crosstrack_states[:, 3], crosstrack_states[:, 4], crosstrack_states[:, 5], label=flyby,
        #         color="C" + str(counter))
        axxy.quiver(
            crosstrack_states[:, 0],
            crosstrack_states[:, 1],
            crosstrack_states[:, 3],
            crosstrack_states[:, 4],
            label=flyby,
            color=f"C{str(counter)}",
        )
        axyz.quiver(
            crosstrack_states[:, 1],
            crosstrack_states[:, 2],
            crosstrack_states[:, 4],
            crosstrack_states[:, 5],
            label=flyby,
            color=f"C{str(counter)}",
        )
        axxz.quiver(
            crosstrack_states[:, 0],
            crosstrack_states[:, 2],
            crosstrack_states[:, 3],
            crosstrack_states[:, 5],
            label=flyby,
            color=f"C{str(counter)}",
        )
    print(windspeeds, magnitudes)
    crosstrack_states = np.array(crosstrack_states)
    print(crosstrack_states.shape)

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_xlim(min(crosstrack_states[:, 0]), max(crosstrack_states[:, 0]))
    # ax.set_ylim(min(crosstrack_states[:, 1]), max(crosstrack_states[:, 1]))
    # ax.set_zlim(min(crosstrack_states[:, 2]), max(crosstrack_states[:, 2]))

    figxy.legend()
    figyz.legend()
    figxz.legend()

    return axxy

def soldir_from_titan(): #Only use for one flyby
    for flyby in flybys:
        tempdf = windsdf[windsdf['Flyby'] == flyby]
        i = pd.to_datetime(tempdf['Bulk Time']).iloc[0]
        et = spice.datetime2et(i)
        sundir, ltime = spice.spkpos('SUN', et, 'IAU_TITAN', "LT+S", 'TITAN')
    return sundir

#flybys = ['t16', 't17', 't20', 't21', 't25', 't26', 't27', 't28', 't29', 't30', 't32', 't42', 't46']
# flybys = ['t16', 't20','t21','t32','t42','t46'] #Weird Ones
# flybys = ['t16', 't17', 't29']
flybys = ['t27']
windsdf = pd.read_csv("crosswinds_full.csv", index_col=0, parse_dates=True)

# crosstrack_latlon_plot()
soldir = soldir_from_titan()
soldir_unorm = spice.unorm(soldir)[0]*1000
axxy = crosstrack_xyz_plot()
print("soldir",spice.unorm(soldir))
axxy.arrow(0,0,soldir_unorm[0],soldir_unorm[1])
plt.show()
