import numpy as np
import pandas as pd
import tools
import tiles
import interp
import computational as cpt
import matplotlib.pyplot as plt
import gsw
from scipy import interpolate
from scipy import integrate
import os

# plt.ion()

time_flag = 'annual'  # 'DJF' # 'annual'
typestat = 'zmean'

seasons = ['DJF', 'MAM', 'JJA', 'SON']

var_stats = ["W", "CT", "SA", "CT_STD", "BVF2_STD", "RSTAR", "CF"]

attributes = {"W": ("weight (dimensionless)", "1", [0., 10000.]),
              "CT": ("conservative temperature", "degC", [-3., 50.]),
              "SA": ("absolute salinity", "g kg-1", [0., 50.]),
              "CT_STD": ("CT standard deviation", "degC", [0., 20.]),
              "BVF2_STD": ("square of Brunt Vaisala Frequency standard deviation", "s-1", [0., 1e-2]),
              "RSTAR": ("compensated density", "kg m-3", [1000., 1032.]),
              "CF": ("compressibility factor  (dimensionless)", "1", [0.99, 1.])}

global_attributes = {
    "title": "World Ocean Climatology of mean temperature, salinity and compensated density"
}

tiles_dir = "%s/%g/stats" % (tiles.tiles_dir, tiles.reso)
var_dir = {v: tiles_dir+"/%s" % v for v in var_stats}
tiles_file = "%s/stats_%s.pkl"  # % (var_dir[var], tile)

zref = tools.zref

threshold = 5e-2


def create_folders():
    for d in [tiles_dir]+list(var_dir.values()):
        if os.path.exists(d):
            pass
        else:
            os.makedirs(d)
    
def compute_avg(j, i, lonr, latr, latdeg, LONr, LATr, resor, data):
    # if True:
    w = cpt.compute_weight(lonr[i], latr[j], LONr, LATr, resor)
    clim = pd.DataFrame(0., columns=var_stats, index=zref)

    profiles_to_use = (w > threshold)
    tags = profiles_to_use.index
    w = w[tags]
    totalweight = w.sum()
    print(" weight=%2.1f" % (totalweight), end="")
    if totalweight < 2:
        clim[:] = np.nan
    else:

        CT = data["CT"].loc[tags, :]
        SA = data["SA"].loc[tags, :]
        BVF2 = data["BVF2"].loc[tags, :]

        bad = CT.isna() | (CT < -3) | (CT > 50) | (SA < 0) | (SA > 50)

        nz = len(zref)
        #weight = w[:, np.newaxis]*np.ones((nz,))
        weight = CT.copy()
        weight.iloc[:, :] = w[:, np.newaxis]
        weight[bad] = 0.
        CT[bad] = 0.
        SA[bad] = 0.

        W = np.sum(weight, axis=0)
        clim.W[:] = W
        z0 = np.sum(CT * weight, axis=0)
        z2 = np.sum(CT*CT * weight, axis=0)

        clim.CT[:] = z0/W
        sigma = np.sqrt((z2-z0*z0/W)/(W-1))
        clim.CT_STD[:] = sigma

        z0 = np.sum(SA * weight, axis=0)
        clim.SA[:] = z0/W

        z0 = np.sum(BVF2 * weight, axis=0)
        z2 = np.sum(BVF2*BVF2 * weight, axis=0)
        sigma = np.sqrt((z2-z0*z0/W)/(W-1))
        clim.BVF2_STD[:] = sigma
        
        if True:
            rhostar, compf = comp_rhostar(clim.SA, clim.CT, latdeg)

            clim.RSTAR[:] = rhostar
            clim.CF[:] = compf

        clim[clim == 0.] = np.nan
    return clim, tags


def comp_rhostar(Si, Ti, lat):
    pi = gsw.p_from_z(-zref, lat)
    cs = gsw.sound_speed(Si, Ti, pi)

    Ri = gsw.rho(Si, Ti, pi)
    g = gsw.grav(lat, pi[0])
    E = np.zeros((len(zref),))
    #plt.plot(Ri, -zref)
    f = interpolate.interp1d(zref, cs)
    def e(x): return -g/f(x)**2
    if True:
        for k, z in enumerate(zref):
            if k == 0:
                r, E[k] = 0., 1.
            else:
                #r1,p = integrate.quad(e,zref[k-1],z,epsrel=1e-1)
                x = np.linspace(zref[k-1], z, 10)
                dx = x[1]-x[0]
                r1 = integrate.trapz(e(x), dx=dx)
                r += r1
                E[k] = np.exp(r)
    return Ri*E, E


def get_grid_on_box(b):
    reso = tiles.reso
    lonmin = np.ceil(b["LONMIN"]/reso)*reso
    lonmax = np.floor(b["LONMAX"]/reso)*reso
    latmin = np.ceil(b["LATMIN"]/reso)*reso
    latmax = np.floor(b["LATMAX"]/reso)*reso
    latmin = max(latmin, -80)  # TODO: replace with min(latglo)
    latmax = min(latmax, 80)  # TODO
    lon = np.arange(lonmin, lonmax+reso, reso)
    lat = np.arange(latmin, latmax+reso, reso)
    return lat, lon


def compute_stats(bb, tile):
    # read more than just one tile => to cope with the halo

    tile_list, rect = tiles.tiles_with_halo(bb, tile)

    argo = tiles.read_argo_tile(tile_list)
    data = interp.read_profiles(tile_list)

    #argo = tiles.extract_in_tile(argo, rect)
    argo = argo[argo.STATUS == "D"]
    for var in data.keys():
        data[var] = data[var].loc[argo.index, :]

    reso = tiles.reso
    zref = tools.zref

    CT = data['CT']
    SA = data['SA']

    # patch TO REMOVE LATER
    CT.iloc[:, 1] = 0.5*(CT.iloc[:, 0]+CT.iloc[:, 2])
    SA.iloc[:, 1] = 0.5*(SA.iloc[:, 0]+SA.iloc[:, 2])

    LON = argo['LONGITUDE']
    LAT = argo['LATITUDE']

    lat, lon = get_grid_on_box(bb[tile])

    LONr = np.deg2rad(LON)
    LATr = np.deg2rad(LAT)
    lonr = np.deg2rad(lon)
    latr = np.deg2rad(lat)
    resor = np.deg2rad(reso)

    nlon, nlat, nz = len(lon), len(lat), len(zref)

    CTbar = np.zeros((nlat, nlon, nz))
    SAbar = np.zeros((nlat, nlon, nz))
    CTstd = np.zeros((nlat, nlon, nz))
    BVF2std = np.zeros((nlat, nlon, nz))
    RHOSTAR = np.zeros((nlat, nlon, nz))
    CF = np.zeros((nlat, nlon, nz))
    W = np.zeros((nlat, nlon, nz))

    monitor_file = "monitor_%s.txt" % tile
    with open(monitor_file, "w") as fid:
        fid.write("MEANSTATE / #profiles: %i / nlat x nlon: %i" % (len(argo), nlat*nlon))
        
    #fig = plt.figure()
    for j in range(nlat):
        for i in range(nlon):
            print("\r j=%2i/%i-%2i/%i" % (j, nlat, i, nlon), end="")
            clim, tags = compute_avg(
                j, i, lonr, latr, lat[j], LONr, LATr, resor, data)
            CTbar[j, i, :] = clim["CT"]
            SAbar[j, i, :] = clim["SA"]
            CTstd[j, i, :] = clim["CT_STD"]
            BVF2std[j, i, :] = clim["BVF2_STD"]
            RHOSTAR[j, i, :] = clim["RSTAR"]
            CF[j, i, :] = clim["CF"]
            W[j, i, :] = clim["W"]
            # fig.canvas.draw()

    mapvar = {"CT": CTbar, "SA": SAbar, "CT_STD": CTstd,
              "BVF2_STD": BVF2std,
              "RSTAR": RHOSTAR, "CF": CF, "W": W}

    print()
    for var in var_stats:
        v = mapvar[var]
        d = var_dir[var]
        f = tiles_file % (d, tile)
        print("write %s" % f)
        pd.to_pickle(v, f)

    os.system("rm %s" % monitor_file)


def read(tile, var, transpose=True):
    d = var_dir[var]
    f = tiles_file % (d, tile)
    print(f)
    if os.path.exists(f):
        data = pd.read_pickle(f)
        if transpose:
            data = np.transpose(data, (2, 0, 1))
    else:
        data = None
    return data
