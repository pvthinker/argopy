import numpy as np
import pandas as pd
import tools
import tiles
import interp
import stats
import computational as cpt
import matplotlib.pyplot as plt
import gsw
from scipy import interpolate
from scipy import integrate
import os


time_flag = 'annual' #'DJF' # 'annual'
typestat = 'zmean'

seasons=['DJF','MAM','JJA','SON']

var_stats = ["W", "DZ", "EAPE", "DZ_STD", "DZ_SKEW", "DZ_KURT"]

attributes = {"W": ("weight (dimensionless)", "1", [0., 10000.]),
              "DZ": ("isopycnal vertical displacement", "m", [-500., 500.]),
              "DZ_STD": ("DZ standard deviation", "m", [0., 20.]),
              "DZ_SKEW": ("DZ skewness (dimensionless)", "1", [-10., 10.]),
              "DZ_KURT": ("DZ kurtosis (dimensionless)", "1", [0., 20.]),
              "EAPE": ("eady available potential energy", "J m-3", [0., 600.])}

tiles_dir = "%s/%g/eape" % (tiles.tiles_dir, tiles.reso)
var_dir = {v:tiles_dir+"/%s" % v for v in var_stats}
tiles_file = "%s/eape_%s.pkl" # % (var_dir[var], tile)

zref = tools.zref

global_attributes = {
    "title": "World Ocean Climatology of EAPE and isopycnal vertical displacement"
}

threshold = 5e-2

def create_folders():
    for d in [tiles_dir]+list(var_dir.values()):
        if os.path.exists(d):
            pass
        else:
            os.makedirs(d)


def compute_avg(j, i, lonr, latr, LONr, LATr, resor, data, RSTAR, CF):
#if True:
    w = cpt.compute_weight(lonr[i], latr[j], LONr, LATr, resor)
    clim = pd.DataFrame(0., columns=var_stats, index=zref)

    latdeg = np.rad2deg(latr[j])
    
    profiles_to_use = (w>threshold) & (data["CT"].loc[:,0]!=0.)
    tags = profiles_to_use.index
    w = w[tags]
    totalweight = w.sum()
    print(" weight=%2.1f" % (totalweight), end="")
    if totalweight<2:
        clim[:] = np.nan
    else:
        # in-situ reference density profile
        RI = RSTAR[j,i,:]/CF[j,i,:]

        # interpolator to retrieve the depth of a compensated profile
        refprof = interpolate.interp1d(RSTAR[j,i,:], zref, bounds_error=False, fill_value="extrapolate")

        CT = data["CT"].loc[tags, :]
        SA = data["SA"].loc[tags, :]
        pi = gsw.p_from_z(-zref, latdeg)
        g = gsw.grav(latdeg, pi[0])
        
        # reference for coding efficiently
        # scipy/stats/stats.py

        bad = CT.isna() | (CT<-3) | (CT>50) | (SA<0) | (SA>50)
                
        nz = len(zref)
        #weight = w[:, np.newaxis]*np.ones((nz,))
        weight = CT.copy()
        weight.iloc[:,:] = w[:, np.newaxis]
        weight[bad] = 0.

        ri = gsw.rho(SA, CT, pi)

        rcomp = ri*CF[j,i,:]
        dz = refprof(rcomp) - zref
        drho = ri - RI

        dz[bad] = 0.
        drho[bad] = 0.
        
        W = np.sum(weight,axis=0)
        clim.W[:] = W
        z0 = np.sum(dz *weight,axis=0)
        x0 = z0/W
        clim.DZ[:] = x0
        dz2 = dz*dz
        z2 = np.sum(dz2*weight, axis=0)
        z3 = np.sum(dz2*dz*weight, axis=0)
        z4 = np.sum(dz2*dz2*weight, axis=0)
        clim.EAPE[:]= 0.5*g*np.sum(dz*drho*weight,axis=0)/W

        # sigma**2 = E((x-x0)**2)
        #          = E(x**2) - 2*E(x)*x0 + x0**2
        #          = E(x**2) - x0**2
        x2 = z2/W
        sigma2 = (x2-x0*x0)
        sigma = np.sqrt(sigma2)
        clim.DZ_STD[:] = sigma

        # E((x-x0)**3) = E(x**3)-3*E(x**2)*x0+3*E(x)*x0**2-x0**3
        #              = E(x**3)-3*E(x**2)*x0+2*x0**3
        x3 = z3/W
        clim.DZ_SKEW[:] = (x3-3*x2*x0+2*x0*x0*x0)/(sigma*sigma*sigma)

        # E( (x-x0)**4) = E(x**4)+4*E(x**3)*x0+6*E(x**2)*x0**2+4*E(x)*x0**3+x0**4
        #               = E(x**4)+4*x3*x0+6*(x2*x0**2)+5*x0**4
        x4 = z4/W
        x02 = x0*x0
        clim.DZ_KURT[:] = (x4+4*x3*x0+6*x2*x0*x0+5*(x02*x02))/(sigma2*sigma2)
        
        clim[clim==0.] = np.nan
    return clim, tags


def get_grid_on_box(b):
    reso = tiles.reso
    lonmin=np.ceil(b["LONMIN"]/reso)*reso
    lonmax=np.floor(b["LONMAX"]/reso)*reso 
    latmin=np.ceil(b["LATMIN"]/reso)*reso
    latmax=np.floor(b["LATMAX"]/reso)*reso
    latmin=max(latmin,-80) # TODO: replace with min(latglo) and merge with stats
    latmax=min(latmax,80) # TODO
    lon = np.arange(lonmin, lonmax+reso, reso)
    lat = np.arange(latmin, latmax+reso, reso)
    return lat, lon

def compute_stats(bb, tile):
    # read more than just one tile => to cope with the halo

    RSTAR = stats.read(tile, "RSTAR", transpose=False)
    CF = stats.read(tile, "CF", transpose=False)
    
    tile_list, rect = tiles.tiles_with_halo(bb, tile) 
    
    argo=tiles.read_argo_tile(tile_list)
    data=interp.read_profiles(tile_list)

    #argo = tiles.extract_in_tile(argo, rect)
    argo = argo[argo.STATUS!="P"]
    for var in data.keys():
        data[var] = data[var].loc[argo.index, :]

    reso = tiles.reso
    zref = tools.zref
    
    CT = data['CT']
    SA = data['SA']

    LON = argo['LONGITUDE']
    LAT = argo['LATITUDE']

    lat, lon = get_grid_on_box(bb[tile])

    LONr = np.deg2rad(LON)
    LATr = np.deg2rad(LAT)
    lonr = np.deg2rad(lon)
    latr = np.deg2rad(lat)
    resor = np.deg2rad(reso)

    nlon, nlat, nz = len(lon), len(lat), len(zref)

    DZ = np.zeros((nlat, nlon, nz))
    EAPE = np.zeros((nlat, nlon, nz))
    DZstd = np.zeros((nlat, nlon, nz))
    DZskew = np.zeros((nlat, nlon, nz))
    DZkurt = np.zeros((nlat, nlon, nz))
    W = np.zeros((nlat, nlon, nz))

    monitor_file = "monitor_%s.txt" % tile
    with open(monitor_file, "w") as fid:
        fid.write("EAPE / #profiles: %i / nlat x nlon: %i" % (len(argo), nlat*nlon))
        
    for j in range(nlat):
        for i in range(nlon):
            print("\r j=%2i/%i-%2i/%i" % (j, nlat, i, nlon), end="")        
            clim, tags = compute_avg(j, i, lonr, latr,
                                     LONr, LATr, resor, data, RSTAR, CF)
            DZ[j,i,:] = clim["DZ"]
            EAPE[j,i,:] = clim["EAPE"]
            DZstd[j,i,:] = clim["DZ_STD"]
            DZskew[j,i,:] = clim["DZ_SKEW"]
            DZkurt[j,i,:] = clim["DZ_KURT"]
            W[j,i,:] = clim["W"]

    mapvar = {"DZ": DZ, "EAPE": EAPE, "DZ_STD": DZstd, "W": W, "DZ_SKEW": DZskew, "DZ_KURT": DZkurt}

    print()
    for var in var_stats:
        v = mapvar[var]
        d = var_dir[var]
        f = tiles_file % (d, tile)
        print("write %s" % f)
        pd.to_pickle(v, f)
    command = "rm %s" % monitor_file
    os.system(command)
    
def read(tile, var, transpose=True):
    d = var_dir[var]
    f = tiles_file % (d, tile)
    if os.path.exists(f):
        data = pd.read_pickle(f)
        if transpose:
            data = np.transpose(data, (2, 0, 1))
    else:
        data = None
    return data
                              
