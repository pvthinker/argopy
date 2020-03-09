import numpy as np
import itertools
import pandas as pd
import tools
import tiles
import general
from netCDF4 import Dataset
import os
import glob

wrong_atlas = "tools.atlas_name should be either 'meanstate' or 'eape'"
assert tools.atlas_name in ["meanstate", "eape"], wrong_atlas

atlas_name = tools.atlas_name

if atlas_name == "meanstate":
    import stats as stats

elif atlas_name == "eape":
    import eape as stats

data_attributes = {
    "period of the climatology": "annual",
    "source": "Argo profiles exclusively"
}

other_attributes = {
    "spatial coverage": "almost global, from 80S to 80N",
    "grid": "Cylindrical equirectangular projection",
    "grid resolution": tiles.reso,
    "grid units": "decimal degrees",
    "author": "Roullet, Guillaume",
    "institution": "Laboratoire d'Oceanographie Physique et Spatiale (LOPS), Univ Brest, CNRS, IRD, Ifremer, IUEM, Brest, France",
    "project funding": "Laboratory recurrent credit allocation",
    "references": "Roullet, Capet and Maze, GRL, 2014",    
    "history": "%s : creation (G. Roullet)" % pd.datetime.today().strftime('%Y-%m-%d'),
    "Conventions": "CF 1.8",
    "comment": "This atlas has been generated with argopy on datarmor, the Ifremer cluster",
    "software name": "argopy",
    "software respository": "http://github.com/pvthinker/argopy"
}

global_attributes = {**stats.global_attributes,
                     **data_attributes,
                     **other_attributes}

reso = tiles.reso

# TODO rename clim into meanstate and/or eape depending on the context

global_dir = "%s/global/atlas" % tools.pargopy
global_atlas_file = global_dir + "/%s_%g.nc" % (atlas_name, reso)

subd_dir = "%s/%g/subdomains" % (tools.pargopy, tiles.reso)
tile_atlas_file = subd_dir + "/%s_%g" % (atlas_name, reso) + "_%s.nc"  # % tile

local_dirs = [global_dir, subd_dir]

longlo = np.arange(-180, 180+reso, reso)
latglo = np.arange(-80, 80+reso, reso)

fill_value = np.nan

l = []
for a in "01":
    for b in "0123":
        for c in "0123":
            l += [a+b+c]

subd_list = l  # [a+b for a,b in itertools.product("01", "0123")]


def create_folders():
    for d in [global_dir, subd_dir]:
        if os.path.exists(d):
            pass
        else:
            os.makedirs(d)


def set_latest_argo_profile(argo):
    count = len(argo[argo.STATUS == "D"])
    latest = argo.JULD[argo.STATUS == "D"].max()
    d = general.juld2date(latest)
    s = pd.datetime(d["YEAR"], d["MONTH"], d["DAY"]).strftime('%Y-%m-%d')
    data_attributes["date of latest Argo profile used"] = s
    data_attributes["number of Argo profile used"] = np.int32(count)
    print(data_attributes)
    attributes = {
        **stats.global_attributes,
        **data_attributes,
        **other_attributes}
    return attributes


def create_empty_netcdf_file(ncfile, lat, lon, set_latest=False):
    with Dataset(ncfile, "w") as nc:
        nc.createDimension("lon", len(lon))
        nc.createDimension("lat", len(lat))
        nc.createDimension("depth", len(tools.zref))

        v = nc.createVariable("lon", float, ("lon",))
        v.long_name = "longitude"
        v.units = "degrees_east"
        v = nc.createVariable("lat", float, ("lat",))
        v.long_name = "latitude"
        v.units = "degrees_north"
        v = nc.createVariable("depth", float, ("depth",))
        v.long_name = "depth"
        v.units = "m"
        v.positive = "down"

        if set_latest:
            argo = tools.read_argodb()
            attributes = set_latest_argo_profile(argo)
        else:
            attributes = global_attributes

        # replace whitespaces with underscores in attribute names
        d = {}
        for k, v in attributes.items():
            newk = k.replace(" ", "_")
            d[newk] = v
        
        nc.setncatts(d)

        for var in stats.var_stats:
            at = stats.attributes[var]
            v = nc.createVariable(var, np.float64,
                                  ("depth", "lat", "lon"),
                                  fill_value = fill_value)
            v.long_name = at[0]
            v.units = at[1]
            v.valid_min = at[2][0]
            v.valid_max = at[2][1]

    with Dataset(ncfile, "a") as nc:
        nc.variables["lon"][:] = lon
        nc.variables["lat"][:] = lat
        nc.variables["depth"][:] = tools.zref


def write_global_from_tiles():
    ncfile = global_atlas_file
    create_empty_netcdf_file(ncfile, latglo, longlo)

    for var in stats.var_stats:
        z3d = assemble_global_var(var)
        with Dataset(ncfile, "r+") as nc:
            nc.variables[var][...] = z3d


def write_global_from_subd():
    ncfile = global_atlas_file
    create_empty_netcdf_file(ncfile, latglo, longlo, set_latest=False)
    
    for var in stats.var_stats:
        at = stats.attributes[var]
        mini = at[2][0]
        maxi = at[2][1]
        for subd in subd_list:
            jj, ii = get_slices(subd)
            #print("%s - %s" % (var, subd))
            print("\r %s - %s" % (var, subd), end="")
            
            with Dataset(tile_atlas_file % subd, "r") as nc:
                z3d = nc.variables[var][...]
                
            z3d[(z3d<mini) | (z3d>maxi)] = fill_value
            
            with Dataset(ncfile, "r+") as nc:
                nc.variables[var][:, jj, ii] = z3d


def write_subd_from_tiles(subd):
    bb = tiles.read_tiles()

    assert subd in subd_list

    print("gather global atlas in %s" % subd)
    tile_list = [k for k in bb.keys() if k[:len(subd)] == subd]

    ncfile = tile_atlas_file % subd
    print("assemble %s" % ncfile)

    jj, ii = get_slices(subd)
    # offset
    i0 = ii.start
    j0 = jj.start
    lonf = longlo[ii]
    latf = latglo[jj]

    nz = len(stats.zref)

    create_empty_netcdf_file(ncfile, latf, lonf)

    for tile in tile_list:
        jj, ii = get_slices(tile)
        #jj = [j-j0 for j in jj]
        #ii = [i-i0 for i in ii]
        ii = slice(ii.start-i0, ii.stop-i0, None)
        jj = slice(jj.start-j0, jj.stop-j0, None)
        nj = jj.stop-jj.start
        ni = ii.stop-ii.start
        z3d = np.zeros((nz, nj, ni))
        #print(longlo[ii], latglo[jj])
        for var in stats.var_stats:
            print("\r %s - %s      " % (tile, var), end="")

            data = stats.read(tile, var)

            # # TODO: remove this dirty fix
            # data[data<-10]=np.nan
            # if var!="W":
            #     data[data>50]=np.nan

            #print( tile, np.shape(z3d), np.shape(data))
            z3d[:, :, :] = data

            with Dataset(ncfile, "r+") as nc:
                #print(np.shape(q), nj, ni)
                #print(jj[0],jj[-1], ii[0],ii[-1])
                #print(tile, z3d.shape, nj, ni, jj, ii)
                for kz in range(nz):
                    nc.variables[var][kz, jj, ii] = z3d[kz][:, :]

    print()


def get_slices(tile):

    b = tiles.get_bb(tile)

    latloc, lonloc = stats.get_grid_on_box(b)

    ii = [i for i, l in enumerate(longlo) if l in lonloc]
    jj = [i for i, l in enumerate(latglo) if l in latloc]
    si = slice(ii[0], ii[-1]+1)
    sj = slice(jj[0], jj[-1]+1)
    return (sj, si)


def fill_global_netcdf_file(ncfile, var):
    print("put %s in %s" % (var, ncfile))
    with Dataset(ncfile, "r+") as nc:
        z3d = assemble_global_var(var)
        nc.variables[var][:, :, :] = z3d


def assemble_global_var(var):
    files = glob.glob(stats.var_dir["W"]+"/*pkl")
    tile_list = [l.split('_')[1][:-4] for l in files]
    print("found %i tiles" % len(tile_list))
    nlat = len(latglo)
    nlon = len(longlo)
    nz = len(tools.zref)

    z3d = np.zeros((nlat, nlon, nz))

    bb = tiles.read_tiles()

    pbs = []
    for tile in tile_list:
        print("\r %8s" % tile, end="")
        jj, ii = get_slices(tile)
        d = stats.read(tile, var, transpose=False)
        d[np.isnan(d)] = fill_value
        #print(len(jj),len(ii), np.shape(d) ,np.shape(z3d))
        try:
            z3d[jj, ii, :] = d
        except:
            z3d[jj, ii, :] = fill_value
            pbs += [tile]
    print("\npb with tiles ", pbs)
    #z3d = np.ma.array(z3d,mask=z3d==fill_value)
    return np.transpose(z3d, (2, 0, 1))


def write_one_tile_netcdf(tile, var):
    jj, ii = get_slices(tile)
    lon = longlo[ii]
    lat = latglo[jj]

    print(lon)
    print(lat)
    nlat = len(lat)
    nlon = len(lon)
    nz = len(tools.zref)

    z3d = np.zeros((nlat, nlon, nz))

    data = stats.read(tile, var, transpose=False)

    print(tile, np.shape(z3d), np.shape(data))

    # d[np.isnan(d)]=fill_value
    try:
        z3d[:, :, :] = data
    except:
        z3d[:, :, :] = fill_value
        print("\npb with tiles ", tile)
    z3d = np.transpose(z3d, (2, 0, 1))
    #z3d = np.ma.array(z3d,mask=z3d==fill_value)

    ncfile = "atlas_%s.nc" % tile
    print("create %s" % ncfile)
    create_empty_netcdf_file(ncfile, lat, lon)
    with Dataset(ncfile, "r+") as nc:
        nc.variables[var][:, :, :] = z3d
    return z3d


def read_one_tile(tile, var):
    ncfile = tile_atlas_file % tile
    with Dataset(ncfile, "r") as nc:
        z3d = nc.variables[var][:, :, :]
        lon = nc.variables["lon"][:]
        lat = nc.variables["lat"][:]
    return lon, lat, z3d
