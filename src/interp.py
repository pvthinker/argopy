import tools
import tiles
import numpy as np
import pandas as pd
import interpolation_tools as it
import matplotlib.pyplot as plt
import itertools
import glob
import os
import sys

zref = tools.zref

zref_var = ['CT', 'SA', 'BVF2'] # "RHOIS" is needed for eape

global_dir = "%s/profiles" % tools.global_dir
global_profiles_file = global_dir + "/%s_%s.pkl"

tiles_dir = "%s/profiles" % tiles.tiles_dir
var_dir = {v:tiles_dir+"/%s" % v for v in zref_var}
tiles_profiles_file = "%s/zref_%s.pkl"

fintiles_wildcard = tiles_dir+"/*/*pkl"
fglobal_wildcard = global_dir+"/*pkl"

local_dirs = [global_dir, tiles_dir]+list(var_dir.values())

def create_folders():
    for d in [global_dir, tiles_dir]+list(var_dir.values()):
        if os.path.exists(d):
            pass
        else:
            os.makedirs(d)

def gather_global_from_tiles():
    bb = tiles.read_tiles()
    subd_list = [a+b for a,b in itertools.product("01", "0123")]
    for subd in subd_list:
        print("update global profiles %s", subd)
        tile_list = [k for k in bb.keys() if k[:2]== subd]    
        data = {}
        for var in zref_var:
            data[var] = []
        for tile in tile_list:
            prof = read_profiles(tile)
            for var in zref_var:
                data[var] += [prof[var]]
        for var in zref_var:
            f = global_profiles_file % (var, subd)
            prof = pd.concat(data[var])
            prof.to_pickle(f)
        
def apply_timestamp():
    """ give the tiles/profiles and global/profiles the same timestamp """
    print("apply timestamp of global/profiles to tiles/profiles")
    #fintiles = glob.glob(tiles_dir+"/*/*pkl")
    #fglobal = glob.glob(global_dir+"/*pkl")
    #files = fintiles+fglobal
    command = "touch %s -r %s" % (fintiles_wildcard, global_dir)
    os.system(command)
    command = "touch %s -r %s" % (fglobal_wildcard, global_dir)    
    os.system(command) 
    # for f in files:
    #     command = "touch %s -r %s" % (f, global_dir)
    #     # TODO: super slow
    #     os.system(command)        
        
def are_tiles_synchronized():    
    tref = os.path.getmtime(global_dir)
    fintiles = glob.glob(tiles_dir+"/*/*pkl")
    fglobal = glob.glob(global_dir+"/*pkl")
    files = fintiles+fglobal
    yes = all([(os.path.getmtime(f) == tref) for f in files])
    return yes

def split_global_into_tiles():
    bb = tiles.read_tiles()
    subd_list = [a+b for a,b in itertools.product("01", "0123")]
    for subd in subd_list:
        print("update tiles profiles %s" % subd)
        tile_list = [t for t in bb.keys() if t[:2] == subd]
        argo = {}
        for tile in tile_list:
            argo[tile] = tiles.read_argo_tile(tile)
        purge = False
        for var in zref_var:
            f = global_profiles_file % (var, subd)
            if os.path.exists(f):
                prof = pd.read_pickle(f)
                for tile in tile_list:
                    print("\r %4s - " % var, end="")
                    idx = argo[tile].index
                    data = {var: prof.loc[idx]}
                    write_profiles(tile, data, variables=[var])
            else:
                print(" %4s - %s is empty" % (var, tile))
                # purge tiles/profiles folder
                for tile in tile_list:
                    f = tiles_profiles_file % (var_dir[var], tile)
                    command = "rm %s" % f
                    print(command)
                    #os.system(command)

                
        print()
    apply_timestamp()
        
def write_profiles(tile, zref_data, variables=zref_var):
    print("write profiles in tile %s" % tile)
    for var in variables:
        f = tiles_profiles_file % (var_dir[var], tile)
        zref_data[var].to_pickle(f)


def read_profiles(tile, verbose=True):
    """ tile is either a tile key or a list of tile keys """
    if type(tile) is list:
        zref_data = {}
        for var in zref_var:
            a = []
            for t in tile:
                f = tiles_profiles_file % (var_dir[var], t)
                assert os.path.exists(f), "%s does not exist" % f
                a += [pd.read_pickle(f)]
            zref_data[var] = pd.concat(a)
    else:
        if verbose: print("read profiles in tile %s" % tile)
        zref_data = {}
        for var in zref_var:
            f = tiles_profiles_file % (var_dir[var], tile)
            if os.path.exists(f):
                zref_data[var] = pd.read_pickle(f)
            else:
                zref_data[var] = pd.DataFrame(columns=zref)
    return zref_data


def check_profile_in_argo(a, d):
    #assert a[a.STATUS=="D"].index.equals(d["CT"].index)
    diff = a[a.STATUS == "D"].index.difference(d["CT"].index)
    return (len(diff) == 0)


def get_iprof(data, iprof):
    """return profile iprof from data=output of read_profile"""
    d = {}
    for k in ['TEMP', 'PSAL', 'PRES', 'TEMP_QC', 'PSAL_QC', 'PRES_QC']:
        d[k] = data[k][iprof, :]
    for k in ['LONGITUDE', 'LATITUDE']:
        d[k] = data[k][iprof]
    return d

def update_profiles_in_tile(tile):
    a = tiles.argotile(tile)
    #a.loc[a.STATUS == "D","STATUS"] = "N"
    a.STATUS[a.STATUS == "D"] = "N"
    #tiles.save_tile(tile, a, h)

    zref_data = read_profiles(tile)

    ok = check_profile_in_argo(a, zref_data)
    if not(ok):
        print("Warning")
        print("profiles flagged as D in argo are actually missing")
        print("pb in tile = %s" % tile)
        sys.exit("I prefer to stop the interpolation")
        
    for wmo in a.groupby('WMO').indices:
        print('interpolate wmo: %i' % wmo, end="")
        aa = a[a.WMO == wmo]
        dac = aa.DAC.iloc[0]
        todo_tags = aa[aa.STATUS == "N"].index
        if len(todo_tags) == 0:
            print(" / nothing to do")
        else:
            print(' / new profiles: %i' % len(todo_tags))
            data = tools.read_profile(dac, wmo, header=True,
                                      data=True, dataqc=True, verbose=False)

            for tag in todo_tags:
                iprof = aa.loc[tag, "IPROF"]
                #print(" - iprof %i" % iprof)
                d = get_iprof(data, iprof)

                zref_d, ok = it.raw_to_zref(d, zref)
                if ok:
                    # update the status in the database
                    a.loc[tag, "STATUS"] = "D"
                    for var in zref_var:
                        zref_data[var].loc[tag, :] = zref_d[var][:]
                else:
                    a.STATUS[tag] = "F"

    tiles.write_argo_tile(tile, a)
    write_profiles(tile, zref_data)
    
if __name__ == '__main__':

    bb = tiles.read_tiles()

    for tile in list(bb.keys())[:12]:

        print('-'*40)
        update_profiles_in_tile(tile)
