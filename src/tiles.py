import numpy as np
import os
import pandas as pd
import tools
import computational as comp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

reso = 0.25

maxload = int(4e5)

tiles_dir = "%s/tiles" % tools.pargopy
backuptiles_dir = "%s/backup" % tiles_dir
argo_dir = "%s/argo" % tiles_dir
argo_file = argo_dir + "/argo_%s.pkl"
trash_file = argo_dir + "/argo_others.pkl"
argo_wildcard = argo_dir + "/argo*.pkl"
tiling_file = tiles_dir + "/tiling.pkl"
tiling_fig = tiles_dir + "/tiling.png"

local_dirs = [tiles_dir, argo_dir, backuptiles_dir]

import interp # circular import! it seems to work

dleft = 1./137
dright = 1./139

def create_folders():
    for d in [tiles_dir, argo_dir, backuptiles_dir]:
        if os.path.exists(d):
            pass
        else:
            os.makedirs(d)

def count_gridpoint_tile(b):
    return int((b['LONMAX']-b['LONMIN'])*(b['LATMAX']-b['LATMIN'])/reso**2)

def setup_tiling(tiles, maxload):

    bb = []
    for b, _argo in tiles:
        na = len(_argo)
        ng = count_gridpoint_tile(b)
        if na*ng < maxload:
            bb += [(b, na, ng)]
        else:
            smallb = split_tile(b)
            smalltiles = []
            for smb in smallb:
                smalltiles += [(smb, extract_in_tile(_argo, smb))]
            bb += setup_tiling(smalltiles, maxload)
    return bb

def split_tile(tile):
    lonmid = (tile['LONMIN']+tile['LONMAX'])/2.
    latmid = (tile['LATMIN']+tile['LATMAX'])/2.
    ne, nw, se, sw = tile.copy(), tile.copy(), tile.copy(), tile.copy()
    ne["ID"] += "3"
    se["ID"] += "1"
    nw["ID"] += "2"
    sw["ID"] += "0"
    ne['LONMIN'] = lonmid
    ne['LATMIN'] = latmid
    nw['LONMAX'] = lonmid
    nw['LATMIN'] = latmid
    se['LONMIN'] = lonmid
    se['LATMAX'] = latmid
    sw['LONMAX'] = lonmid
    sw['LATMAX'] = latmid
    return [sw, se, nw, ne]

def extract_in_tile(_argo, tile, margin=0):
    lon0 = tile['LONMIN']
    lon1 = tile['LONMAX']
    lat0 = tile['LATMIN']
    lat1 = tile['LATMAX']
    if margin > 0:
        lat = 0.5*(lat0+lat1)
        cff = 1./np.abs(np.cos(np.deg2rad(lat)))
        lon0 -= margin*cff
        lon1 += margin*cff
        lat0 -= margin
        lat1 += margin
        # ci-dessous pb avec la dateline=>TODO
    if False:#(lon0 < -180) ^ (lon1 > 180): # xor
        test = (((_argo.LONGITUDE < lon0)
                 | (_argo.LONGITUDE > lon1))
                 & (_argo.LATITUDE >= lat0)
                 & (_argo.LATITUDE <= lat1))
    else:
        test = ((_argo.LONGITUDE >= lon0)
                 & (_argo.LONGITUDE <= lon1)
                 & (_argo.LATITUDE >= lat0)
                 & (_argo.LATITUDE <= lat1))
        
    return _argo[test]

def sketch_tiles(ax, bb, tile_list):
    #plt.figure()
    for tile in tile_list:
        b = bb[tile]
        draw_tile(ax, b)
        lon0 = (b['LONMIN']+b['LONMAX'])/2
        lat0 = (b['LATMIN']+b['LATMAX'])/2
        ax.text(lon0, lat0, tile, horizontalalignment="center")


def plot_tiles(tiles, maxload):
    print("prepare the tiling figure", end="")
    x = np.arange(-180, 180, reso)
    y = np.arange(-90, 90+reso, reso)

    [xx, yy] = np.meshgrid(x, y)

    z2d = np.zeros_like(xx)

    for tile, b in tiles.items():
        na = b["nprofiles"]
        ng = b["cells"]
#        b, na, ng = zb
        idx = np.where((xx >= b['LONMIN']) &
                       (xx <= b['LONMAX']) &
                       (yy >= b['LATMIN']) &
                       (yy <= b['LATMAX']))
        z2d[idx[0], idx[1]] = na*ng

    fig, ax = plt.subplots(1)
    fig.set_size_inches([12, 5])
    # fig = plt.gcf()
    # fig.get_size_inches()
    #plt.clf()
    im = ax.pcolor(x, y, z2d, cmap=plt.get_cmap('YlOrRd'), vmax=maxload*1.2)
    cbar = fig.colorbar(im)
    labels = ['{:,.0f}'.format(x)
              for x in np.arange(0, maxload*1.2, maxload*.2)]
    cbar.ax.set_yticklabels(labels, fontsize=16)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('(#profiles) * (#grid points)  /   #tiles = %i' %
              len(tiles), fontsize=16)
    ax.axis('tight')
    fig.tight_layout()
    for tile, b in tiles.items():
        lon0 = (b['LONMIN']+b['LONMAX'])/2
        lat0 = (b['LATMIN']+b['LATMAX'])/2
        draw_tile(ax, b)
        if len(tile)<=4:
            ax.text(lon0, lat0, tile, horizontalalignment="center", fontsize=7)

    print(" / save it in %s" % tiling_fig)
    plt.savefig(tiling_fig)
    
def draw_tile(ax, b):
    x0 = b['LONMIN']
    x1 = b['LONMAX']
    y0 = b['LATMIN']
    y1 = b['LATMAX']
    rect = patches.Rectangle( (x0,y0), x1-x0, y1-y0, edgecolor='k',
                              fill=False)
    ax.add_patch(rect)
    #plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k')

def get_bb(tile):
    lonmin = -180. -dleft
    lonmax = +180. +dright
    latmin = -90. -dleft
    latmax = +90. + dright
    s=tile[0]
    lonmid = (lonmin+lonmax)/2
    if s=="0":
        lonmax = lonmid
    else:
        lonmin = lonmid
    for s in tile[1:]:
        lonmid = (lonmin+lonmax)/2
        latmid = (latmin+latmax)/2
        if s=="0":
            lonmax, latmax = lonmid, latmid
        elif s=="1":
            lonmin, latmax = lonmid, latmid
        elif s=="2":
            lonmax, latmin = lonmid, latmid
        elif s=="3":
            lonmin, latmin = lonmid, latmid
    return {"LONMIN": lonmin, "LONMAX": lonmax,
            "LATMIN": latmin, "LATMAX": latmax}
        
def gettile(ids, lat, lon):
    lon = (lon+180) % 360 -180
    lat = max(-90,min(lat,90))
    lon += (180+dleft) 
    lat += (90+dleft)
    dd = 180.+(dright+dleft)/2.
    s = str(int( (lon)/dd)) 
    lon %= dd 
    dd = 90.+(dright+dleft) 
    coef = 4 
    maxdepth = 10
    k = 0
    while not(s in ids) and (k<maxdepth):
        i = int(lon//dd) 
        j = int(lat//dd) 
        s += str(i+j*2) 
        lon -= i*dd 
        lat -= j*dd 
        #print(j,i,j) 
        #print(lat,lon, dd, s) 
        dd /= 2. 
        #coef *= 4
        k += 1
    assert s in ids, (s, lat, lon)
    return s

def gettilevec(ids, lats, lons):
    tile_list = [gettile(ids, lat, lon) for lat, lon in zip(lats, lons)]
    return list(set(tile_list))

def convert_tiles_to_dict(tiles):
    bb = {}
    for b in tiles.copy():
        id = b[0].pop("ID")
        b[0]["nprofiles"] = b[1]
        b[0]["cells"] = b[2]
        bb[id] = b[0]
    return bb

def hashtiles(bb):
    return hash(tuple(bb.keys()))

def write_tiles(bb):
    f=tiling_file
    ok = True
    if os.path.exists(f):
        bb_old = read_tiles()
        if hashtiles(bb_old) != hashtiles(bb): ok = False
    if ok:
        pd.to_pickle(bb, f)
    else:
        print("Warning")
        print("you are about to create a new tiling")
        tiles_ok = are_tiles_synchronized()
        profiles_ok = interp.are_tiles_synchronized()
        if True:
            print("global and tiles are synchronized")
            backup_current_tiles()
            pd.to_pickle(bb, f)
        else:
            print("check that it is ok")
            if not(tiles_ok): print("synchronize argo")
            if not(profiles_ok): print("synchronize profiles")
            sys.exit("I prefer to stop")

def backup_current_tiles():
    bb_old = read_tiles()
    h = hashtiles(bb_old)
    d = "%s/%s" % (backuptiles_dir, h)
    print("backup the current tiling in %s " % d)    
    os.makedirs(d)
    command = "mv %s/tiling.* %s" % (tiles_dir, d)

    print("purge %s" % tiles_dir)
    os.system(command)
    command = "rm -f %s/*pkl" % argo_dir
    os.system(command)
    command = "rm -f %s/*/*pkl" % interp.tiles_dir
    os.system(command)

def read_tiles():
    f=tiling_file
    return pd.read_pickle(f)

def write_argo_tile(tile, a):
    f = argo_file % tile
    print("write argo tile: %s" % f)
    pd.to_pickle(a, f)

def read_argo_tile(tile):
    """ tile is either a tile key or a list of tile keys """
    if type(tile) is list:
        a = []
        for t in tile:
            f = argo_file % t
            print(f)
            a += [pd.read_pickle(f)]
        argo = pd.concat(a)
    else:
        f = argo_file % tile
        argo = pd.read_pickle(f)
    return argo

def get_subtiles(maintile, bb):
    """ return all tiles contained in main tile"""
    n=len(maintile)
    subb = [k for k in bb.keys() if k[:n]==maintile]
    return subb

def apply_timestamp():
    """ give the argo_tiles and argodb the same timestamp """
    bb = read_tiles()
    fargo = tools.argodb_file
    print("apply timestamp to argo tiles")
    command = "touch %s -r %s" % (argo_wildcard, fargo)
    os.system(command)

def are_tiles_synchronized():
    bb = read_tiles()
    fargo = tools.argodb_file
    tref = os.path.getmtime(fargo)
    yes = all([(os.path.getmtime(argo_file % tile) == tref) for tile in bb.keys()])
    return yes
    
def gather_global_from_tiles():
    bb = read_tiles()
    argos = []
    for tile in bb.keys():
        print("\r%10s" % tile, end="")
        a = read_argo_tile(tile)
        argos += [a]
    f = trash_file
    a = pd.read_pickle(f)
    argos += [a]
    argo = pd.concat(argos)
    tools.write_argodb(argo)
    apply_timestamp()
    
def split_global_into_tiles(bb, argo):
    """ split the argo database into tiles defined in bb
    and write the sub database into pkl files"""
    direc = tiles_dir
        
    for tile in list(bb.keys()):
        lon0 = bb[tile]["LONMIN"]
        lon1 = bb[tile]["LONMAX"]
        lat0 = bb[tile]["LATMIN"]
        lat1 = bb[tile]["LATMAX"]
        a = argo[(argo.LONGITUDE >= lon0) 
                 & (argo.LONGITUDE < lon1) 
                 & (argo.LATITUDE >= lat0) 
                 & (argo.LATITUDE < lat1)]
        #print(tile)
        f = argo_file % tile 
        #print(tile, len(a), f)
        print("\r%9s - %4i" % (tile, len(a)), end="")
        pd.to_pickle(a, f)
    print()
    # do a trash pkl for profiles that have LON/LAT outside the sphere
    lon0, lon1 = -180., 180.
    lat0, lat1 = -90., 90.
    a = argo[(argo.LONGITUDE < lon0) 
             | (argo.LONGITUDE > lon1) 
             | (argo.LATITUDE < lat0) 
             | (argo.LATITUDE > lat1)]
    a.STATUS = "E"
    f = trash_file
    print("found %i profiles outside the sphere => argodb_trash.pkl" % len(a))
    pd.to_pickle(a, f)
    apply_timestamp()
    
def argotile(tile):
    f = argo_file % tile
    assert os.path.exists(f)
    print("read %s" % f)
    return pd.read_pickle(f)

def argotiles(tile_list):
    a = []
    for tile in tile_list:
        a += [argotile(tile)]
    return pd.concat(a)

def extract_a_block(bb, block):
    """ block is 0, 12, etc """
    n = len(block)
    return [b for b in bb.keys() if b[:n]==block]

def find_tiles(bb, rectangle):
    """ find the tiles that have a non zero intersection with the rectangle """
    r = rectangle
    print("rect=",r)
    n = 40
    lon = list(np.linspace(r["LONMIN"], r["LONMAX"], n))
    lat = list(np.linspace(r["LATMIN"], r["LATMAX"], n))
    lons = lon+[lon[0]]*n+lon+[lon[-1]]*n
    lats = [lat[0]]*n+lat+[lat[-1]]*n+lat
    tile_list = gettilevec(bb.keys(), lats, lons)
    return list(set(tile_list))

def tiles_around_point(bb, lon, lat, radius):
    cff = np.cos(lat*np.pi/180)
    lon0 = lon-radius/cff
    lon1 = lon+radius/cff
    lat0 = lat-radius
    lat1 = lat+radius
    rect = {"LONMIN": lon0, "LONMAX": lon1, "LATMIN": lat0, "LATMAX": lat1}
    return find_tiles(bb, rect)

def tiles_with_halo(bb, tile, coef=3):
    """ return the list of tiles that contain tile + its halo
    halowidth is coef*reso """
    rect = bb[tile].copy()
    rect["LATMIN"] -= coef*reso
    rect["LATMAX"] += coef*reso
    
    lat = 0.5*(rect["LATMIN"]+rect["LATMAX"])
    cff = np.cos(lat*np.pi/180)
    
    rect["LONMIN"] -= coef*reso/cff
    rect["LONMAX"] += coef*reso/cff

    tile_list = [tile]+find_tiles(bb, rect)
    return (list(set(tile_list)), rect)

def profiles_around_point(bb, lon, lat, radius):
    """ extract all profiles within a certain distance from a given point """
    cff = np.pi/180.
    tile_list = tiles_around_point(bb, lon, lat, radius)
    a = argotiles(tile_list)
    d = comp.dist_sphe(a.LONGITUDE*cff, a.LATITUDE*cff, lon*cff, lat*cff)
    d /= cff
    a["DIST"] = d
    return a[d<radius]


def define_tiles():
    argo = tools.read_argodb()

    midlon = (dright-dleft)/2.
    tile = {}
    tile['ID'] = "0"
    tile['LONMIN'] = -180. -dleft#-1./17
    tile['LONMAX'] = midlon#1./127.
    tile['LATMIN'] = -90.-dleft#-1./17
    tile['LATMAX'] = 90.+dright#+1./19

    tile0 = tile.copy()
    tile1 = tile.copy()
    tile1['ID'] = "1"
    tile1['LONMIN'] = midlon#1./127
    tile1['LONMAX'] = 180.+dright#+1./17

    twotile = [(tile0, extract_in_tile(argo, tile0)),
              (tile1, extract_in_tile(argo, tile1))]

    tiles = setup_tiling(twotile, maxload)
    bb = convert_tiles_to_dict(tiles)
    write_tiles(bb)
    split_global_into_tiles(bb, argo)
    plot_tiles(bb, maxload)
    
    
if __name__ == '__main__':
    pass
    #define_tiles()
