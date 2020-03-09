import numpy as np

def dist_sphe(x, y, lon, lat):
    """Compute the spherical arc between two points on the unit sphere"""
    return np.arccos(np.sin(lat)*np.sin(y)+np.cos(lat)*np.cos(y)*np.cos(lon-x))

def compute_weight(x, y, lon, lat, reso, degrees=False):
    """Compute the weight between points (x, y) and point (lon, lat) with
    a gaussian filter """
    if degrees:
        cff=np.pi/180
        dist = dist_sphe(x*cff, y*cff, lon*cff, lat*cff)
        weight = np.exp(-0.5*(dist/(reso*cff))**2)
    else:
        dist = dist_sphe(x, y, lon, lat)
        weight = np.exp(-0.5*(dist/reso)**2)
    return weight
