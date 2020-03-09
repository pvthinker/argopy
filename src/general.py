import numpy as np
import jdcal

calendar_start = 2433295.5  # January 1st 1950

def juld2date(juld):
    if 'float' in str(type(juld)):
        juld = [juld]
    elif hasattr(juld, '__iter__'):
        pass
            
    else:
        raise ValueError('juld must be an int, a float or be iterable')

    ndays = len(juld)
    year = np.zeros((ndays,), dtype=int)
    month = np.zeros((ndays,), dtype=int)
    day = np.zeros((ndays,), dtype=int)
    dayfrac = np.zeros((ndays,))

    for k, jd in enumerate(juld):
        year[k], month[k], day[k], dayfrac[k] = jdcal.jd2jcal(
            calendar_start, jd)

    if len(juld) == 1:
        year, month, day, dayfrac = year[0], month[0], day[0], dayfrac[0]

    return {'YEAR': year, 'MONTH': month, 'DAY': day, 'DAYFRAC': dayfrac}


def formatlonlat(lon, lat, digits=2, compact=False):
    """ output lon lat in string with degree NSEW """

    if hasattr(lon, "__iter__") and hasattr(lat, "__iter__"):
        assert len(lon)==len(lat)
        pos = [formatlonlat(lo, la) for lo, la in zip(lon, lat)]
    else:
        slat = "N" if lat>=0 else "S"
        slon = "E" if lat>=0 else "W"
        number = "%" + ".%if" % digits
        if compact:
            template=number+"%s"+number+"%s"
        else:
            template = number+"°%s - "+number+"°%s"
        pos = template % (abs(lat), slat, abs(lon), slon)
    return pos

def midpoint(bb, tile):
    b = bb[tile]
    lon = 0.5*(b["LONMIN"]+b["LONMAX"])
    lat = 0.5*(b["LATMIN"]+b["LATMAX"])
    return lon, lat
    
