"""
"""

import numpy as np
import pandas as pd
import gsw

#import general_tools as tools
import database as db
#import param as param
from numba import jit
import warnings
warnings.filterwarnings("ignore")

def raw_to_zref(d, zref):
    temp = d['TEMP']
    psal = d['PSAL']
    pres = d['PRES']
    temp_qc = d['TEMP_QC']
    psal_qc = d['PSAL_QC']
    pres_qc = d['PRES_QC']
    lon = d['LONGITUDE']
    lat = d['LATITUDE']

    klist, ierr = remove_bad_qc(d)
    if ierr == 0:
        Tis = temp[klist]
        SP = psal[klist]
        p = pres[klist]

        CT, SA, z = insitu_to_absolute(Tis, SP, p, lon, lat, zref)
        Ti, Si, dTidz, dSidz = interp_at_zref(CT, SA, z, zref, klist)
        pi = gsw.p_from_z(-zref, lat)
        #Ri = gsw.rho(Si, Ti, pi)
        Ri, alpha, beta = gsw.rho_alpha_beta(Si, Ti, pi)
        g = gsw.grav(lat, pi)
        BVF2i = g*(beta*dSidz-alpha*dTidz)
        flag = True
    else:
        zero = zref*0.
        Ti, Si, Ri, BVF2i = zero, zero, zero, zero
        flag = False
    return {'CT': Ti, 'SA': Si, 'RHO': Ri, 'BVF2': BVF2i}, flag




def remove_bad_qc(d):
    """Return the index list of data for which the three qc's are 1
    and the error flag ierr

    ierr = 0 : no pb

    ierr = 1 : too few data in the profile

    :rtype: list, int"""
    temp = d['TEMP']
    psal = d['PSAL']
    pres = d['PRES']
    temp_qc = d['TEMP_QC']
    psal_qc = d['PSAL_QC']
    pres_qc = d['PRES_QC']

    
    maskarraytype = np.ma.core.MaskedArray
    keys = [temp, psal, pres, temp_qc, psal_qc, pres_qc]
    for key in keys:
        if key.any == maskarraytype:
            key = key.compressed()
    # klist = [k for k in range(len(pres)) if (temp_qc[k] == '1') and (
    #     sal_qc[k] == '1') and (pres_qc[k] == '1')]
    klist = [k for k in range(len(pres))
             if temp_qc[k]+psal_qc[k]+pres_qc[k] == '111']
    ierr = 0
    p = pres[klist]
    ierr = check_pressure(p)

    return klist, ierr


def insitu_to_absolute(Tis, SP, p, lon, lat, zref):
    """Transform in situ variables to TEOS10 variables

    :rtype: float, float, float"""
    #  SP is in p.s.u.
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    CT = gsw.CT_from_t(SA, Tis, p)
    z = -gsw.z_from_p(p, lat)
    return(CT, SA, z)



def interp_at_zref(CT, SA, z, zref, klist):
    """Interpolate CT, SA, dCT/dz and dSA/dz from their native depths z to
    zref

    Method: we use piecewise Lagrange polynomial interpolation

    For each zref[k], we select a list of z[j] that are close to
    zref[k], imposing to have z[j] that are above and below zref[k]
    (except near the boundaries)

    If only two z[j] are found then the result is a linear interpolation

    If n z[j] are found then the result is a n-th order interpolation.

    For interior points we may go up to 6-th order

    For the surface level (zref==0), we do extrapolation

    For the bottom level (zref=2000), we do either extrapolation or
    interpolation if data deeper than 2000 are available.

    :rtype: float, float, float, float

    """

    nref = len(zref)
    CTi = np.zeros((nref,), dtype=float)
    SAi = np.zeros((nref,), dtype=float)
    dCTdzi = np.zeros((nref,), dtype=float)
    dSAdzi = np.zeros((nref,), dtype=float)


    nbpi, ks = select_depth(zref, z, klist)
    nupper = np.zeros((nref,), dtype=int)
    nlower = np.zeros((nref,), dtype=int)

    # count the number of data that are lower and upper than zref[k]
    for k in range(nref):
        if k > 0:
            nlower[k] += nbpi[k-1]
        if k > 1:
            nlower[k] += nbpi[k-2]
        if k < nref:
            nupper[k] += nbpi[k]
        if k < nref-1:
            nupper[k] += nbpi[k+1]

    # for each zref, form the list of z[j] used for the interpolation
    # if the list has at least two elements (a linear interpolation is possible)
    # then do it, otherwise, skip that depth
    order = 3
    for k in range(nref):
        idx = []
        idx2 = []
        klow = []
        kupper = []
        if k == 0:
            if nupper[k] >= 2:
                kupper = ks[0]+ks[1]+ks[2]
                idx = kupper[:2]
        elif k == 1:
            if (nlower[k] >= 1) and (nupper[k] >= 1):
                klow = ks[0]
                kupper = ks[1]+ks[2]
                nelem = min(order, min(len(klow), len(kupper)))
                idx = klow[-nelem:]+kupper[:nelem]
        elif k == (nref-1):
            if (nupper[k]==0):
                # fix to prevent oscillation at level 2000m when
                # a high resolution profile stops just before 2000
                # e.g.
                # coriolis/6902548/113
                # aoml/4902979/58
                #kk1 = (ks[k-2]+ks[k-1])[-2]
                #kk0 = (ks[k-2]+ks[k-1])[-1]
                idx = []#(ks[k-2]+ks[k-1])[-2:] 
                # if (2*z[kk0]-z[kk1]) > zref[k]:
                #     # linear interpolation
                #     idx = (ks[k-2]+ks[k-1])[-2:]
                # else:
                #     # too far, drop the point
                #     idx = []#(ks[k-2]+ks[k-1])[-2:]       
            elif (nlower[k]+nupper[k]) >= 2:
                klow = ks[k-2]+ks[k-1]
                kupper = ks[k]
                nelem = min(order, min(len(klow), len(kupper)))
                idx = klow[:-nelem]+kupper[:nelem]
        elif k == (nref-2):
            if (nlower[k] >= 1) and (nupper[k] >= 1):
                klow = ks[k-2]+ks[k-1]
                kupper = ks[k]+ks[k+1]
                nelem = min(order, min(len(klow), len(kupper)))
                idx = klow[-nelem:]+kupper[:nelem]
        else:
            if (nlower[k] >= 1) and (nupper[k] >= 1):
                klow = (ks[k-2]+ks[k-1])
                kupper = (ks[k]+ks[k+1])
                # avoid having larger than 6th order interpolation
                nelem = min(order, min(len(klow), len(kupper)))
                dzlow = np.median(np.diff(z[klow]))
                dzupp = np.median(np.diff(z[kupper]))
                #print("*", dzlow, dzupp)
                if abs(dzupp-dzlow)>5:
                    nelem = 1
                idx = klow[-nelem:]+kupper[:nelem]
                # data points need to be contiguous
                # otherwise it means we are extrapolating
                # in an intervale that was flaged bad
                if kupper[0]-klow[-1]>1: idx=[]
                # if (nupper[k]==1) or (nlower[k]==1):
                #     # fix to prevent outlier when the vertical
                #     # resolution suddenly switch from 1 meter to 50m
                #     # e.g. this happens at 1000m for aoml/5904132
                #     idx =  (ks[k-2]+ks[k-1])[-1:]+(ks[k]+ks[k+1])[:1]
                # else:
                #     idx = (ks[k-2]+ks[k-1])[-3:]+(ks[k]+ks[k+1])[:3]

        # if (len(klow)>0) and (len(kupper)>0):
        #     if kupper[0]-klow[-1]>1: idx=[]
        #print(zref[k], z[idx], nlower[k], nupper[k], idx)

        if len(idx) >= 2:
            cs, ds = lagrangepoly(zref[k], z[idx])
            # the meaning of the weights computed by lagrangepoly should
            # be clear in the code below
            #
            # cs[i] (resp. ds[i]) is the weight to apply on CT[idx[i]]
            # sitting at z[idx[i]] to compute CT (resp. dCT/dz) at zref[k]
            #
            CTi[k] = np.sum(cs*CT[idx])
            SAi[k] = np.sum(cs*SA[idx])
            dCTdzi[k] = np.sum(ds*CT[idx])
            dSAdzi[k] = np.sum(ds*SA[idx])
        else:
            CTi[k] = np.nan
            SAi[k] = np.nan
            dCTdzi[k] = np.nan
            dSAdzi[k] = np.nan
    #print(ks)
    return CTi, SAi, dCTdzi, dSAdzi


def select_depth(zref, z, klist):
    """Return the number of data points we have between successive zref.

    for each intervale k, we select the z_j such that

    zref[k] <= z_j < zref[k+1], for k=0 .. nref-2

    zref[nref-1] <= z_j < zextra, for k=nref-1

    and return

    nbperintervale[k] = number of z_j

    kperint[k] = list of j's


    with zextra = 2*zref[-1] - zref[-2]

    :rtype: int, list

    """
    nz = len(z)
    nref = len(zref)
    zextra = 2*zref[-1]-zref[-2]
    zrefextended = list(zref)+[zextra]
    nbperintervale = np.zeros((nref,), dtype=int)
    kperint = []
    zprev = -1.
    j = 0
    z=[x if k in klist else np.nan for k,x in enumerate(z) ]
    #print('*'*10, z, j, nz)
    for k, z0 in enumerate(zrefextended[1:]):
        n = 0
        ks = []
        if j==nz:
            pass
        else:
            while np.isnan(z[j]) and (j<(nz-1)):
                j +=1
        while (j < nz) and (z[j] < z0):
            # for a few profiles it may happens that two consecutive
            # data sit at the same depth this causes a division by
            # zero in the interpolation routine.  Here we fix this by
            # simply skipping depths that are already used.
            if z[j] > zprev:
                if j in klist:
                    n += 1
                    ks.append(j)
            zprev = z[j]
            j += 1
        nbperintervale[k] = n
        kperint.append(ks)
        #print(z0, ks, z[j])
    #print("DONE!!!")
    return nbperintervale, kperint

@jit
def lagrangepoly(x0, xi):
    """Weights for polynomial interpolation at x0 given a list of xi
    return both the weights for function (cs) and its first derivative
    (ds)

    Example:
    lagrangepoly(0.25, [0, 1])
    >>> [0.75, 0.25,], [1, -1]

    :rtype: float, float

    """
    xi = np.asarray(xi)
    ncoef = len(xi)
    cs = np.ones((ncoef,))
    ds = np.zeros((ncoef,))

    denom = np.zeros((ncoef, ncoef))
    for i in range(ncoef):
        for j in range(ncoef):
            if i != j:
                dx = xi[i]-xi[j]
                if dx == 0:
                    # should not happen because select_depth removes
                    # duplicate depths
                    #  raise ValueError('division by zero in lagrangepoly')
                    print('WARNING, division by zero in lagrangepoly')
                else:
                    denom[i, j] = 1./dx

    # for the derivative, see
    # https://en.wikipedia.org/wiki/Lagrange_polynomial#Derivatives
    for i in range(ncoef):
        for j in range(ncoef):
            if i != j:
                cff = 1.
                cs[i] *= (x0-xi[j])*denom[i, j]
                for k in range(ncoef):
                    if (k != i) and (k != j):
                        cff *= (x0-xi[k])*denom[i, k]
                ds[i] += cff*denom[i, j]
    return cs, ds


@jit
def try_to_remove_duplicate_pressure(p):
    """
    :param p: list of pressures

    Fonction utilisée pour trouver d'éventuelles pressions dupliquées

    :rtype: list
    """
    idx = [0]+[l+1 for l, x in enumerate(p[1:]) if p[l] < x]
    return idx


@jit
def check_pressure(p):
    """
    :param p: list of pressures

    Fonction utilisée pour vérifier les valeurs de pression et les flagger comme
    mauvaises si elles ne conviennent pas

    :rtype: int
    """

    dp = np.diff(p)
    if np.all(dp > 0):
        ierr = 0
    else:
        npb = np.sum(np.diff(p) <= 0)
        if len(p) < 5:
            print(': only %i p points' % (len(p)))
            ierr = 1
            # print(p)
        else:
            idxf = try_to_remove_duplicate_pressure(p)
            p = p[idxf]
            dp = np.diff(p)
            if np.all(dp > 0):
                ierr = 0
                print(': fixed %i pbs' % npb)
                # print(p0, p)
            else:
                if len(p) > 100:
                    ierr = 1
                    print(': unfixed')
                else:
                    ierr = 1
                    print(': unfixed [hr profile]')

    return ierr
