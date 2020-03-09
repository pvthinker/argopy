import numpy as np
import glob
import pandas as pd
import os
from netCDF4 import Dataset
import socket

atlas_name = "meanstate" # or "eape"

hostname = socket.gethostname()

if (hostname[:8] == "datarmor") or (hostname[::2][:3] == "rin"):
    # login node is datarmor3
    # computational nodes are rXiYnZ
    gdac = "/home/ref-argo/gdac/dac"
    pargopy = "/home1/datawork/groullet/argopy"

elif hostname in ["altair", "libra"]:
    gdac = "/net/alpha/exports/sciences/roullet/Argo/dac"
    pargopy = "/net/alpha/exports/sciences/roullet/Argo"

else:
    raise ValueError("Configure tools.py before using Argopy")


daclist = ["aoml", "bodc", "coriolis", "csio",
           "csiro", "incois", "jma", "kma",
           "kordi", "meds", "nmdis"]

zref = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
                 100., 110., 120., 130., 140., 150., 160., 170.,
                 180., 190., 200., 220., 240., 260., 280, 300.,
                 320., 340., 360., 380., 400., 450., 500., 550.,
                 600., 650., 700., 750., 800., 850., 900., 950.,
                 1000., 1050., 1100., 1150., 1200., 1250., 1300.,
                 1350., 1400., 1450., 1500., 1550., 1600., 1650.,
                 1700., 1750., 1800., 1850., 1900., 1950.,
                 2000.])

argodb_keys = ["DAC", "WMO", "IPROF", "N_LEVELS", "DATA_MODE", "LONGITUDE", "LATITUDE", "JULD", "STATUS"]

global_dir = "%s/global" % pargopy
argodb_dir = "%s/argo" % global_dir
argodb_file = "%s/argo_global.pkl" % argodb_dir

argo_file = gdac+"/%s/%i/%i_prof.nc"
    
def create_folders():
    for d in [global_dir, argodb_dir]:
        if os.path.exists(d):
            pass
        else:
            os.makedirs(d)
        

def unmask(data):
    """ transform masked array into regular numpy array """
    data_out = {}
    for k in data.keys():
        if type(data[k]) is np.ma.core.MaskedArray:
            data_out[k] = data[k].data
        else:
            data_out[k] = data[k]
    return data_out


def bytes2str(data):
    """ byte strings into strings"""
    data_out = {}
    for k in data.keys():
        data_out[k] = data[k]
        if type(data[k]) is np.ndarray:
            firstelem = data_out[k].ravel()[0]
            #print(k, type(firstelem))
            if type(firstelem) is np.bytes_:
                data_out[k] = np.asarray(data[k].data, dtype=str)
    return data_out


def get_all_wmos():
    print("retrieve all wmos in the DAC ", end="")
    wmos = []
    dacs = []
    for dac in daclist:
        prfiles = glob.glob("{}/{}/*/*_prof.nc".format(gdac, dac))
        wmos += [int(f.split("/")[-2]) for f in prfiles]
        dacs += [dac for f in prfiles]

    nwmos = len(wmos)
    print("/ found: %i" % nwmos)
    return (dacs, wmos)


def write_argodb(argo):
    f = argodb_file
    print("write %s " % f)
    pd.to_pickle(argo, f)


def read_argodb():
    d = argodb_dir
    f = argodb_file
    if os.path.exists(f):
        print("read %s " % f)
        argo = pd.read_pickle(f)
    else:
        if os.path.exists(d):
            pass
        else:
            os.makedirs(d)    
        print("Creation of the empty argo database: %s" % f)
        argo = pd.DataFrame(columns=argodb_keys)
        write_argodb(argo)
    return argo



def update_argodb(argo, dacs, wmos):
    idx0 = argo.index
    print("update argo with %i wmos" % len(wmos))
    for dac, wmo in zip(dacs, wmos):
        print("\r%9s - %8i" % (dac, wmo), end="")
        a0 = argo[(argo.DAC == dac) & (argo.WMO == wmo)]
        output = read_profile(dac, wmo,
                              header=True, headerqc=True,
                              verbose=False, path=gdac)
        # print(output.keys())
        # print(len(output["JULD"]))
        # print(output)
        nprof = output["N_PROF"]
        nlevels = output["N_LEVELS"]
        tags = [hash((dac, wmo, i)) for i in range(nprof)]

        a1 = pd.DataFrame(columns=argodb_keys)
        for k in ["JULD", "LONGITUDE", "LATITUDE", "DATA_MODE"]:
            a1[k] = output[k]
        a1.STATUS = "N"
        a1.DAC = [dac]*nprof
        a1.WMO = [wmo]*nprof
        a1.IPROF = np.arange(nprof)
        a1.N_LEVELS = [nlevels]*nprof
        a1.index = tags

        data = {k: output[k] for k in ["POSITION_QC", "JULD_QC"]}
        qc = pd.DataFrame(data=data, index=a1.index)

        bad_jul = qc[(qc.POSITION_QC == "1") & (qc.JULD_QC != "1")].index
        bad_pos = qc[(qc.POSITION_QC != "1")].index

        newtags = a1.index.difference(a0.index)
        print("===>newtags: %i" % len(newtags))
        argo = pd.concat([argo, a1.loc[newtags, :]])
        
        argo.loc[bad_jul, "STATUS"] = "T"
        argo.loc[bad_pos, "STATUS"] = "L"
        
    print()    
    return argo

def read_profile(dac, wmo, iprof=None,
                 header=False, data=False,
                 headerqc=False, dataqc=False,
                 shortheader=False,
                 verbose=True, path=None):
    """
    :param dac: DAC du profil recherche
    :param wmo: WMO du profil recherche
    :param iprof: Numero du profil recherche
    :param header: Selectionne seulement LATITUDE, LONGITUDE et JULD
    :param headerqc: Selectionne seulement POSITION_QC et JULD_QC
    :param data: Selectionne TEMP, PSAL et PRES
    :param dataqc: Selectionne TEMP_QC, PSAL_QC et PRES_QC
    :param verbose: ???

    Les valeurs selectionnee grace aux arguments passes a la fonction definissent
    la DataFrame que retournera celle-ci.

    Basic driver to read the \*_prof.nc data file

    The output is a dictionnary of vectors
    - read one or all profiles read the header (lat, lon, juld) or not
    - read the data or not always return IDAC, WMO, N_PROF, N_LEVELS
    - and DATA_UPDATE (all 5 are int)

    :rtype: dict
    """
    key_header = ["LATITUDE", "LONGITUDE", "JULD"]
    key_headerqc = ["POSITION_QC", "JULD_QC"]
    key_data = ["TEMP", "PSAL", "PRES"]
    key_dataqc = ["TEMP_QC", "PSAL_QC", "PRES_QC"]

    filename = argo_file % (dac, wmo, wmo)

    if verbose:
        print(filename)
        # print("/".join(filename.split("/")[-3:]))

    output = {}

    required_keys = set(["TEMP", "PSAL", "PRES"])

    if (os.path.isfile(filename)):
        with Dataset(filename, "r", format="NETCDF4") as f:
            output["DAC"] = dac
            output["WMO"] = wmo
            output["N_PROF"] = len(f.dimensions["N_PROF"])
            output["N_LEVELS"] = len(f.dimensions["N_LEVELS"])
            # DATE_UPDATE is an array of 14 characters in the *_prof.nc
            # we transform it into an int
            # YYYYMMDDhhmmss
            # print(filename)
            dateupdate = f.variables["DATE_UPDATE"][:]
            if type(dateupdate) is np.ma.core.MaskedArray:
                dateupdate = [c.decode("utf-8") for c in dateupdate.data]
            output["DATE_UPDATE"] = "".join(dateupdate)

            if shortheader:
                pass
            else:
                keyvar = set(f.variables.keys())

                if required_keys.issubset(keyvar):
                    output["TSP_QC"] = "1"
                else:
                    output["TSP_QC"] = "2"

                if header or headerqc or data or dataqc:
                    if iprof is None:
                        idx = range(output["N_PROF"])
                        output["IPROF"] = np.arange(output["N_PROF"])
                    else:
                        idx = iprof
                        output["IPROF"] = iprof

                if header:
                    for key in key_header:
                        output[key] = f.variables[key][idx]
                        output["DATA_MODE"] = np.asarray(
                            [c for c in f.variables["DATA_MODE"][idx]])

                if headerqc:
                    for key in key_headerqc:
                        output[key] = f.variables[key][idx]

                if data:
                    for key in key_data:
                        if output["TSP_QC"] == "1":
                            output[key] = f.variables[key][idx, :]
                        else:
                            output[key] = np.NaN+np.zeros(
                                (output["N_PROF"], output["N_LEVELS"]))

                if dataqc:
                    for key in key_dataqc:
                        if output["TSP_QC"] == "1":
                            output[key] = f.variables[key][idx]
                        else:
                            output[key] = np.zeros((output["N_PROF"],
                                                    output["N_LEVELS"]),
                                                   dtype=str)
        output = bytes2str(unmask(output))
    return output

if __name__ == '__main__':
    dacs, wmos = get_all_wmos()
    argo = read_argodb()
    argo = update_argodb(argo, dacs, wmos)
    write_argodb(argo)
