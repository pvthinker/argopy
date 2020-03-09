import pandas as pd
import tools
import tiles
import init
import masternslave as mns
import os

work_dir = "%s/work" % tiles.tiles_dir
result_file = work_dir + "/result_%04i.pkl"
task_file = work_dir + "/task_%04i.pkl"

local_dirs = [work_dir]

def create_folders():
    if os.path.exists(work_dir):
        pass
    else:
        os.makedirs(work_dir)

def update_with_new_wmos():
    dacs, wmos = tools.get_all_wmos()
    # if debug:
    #     dacs = dacs[::100]
    #     wmos = wmos[::100]
    argo = tools.read_argodb()

    all_wmos = set(wmos)
    known_wmos = set(argo.WMO)
    new=list(all_wmos.difference(known_wmos))
    new_dacs = [d for d,w in zip(dacs, wmos) if w in new]
    new_wmos = [w for w in wmos if w in new]

    if len(new_wmos)>200:
        raise ValueError("Too many new wmos, update manually")

    a=tools.update_argodb(argo, new_dacs, new_wmos)
    tools.write_argodb(a)

def master_job(nslaves, debug=False):
    # define the master director
    master = mns.Master(nslaves)

    init.init()
    
    dacs, wmos = tools.get_all_wmos()
    # if debug:
    #     dacs = dacs[::100]
    #     wmos = wmos[::100]
    argo = tools.read_argodb()

    known_wmos = set(argo.WMO)

    if len(wmos) > len(known_wmos):
        update_with_new_wmos()
    
    nwmos = len(wmos)
    assert nwmos>=nslaves
    print("number of nwmos: %i" % nwmos)
    # define tasks
    print("define the tasks")

    if False:
        task_size = (nwmos // nslaves)
        for itask in range(0, nslaves):
            istr = itask*task_size
            iend = istr+task_size
            if itask == nslaves-1:
                iend = nwmos
            d = dacs[istr:iend]
            w = wmos[istr:iend]
            a = argo[argo.WMO.isin(w)]
            #a = pd.concat([argo[argo.WMO==x] for x in w])        

            task = (a, d, w)
            print('task %02i : %i' % (itask, len(w)))
            f = task_file % itask
            pd.to_pickle(task, f)

    # master defines the tasks
    master.barrier(0)

    print("slaves are working")
    # slaves work

    master.barrier(1)
    print("gather the results")
    # # send tasks to slaves
    # for islave in range(nslaves):
    #     master.send(islave, islave)

    # gather DataFrame
    argos = []
    for itask in range(nslaves):
        f = result_file % itask
        assert os.path.exists(f)
        a = pd.read_pickle(f)
        argos += [a]
    argo = pd.concat(argos)

    print("number of profiles in the the database: %i" % len(argo))
    print("write argo_global.pkl")
    
    tools.write_argodb(argo)

    print("define tiles")

    tiles.define_tiles()
    
    # clean up workdir
    print("clean up %s" % work_dir)
    os.system("rm -Rf %s/*.pkl" % work_dir)
    
    # master gathers the dataframes
    master.barrier(2)


def slave_job(myrank):
    # define the slave
    slave = mns.Slave(myrank)

    itask = slave.islave

    # master defines the tasks
    slave.barrier(0)

    f = task_file % itask
    task = pd.read_pickle(f)

    argo = task[0]
    dacs = task[1]
    wmos = task[2]

    argo = tools.update_argodb(argo, dacs, wmos)

    f = result_file % itask
    pd.to_pickle(argo, f)

    # slaves work
    slave.barrier(1)

    # master gathers the dataframes
    slave.barrier(2)

def single_proc_update():
    dacs, wmos = tools.get_all_wmos()
    argo = tools.read_argodb()
    argo = tools.update_argodb(argo, dacs, wmos)
    return argo

if __name__ == '__main__':

    myrank, nslaves = mns.setup()
    print("number of slaves: %i" % nslaves)
    if nslaves == 0:
        print("Starting the refresh the database")
        master_job(55)
        #single_proc_update()
        
        #nslaves = 10        
        #master_job(nslaves, debug=True)
        #for myrank in range(1, nslaves+1):
        #    slave_job(myrank)
    else:
        if myrank == 0:
            print("Starting the refresh the database")
            master_job(nslaves)

        else:
            slave_job(myrank)
