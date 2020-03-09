import numpy as np
import pandas as pd
import tools
import masternslave as mns
import os
import tiles
import eape


def workload(bb):
    work = {}
    for tile, b in bb.items():
        work[tile] = b["cells"]*b["nprofiles"]
    return work

def define_tasks(resume=False):
    argo = tools.read_argodb()
    bb = tiles.read_tiles()
    keys = list(bb.keys())
    work = workload(bb)
    if resume:
        d = eape.var_dir["CT"]
        keys = [k for k in keys
                if not(os.path.exists(eape.tiles_file % (d, k)))]


    weight = [work[k] for k in keys]
    idx = np.argsort(weight)

    tasks = list(idx[::-1])
    tile_list = [keys[t] for t in tasks]
    #print(tasks)
    #print(tile_list)
    return (tasks, keys)
    
def master_job(nslaves, resume=False):
    # define the master director
    master = mns.Master(nslaves, verbose=False)

    eape.create_folders()
    
    obj = define_tasks(resume=resume)
    tasks, keys = mns.bcast(obj)

    bb = tiles.read_tiles()
    bb = mns.bcast(bb)
    
    # master defines the tasks
    #print("MASTER has tasks: ", tasks)
    master.barrier(0)


    # slaves work
    master.async_distrib(tasks)
    
    master.barrier(1)
    
    # master gathers the dataframes
    master.barrier(2)


def slave_job(myrank):
    # define the slave
    slave = mns.Slave(myrank, verbose=False)
    # master defines the tasks
    tasks, keys = mns.bcast(None)
    bb = mns.bcast(None)
    slave.barrier(0)
    
    #pd.read_pickle(file_tiles_to_stats) 
    done = False
    #print('slave #%3i is ready to go' % (slave.islave))
    # do_stats_task(slave.islave)
    while not(done):
        itask = slave.get_async_task()
        # itask = slave.islave
        if itask > -1:            
            tile = keys[itask]
            print('** slave #%3i processes task %i / tile %s' % (slave.islave, itask, tile))
            eape.compute_stats(bb, tile)
            
        else:
            done = True

    # slaves work
    slave.barrier(1)

    # master gathers the dataframes
    slave.barrier(2)

def monoproc_job():
    tasks, keys = define_tasks(resume=True)
    
    bb = tiles.read_tiles()

    for itask in tasks:
        tile = keys[itask]
        print('** processes tile %s' % tile)
        eape.compute_stats(bb, tile)


if __name__ == '__main__':
    print("Start EAPE")
    myrank, nslaves = mns.setup()

    if nslaves == 0:
        monoproc_job()
        #for myrank in range(1, nslaves+1):
        #    slave_job(myrank)
    else:
        if myrank == 0:
            master_job(nslaves, resume=False)

        else:
            slave_job(myrank)
