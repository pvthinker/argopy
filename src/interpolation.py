import numpy as np
import pandas as pd
import tools
import masternslave as mns
import os
import interp
import tiles

file_tiles_to_interpolate = "interpo_tiles.pkl"

def workload(bb):
    work = {}
    for tile in bb.keys():
        a = tiles.read_argo_tile(tile)
        work[tile] = a.N_LEVELS[a.STATUS=="N"].sum()
    return work

def master_job(nslaves, resume=False):
    # define the master director
    master = mns.Master(nslaves)


    argo = tools.read_argodb()
    bb = tiles.read_tiles()
    keys = list(bb.keys())


    work = workload(bb)
    if resume:
        missing = [k for k in keys if not(os.path.exists(interp.tiles_profiles_file % (interp.var_dir["CT"], k)))]
        weight = [work[k] for k in missing]
        idx = np.argsort(weight)
        tasks = idx[::-1]
        keys = missing

    else:
        weight = [work[k] for k in keys]
        idx = np.argsort(weight)
        tasks = idx[::-1]            
        #tiles.split_global_into_tiles(bb, argo)

    print(tasks)
    pd.to_pickle(keys, file_tiles_to_interpolate) 

    
    # master defines the tasks
    master.barrier(0)


    # slaves work
    master.async_distrib(tasks)
    
    master.barrier(1)
    
    # gather DataFrame
    tiles.gather_global_from_tiles()
    # gather profiles
    interp.gather_global_from_tiles()
    
    # master gathers the dataframes
    master.barrier(2)


def slave_job(myrank):
    # define the slave
    slave = mns.Slave(myrank)
    
    # master defines the tasks
    slave.barrier(0)
    keys = pd.read_pickle(file_tiles_to_interpolate) 
    done = False
    # print('slave #%3i processes task %i' % (slave.islave, slave.islave))
    # do_stats_task(slave.islave)
    while not(done):
        itask = slave.get_async_task()
        # itask = slave.islave
        if itask > -1:            
            tile = keys[itask]
            print('** slave #%3i processes task %i / tile %s' % (slave.islave, itask, tile))
            interp.update_profiles_in_tile(tile)
            
        else:
            done = True

    # slaves work
    slave.barrier(1)

    # master gathers the dataframes
    slave.barrier(2)


if __name__ == '__main__':
    print("Start the interpolation")
    myrank, nslaves = mns.setup()

    if nslaves == 0:
        print("pb nslaves==0")
        exit(0)
        nslaves = 10        
        master_job(nslaves)
        #for myrank in range(1, nslaves+1):
        #    slave_job(myrank)
    else:
        if myrank == 0:
            master_job(nslaves, resume=False)

        else:
            slave_job(myrank)
