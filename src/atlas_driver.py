import numpy as np
import pandas as pd
import tools
import masternslave as mns
import os
import interp
import tiles
import sys

atlas_name = sys.argv[-1]

wrong_atlas = "execute atlas_driver.py with either 'meanstate' or 'eape' in argument"
assert atlas_name in ["eape", "meanstate"], wrong_atlas

tools.atlas_name = atlas_name

import atlas

    
def master_job(nslaves):
    # define the master director
    master = mns.Master(nslaves, verbose=False)

    atlas.create_folders()
    
    
    tasks = [k for k, s in enumerate(atlas.subd_list)]

    # master defines the tasks
    #print("MASTER has tasks: ", tasks)
    master.barrier(0)


    # slaves work
    master.async_distrib(tasks)
    
    master.barrier(1)

    atlas.write_global_from_subd()
    # master gathers the dataframes
    master.barrier(2)


def slave_job(myrank):
    # define the slave
    slave = mns.Slave(myrank, verbose=False)
    # master defines the tasks

    slave.barrier(0)
    
    #pd.read_pickle(file_tiles_to_stats) 
    done = False
    #print('slave #%3i is ready to go' % (slave.islave))
    # do_stats_task(slave.islave)
    while not(done):
        itask = slave.get_async_task()
        # itask = slave.islave
        if itask > -1:            
            subd = atlas.subd_list[itask]
            print('** slave #%3i processes task %i / subd %s' % (slave.islave, itask, subd))
            atlas.write_subd_from_tiles(subd)
            
        else:
            done = True

    # slaves work
    slave.barrier(1)

    # master gathers th 
    slave.barrier(2)

def monoproc_job():

    for subd in atlas.subd_list:
        atlas.write_subd_from_tiles(subd)
        
    atlas.write_global_from_subd()

if __name__ == '__main__':
    try:
        print("Start atlas of %s" % atlas_name)
        myrank, nslaves = mns.setup()

        if nslaves == 0:
            monoproc_job()
            #for myrank in range(1, nslaves+1):
            #    slave_job(myrank)
        else:
            if myrank == 0:
                master_job(nslaves)

            else:
                slave_job(myrank)
    except:
        mms.abort()
