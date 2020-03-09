# -*- coding: utf-8 -*-
"""

"""

from mpi4py import MPI
import numpy as np
import time
import os


comm = MPI.COMM_WORLD

def bcast(data):
    print("bcast / rank =%i" % comm.Get_rank())
    return comm.bcast(data, root=0)

def abort():
    comm.Abort()
    
class Master():
    def __init__(self, nslaves, verbose=False):
        self.master = True
        self.slave = False
        self.myrank = 0
        self.nslaves = nslaves
        self.verbose = verbose
        # list of buffers used by master to receive msg from slaves
        self.answer = [np.zeros((1,), dtype=int) for k in range(nslaves)]
        self.slavestate = [1]*self.nslaves
        # list of irecv managed by master
        self.reqr = []
        self.record = []
        self.status = MPI.Status()
        if self.verbose:
            print('Hello I\'m the master, I\'ve %i slaves under my control'
                  % nslaves)

    def barrier(self, ibarrier):
        if self.verbose:
            print('master reach barrier %i' % ibarrier)
        comm.Barrier()

    def getavailableslave(self):
        """
        :param slavestate: Give the state of the slave

        Return the index of a slave that is awaiting a task.  A busy slave
        has a state == 0. If all slaves are busy then wait until a msg is
        received, the msg is sent upon task completion by a slave. Then
        determin who sent the msg. The msg is collected in the answer
        array. By scanning it, we determine who sent the message.

        :rtype: int
        """

        islave = 0
        while (islave < self.nslaves) and (self.slavestate[islave] == 0):
            islave += 1

        if islave == self.nslaves:
            # all slaves are busy, let's wait for the first
            # print('waiting ...', len(reqr), answer)
            MPI.Request.Waitany(self.reqr, self.status)
            # print('ok a slave is available', answer)
            islave = 0
            while (islave < self.nslaves) and (self.answer[islave][0] == 0):
                islave += 1
            self.slavestate[islave] = 1  # available again
            # note  slave send the message int(1)
            # master is getting the msh in the answer array
            # but the received msg is never int(1) !!!
            self.answer[islave][0] = 0
            # todo: remove the reqr that is done

        else:
            pass
        # print('=> %i is available' % islave)
        return islave

    def async_distrib(self, tasks):
        if type(tasks) is int:
            ntasks = tasks
            tasks = range(ntasks)
        elif hasattr(tasks, "__iter__"):
            ntasks = len(tasks)
        else:
            print("tasks should be int or list")
            comm.Abort()
            
        self.record = [-1]*ntasks
        # distribute the ntasks
        for itask in tasks:
            islave = self.getavailableslave()
            if self.verbose:
                print('master sends task %i to #%3i' % (itask, islave))
            self.record[itask] = islave
            comm.isend(int(itask), dest=islave+1, tag=islave)

            self.reqr.append(comm.Irecv(
                self.answer[islave], source=islave+1, tag=islave))
            self.slavestate[islave] = 0
        # tell all slaves that they are done
        for islave in range(self.nslaves):
            comm.isend(int(-1), dest=islave+1, tag=islave)

    def summary(self):
        for k, islave in enumerate(self.record):
            print('task %5i done by #%3i' % (k, islave))


class Slave():
    def __init__(self, myrank, verbose=False):
        self.master = False
        self.slave = True
        self.myrank = myrank
        self.islave = myrank-1
        self.record = []
        self.verbose = verbose
        if self.verbose:
            print('slave #%3i starts' % self.islave)

    def barrier(self, ibarrier):
        if self.verbose:
            print('slave #%3i reach barrier %i' % (self.islave, ibarrier))
        comm.Barrier()

    def get_async_task(self):
        if len(self.record) == 0:
            # first task
            pass
        else:
            # already a task done
            # tell master that it's done
            comm.isend(int(1), dest=0, tag=self.islave)
        # wait for a new task
        if self.verbose:
            print('slave #%3i waits' % self.islave)
        msg = comm.recv(source=0, tag=self.islave)
        if type(msg) is int:
            itask = msg
        else:
            print("msg=", msg, "type=", type(msg))
            print("COMMPB with slave %i" % self.islave)
            with open("comm_pb_%i.txt" % self.islave, "w") as fid:
                fid.write(msg)
            itask = -1
        if self.verbose:
            print('slave #%3i gets task %i' % (self.islave, itask))
        if itask > -1:
            self.record += [itask]
        return itask


def setup():
    myrank = comm.Get_rank()
    nslaves = comm.Get_size()-1
    return myrank, nslaves

def ordering_tasks(tasks, taskfilename):
    """
    :param tasks: List containing the tasks to do
    
    Sort the tasks according to their workload
    workload is proportional to size of the task file
    
    :rtype: list of int
    """

    workload = [os.path.getsize(taskfilename(t)) for t in tasks]
    idx = np.argsort(workload)

    return [tasks[i] for i in idx[::-1]]

if __name__ == '__main__':

    ntasks = 1600

    myrank, nslaves = setup()

    if myrank == 0:
        master = Master(nslaves)
        os.system("rm -f monitor_mns*txt")
        msg = bcast(({3:"Hello!"}, ntasks))
        master.barrier(0)
        master.async_distrib(ntasks)
        master.barrier(1)
        master.summary()
        master.barrier(2)

    else:
        slave = Slave(myrank)
        msg = bcast(None)
        print(myrank, msg)
        slave.barrier(0)
        done = False
        mode = "w"
        while not(done):
            itask = slave.get_async_task()
            with open("monitor_mns_%i.txt" % myrank, mode) as fid:
                fid.write("%i\n" % itask)
            mode = "a"
            # print('slave #%3i does task %i' % (slave.islave, itask))
            if itask > -1:
                time.sleep(np.random.randint(3)/1000.)
            else:
                done = True
        slave.barrier(1)
        slave.barrier(2)
        #print(slave.islave, slave.record)
