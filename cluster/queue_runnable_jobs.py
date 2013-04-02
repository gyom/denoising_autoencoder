#!/bin/env python

import cluster
import cluster.configuration
from   cluster.configuration import Configuration
import cluster.database
from   cluster.database import Database
import cluster.job
from   cluster.job import Job
from   cluster.job import filter_runnable_jobs


import os, sys

def usage():
    pass

def main(argv):
    """
    """

    import getopt

    try:
        opts, args = getopt.getopt(argv[1:], "hv", ["activate_jobdispatch"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    activate_jobdispatch = False

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--activate_jobdispatch"):
            activate_jobdispatch = True
        #elif o in ("--want_early_termination"):
        #    want_early_termination = ((a == "True") or (a == "true") or (a=="1"))
        else:
            assert False, "unhandled option"

    print activate_jobdispatch

    # Make sure that we're on maggie46 where jobdispatch
    # can be called. Otherwise we shouldn't do a thing.
    # That thing might also work with zappa-whatever.
    if activate_jobdispatch:
        import socket
        assert socket.gethostname() == "maggie46.iro.umontreal.ca"

    config = Configuration()
    database = Database(config)

    # filter runnable jobs and then add them to
    #         assert self.config_contents.has_key('jobs_runnable_queue')
    # then launch the appropriate number of jobdispatch of run_next_job.py ?

    all_jobs = database.get_all_jobs()
    all_runnable_jobs = filter_runnable_jobs(all_jobs)
    N = len(all_runnable_jobs)

    #for job in all_runnable_jobs:
    #    job.status = "queued"
    #    job.save()
    #    config.redis_server.zadd(config.config_contents['jobs_runnable_queue'], job.id, 0)

    # At this point we have already made sure
    # that we're running on maggie46.
    if activate_jobdispatch:
        import subprocess
        export_cmd = "export UMONTREAL_CLUSTER_CONFIG=%s" % (config.config_file,)
        full_cmd = "jobdispatch --repeat_jobs=%d '%s; %s'" % (N, 
                                                              export_cmd,
                                                              config.config_contents['script_run_next_job'])
        print full_cmd
        # uncomment this
        #subprocess.check_output(full_cmd, shell=True)


if __name__ == "__main__":
    main(sys.argv)
