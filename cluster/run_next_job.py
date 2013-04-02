
#!/bin/env python

import cluster
import cluster.configuration
from   cluster.configuration import Configuration
import cluster.database
from   cluster.database import Database
import cluster.job
from   cluster.job import Job

import os, sys
import subprocess

def main(argv):
    """
    """

    config = Configuration()
    database = Database(config)

    if config.redis_server.scard(config.config_contents['jobs_runnable_queue']) > 0:
        job_id = config.redis_server.spop(config.config_contents['jobs_runnable_queue']) > 0:

    job = database.load_job(job_id)
    assert job_id == job.id

    job.status = "running"
    database.save_job(job)

    # Need to add something with "tee" to redirect
    # stdout and stderr to files in the right directory.
    #
    # command > >(tee stdout.log) 2> >(tee stderr.log >&2)
    #
    stdout_path = os.path.join(job.dir(), "stdout.log")
    stderr_path = os.path.join(job.dir(), "stderr.log")
    export_cmd = "export UMONTREAL_CLUSTER_JOB_ID=%d" % (job_id,)

    #full_cmd = "%s ; %s" % (export_cmd, job.cmdline)
    full_cmd = "%s ; %s > >(tee '%s') 2> >(tee '%s' >&2)" % (export_cmd, job.cmdline, stdout_path, stderr_path)

    print full_cmd
    # uncomment this
    subprocess.check_output(full_cmd, shell=True)

    job.status = "success"
    database.save_job(job)


if __name__ == "__main__":
    main(sys.argv)
