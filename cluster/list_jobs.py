#!/bin/env python

import cluster
import cluster.configuration
from   cluster.configuration import Configuration
import cluster.database
from   cluster.database import Database
import cluster.job
from   cluster.job import Job

config = Configuration()
database = Database(config)

all_jobs = database.get_all_jobs()

print all_jobs

for job in all_jobs:
    print str(job)

# export UMONTREAL_CLUSTER_CONFIG=$HOME/umontreal/denoising_autoencoder/cluster/test/basic_functionality_config.json, python basic_functionality.py