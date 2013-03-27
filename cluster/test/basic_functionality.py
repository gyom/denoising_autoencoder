#!/bin/env python

import cluster
import cluster.configuration
from   cluster.configuration import Configuration
import cluster.database
from   cluster.database import Database
import cluster.job
from   cluster.job import Job

#config = cluster.configuration.Configuration()
#database = cluster.database.Database(config)

config = Configuration()
database = Database(config)

j0 = Job(id = -1, config = config, database = database)
j1 = Job(id = -1)
j2 = Job(id = -1)

j2.dependencies = [j0.id, j1.id]

database.save_job(j0)
database.save_job(j1)
database.save_job(j2)


# export UMONTREAL_CLUSTER_CONFIG=$HOME/umontreal/denoising_autoencoder/cluster/test/basic_functionality_config.json, python basic_functionality.py
