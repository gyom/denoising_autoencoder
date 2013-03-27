
import cluster
import os

class Job(object):
    def __init__(self, id = None, config = None, database = None):
        self.__the_keys__ = []

        import cluster.configuration
        import cluster.database
        if config is None:
            self.config = cluster.configuration.Configuration()
            assert self.config is not None
        else:
            self.config = config

        if database is None:
            self.database = cluster.database.Database(self.config)
            assert self.database is not None
        else:
            self.database = database

        # Scenario 1. We were called from the command line with 
        # the value of UMONTREAL_CLUSTER_JOB_ID set to something.
        # We are calling Job() to basically get the details of *this* job.
        #
        # Scenario 2. We are creating a new job object and we
        # will fill in the values manually to later save it to
        # the database.
        #
        # Scenario 3. We are creating an empty shell in which the
        # details will be filled by an immediate call to from_dict.

        if (id is None) and os.environ.has_key("UMONTREAL_CLUSTER_JOB_ID"):
            self.id = os.environ["UMONTREAL_CLUSTER_JOB_ID"]
            self.from_dict(self.database.load_job(self.id))
        elif id == -1:
            # generate a new id
            self.id = self.database.get_next_free_job_id()
            self.status = "not_run"
            self.parent = None
            self.dependencies = []
            self.cmdline = ""
            self.__the_keys__ += ["id", "status", "parent", "dependencies", "cmdline"]
        else:
            # this object will be constructed elsewhere
            pass

    def from_dict(self, job_desc):
        "job_desc is an object loaded from a json object"
        assert job_desc.has_key("id")
        assert job_desc.has_key("status")
        assert job_desc.has_key("parent")
        assert job_desc.has_key("dependencies")
        assert job_desc.has_key("cmdline")

        for (k,v) in job_desc.items():
            setattr(self, k, v)
            self.__the_keys__.append(k)
            # print k, v
        return self

    def to_dict(self):
        "returns a structure that can be serialized into a json object"
        r = {}
        for k in self.__the_keys__:
            r[k] = getattr(self, k)
        return r

    def dir(self):
        return os.path.join(self.config.config_contents['root_dir_jobs'], "%0.8d")

    def save(self):
        self.database.save_job(self)

    def __str__(self):
        s = "Job %d. Status : %s. Parent : %s. Depends on : %s.\n    cmdline : %s" % (self.id, self.status, str(self.parent), self.dependencies, self.cmdline)
        return s



def filter_runnable_jobs(L):

    done_job_id_list = [job.id for job in L if job.status == "success"]

    def can_this_job_run(job):
        for p in job.dependencies:
            if p not in done_job_id_list:
                return False
        return True

    return [job for job in L if can_this_job_run(job) and job.status == "not_run"]



