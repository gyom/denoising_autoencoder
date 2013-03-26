
import cluster
import cluster.job
from cluster.job import Job

import json

class Database():
    def __init__(self, config):
        self.config = config
        assert self.config is not None
        #print "Database __init__ with config being "
        #print config
        
    def get_next_free_job_id(self):
        return self.config.redis_server.incr(self.config.config_contents['jobs_id_counter'])

    def get_all_job_ids(self):
        i = 0
        #N = self.config.redis_server.llen(self.config.config_contents['jobs_list'])
        N = self.config.redis_server.scard(self.config.config_contents['jobs_list'])
        if N == 0:
            return []
        else:
            #return self.config.redis_server.lrange(self.config.config_contents['jobs_list'], 0, N-1)
            return self.config.redis_server.srandmember(self.config.config_contents['jobs_list'], N)

    def get_all_jobs(self):
        return [self.load_job(job_id) for job_id in self.get_all_job_ids()]

    def load_job(self, job_id, want_job_object = True):
        if want_job_object:
            contents = self.config.redis_server.get(job_id)
            if contents is not None:
                return Job().from_dict(json.loads(contents))
            else:
                return None
        else:
            contents = self.config.redis_server.get(job_id)
            if contents is not None:
                return json.loads(contents)
            else:
                return None

    def save_job(self, job):
        self.config.redis_server.set(job.id, json.dumps(job.to_dict()))
        self.config.redis_server.sadd(self.config.config_contents['jobs_list'], job.id)
        print "saved job %d to database" % job.id
        