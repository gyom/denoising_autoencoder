
import os
import re
import redis

class Configuration():
    def __init__(self):
        self.config_file = os.environ["UMONTREAL_CLUSTER_CONFIG"]
        assert(os.path.exists(self.config_file))
        if re.match(r"^.+\.json$", self.config_file):
            import json
            self.config_contents = json.load(open(self.config_file, "r"))
        else:
            raise("Filed to identify the kind of resource file that we are using for UMONTREAL_CLUSTER_CONFIG")

        self.process_config()
        self.validate_config()
        # print "Done constructing Configuration object."

    def process_config(self):
        assert type(self.config_contents) == type({})

        assert self.config_contents.has_key('redis_server_host')
        if not self.config_contents.has_key('redis_server_port'):
            self.config_contents['redis_server_port'] = 6379
        if not self.config_contents.has_key('redis_server_password'):
            self.config_contents['redis_server_password'] = None


        self.redis_server = redis.Redis(self.config_contents['redis_server_host'], port = self.config_contents['redis_server_port'], password = self.config_contents['redis_server_password'])


    def validate_config(self):
        assert self.config_contents.has_key('root_dir_jobs')
        assert os.path.exists(self.config_contents['root_dir_jobs'])
        assert os.path.isdir(self.config_contents['root_dir_jobs'])

        assert self.config_contents.has_key('jobs_id_counter')
        assert self.config_contents.has_key('jobs_list')

        if not self.redis_server.ping():
            print "Cannot ping server. Exiting."
            quit()




