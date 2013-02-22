#!/bin/env python

import numpy as np
import sys, os, time
import subprocess

import redis

def usage():
    print "-- python run_commandline_from_redis.py --server=localhost --port=6379 --queue=some_job_02032 --output_queue=whatever"
    print ""

def main(argv):
    """
    This function wants to be as agnostic as possible concerning
    the command-line script run.

    The command-line should be found in a list of elements.
    The name of that list is given by the 'queue' argument.

    You can also use an output queue if you want to store the output
    of the commands run. Naturally, this is not expected to be the
    way to store the results. The command-line called should be doing
    some other form of logging.

    Populating the redis database should be done elsewhere.
    """

    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server=", "port=", "queue=", "output_queue="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    redis_server = None
    redis_port = None
    redis_queue_key = None
    redis_output_queue_key = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--server"):
            redis_server = a
        elif o in ("--port"):
            redis_port = int(a)
        elif o in ("--queue"):
            redis_queue_key = a
        elif o in ("--output_queue"):
            redis_output_queue_key = a
        else:
            assert False, "unhandled option"
 
    assert redis_server
    assert redis_port
    assert redis_queue_key

    r_server = redis.Redis(redis_server, redis_port)

    if not r_server.ping():
        print "Failed to ping the server. Exiting."
        quit()

    print "We have %d jobs in the queue." % r_server.llen(redis_queue_key)

    command = r_server.lpop(redis_queue_key)
    if command == None:
        print "No jobs in queue. Exiting."
        quit()
    else:
        print "The command to be executed is \n    %s\n" % (command,)

    print "=============================="
    # You need the "shell=True" here because you rely
    # on the shell to tell the arguments apart.
    # You can't use split() here without ruining some
    # arguments using spaces in quotes.
    command_output = subprocess.check_output(command, shell=True)
    print command_output
    print "=============================="

    if redis_output_queue_key:
        r_server.rpush(redis_output_queue_key, command_output)

    print "Done."


if __name__ == "__main__":
    main(sys.argv)