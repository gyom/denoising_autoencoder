

def conj(E, o):
    if type(E) == list:
        return E + [o]
    elif type(E) == dict:
        return dict(E.items() + [o])
    else:
        #dict(E.items() + [o])
        raise BaseException("Unsure how to process the arguments given to conj. type was %s" % str(type(E)),)


def get_dict_key_or_default(D, key, default, want_error_if_missing = False):
    if D.has_key(key):
        return D[key]
    else:
        if not want_error_if_missing:
            return default
        else:
            raise("Cannot find key %s in dictionary." % (key,))


import sys, time
def make_progress_logger(prefix):

    start_time = time.time()
    previous_time = start_time
    def f(p):
        current_time = time.time()
        # Don't write anything if you're calling this too fast.
        if (int(current_time - f.previous_time) > 5) and p > 0.0:

            finish_time = f.start_time + (current_time - f.start_time) / p
            time_left = finish_time - current_time

            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write("%s progress : %d %% . Estimated time left : %d secs." % (prefix, int(100*p), time_left))
            sys.stdout.flush()

            f.previous_time = current_time

    f.start_time = start_time
    f.previous_time = previous_time

    return f
