

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