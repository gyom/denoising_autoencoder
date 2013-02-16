

def conj(E, o):
    if type(E) == list:
        return E + [o]
    elif type(E) == dict:
        return dict(E.items() + [o])
    else:
        #dict(E.items() + [o])
        raise BaseException("Unsure how to process the arguments given to conj. type was %s" % str(type(E)),)
