import theano

def set_theano_fast_compile():
    theano.config.mode = 'FAST_COMPILE'


def set_theano_fast_run():
    theano.config.mode = 'FAST_RUN'
