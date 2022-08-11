import sys, os
from contextlib import contextmanager

# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/14797594#14797594
# https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262

@contextmanager
def no_output():
    sys.stdout.flush()
    _origstdout = sys.stdout
    _oldstdout_fno = os.dup(sys.stdout.fileno())
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _newstdout = os.dup(1)
    os.dup2(_devnull, 1)
    os.close(_devnull)
    sys.stdout = os.fdopen(_newstdout, 'w')
    yield
    sys.stdout.close()
    sys.stdout = _origstdout
    sys.stdout.flush()
    os.dup2(_oldstdout_fno, 1)
    os.close(_oldstdout_fno)  # Added