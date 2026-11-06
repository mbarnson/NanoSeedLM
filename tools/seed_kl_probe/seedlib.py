# seedlib.py - ctypes wrapper for libseed.so (probe).
import ctypes, os, numpy as np
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libseed.so"))
_lib.seed_setup.restype = ctypes.c_uint32
_lib.seed_setup.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_f = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_u = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
_i = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_lib.seed_search.argtypes = [_f, _f, ctypes.c_int, _f, _u, _i, _f]
def setup(P, K, stride=1):
    t = _lib.seed_setup(P, K, stride)
    if not t: raise RuntimeError("no primitive taps")
    return t
def search(A, w):
    """A: 8x8 lower triangular; w: [nb, 8]. Returns decoded weights [nb, 8], seeds, exponents, |A(w - wd)|^2."""
    A = np.ascontiguousarray(A, np.float32).reshape(64); w = np.ascontiguousarray(w, np.float32)
    nb = w.shape[0]
    wd = np.zeros((nb, 8), np.float32); s = np.zeros(nb, np.uint32); e = np.zeros(nb, np.int32); er = np.zeros(nb, np.float32)
    _lib.seed_search(A, w, nb, wd, s, e, er)
    return wd, s, e, er
