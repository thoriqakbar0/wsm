"""Atomic checkpoint publication without replacing an existing destination."""
import ctypes
import errno
import os

from .errors import WeightlessError


def _publish(source, destination):
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        raise WeightlessError("unsupported_platform", "Atomic exclusive export requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise WeightlessError("export_exists", "Destination already exists")
        raise OSError(code, os.strerror(code), str(destination))
