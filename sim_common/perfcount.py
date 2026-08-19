"""Linux hardware performance counters."""

from __future__ import annotations

import ctypes
import os
import platform
import sys

__all__ = ["InstructionCount"]

_PERF_TYPE_HARDWARE = 0
_PERF_COUNT_HW_INSTRUCTIONS = 1
_PERF_EVENT_IOC_ENABLE = 0x2400
_PERF_EVENT_IOC_DISABLE = 0x2401
_NR_PERF_EVENT_OPEN_X86_64 = 298
_SUPPORTED = sys.platform == "linux" and platform.machine() == "x86_64"


class _PerfEventAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("disabled", ctypes.c_uint64, 1),
        ("inherit", ctypes.c_uint64, 1),
        ("pinned", ctypes.c_uint64, 1),
        ("exclusive", ctypes.c_uint64, 1),
        ("exclude_user", ctypes.c_uint64, 1),
        ("exclude_kernel", ctypes.c_uint64, 1),
        ("exclude_hv", ctypes.c_uint64, 1),
        ("_reserved_flags", ctypes.c_uint64, 57),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("bp_addr", ctypes.c_uint64),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True) if _SUPPORTED else None
if _LIBC is not None:
    _LIBC.syscall.restype = ctypes.c_long
    _LIBC.ioctl.restype = ctypes.c_int


class InstructionCount:
    """Hardware instructions retired between ``__enter__`` and ``__exit__``.

    Uses ``perf_event_open(PERF_TYPE_HARDWARE,
    PERF_COUNT_HW_INSTRUCTIONS)`` via ctypes. ``available()`` returns false on
    non-Linux, on paranoid-kernel hosts, or when the syscall fails.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._count: int | None = None

    @staticmethod
    def _open() -> int:
        assert _LIBC is not None
        attr = _PerfEventAttr(
            type=_PERF_TYPE_HARDWARE,
            size=ctypes.sizeof(_PerfEventAttr),
            config=_PERF_COUNT_HW_INSTRUCTIONS,
            disabled=1,
            exclude_kernel=1,
            exclude_hv=1,
        )
        fd = _LIBC.syscall(
            _NR_PERF_EVENT_OPEN_X86_64,
            ctypes.byref(attr),
            0,
            -1,
            -1,
            0,
        )
        if fd == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return int(fd)

    @staticmethod
    def _ioctl(fd: int, request: int) -> None:
        assert _LIBC is not None
        if _LIBC.ioctl(fd, request, 0) == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    @staticmethod
    def available() -> bool:
        if not _SUPPORTED:
            return False
        try:
            fd = InstructionCount._open()
        except OSError:
            return False
        os.close(fd)
        return True

    def __enter__(self) -> InstructionCount:
        assert self._fd is None
        self._count = None
        self._fd = self._open()
        try:
            self._ioctl(self._fd, _PERF_EVENT_IOC_ENABLE)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self._fd is not None
        fd = self._fd
        try:
            self._ioctl(fd, _PERF_EVENT_IOC_DISABLE)
            data = os.read(fd, 8)
            assert len(data) == 8
            self._count = int.from_bytes(data, byteorder=sys.byteorder)
        finally:
            os.close(fd)
            self._fd = None

    @property
    def count(self) -> int:
        assert self._count is not None
        return self._count
