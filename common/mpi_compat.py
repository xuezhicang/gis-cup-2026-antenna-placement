"""Use mpi4py when it is installed; otherwise run as a single process.

Every step script does ``from common.mpi_compat import MPI`` instead of
``from mpi4py import MPI``.  With mpi4py installed nothing changes and the
scripts can be started with ``mpirun -np N``.  Without it, a minimal
one-process stand-in provides the collective calls the scripts use, so
``workers = 1`` needs no MPI installation at all.
"""

from __future__ import annotations

import sys
import time

try:
    from mpi4py import MPI  # noqa: F401
except ImportError:

    class _SingleProcessComm:
        """The subset of ``MPI.Comm`` used by this project, for one process."""

        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 1

        def Barrier(self) -> None:
            return None

        def bcast(self, obj, root: int = 0):
            return obj

        def gather(self, obj, root: int = 0):
            return [obj]

        def allgather(self, obj):
            return [obj]

        def reduce(self, obj, op=None, root: int = 0):
            return obj

        def allreduce(self, obj, op=None):
            return obj

        def send(self, *args, **kwargs):
            raise RuntimeError("point-to-point messages need mpi4py and more than one process")

        def recv(self, *args, **kwargs):
            raise RuntimeError("point-to-point messages need mpi4py and more than one process")

        def Abort(self, errorcode: int = 1) -> None:
            sys.exit(errorcode)

    class _Status:
        def Get_source(self) -> int:
            return 0

        def Get_tag(self) -> int:
            return 0

    class _SingleProcessMPI:
        COMM_WORLD = _SingleProcessComm()
        SUM = "sum"
        MAX = "max"
        LOR = "lor"
        ANY_SOURCE = -1
        ANY_TAG = -1
        Status = _Status

        @staticmethod
        def Wtime() -> float:
            return time.perf_counter()

    MPI = _SingleProcessMPI()

__all__ = ["MPI"]
