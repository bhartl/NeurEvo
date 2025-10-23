# --- Optional MPI import with graceful fallback ---
from typing import TYPE_CHECKING
import os

_FORCE_NO_MPI = os.getenv("MINDCRAFT_DISABLE_MPI", "0") == "1"
_HAVE_MPI = False

try:
    if _FORCE_NO_MPI:
        raise ImportError("MPI disabled via MINDCRAFT_DISABLE_MPI=1")

    from mpi4py import MPI  # type: ignore
    _HAVE_MPI = True

except Exception:
    class _Op:
        SUM = "SUM"
        MAX = "MAX"
        MIN = "MIN"
        PROD = "PROD"

    class _DummyComm:
        """Minimal stand-in for mpi4py.MPI.Comm used in ES.
        Provides the attributes/methods this module relies on.
        """
        def __init__(self):
            self._rank = 0
            self._size = 1

        # attribute-style access (es_method.init_MPI reads .size/.rank)
        @property
        def rank(self):  # noqa: N802 (match mpi4py property name used in file)
            return self._rank

        @property
        def size(self):
            return self._size

        # method-style access (top-level used Get_rank/Get_size originally)
        def Get_rank(self):
            return self._rank

        def Get_size(self):
            return self._size

        # collectives used in this file (safe no-ops for single process)
        def bcast(self, x, root=0):
            return x

        def gather(self, x, root=0):
            return [x]

        def scatter(self, xs, root=0):
            if isinstance(xs, (list, tuple)) and len(xs) > 0:
                return xs[0]
            return xs

        def Barrier(self):
            pass

        def allreduce(self, x, op=None):
            return x

        def reduce(self, x, op=None, root=0):
            return x

        # common helper; harmless here
        def split(self, color=0, key=0):
            return self

       # Default handler for missing MPI methods
        def __getattr__(self, name):
            raise AttributeError(
                f"MPI method '{name}' was called, but mpi4py is not available "
                f"and the dummy class doesn't implement the method-call.\n"
                "→ Either install mpi4py (and a system MPI such as OpenMPI or MPICH), "
                "or disable distributed features in your script."
            )

    class _DummyMPI:
        """Expose a minimal subset of the mpi4py.MPI API."""
        COMM_WORLD = _DummyComm()
        Op = _Op
        SUM = _Op.SUM
        MAX = _Op.MAX
        MIN = _Op.MIN
        PROD = _Op.PROD

    MPI = _DummyMPI()

    if not _FORCE_NO_MPI:
        import warnings
        warnings.warn(
            "mindcraft.util.mpi: mpi4py not found — running in single-process (serial) mode.",
            category=RuntimeWarning,
            stacklevel=2,
        )

# Expose class-level defaults (original file expects these names)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
is_root = (rank == 0)

if TYPE_CHECKING:
    from mpi4py import MPI as _T_MPI  # noqa: F401

__all__ = ["MPI", "comm", "rank", "size", "is_root", "_HAVE_MPI"]
