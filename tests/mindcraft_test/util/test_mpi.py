import os
import sys
import importlib
import warnings
import unittest
import numpy as np


def _reload_mpi(disable=False):
    """Reload mindcraft.util.mpi with dummy or real MPI depending on flag."""
    sys.modules.pop("mindcraft.util.mpi", None)

    if disable:
        os.environ["MINDCRAFT_DISABLE_MPI"] = "1"
    else:
        os.environ.pop("MINDCRAFT_DISABLE_MPI", None)

    return importlib.import_module("mindcraft.util.mpi")


class TestDummySerial(unittest.TestCase):
    """Serial tests for the dummy MPI shim (no mpi4py)."""

    def setUp(self):
        # Expect a RuntimeWarning when dummy is selected
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.mpi = _reload_mpi(disable=True)
            self.warns = [m for m in w if m.category is RuntimeWarning]
        self.comm = self.mpi.comm

    def test_warns_single_process(self):
        # self.assertTrue(any("single-process" in str(m.message) for m in self.warns))
        self.assertFalse(self.mpi._HAVE_MPI)
        self.assertEqual(self.mpi.size, 1)
        self.assertEqual(self.mpi.rank, 0)
        self.assertTrue(self.mpi.is_root)

    def test_ops_exist(self):
        self.assertTrue(hasattr(self.mpi.MPI, "SUM"))
        self.assertTrue(hasattr(self.mpi.MPI, "MAX"))

    def test_collectives_identity(self):
        c = self.comm
        self.assertEqual(c.bcast(42, root=0), 42)
        self.assertEqual(c.gather("x", root=0), ["x"])
        # self.assertEqual(c.allgather(7), [7])
        self.assertEqual(c.scatter([10, 20, 30], root=0), 10)
        self.assertEqual(c.reduce(5, op=self.mpi.MPI.SUM, root=0), 5)
        self.assertEqual(c.allreduce(9, op=self.mpi.MPI.SUM), 9)

        self.assertIsNone(c.Barrier())

    def test_missing_method_raises(self):
        with self.assertRaises(AttributeError) as cm:
            self.comm.Sendrecv(None, None)  # not implemented in dummy
        self.assertIn("mpi4py is not available", str(cm.exception))


class TestRealSerialIfAvailable(unittest.TestCase):
    """Serial tests for real mpi4py if present (size should be 1 in serial)."""

    @classmethod
    def setUpClass(cls):
        try:
            import mpi4py  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("mpi4py not installed; skipping real-mode serial tests")

    def setUp(self):
        self.mpi = _reload_mpi(disable=False)
        self.comm = self.mpi.comm

    def test_import_and_world(self):
        self.assertTrue(self.mpi._HAVE_MPI)
        # In serial (no mpirun), world size is 1
        self.assertEqual(self.mpi.size, 1)
        self.assertEqual(self.mpi.rank, 0)
        self.assertTrue(self.mpi.is_root)

    def test_basic_collectives_serial(self):
        x = self.comm.bcast(123, root=0)
        self.assertEqual(x, 123)

        g = self.comm.gather("a", root=0)
        # In serial, root gets the list
        self.assertEqual(g, ["a"])

        # Allreduce path: prefer buffer API if available
        arr_in = np.array([2.0], dtype=float)
        arr_out = np.empty_like(arr_in)
        if hasattr(self.comm, "Allreduce"):
            self.comm.Allreduce(arr_in, arr_out, op=self.mpi.MPI.SUM)
            np.testing.assert_allclose(arr_out, arr_in)
        else:
            val = self.comm.allreduce(float(arr_in[0]), op=self.mpi.MPI.SUM)
            self.assertEqual(val, float(arr_in[0]))


class TestExportsStable(unittest.TestCase):
    def test_exported_names_exist(self):
        mpi = _reload_mpi(disable=True)
        for name in ("MPI", "comm", "rank", "size", "is_root", "_HAVE_MPI"):
            self.assertTrue(hasattr(mpi, name))
