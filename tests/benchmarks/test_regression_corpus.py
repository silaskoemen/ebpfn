import io
import os
import pickle

import numpy as np
import pytest
from benchmarks.data.regression_corpus import _encode, _load_npy, _subsample


def _npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=True)
    return buffer.getvalue()


def test_load_npy_reads_numeric_and_object_arrays() -> None:
    numeric = np.arange(6, dtype=np.float64).reshape(3, 2)
    np.testing.assert_array_equal(_load_npy(_npy(numeric)), numeric)
    labels = np.array([["a", "b"], ["c", None]], dtype=object)
    np.testing.assert_array_equal(_load_npy(_npy(labels)), labels)


def test_load_npy_refuses_foreign_callables() -> None:
    class Exploit:
        def __reduce__(self) -> tuple:
            return (os.system, ("true",))

    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "|O", "fortran_order": False, "shape": (1,)})
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        _load_npy(header.getvalue() + pickle.dumps(Exploit()))


def test_encode_fills_numeric_gaps_and_codes_categoricals() -> None:
    numeric = np.array([[1.0], [np.nan], [3.0]])
    categorical = np.array([["x"], [None], ["x"]], dtype=object)
    np.testing.assert_array_equal(_encode(numeric, categorical), [[1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])


def test_subsample_at_a_smaller_cap_is_a_prefix_of_a_larger_one() -> None:
    x = np.arange(200, dtype=np.float64).reshape(100, 2)
    y = np.arange(100, dtype=np.float64)
    y[3] = np.nan
    small_x, small_y = _subsample(x, y, "src/name", 10)
    large_x, large_y = _subsample(x, y, "src/name", 40)
    np.testing.assert_array_equal(large_x[:10], small_x)
    np.testing.assert_array_equal(large_y[:10], small_y)
    assert np.isfinite(large_y).all()
