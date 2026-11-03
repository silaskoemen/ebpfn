"""The extended regression corpus: TabArena and TALENT regression datasets, fetched one at a time.

Studies score at most ``rows`` rows per dataset, so nothing larger is kept: each source
dataset is fetched on demand, a fixed random subsample is written to ``CORPUS_DIR`` as a
small parquet, and the raw download is discarded. The subsample is a prefix of one seeded
permutation, so a slice built at a larger cap truncates to the slice a smaller cap builds. TabArena comes from its OpenML suite;
TALENT's regression datasets are read member by member out of its published archive with
HTTP range requests, so the archive itself is never downloaded.

Features are stored as floats -- categoricals as ordinal codes (missing gets its own
code), numeric gaps as the column mean -- so every model in a study sees identical inputs.

Build from project root (idempotent; slices already holding ``--rows`` rows are skipped):
    pixi run python -m benchmarks.data.regression_corpus [--rows 2048]
"""

import argparse
import io
import json
import pickle
import shutil
import tempfile
import urllib.request
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

CORPUS_DIR = Path("data/processed/corpus")
SEED = 0
ROWS = 2048  # default cap; realistic sizes, still quick to score on a laptop
MIN_ROWS = 256
MAX_FEATURES = 300
TABARENA_SUITE = "tabarena-v0.1"
TALENT_ARCHIVE = (
    "https://drive.usercontent.google.com/download?id=1Rr2uZsZzabshoa0KX0uCne-qMWFpbyAo&export=download&confirm=t"
)
TALENT_SPLITS = ("train", "val", "test")
# Slices that repeat an earlier dataset (original corpus > TabArena > TALENT), matched on
# target scale, deciles and row count: same data, a row subset of it, or another target
# transform of it. Kept on disk, left out of the corpus.
DUPLICATES = frozenset(
    {
        "tabarena/airfoil_self_noise",
        "talent/airfoil_self_noise",
        "tabarena/concrete_compressive_strength",
        "talent/concrete_compressive_strength",
        "tabarena/physiochemical_protein",
        "talent/Physicochemical_r",
        "tabarena/superconductivity",
        "talent/Superconductivty",
        "talent/kin8nm",
        "talent/Another-Dataset-on-used-Fiat-500-(1538-rows)",
        "talent/Diamonds",
        "talent/Food_Delivery_Time",
        "talent/healthcare_insurance_expenses",
        "talent/MiamiHousing2016",
        "talent/qsar_fish_toxicity",
        "talent/houses",
        "talent/wine+quality",
        "talent/Wine_Quality_red",
        "talent/Wine_Quality_white",
        "talent/volume",  # Facebook_Comment_Volume under another target window
    }
)
_ZIP_BUFFER = 1 << 20
# The only callables a TALENT object array may reference: numpy's array reconstruction.
_SAFE_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer: "WriteableBuffer", /) -> int:
        buffer = memoryview(buffer).cast("B")
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if (module, name) not in _SAFE_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"refusing {module}.{name} in a downloaded array")
        return super().find_class(module, name)


def _load_npy(raw: bytes) -> np.ndarray:
    """``np.load`` for an untrusted ``.npy``: object arrays may only rebuild numpy arrays."""
    buffer = io.BytesIO(raw)
    version = np.lib.format.read_magic(buffer)
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _, dtype = read_header(buffer)
    if not dtype.hasobject:
        buffer.seek(0)
        return np.load(buffer, allow_pickle=False)
    array = _ArrayUnpickler(buffer).load()
    if not isinstance(array, np.ndarray) or array.shape != shape:
        raise ValueError(f"object array does not match its header {shape}")
    return array


def _encode(numeric: np.ndarray, categorical: np.ndarray) -> np.ndarray:
    """Float feature matrix: numeric gaps as the column mean, categoricals as ordinal codes."""
    numeric = numeric.astype(np.float64)
    if numeric.size:
        means = np.nanmean(np.where(np.isfinite(numeric), numeric, np.nan), axis=0)
        numeric = np.where(np.isfinite(numeric), numeric, np.nan_to_num(means))
    codes = []
    for column in categorical.T:
        labels = np.array(["" if value is None or value != value else str(value) for value in column])
        _, code = np.unique(labels, return_inverse=True)
        codes.append(code.astype(np.float64))
    return np.column_stack([numeric, *codes]) if codes else numeric


def _subsample(x: np.ndarray, y: np.ndarray, key: str, rows: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    index = np.random.default_rng([SEED, zlib.crc32(key.encode())]).permutation(len(y))[:rows]
    return x[index], y[index]


def iter_tabarena() -> Iterator[tuple[str, dict, np.ndarray, np.ndarray]]:
    """``(name, metadata, x, y)`` per TabArena regression task, each fetched into a throwaway cache."""
    import openml
    import pandas as pd

    for task_id in openml.study.get_suite(TABARENA_SUITE).tasks or []:
        cache = Path(tempfile.mkdtemp(prefix="openml-"))
        try:
            openml.config.set_root_cache_directory(str(cache))
            task = openml.tasks.get_task(task_id, download_splits=False)
            if not isinstance(task, openml.tasks.OpenMLRegressionTask):
                continue
            dataset = task.get_dataset()
            features, target, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            assert isinstance(features, pd.DataFrame)
            assert target is not None
            categorical = [c for c in features.columns if str(features[c].dtype) in ("category", "object", "bool")]
            numeric = [c for c in features.columns if c not in categorical]
            x = _encode(
                features[numeric].to_numpy(dtype=np.float64, na_value=np.nan),
                features[categorical].to_numpy(dtype=object),
            )
            meta = {
                "task_id": int(task_id),
                "dataset_id": int(task.dataset_id),
                "n_total": len(target),
                "n_categorical": len(categorical),
            }
            yield str(dataset.name), meta, x, np.asarray(target, dtype=np.float64)
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def iter_talent() -> Iterator[tuple[str, dict, np.ndarray, np.ndarray]]:
    """``(name, metadata, x, y)`` per TALENT regression dataset, read straight out of the remote archive."""
    archive = zipfile.ZipFile(io.BufferedReader(_RemoteFile(TALENT_ARCHIVE), buffer_size=_ZIP_BUFFER))
    members = set(archive.namelist())
    for info_path in sorted(m for m in members if m.endswith("/info.json")):
        info = json.loads(archive.read(info_path))
        if info["task_type"] != "regression":
            continue
        folder = info_path.removesuffix("info.json")

        def splits(prefix: str, folder: str = folder) -> list[np.ndarray]:
            return [
                _load_npy(archive.read(f"{folder}{prefix}_{s}.npy"))
                for s in TALENT_SPLITS
                if f"{folder}{prefix}_{s}.npy" in members
            ]

        y = np.concatenate(splits("y")).astype(np.float64)
        if y.ndim > 1:  # some targets are stored as one column
            if y.shape[1] != 1:
                raise ValueError(f"{folder}: expected one target, got {y.shape[1]}")
            y = y[:, 0]
        numeric = np.concatenate(splits("N")) if info["n_num_features"] else np.empty((len(y), 0))
        categorical = np.concatenate(splits("C")) if info["n_cat_features"] else np.empty((len(y), 0), dtype=object)
        meta = {"source_url": info.get("source"), "n_total": len(y), "n_categorical": int(info["n_cat_features"])}
        yield folder.rstrip("/").split("/")[-1], meta, _encode(numeric, categorical), y


def slice_path(source: str, name: str) -> Path:
    return CORPUS_DIR / f"{source}__{name.replace('/', '_')}.parquet"


def build(rows: int = ROWS) -> None:
    """Write every eligible dataset's subsample to ``CORPUS_DIR`` (skips slices already holding ``rows``)."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CORPUS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for source, datasets in (("tabarena", iter_tabarena), ("talent", iter_talent)):
        for name, meta, x, y in datasets():
            key = f"{source}/{name}"
            if x.shape[1] > MAX_FEATURES or np.isfinite(y).sum() < MIN_ROWS:
                manifest[key] = meta | {"skipped": f"{x.shape[1]} features, {int(np.isfinite(y).sum())} rows"}
                print(f"skip {key}: {manifest[key]['skipped']}", flush=True)
                continue
            path = slice_path(source, name)
            target = min(rows, int(np.isfinite(y).sum()))
            if not path.exists() or manifest.get(key, {}).get("rows", 0) < target:
                xs, ys = _subsample(x, y, key, rows)
                pl.DataFrame({**{f"x{j}": xs[:, j] for j in range(xs.shape[1])}, "y": ys}).write_parquet(path)
                print(f"wrote {key}: {xs.shape}", flush=True)
            manifest[key] = meta | {"rows": max(target, manifest.get(key, {}).get("rows", 0)), "features": x.shape[1]}
            manifest_path.write_text(json.dumps(manifest, indent=1))


def load_corpus(rows: int = ROWS) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(source/name, x, y)`` for every slice in ``CORPUS_DIR`` that is not a ``DUPLICATES`` entry, cut to ``rows``."""
    datasets = []
    for path in sorted(CORPUS_DIR.glob("*__*.parquet")):
        source, name = path.stem.split("__", 1)
        if f"{source}/{name}" in DUPLICATES:
            continue
        frame = pl.read_parquet(path).head(rows)
        datasets.append(
            (
                f"{source}/{name}",
                frame.drop("y").to_numpy().astype(np.float64),
                frame["y"].to_numpy().astype(np.float64),
            )
        )
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=ROWS, help="rows kept per dataset")
    build(parser.parse_args().rows)
