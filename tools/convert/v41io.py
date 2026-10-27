"""Readers for the files the DeepSeek-V4.1 template converter works with.

GGUF: the header (key/values and tensor infos) of an existing mx-llama.cpp GGUF, with where each tensor's data starts.
Safetensors: the header of a checkpoint shard, local (read with pread, dropped from the page cache after reading so a
long conversion does not push out the server's hot pages) or remote (HTTP range requests against the Hugging Face
resolve URL, so tensors of a checkpoint can be compared without downloading it).
"""

import json
import os
import struct
import time
import urllib.request

GGUF_TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16", 39: "MXFP4"}
# bytes per block and elements per block of the GGUF tensor types the template uses
GGUF_BLOCK = {0: (4, 1), 1: (2, 1), 8: (34, 32), 30: (2, 1), 39: (17, 32)}

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class _Reader:
    def __init__(self, f):
        self.f = f

    def get(self, fmt):
        fmt = "<" + fmt
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def string(self):
        return self.f.read(self.get("Q")).decode("utf-8")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.get("I"), self.get("Q")
            return [self.value(et) for _ in range(n)]
        return self.get(_SCALAR[t])


def fetch(url, headers=None, tries=6):
    """GET with retries and backoff: the CDN behind the Hugging Face resolve URLs resets connections now and then."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as resp:
                return resp.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def tensor_nbytes(dims, ggml_type):
    bsize, belems = GGUF_BLOCK[ggml_type]
    n = 1
    for d in dims:
        n *= d
    assert n % belems == 0
    return n // belems * bsize


def read_gguf(path):
    """Key/values, tensor infos (name, dims, type, offset, nbytes) and the data section start of one GGUF file."""
    with open(path, "rb") as f:
        r = _Reader(f)
        assert f.read(4) == b"GGUF", path
        version = r.get("I")
        n_tensors, n_kv = r.get("Q"), r.get("Q")
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.get("I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.get("Q") for _ in range(r.get("I"))]
            ggml_type = r.get("I")
            offset = r.get("Q")
            tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset,
                            "nbytes": tensor_nbytes(dims, ggml_type)})
        alignment = kv.get("general.alignment", 32)
        header_end = f.tell()
    data_start = (header_end + alignment - 1) // alignment * alignment
    return {"path": path, "version": version, "kv": kv, "tensors": tensors, "alignment": alignment,
            "header_end": header_end, "data_start": data_start, "size": os.path.getsize(path)}


class SafetensorsShard:
    """One safetensors shard: tensor names, dtypes and shapes, and reads of a tensor's bytes or a byte range of it."""

    def __init__(self, location):
        self.location = location
        self.remote = location.startswith("http")
        self.fd = None if self.remote else os.open(location, os.O_RDONLY)
        n = struct.unpack("<Q", self._read(0, 8))[0]
        header = json.loads(self._read(8, n))
        self.metadata = header.pop("__metadata__", None)
        self.base = 8 + n
        self.tensors = header

    def _read(self, offset, size):
        if self.remote:
            for _ in range(3):
                data = fetch(self.location, {"Range": f"bytes={offset}-{offset + size - 1}"})
                if len(data) == size:
                    return data
            raise IOError(f"short read from {self.location}")
        out = bytearray()
        while len(out) < size:
            chunk = os.pread(self.fd, min(size - len(out), 1 << 28), offset + len(out))
            if not chunk:
                raise IOError(f"short read from {self.location}")
            out += chunk
        os.posix_fadvise(self.fd, offset, size, os.POSIX_FADV_DONTNEED)
        return bytes(out)

    def info(self, name):
        t = self.tensors[name]
        return t["dtype"], t["shape"], t["data_offsets"][1] - t["data_offsets"][0]

    def read(self, name, start=0, size=None):
        """Bytes [start, start + size) of tensor name."""
        t = self.tensors[name]
        begin, end = t["data_offsets"]
        size = end - begin - start if size is None else size
        assert 0 <= start and start + size <= end - begin, (name, start, size)
        return self._read(self.base + begin + start, size)


class Checkpoint:
    """A Hugging Face checkpoint by its weight map: a local directory, or a repo id with a revision read remotely."""

    def __init__(self, source, revision="main"):
        self.local = os.path.isdir(source)
        if self.local:
            self.root = source
            index = json.load(open(os.path.join(source, "model.safetensors.index.json")))
        else:
            self.root = f"https://huggingface.co/{source}/resolve/{revision}"
            index = json.loads(fetch(self.root + "/model.safetensors.index.json"))
        self.weight_map = index["weight_map"]
        self._shards = {}

    def shard(self, name):
        file = self.weight_map[name]
        if file not in self._shards:
            self._shards[file] = SafetensorsShard(os.path.join(self.root, file) if self.local else f"{self.root}/{file}")
        return self._shards[file]

    def info(self, name):
        return self.shard(name).info(name)

    def read(self, name, start=0, size=None):
        return self.shard(name).read(name, start, size)
