"""Routed experts streamed from SSD into preallocated slabs in unified memory.

Each expert is one page-aligned record on disk (format.ExpertLayout). For every distinct layout
the store owns one *slab*: nine MLX arrays [slots, ...] (w1/w3/w2 x packed/scales/biases),
allocated once. A miss takes an LRU slot and a single `preadv` on an F_NOCACHE descriptor scatters
the record straight into that slot of the nine arrays -- unified memory makes MLX allocations
directly writable, so there is no staging copy, no per-miss allocation and no MLX call off the
compute thread (MLX streams are per-thread). The model then runs all routed experts of a layer
with one `gather_qmm` per projection over slot indices.

The OS page cache is bypassed (F_NOCACHE): it would hold a second copy of experts we already hold,
which a 24 GB machine cannot afford. Measured on an M4 MacBook Air: 2.8-3.15 GB/s.

Slot safety: slots served by the last two `get_slots` calls stay pinned, because the compute that
reads them is lazy until the next routing sync; eviction never hands those out.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx

from . import format as F
from .model.model import ExpertWeights

F_NOCACHE = 48
_DT = {"uint32": mx.uint32, "bfloat16": mx.uint16}  # bf16 slabs are stored as uint16 bits


class Slab:
    def __init__(self, layout: F.ExpertLayout, slots: int):
        self.layout, self.slots = layout, slots
        self.parts = {name: mx.zeros((slots, *shape), dtype=_DT[dtype]) for name, shape, dtype, _, _ in layout.parts()}
        mx.eval(list(self.parts.values()))
        self.nbytes = {name: n for name, _, _, _, n in layout.parts()}
        self.order = [name for name, *_ in layout.parts()]
        self.mem = {name: memoryview(a).cast("B") for name, a in self.parts.items()}
        self.bf16 = {name: a.view(mx.bfloat16) for name, a in self.parts.items() if a.dtype == mx.uint16}
        mx.eval(list(self.bf16.values()))  # evaluated views share the slab's memory (SSD writes stay visible)
        self.lru: OrderedDict[tuple[str, int], int] = OrderedDict()  # (key, expert) -> slot
        self.free = list(range(slots))
        self.pins: list[set[int]] = [set(), set()]
        self.busy: set[int] = set()  # slots of a get_slots call in progress (speculation runs concurrently)

    def iovecs(self, slot: int) -> list[memoryview]:
        return [self.mem[n][slot * self.nbytes[n] : (slot + 1) * self.nbytes[n]] for n in self.order]

    def proj(self, w: str):
        """(packed, scales, biases, bits) of projection w1/w3/w2 across all slots."""
        bits = {"w1": self.layout.bits_w1, "w3": self.layout.bits_w3, "w2": self.layout.bits_w2}[w]
        return self.parts[f"{w}.q"], self.bf16[f"{w}.s"], self.bf16[f"{w}.b"], bits


class _Cancelled(Exception):
    pass


class _Req:
    """One expert record read, split into chunks; Future-like (done/result) plus promote()."""

    __slots__ = ("io", "pending", "left", "error", "event", "high")

    def __init__(self, io, n: int, high: bool):
        self.io, self.left, self.error, self.high = io, n, None, high
        self.pending: list = []
        self.event = threading.Event()

    def done(self) -> bool:
        return self.event.is_set()

    def result(self):
        if not self.event.is_set():
            if not self.high:
                self.io.promote(self)
            self.event.wait()
        if self.error is not None:
            raise self.error
        return None

    def _chunk_done(self, exc=None):
        if exc is not None and self.error is None:
            self.error = exc
        self.left -= 1
        if self.left == 0:
            self.event.set()


class PrioIO:
    """Chunked two-priority SSD read scheduler. Demand chunks are always taken before speculative
    ones, and at most SPEC_CONC speculative chunks are on the SSD at once, so a demand read never
    queues behind a backlog of guesses; a guess that turns out to be needed is promoted."""

    CHUNK = int(os.environ.get("DEEPINFER_IO_CHUNK", str(2 << 20)))
    SPEC_CONC = int(os.environ.get("DEEPINFER_SPEC_CONC", "4"))

    def __init__(self, threads: int):
        self.cv = threading.Condition()
        self.hi: deque = deque()
        self.lo: deque = deque()
        self.active_lo = 0
        self.workers = [threading.Thread(target=self._work, name=f"expert-io-{i}", daemon=True) for i in range(threads)]
        for t in self.workers:
            t.start()

    def submit(self, fd: int, bufs: list, off: int, total: int, high: bool) -> _Req:
        cuts = list(range(0, total, self.CHUNK)) + [total]
        cuts = sorted({min(total, (c // F.ALIGN) * F.ALIGN) for c in cuts} | {total})
        req = _Req(self, len(cuts) - 1, high)
        jobs = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            sel, pos = [], 0
            for v in bufs:
                lo, hi = max(a, pos), min(b, pos + len(v))
                if lo < hi:
                    sel.append(v[lo - pos : hi - pos])
                pos += len(v)
            jobs.append((req, fd, sel, off + a, b - a))
        with self.cv:
            if high:
                self.hi.extend(jobs)
            else:
                self.lo.extend(jobs)
                req.pending = jobs
            self.cv.notify_all()
        return req

    def promote(self, req: _Req):
        with self.cv:
            if req.high:
                return
            req.high = True
            moved = [j for j in self.lo if j[0] is req]
            if moved:
                self.lo = deque(j for j in self.lo if j[0] is not req)
                self.hi.extendleft(reversed(moved))
                self.cv.notify_all()

    def cancel(self, req: _Req) -> bool:
        """Drop a speculative read none of whose chunks has started. False if any chunk is (or was)
        on the SSD -- the caller then lets it finish (a running chunk may still write its slot)."""
        with self.cv:
            if req.high or req.event.is_set():
                return False
            pend = [j for j in self.lo if j[0] is req]
            if len(pend) != len(req.pending):
                return False
            self.lo = deque(j for j in self.lo if j[0] is not req)
            req.error = _Cancelled()
            req.left = 0
            req.event.set()
            return True

    def _work(self):
        while True:
            with self.cv:
                while not self.hi and not (self.lo and self.active_lo < self.SPEC_CONC):
                    self.cv.wait()
                if self.hi:
                    job, spec = self.hi.popleft(), False
                else:
                    job, spec = self.lo.popleft(), True
                    self.active_lo += 1
            req, fd, sel, off, want = job
            exc = None
            try:
                while want:
                    n = os.preadv(fd, sel, off)
                    if n <= 0:
                        raise IOError(f"read failed at {off}")
                    off, want = off + n, want - n
                    while sel and n >= len(sel[0]):
                        n -= len(sel[0])
                        sel.pop(0)
                    if sel and n:
                        sel[0] = sel[0][n:]
            except Exception as e:  # noqa: BLE001 - reported through the request
                exc = e
            with self.cv:
                if spec:
                    self.active_lo -= 1
                    if self.lo:
                        self.cv.notify_all()
                req._chunk_done(exc)



class _NReq:
    """A native read request (mkio): Future-like done()/result(), promote via result()."""

    __slots__ = ("io", "id", "fin", "high", "gen", "__weakref__")

    def __init__(self, io, rid: int, high: bool):
        self.io, self.id, self.fin, self.high = io, rid, 0, high
        self.gen = io.lib.mkio_gen(rid) if io.tracing else -1  # unique request id, for tracing

    def done(self) -> bool:
        if self.fin == 0:
            self.fin = self.io.lib.mkio_done(self.id)
        return self.fin != 0

    @property
    def error(self):
        return None if self.fin >= 0 else IOError("read failed or cancelled")

    def wait_us(self, us: int) -> bool:
        """Wait at most `us` microseconds (GIL released); True once the request has finished."""
        if self.fin == 0:
            self.fin = self.io.lib.mkio_wait_us(self.id, us)
        return self.fin != 0

    def result(self):
        if self.fin == 0:
            self.fin = self.io.lib.mkio_wait(self.id)  # ctypes drops the GIL for the call
        if self.fin < 0:
            raise IOError("expert read failed or was cancelled")
        return None

    def __del__(self):
        try:
            if self.done():
                self.io.lib.mkio_release(self.id)
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


def _mkio_lib():
    """Path of the native IO library, built from native/mkio.c on first use (clang); None if it
    cannot be built (the Python PrioIO is used then)."""
    import subprocess

    src = Path(__file__).parent / "native" / "mkio.c"
    lib = src.with_name("libmkio.dylib")
    try:
        if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["clang", "-O2", "-shared", "-fPIC", "-o", str(lib), str(src)], check=True, capture_output=True)
        return lib
    except Exception:  # noqa: BLE001 - no compiler: fall back
        return None


class NativeIO:
    """Same contract as PrioIO, implemented in C (native/mkio.c): no Python per chunk, no GIL
    traffic from the I/O workers."""

    def __init__(self, threads: int):
        import ctypes

        lib = ctypes.CDLL(str(_mkio_lib()))
        lib.mkio_init.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
        lib.mkio_submit.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_longlong),
                                    ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
        for f in ("mkio_done", "mkio_wait", "mkio_cancel"):
            getattr(lib, f).argtypes = [ctypes.c_int]
            getattr(lib, f).restype = ctypes.c_int
        for f in ("mkio_promote", "mkio_release"):
            getattr(lib, f).argtypes = [ctypes.c_int]
            getattr(lib, f).restype = None
        lib.mkio_submit_tagged.argtypes = lib.mkio_submit.argtypes + [ctypes.c_longlong]
        lib.mkio_wait_us.argtypes, lib.mkio_wait_us.restype = [ctypes.c_int, ctypes.c_longlong], ctypes.c_int
        lib.mkio_gen.argtypes, lib.mkio_gen.restype = [ctypes.c_int], ctypes.c_long
        lib.mkio_now.argtypes, lib.mkio_now.restype = [], ctypes.c_longlong
        lib.mkio_trace_enable.argtypes, lib.mkio_trace_enable.restype = [ctypes.c_int], ctypes.c_int
        lib.mkio_trace_disable.argtypes, lib.mkio_trace_disable.restype = [], None
        for f in ("mkio_trace_chunks", "mkio_trace_reqs"):
            getattr(lib, f).argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
            getattr(lib, f).restype = ctypes.c_int
        self.lib, self.ct = lib, ctypes
        self.tracing = False
        if lib.mkio_init(threads, PrioIO.CHUNK, PrioIO.SPEC_CONC) != 0:
            raise RuntimeError("mkio_init failed")

    def submit_addrs(self, fd: int, addrs, lens, off: int, total: int, high: bool, tag: int = -1) -> _NReq:
        ct = self.ct
        n = len(addrs)
        a = (ct.c_void_p * n)(*addrs)
        l_ = (ct.c_longlong * n)(*lens)
        rid = self.lib.mkio_submit_tagged(fd, a, l_, n, off, total, 1 if high else 0, tag)
        if rid < 0:
            raise RuntimeError("mkio_submit: request table full")
        return _NReq(self, rid, high)

    # -- tracing (mac-deploy-tests/mk/trace_decode.py): chunk and request events, CLOCK_UPTIME_RAW ns
    def trace_start(self, cap: int = 1 << 18) -> None:
        if self.lib.mkio_trace_enable(cap) != 0:
            raise MemoryError("mkio_trace_enable")
        self.tracing = True

    def trace_stop(self):
        """Stop tracing; returns (chunks [n, 9]: tq t0 t1 gen tag thr lo a b, requests [m, 6]:
        t gen tag kind(0 submit, 1 promote, 2 cancel) high total) as numpy int64 arrays."""
        import numpy as _np

        self.lib.mkio_trace_disable()
        self.tracing = False
        out = []
        for f, w in (("mkio_trace_chunks", 9), ("mkio_trace_reqs", 6)):
            buf = _np.zeros((1 << 18) * w, dtype=_np.int64)
            n = getattr(self.lib, f)(buf.ctypes.data_as(self.ct.POINTER(self.ct.c_longlong)), 1 << 18)
            out.append(buf[: n * w].reshape(n, w).copy())
        return out[0], out[1]

    def promote(self, req: _NReq):
        if not req.high:
            req.high = True
            self.lib.mkio_promote(req.id)

    def cancel(self, req: _NReq) -> bool:
        if req.high:
            return False
        ok = self.lib.mkio_cancel(req.id) == 1
        if ok:
            req.fin = -1
        return ok


class ExpertStore:
    # Let macOS's file cache keep evicted experts (a second-level cache: a "miss" becomes a memory
    # copy). Peer projects measured +22% on a 24 GB M4 with it on and a smaller slab.
    PAGE_CACHE = os.environ.get("DEEPINFER_PAGE_CACHE", "0") == "1"
    IO_SPLIT = int(os.environ.get("DEEPINFER_IO_SPLIT", "1"))  # parallel reads per expert record
    PRIO_IO = os.environ.get("DEEPINFER_PRIO_IO", "1") == "1"  # chunked two-priority scheduler (PrioIO)
    NATIVE_IO = os.environ.get("DEEPINFER_NATIVE_IO", "1") == "1"  # ... implemented in C (native/mkio.c)
    HOT_WAIT = os.environ.get("DEEPINFER_HOT_WAIT", "1") == "1"  # keep the GPU clocked up while waiting for reads

    def __init__(self, root: Path, budget_bytes: int, threads: int = 8):
        root = Path(root)
        meta = json.loads((root / "layout.json").read_text())["experts"]
        self.layouts = {
            k: F.ExpertLayout(v["bits_w1"], v["bits_w3"], v["bits_w2"], v.get("dim", F.DIM), v.get("inter", F.INTER))
            for k, v in meta.items()
        }
        self.paths = {k: root / "experts" / f"{k}.bin" for k in self.layouts}
        self.fds: dict[str, int] = {}
        self.budget = budget_bytes
        # one slab per distinct layout, sized by its share of the experts' bytes
        groups: dict[F.ExpertLayout, list[str]] = {}
        for k, lay in self.layouts.items():
            groups.setdefault(lay, []).append(k)
        total = sum(len(ks) * lay.stride for lay, ks in groups.items())
        self.slabs: dict[str, Slab] = {}
        for lay, ks in groups.items():
            slots = max(96, int(budget_bytes * len(ks) * lay.stride / total) // lay.stride)
            slab = Slab(lay, slots)
            for k in ks:
                self.slabs[k] = slab
        self.used = sum(s.slots * s.layout.stride for s in set(self.slabs.values()))
        self.inflight: dict[tuple[str, int], Future] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="expert-io")
        self.prio = None
        if self.PRIO_IO:
            self.prio = NativeIO(threads) if (self.NATIVE_IO and _mkio_lib() is not None) else PrioIO(threads)
        self._addr = {}  # slab id -> per-part base addresses (native IO)
        self.spec_pool = ThreadPoolExecutor(1, thread_name_prefix="expert-spec")
        self._pad = threading.local()
        self.stats = {"hits": 0, "misses": 0, "prefetch_hits": 0, "bytes": 0, "spec_bytes": 0, "wait_s": 0.0}
        self._prefetched: set[tuple[str, int]] = set()
        self._spec_inflight: dict[str, set[int]] = {}
        self.trace_keys: dict[str, int] = {}  # layer key -> id in IO trace tags (NativeIO tracing only)

    # -- IO

    def _fd(self, key: str) -> int:
        fd = self.fds.get(key)
        if fd is None:
            fd = os.open(self.paths[key], os.O_RDONLY)
            if not self.PAGE_CACHE:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[key] = fd
        return fd

    def _iov_addrs(self, slab, slot: int, lay):
        """Raw addresses / lengths of a slot's record parts (+ alignment padding scratch)."""
        import numpy as np

        base = self._addr.get(id(slab))
        if base is None:
            base = [np.frombuffer(slab.mem[n], dtype=np.uint8).ctypes.data for n in slab.order]
            self._addr[id(slab)] = base
        addrs = [b + slot * slab.nbytes[n] for b, n in zip(base, slab.order)]
        lens = [slab.nbytes[n] for n in slab.order]
        pad = lay.stride - lay.nbytes
        if pad:
            pb = self._padbuf(lay)
            if not hasattr(self, "_padaddr"):
                self._padaddr = {}
            if pad not in self._padaddr:
                import ctypes

                self._padaddr[pad] = ctypes.addressof((ctypes.c_char * pad).from_buffer(pb))
            addrs.append(self._padaddr[pad])
            lens.append(pad)
        return addrs, lens

    def _padbuf(self, lay) -> bytearray:
        """Scratch for a record's alignment padding (written by reads, never used)."""
        n = lay.stride - lay.nbytes
        buf = getattr(self, "_padbufs", {}).get(n)
        if buf is None:
            self._padbufs = getattr(self, "_padbufs", {})
            buf = self._padbufs[n] = bytearray(n)
        return buf

    def _read(self, key: str, e: int, slot: int) -> None:
        """IO thread: one preadv of the whole page-aligned record, scattered into the slab."""
        slab, lay = self.slabs[key], self.layouts[key]
        pad = lay.stride - lay.nbytes
        if getattr(self._pad, "buf", None) is None or len(self._pad.buf) < pad:
            self._pad.buf = bytearray(max(pad, 1))
        bufs = slab.iovecs(slot) + ([memoryview(self._pad.buf)[:pad]] if pad else [])
        off, want = e * lay.stride, lay.stride
        while want:
            n = os.preadv(self._fd(key), bufs, off)
            if n <= 0:
                raise IOError(f"read failed on {key} expert {e} at {off}")
            off, want = off + n, want - n
            while bufs and n >= len(bufs[0]):  # short read: drop what was filled, continue
                n -= len(bufs[0])
                bufs.pop(0)
            if bufs and n:
                bufs[0] = bufs[0][n:]

    def _read_part(self, key: str, e: int, slot: int, i: int, n: int) -> None:
        """Bytes [i/n, (i+1)/n) of a record (16 KB-aligned cut points), into the right iovecs."""
        slab, lay = self.slabs[key], self.layouts[key]
        stride = lay.stride
        a = (stride * i // n) // F.ALIGN * F.ALIGN
        b = stride if i == n - 1 else (stride * (i + 1) // n) // F.ALIGN * F.ALIGN
        if b <= a:
            return
        if getattr(self._pad, "buf", None) is None or len(self._pad.buf) < stride - lay.nbytes:
            self._pad.buf = bytearray(max(stride - lay.nbytes, 1))
        bufs = slab.iovecs(slot) + ([memoryview(self._pad.buf)[: stride - lay.nbytes]] if stride > lay.nbytes else [])
        sel, pos = [], 0  # the iovec slices covering [a, b)
        for v in bufs:
            lo, hi = max(a, pos), min(b, pos + len(v))
            if lo < hi:
                sel.append(v[lo - pos : hi - pos])
            pos += len(v)
        off, want = e * stride + a, b - a
        while want:
            got = os.preadv(self._fd(key), sel, off)
            if got <= 0:
                raise IOError(f"read failed on {key} expert {e}")
            off, want = off + got, want - got
            while sel and got >= len(sel[0]):
                got -= len(sel[0])
                sel.pop(0)
            if sel and got:
                sel[0] = sel[0][got:]

    # -- slots

    def _take_slot(self, slab: Slab, optional: bool, protect: set[int] = frozenset()) -> int | None:
        if slab.free:
            return slab.free.pop()
        pinned = slab.pins[0] | slab.pins[1] | slab.busy | protect
        for k, slot in slab.lru.items():  # oldest first
            if slot in pinned:
                continue
            fut = self.inflight.get(k)
            if fut is not None:
                if not fut.done():
                    continue  # a read still landing in this slot
                del self.inflight[k]  # finished read nobody asked for (a lookahead misprediction)
            del slab.lru[k]
            self._prefetched.discard(k)
            return slot
        if optional:
            return None
        raise MemoryError("expert cache too small for one layer's experts; raise --cache-gb")

    def prefetch(self, key: str, ids, optional: bool = False, cold: bool = False) -> None:
        """Start reads for experts not resident. `optional` (lookahead) never evicts pinned
        slots and silently skips when the slab is full. Experts of this same request that are
        already resident are never evicted to make room for the others. `cold` inserts at the
        LRU end that is evicted first, so a wrong guess displaces nothing that was used."""
        slab = self.slabs[key]
        ids = [int(e) for e in ids]
        with self.lock:
            protect = {slab.lru[(key, e)] for e in ids if (key, e) in slab.lru}
            for e in ids:
                k = (key, e)
                if k in slab.lru:
                    continue
                slot = self._take_slot(slab, optional, protect)
                if slot is None:
                    return
                slab.lru[k] = slot
                if cold:
                    slab.lru.move_to_end(k, last=False)
                protect.add(slot)
                if isinstance(self.prio, NativeIO):
                    lay = self.layouts[key]
                    addrs, lens = self._iov_addrs(slab, slot, lay)
                    tag = -1
                    if self.prio.tracing:  # layer key id, expert, speculative flag
                        kid = self.trace_keys.setdefault(key, len(self.trace_keys))
                        tag = (kid << 20) | (e << 1) | (1 if optional else 0)
                    self.inflight[k] = self.prio.submit_addrs(self._fd(key), addrs, lens, e * lay.stride, lay.stride,
                                                              high=not optional, tag=tag)
                elif self.prio is not None:
                    lay = self.layouts[key]
                    bufs = slab.iovecs(slot) + ([memoryview(self._padbuf(lay))] if lay.stride > lay.nbytes else [])
                    self.inflight[k] = self.prio.submit(self._fd(key), bufs, e * lay.stride, lay.stride, high=not optional)
                elif self.IO_SPLIT > 1:
                    parts = [self.pool.submit(self._read_part, key, e, slot, i, self.IO_SPLIT) for i in range(self.IO_SPLIT)]
                    self.inflight[k] = self.pool.submit(lambda ps: [p.result() for p in ps], parts)
                else:
                    self.inflight[k] = self.pool.submit(self._read, key, e, slot)
                self.stats["bytes"] += self.layouts[key].nbytes
                if optional:
                    self.stats["spec_bytes"] += self.layouts[key].nbytes
                if optional:
                    self._prefetched.add(k)
                    if self.prio is not None:
                        self._spec_inflight.setdefault(key, set()).add(e)

    SPEC_WAIT = True  # ablation switches: issue guesses only after the demand reads land ...
    SPEC_COLD = True  # ... and insert them at the cold end of the LRU

    def speculate(self, key: str, ids, after) -> None:
        """Idle-window speculation: read `ids` of layer `key` only once the reads in `after`
        (the current layer's demand misses) have landed, i.e. while the GPU computes and the SSD
        would otherwise be idle; inserted cold, so mispredictions never evict a used expert."""
        futs = [f for f in after if f is not None] if self.SPEC_WAIT else []
        if not self.SPEC_WAIT or self.prio is not None:  # PrioIO: guesses are low priority already
            self.prefetch(key, ids, optional=True, cold=self.SPEC_COLD)
            return

        def go():
            for f in futs:
                try:
                    f.result()
                except Exception:  # noqa: BLE001 - the demand path reports read errors
                    pass
            self.prefetch(key, ids, optional=True, cold=self.SPEC_COLD)

        self.spec_pool.submit(go)

    def promote(self, key: str, ids) -> None:
        """Needed experts whose speculative reads are still in flight jump the demand queue now
        (instead of when the decode thread starts waiting for them)."""
        if self.prio is None:
            return
        with self.lock:
            for e in ids:
                fut = self.inflight.get((key, int(e)))
                if fut is not None and not getattr(fut, "high", True) and not fut.done():
                    self.prio.promote(fut)

    def resident(self, key: str, e: int) -> bool:
        k = (key, int(e))
        fut = self.inflight.get(k)
        return k in self.slabs[key].lru and (fut is None or fut.done())

    def resident_ids(self, key: str) -> list[int]:
        """Experts of layer `key` whose data is in the slab now (no read in flight)."""
        slab = self.slabs[key]
        with self.lock:
            return [e for (k, e) in slab.lru if k == key and ((k, e) not in self.inflight or self.inflight[(k, e)].done())]

    def cancel_unneeded(self, key: str, ids) -> None:
        """Layer `key` has routed: its speculative reads that were not selected and have not started
        are dropped (slot freed), so the SSD does not spend bandwidth on resolved guesses."""
        pend = self._spec_inflight.get(key)
        if not pend or self.prio is None:
            return
        keep = set(int(e) for e in ids)
        slab = self.slabs[key]
        with self.lock:
            for e in list(pend):
                k = (key, e)
                fut = self.inflight.get(k)
                if fut is None or fut.done():
                    pend.discard(e)
                    continue
                if e in keep:
                    continue
                if self.prio.cancel(fut):
                    pend.discard(e)
                    del self.inflight[k]
                    slot = slab.lru.pop(k, None)
                    if slot is not None:
                        slab.free.append(slot)
                    self._prefetched.discard(k)
                    self.stats["bytes"] -= self.layouts[key].nbytes
                    self.stats["spec_bytes"] -= self.layouts[key].nbytes
                    self.stats["spec_cancelled"] = self.stats.get("spec_cancelled", 0) + 1

    def get_slots(self, key: str, ids) -> tuple[Slab, list[int]]:
        """Make `ids` resident; return their slab and slots (blocking until loaded)."""
        slab, slots, finish = self.reserve_slots(key, ids)
        finish()
        return slab, slots

    def reserve_slots(self, key: str, ids):
        """get_slots in two halves: start (or keep) the reads of `ids` and return (slab, slots, finish)
        right away -- a slot is fixed when its read is issued and stays reserved (busy) until finish() --
        so the caller can build the graph that reads those slots while the data lands; finish() blocks
        until it has (hot-waiting) and releases the reservation into the usual two-call pin."""
        slab = self.slabs[key]
        ids = [int(e) for e in ids]
        with self.lock:
            for e in ids:
                k = (key, e)
                fut = self.inflight.get(k)
                if k in slab.lru and (fut is None or fut.done()):
                    self.stats["prefetch_hits" if k in self._prefetched else "hits"] += 1
                else:
                    self.stats["misses"] += 1
                self._prefetched.discard(k)
        self.prefetch(key, ids)
        with self.lock:
            slots = [slab.lru[(key, e)] for e in ids]
            mine = set(slots)
            slab.busy |= mine

        def finish():
            for e in ids:
                k = (key, e)
                fut = self.inflight.get(k)
                if fut is not None:
                    t = time.perf_counter()
                    if self.HOT_WAIT and not fut.done():
                        if hasattr(fut, "high") and not fut.high:
                            self.prio.promote(fut)
                        from .model import mk

                        mk.hot_wait(fut)
                    else:
                        fut.result()
                    self.stats["wait_s"] += time.perf_counter() - t
                    with self.lock:
                        self.inflight.pop(k, None)
                with self.lock:
                    slab.lru.move_to_end(k)
            with self.lock:
                slab.pins = [slab.pins[1], set(slots)]
                slab.busy -= mine

        return slab, slots, finish

    def all_done(self, key: str, ids) -> bool:
        """No read of `ids` (layer `key`) is still in flight."""
        with self.lock:
            return all((f := self.inflight.get((key, int(e)))) is None or f.done() for e in ids)

    def get(self, key: str, ids) -> list[ExpertWeights]:
        """Per-expert views (tests, benchmarks); the model uses get_slots + gather_qmm."""
        slab, slots = self.get_slots(key, ids)
        out = []
        for s in slots:
            out.append(ExpertWeights(*[
                (q[s], sc[s], b[s], bits, "affine") for q, sc, b, bits in (slab.proj("w1"), slab.proj("w3"), slab.proj("w2"))
            ]))
        return out

    def hit_rate(self) -> float:
        s = self.stats
        n = s["hits"] + s["misses"] + s["prefetch_hits"]
        return (s["hits"] + s["prefetch_hits"]) / n if n else 0.0
